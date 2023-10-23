import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle

from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *
import logging
import json
import time

def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset) 
    prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_ratio * num_classes)
    train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers)
    test_loader = load.dataloader(args.dataset, args.test_batch_size, False, args.workers)

    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    if args.model_class == 'default':
        model = load.model(args.model, args.model_class)(input_shape, 
                                                        num_classes, 
                                                        args.dense_classifier, 
                                                        args.pretrained,
                                                        args.n_layers,
                                                        args.n_neurons).to(device)
    else:
        model = load.model(args.model, args.model_class)(input_shape, 
                                                        num_classes, 
                                                        args.dense_classifier, 
                                                        args.pretrained).to(device)
    

    if args.reinitialize:
        if args.reinit_mode != None:
            model._initialize_weights(args.reinit_mode)
        else:
            model._initialize_weights()

    # model = nn.DataParallel(model)
    
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)


    ## Pre-Train ##
    print('Pre-Train for {} epochs.'.format(args.pre_epochs))
    pre_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                 test_loader, device, args.pre_epochs, args.verbose)
    file_path = f'./Reproduced_Results/Masks/{args.dataset}_{args.model}/{args.init_type}/pre_epoch_{args.pre_epochs}/compression_{int(args.compression*100)}/{args.pruner}/seed_{args.seed}'
        
    # If load pre-found mask
    if args.load_mask:
        file_path = f'./Reproduced_Results/Masks/{args.dataset}_{args.model}/{args.init_type}/pre_epoch_{args.pre_epochs}/compression_{int(args.compression*100)}/{args.pruner}/seed_{args.seed}'
        if not os.path.exists(file_path):
            print('No existing mask')
            print('Go to pruning ')
        else: # Load mask
            print('Loading mask')
            with open(f'{file_path}/{args.pruner}_{int(args.compression*100)}.pkl', 'rb') as f:
                data = pickle.load(f)
            masks = data['mask']
            i = 0
            for name, mask in model.named_buffers():
                if 'weight_mask' in name and len(mask.shape) in [2,4]:
                    print(f'Load {name}')
                    # print(mask.shape)
                    # print(masks[i].shape)
                    mask.data.copy_(torch.tensor(masks[i]))
                    i += 1
    else:
        ## Prune ##
        if args.pruner == 'npb':
            print('Optimization pruner for Node-Path Balancing Principle')
            from Pruners.pruners import Optimization_Pruner as npb
            sparsity = 10**(-float(args.compression))

            scale_weight = None
            # Ablation studies for different weight initialization for mask variable
            if args.is_scale_weight is not None:    
                cloned_model = copy.deepcopy(model)
                pruner = load.pruner(args.scale_weight_method)(generator.masked_parameters(cloned_model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
                scale_weight = get_scores(cloned_model, loss, pruner, prune_loader, device, sparsity, 
                                        args.compression_schedule, args.mask_scope, 
                                        args.prune_epochs, args.prune_train_mode, args)
                del cloned_model

            pruner = npb(initializer=args.init_mode, alpha=args.alpha, 
                        beta=args.beta, sparsity=sparsity, 
                        device=device, max_param_per_kernel=args.max_p,
                        min_param_to_node=args.min_n,
                        is_scale_weight=args.is_scale_weight, 
                        chunk_size=args.chunk_size,
                        node_constraint=args.is_node_constraint,
                        loss=loss, dataloader=prune_loader,
                        scale_weight=scale_weight)
            

            file_name = None
            if args.store_mask:
                file_path = f'./Reproduced_Results/Masks/{args.dataset}_{args.model}/{args.init_type}/pre_epoch_{args.pre_epochs}/compression_{int(args.compression*100)}/{args.pruner}/seed_{args.seed}'
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                file_name = f'{file_path}/{args.pruner}_{args.init_mode}_{int(args.compression*100)}_{args.chunk_size}_{args.alpha}_{args.beta}_{args.max_p}.pkl'
            
            pruning_start_time = time.time()
            model = pruner.prune(model, input_shape, args.store_mask, file_name)
            pruning_end_time = time.time()
            pruning_time = pruning_end_time - pruning_start_time
            print("The pruning time is: \t", pruning_time)

            # if args.reinitialize:
            #     if args.reinit_mode != None:
            #         model._initialize_weights(args.reinit_mode)
            #     else:
            #         model._initialize_weights()
            if args.shuffle:
                print('Shuffling mask')
                tmp = None
                with torch.no_grad():
                    for name, mask in model.named_buffers():
                        if 'weight' in name and len(mask.shape) in [2,4]:
                            shape = mask.shape
                            gen = torch.Generator()
                            N = mask.nelement()
                            if args.shuffle_seed != 1: gen.manual_seed(args.shuffle_seed)
                            perm = torch.normal(torch.zeros(N), torch.ones(N), generator=gen)
                            perm = torch.argsort(perm)
                            tmp = mask.reshape(-1)[perm].reshape(shape)
                            mask.data.copy_(tmp)

        elif args.pruner == 'rand_erk':
            print('Random pruning with ERK initialization')
            from Pruners.pruners import Random_ERK as rand_erk
            sparsity = 10**(-float(args.compression))
            pruner = rand_erk(initializer=args.init_mode, device=device, sparsity=sparsity) 
            model = pruner.prune(model, input_shape)


        else:
            print('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs))
            pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
            # args.compression = 10**(arg.compression)
            sparsity = 10**(-float(args.compression))
            pruning_start_time = time.time()
            prune_loop(model, loss, pruner, prune_loader, device, sparsity, 
                    args.compression_schedule, args.mask_scope, 
                    args.prune_epochs, args.reinitialize, args.prune_train_mode, 
                    args.shuffle, args.invert, args.store_mask, args.pruner, 
                    args.compression, args.dataset, args)
            pruning_end_time = time.time()
            pruning_time = pruning_end_time - pruning_start_time
            print("The pruning time is: \t", pruning_time)
        
        prune_result = metrics.summary_flop_only(model, 
                                    metrics.flop(model, input_shape, device),
                                    lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
        total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
        possible_params = prune_result['size'].sum()
        total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
        possible_flops = prune_result['flops'].sum()
        print(prune_result)
        print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
        print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))

        saved_pruning_time_file_path = f'./Reproduced_Results/Masks/{args.dataset}_{args.model}/compression{int(args.compression*100)}/'
        if not os.path.exists(saved_pruning_time_file_path):
            os.makedirs(saved_pruning_time_file_path)
        if args.pruner == 'snip' and args.prune_epochs == 100:
            pruner_name = 'iter_snip'
        elif args.pruner == 'npb':
            pruner_name = args.pruner + '_' + args.init_mode
        else:
            pruner_name = args.pruner   
        with open(f'{saved_pruning_time_file_path}/{pruner_name}_{int(args.compression*100)}_pruning_time_and_flop.txt', 'w') as f:
            f.write(f'The pruning time is: \t {pruning_time}\n')
            f.write("FLOP Sparsity: {}/{} ({:.4f})\n".format(total_flops, possible_flops, total_flops / possible_flops))
            f.write(prune_result.to_string().replace('\n', '\n\t'))

    if args.save_model:
        saved_model_file_path = f'./Reproduced_Results/Masks/{args.dataset}_{args.model}/compression{int(args.compression*100)}/'
        if not os.path.exists(saved_model_file_path):
            os.makedirs(saved_model_file_path)
        lip_before_train = metrics.naive_lip_l2(model, test_loader)
        torch.save(model.state_dict(),"{}/init_model_{}_{}_alpha_{}_beta_{}.pt".format(saved_model_file_path, args.pruner, int(args.compression*100), args.alpha, args.beta))
    
    df, eff_paths, eff_neurons, eff_params, unpruned_params = metrics.measure_node_path(model, input_shape)
    print(f'The number of effective nodes is:\t {eff_neurons}')
    print(f'The number of effective paths is:\t {eff_paths}')
    print(f'The number of effective params is:\t {eff_params}')
    print(f'The number of unpruned params is:\t {unpruned_params}')
    print(f'')
    all_params = sum([p.numel() for p in model.parameters()])
    saved_params = sum([mask.sum().item() for mask in model.buffers()])
    print(f'All parameters are {all_params}')
    print(f'Saved parameters is {saved_params}')
    
    ## Post-Train ##
    print('Post-Training for {} epochs.'.format(args.post_epochs))
    if args.is_addin:
        post_result = random_train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                  test_loader, device, args.post_epochs, args.verbose, args)
    else:
        post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                    test_loader, device, args.post_epochs, args.verbose, args.wandb) 

    ## Display Results ##
    try:
        frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
        train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])
        prune_result = metrics.summary(model, 
                                    pruner.scores,
                                    metrics.flop(model, input_shape, device),
                                    lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
        total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
        possible_params = prune_result['size'].sum()
        total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
        possible_flops = prune_result['flops'].sum()
        print("Train results:\n", train_result)
        print("Prune results:\n", prune_result)
        print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
        print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))
    except:
        pass

    if args.save_result:
        pruner_name = args.pruner
        if args.shuffle:
            pruner_name = 'shuffled_' + args.pruner
        if args.pruner in ['grasp', 'snip'] and args.prune_epochs == 100:
            pruner_name = 'iterative_' + pruner_name
        if args.pruner == 'synflow' and args.prune_epochs == 1:
            pruner_name = 'oneshot_' + pruner_name
        if args.pruner == 'impsynflow':
            pruner_name = f'impsynflow_{args.delta_t}_{args.max_t}'
        # if args.seed != 1:
        #     pruner_name = pruner_name + f'_seed_{args.seed}'

        if args.shuffle:
            if args.model != 'fc':
                file_path = f'./Reproduced_Results/Masks/{args.dataset}_{args.model}/shuffling/pre_epoch_{args.pre_epochs}/compression_{int(args.compression*100)}/{pruner_name}/seed_{args.shuffle_seed}'
            else:
                file_path = f'./Reproduced_Results/Masks/{args.dataset}_{args.model}/shuffling/pre_epoch_{args.pre_epochs}/MLP_{args.n_layers}_layers_{args.n_neurons}/compression_{int(args.compression*100)}/{pruner_name}/seed_{args.seed}'

        else:
            if args.model != 'fc':
                file_path = f'./Reproduced_Results/Masks/{args.dataset}_{args.model}/{args.init_type}/pre_epoch_{args.pre_epochs}/compression_{int(args.compression*100)}/{pruner_name}/seed_{args.seed}'
            else:
                file_path = f'./Reproduced_Results/Masks/{args.dataset}_{args.model}/{args.init_type}/pre_epoch_{args.pre_epochs}/MLP_{args.n_layers}_layers_{args.n_neurons}/compression_{int(args.compression*100)}/{pruner_name}/seed_{args.seed}'
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if args.pruner == 'npb':
            file_name = f'{file_path}/{pruner_name}_{args.init_mode}_{int(args.compression*100)}_{args.chunk_size}_{args.alpha}_{args.beta}_{args.max_p}_performance.log'
        else:    
            file_name = f'{file_path}/{pruner_name}_{int(args.compression*100)}_performance.log'
        
        if os.path.exists(file_name):
            os.remove(file_name)

        logging.basicConfig(filename=file_name, format='%(asctime)s %(message)s', filemode='a')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        logger.info('The number of effective nodes is:\t {}'.format(eff_neurons))
        logger.info('The number of effective paths is:\t {}'.format(eff_paths))
        logger.info('The number of effective params is:\t{}'.format(eff_params))
        logger.info('The number of unpruned params is:\t{}'.format(unpruned_params))
        all_params = sum(p.numel() for p in model.parameters())
        all_unpruned_params = sum([mask.sum().item() for mask in model.buffers()])
        logger.info("Effective Sparsity is: {}/{} ({:.4f})".format(eff_params, all_params, eff_params / all_params))
        logger.info("Actual Sparsity is: {}/{} ({:.4f})".format(all_unpruned_params, all_params, all_unpruned_params / all_params))
        print("Effective Sparsity is: {}/{} ({:.4f})".format(eff_params, all_params, eff_params / all_params))
        print("Actual Sparsity is: {}/{} ({:.4f})".format(all_unpruned_params, all_params, all_unpruned_params / all_params))
        
        if args.wandb is not None:
            args.wandb.log({
                'Effective nodes': eff_neurons,
                'Effective paths': eff_paths,
                'Effective params': eff_params,
                'Effective Sparsity': eff_params/all_params,
                'Actual Sparsity': unpruned_params/all_params
            })

        try:
            logger.info('Train results:\n {}'.format(train_result.to_string().replace('\n', '\n\t')))
            logger.info('Prune results:\n {}'.format(prune_result.to_string().replace('\n', '\n\t')))
            logger.info("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
            logger.info("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))
        except:
            pass

    print(f'The number of effective nodes is:\t {eff_neurons}')
    print(f'The number of effective paths is:\t {eff_paths}')
    print(f'The number of effective params is:\t {eff_params}')
    print(f'The number of unpruned params is:\t {unpruned_params}')
    ## Save Results and Model ##
    if args.save_model and args.post_epochs != 0:
        print('Saving results.')
        # pre_result.to_pickle("{}/pre-train.pkl".format(args.result_dir))
        # post_result.to_pickle("{}/post-train.pkl".format(args.result_dir))
        # prune_result.to_pickle("{}/compression.pkl".format(args.result_dir))
        # torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
        # torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
        # torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))
        # pre_result.to_pickle("{}/pre-train.pkl".format(file_path))
        # post_result.to_pickle("{}/post-train.pkl".format(file_path))
        # prune_result.to_pickle("{}/compression.pkl".format(file_path))
        lip_after_train = metrics.naive_lip_l2(model, test_loader)
        torch.save(model.state_dict(),"{}/model_{}_{}_alpha_{}_beta_{}.pt".format(saved_model_file_path, args.pruner, int(args.compression*100), args.alpha, args.beta))
        # torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(file_path))
        # torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(file_path))
        lip_res = {'Lipschitz Before Training': str(lip_before_train), "Lipschitz After Training": str(lip_after_train)}
        print(lip_res)
        with open(f'{saved_model_file_path}/lipschitz_result_model_{args.pruner}_{int(args.compression*100)}_alpha_{args.alpha}_beta_{args.beta}.json', 'w') as f:
            json.dump(lip_res, f)

