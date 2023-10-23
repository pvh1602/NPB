import torch
import numpy as np

class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # # Set score for masked parameters to -inf 
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)] 
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))
    
    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)


    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def perm(self, N, seed: int = None):
        """Generate a tensor with the numbers 0 through N-1 ordered randomly."""

        gen = torch.Generator()
        if seed is not None: gen.manual_seed(seed)
        perm = torch.normal(torch.zeros(N), torch.ones(N), generator=gen)
        return torch.argsort(perm)

    def shuffle(self, seed=None):
        tmp = None
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = self.perm(mask.nelement(), seed)
            tmp = mask.reshape(-1)[perm].reshape(shape)
            mask.copy_(tmp)
        
    def invert(self):
        for v in self.scores.values():
            v.div_(v**2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0 
        for mask, _ in self.masked_parameters:
             remaining_params += mask.detach().cpu().numpy().sum()
             total_params += mask.numel()
        return remaining_params, total_params

    def get_layer_sparsity_dict(self, sparsity, layer_names, scope='global'):
        """
        Use global pruning and then return the sparsity level of each layer
        """        
        sparsity_dict = {}
        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            idx = 0
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)] 
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                m = torch.where(score <= threshold, zero, one)
                sparsity = 1 - m.sum() / mask.numel()
                sparsity_dict[layer_names[idx]] = sparsity
                idx += 1
        return sparsity_dict

class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)
    
    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False
        
        # for i, v in enumerate(self.scores.values()):
        #     print(f'norm of layer {i+1} is {torch.norm(v)}')

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device):

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=True)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            
            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()
        
        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device, imp=False):
        
        if imp:
            for _, p in self.masked_parameters:
                self.scores[id(p)] = torch.randn_like(p)
            return 
            
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0,:].shape)
        input = torch.ones([1] + input_dim).to(device)#, dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()
        
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)


from Pruners.utils import *
# class Improve_Synflow(Pruner)

class Random_ERK():
    def __init__(self, initializer: str, device, sparsity) -> None:
        self.initializer = initializer
        self.device = device
        self.sparsity = sparsity

    def prune(self, net, input_shape):
        cloned_net = copynet(net)
        sparsity_dict = ERK_sparsify(cloned_net, sparsity=1-self.sparsity)
        for name, mask in cloned_net.named_buffers():
            if 'weight' in name and len(mask.shape) in [2,4]:
                mask.data.copy_((torch.rand(mask.shape) > sparsity_dict[name]).int().data.to(self.device))
        
        for (n, m), (name, mask) in zip(net.named_buffers(), cloned_net.named_buffers()):
            m.copy_(mask)

        return net

import time
class Optimization_Pruner():
    def __init__(self, initializer: str,  alpha: float, beta: float, 
                sparsity: float, device, max_param_per_kernel=None, 
                min_param_to_node=None, is_scale_weight=False,
                chunk_size=32, node_constraint=False,
                loss=None, dataloader=None, scale_weight=None) -> None:
        
        self.initializer = initializer
        self.alpha = alpha
        self.beta = beta
        self.sparsity = sparsity
        self.device = device 
        self.max_param_per_kernel = max_param_per_kernel
        self.min_param_to_node = min_param_to_node
        self.is_scale_weight = is_scale_weight
        self.chunk_size = chunk_size
        self.node_constraint = node_constraint
        self.loss = loss
        self.dataloader = dataloader
        self.scale_weight = scale_weight

    def prune(self, net, input_shape, is_store_mask=False, file_name=None):
        layer_id = 0
        estimate_time = 0
        c, h, w = input_shape

        cloned_net = copynet(net)
        cloned_net.double()
        input_ = torch.ones((1,c,h,w)).double()
        prev = input_
        cloned_net.cpu()

        is_resnet20 = False
        if cloned_net.__class__.__name__ == 'ResNet' and h == 32:
            is_resnet20 = True

        saved_params = {}
        i = 0
        for name, param in cloned_net.named_parameters():
            if 'weight' in name and len(param.shape) in [2,4]:
                name = name + '_mask'
                if self.is_scale_weight:
                    saved_params[name] = self.scale_weight[i]   
                else:
                    saved_params[name] = None
                i += 1

        for param in cloned_net.parameters():
            param.data.copy_(torch.ones_like(param))
        if self.initializer == 'ERK': 
            sparsity_dict = ERK_sparsify(cloned_net, sparsity=1-self.sparsity)
        elif self.initializer == 'uniform':
            sparsity_dict = uniform_sparsify(cloned_net, sparsity=1-self.sparsity)
        elif self.initializer == 'ERK+':
            sparsity_dict = ERK_plus_sparsify(cloned_net, sparsity=1-self.sparsity)
        elif self.initializer in ['mag', 'snip', 'grasp', 'rand']:
            sparsity_dict = pai_sparsify(net, self.initializer, sparsity=self.sparsity,
                                        loss=self.loss, dataloader=self.dataloader, device=self.device)
        else:
            raise ValueError('Wrong initializer in args.init_mode argument')
        for name, mask in cloned_net.named_buffers():
            # print(name)
            
            if 'weight_mask' in name and len(mask.shape) == 4 and 'bn' not in name:
                print(f'Considering layer {layer_id} with name {name}')
                # print(f'input vector is {torch.sum(prev[0], dim=(1,2))}')
                # print(f'sparsity at layer {name} is {sparsity_dict[name]}')

                if layer_id == 0: # Input layer
                    start_time = time.time()
                    new_mask = optimize_layerwise(mask, prev[0], sparsity=sparsity_dict[name], 
                                                alpha=self.alpha, beta=self.beta, 
                                                max_param_per_kernel=None,
                                                init_weight=saved_params[name])
                    # break
                    mask.copy_(new_mask)
                    estimate_time = estimate_time + time.time() - start_time
                else:
                    # pass
                    c_out, c_in, _, _ = mask.shape
                    if (c_out*c_in > 128*128) or (is_resnet20 and c_out*c_in > 64*32):      # Using Chunking
                        n_chunks = int(c_out / self.chunk_size)
                        new_mask = copy.deepcopy(mask)
                        chunked_masks = []
                        for idx in range(n_chunks):
                            start_time = time.time()
                            start_c_out = idx * self.chunk_size
                            end_c_out = (idx+1) * self.chunk_size
                            # print(f'Consider C_out from {start_c_out} to {end_c_out}')
                            chunked_mask = copy.deepcopy(new_mask[start_c_out:end_c_out, :, :, :])
                            chunked_sparsity = sparsity_dict[name]
                            if self.is_scale_weight:
                                chunked_init_weight = saved_params[name][start_c_out:end_c_out, :, :, :]
                            else:
                                chunked_init_weight = None
                            if 'shortcut' in name:
                                # print(name)
                                chunked_mask = optimize_layerwise(chunked_mask, prev[0], sparsity=sparsity_dict[name], 
                                            alpha=self.alpha, init_weight=chunked_init_weight, node_constraint=self.node_constraint)
                            else:
                                # pass
                                chunked_mask = optimize_layerwise(chunked_mask, prev[0], sparsity=chunked_sparsity, 
                                    alpha=self.alpha, beta=self.beta, 
                                    max_param_per_kernel=self.max_param_per_kernel,
                                    min_param_to_node=self.min_param_to_node, 
                                    init_weight=chunked_init_weight, 
                                    node_constraint=self.node_constraint)
                            mask[start_c_out:end_c_out, :, :, :].copy_(chunked_mask)
                            end_time = time.time()
                            # break
                            # print(f'Chunk mask sum {chunked_mask.sum()}')
                            # print(f'mask sum {mask.sum()} and numel {mask.numel()}')
                        estimate_time = estimate_time + end_time - start_time + 10
                        
                    else:   # small size
                        start_time = time.time()
                        if 'shortcut' in name:
                            # pass
                            mask.copy_(optimize_layerwise(mask, prev[0], sparsity=sparsity_dict[name], 
                                        alpha=self.alpha, init_weight=saved_params[name], 
                                        node_constraint=self.node_constraint))
                        else:
                            # pass
                            mask.copy_(optimize_layerwise(mask, prev[0], sparsity=sparsity_dict[name], 
                                        alpha=self.alpha, beta=self.beta, 
                                        max_param_per_kernel=self.max_param_per_kernel,
                                        min_param_to_node=self.min_param_to_node, init_weight=saved_params[name], 
                                        node_constraint=self.node_constraint))
                        end_time = time.time()
                        estimate_time = estimate_time + end_time - start_time
                layer_id += 1
                actual_sparsity = 1 - mask.sum().item() / mask.numel()
                print(f'Desired sparsity is {sparsity_dict[name]} and optimizer finds sparsity is {actual_sparsity}')

                if 'shortcut' in name:
                    if self.max_param_per_kernel > 5:
                        self.max_param_per_kernel -= 2
                        
            elif 'weight_mask' in name and len(mask.shape) == 2:    # Linear layer
                print(f'Considering layer {layer_id} with name {name}')
                start_time = time.time()
                f_out, f_in = mask.shape
                if (f_out*f_in > 512*10):
                    n_chunks = int(f_out / 10)
                    new_mask = copy.deepcopy(mask)
                    for idx in range(n_chunks):
                        start_f_out = idx * 10
                        end_f_out = (idx+1) * 10
                        # print(f'Consider C_out from {start_f_out} to {end_f_out}')
                        chunked_mask = copy.deepcopy(new_mask[start_f_out:end_f_out, :])
                        chunked_sparsity = sparsity_dict[name]
                        if self.is_scale_weight:
                            chunked_init_weight = saved_params[name][start_f_out:end_f_out, :]
                        else:
                            chunked_init_weight = None
                        chunked_mask = optimize_layerwise(chunked_mask, prev[0], sparsity=chunked_sparsity, 
                                    alpha=self.alpha, beta=0, init_weight=chunked_init_weight)

                        mask[start_f_out:end_f_out, :].copy_(chunked_mask)
                else:
                    mask.copy_(optimize_layerwise(mask, prev[0], sparsity=sparsity_dict[name], 
                                                    alpha=self.alpha, beta=0, 
                                                    init_weight=saved_params[name]))
                layer_id += 1
                actual_sparsity = 1 - mask.sum().item() / mask.numel()
                end_time = time.time()
                estimate_time = estimate_time + end_time - start_time
                print(f'Desired sparsity is {sparsity_dict[name]} and optimizer finds sparsity is {actual_sparsity}')

            else:
                # print(f'Ignore layer {name}')
                continue

            
            
            inter_inputs = get_intermediate_inputs(cloned_net, input_)
            try:
                if cloned_net.__class__.__name__ == 'VGG':
                    if layer_id in [1, 4, 9, 14]:
                        if layer_id != 1:
                            self.max_param_per_kernel -= 2
                        layer_id += 1
                prev = inter_inputs[layer_id][0].detach().requires_grad_(False)
            except:
                print('Done Pruning!')
        

        cloned_net = fine_tune_mask(cloned_net, input_shape)
        count_ineff_param(cloned_net, input_shape)

        cloned_net.float()

        # Copy mask
        cloned_net.to(self.device)
        for (n, m), (name, mask) in zip(net.named_buffers(), cloned_net.named_buffers()):
            m.copy_(mask)

        print("Estimate pruning time is: ", estimate_time)

        if is_store_mask:
            if file_name is not None:
                try:
                    store_mask(net, file_name)
                except:
                    raise RuntimeError('There is something wrong with store mask function!')
            else:
                print('No store mask file name')
        return net
            

