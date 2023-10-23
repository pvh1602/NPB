import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from prune import * 
from Layers import layers

def summary(model, scores, flops, prunable):
    r"""Summary of compression results for a model.
    """
    rows = []
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            pruned = prunable(module) and id(param) in scores.keys()
            if pruned:
                sparsity = getattr(module, pname+'_mask').detach().cpu().numpy().mean()
                score = scores[id(param)].detach().cpu().numpy()
            else:
                sparsity = 1.0
                score = np.zeros(1)
            shape = param.detach().cpu().numpy().shape
            flop = flops[name][pname]
            score_mean = score.mean()
            score_var = score.var()
            score_sum = score.sum()
            score_abs_mean = np.abs(score).mean()
            score_abs_var  = np.abs(score).var()
            score_abs_sum  = np.abs(score).sum()
            rows.append([name, pname, sparsity, np.prod(shape), shape, flop,
                         score_mean, score_var, score_sum, 
                         score_abs_mean, score_abs_var, score_abs_sum, 
                         pruned])

    columns = ['module', 'param', 'sparsity', 'size', 'shape', 'flops', 'score mean', 'score variance', 
               'score sum', 'score abs mean', 'score abs variance', 'score abs sum', 'prunable']
    return pd.DataFrame(rows, columns=columns)

def summary_flop_only(model, flops, prunable):
    r"""Summary of compression results for a model.
    """
    rows = []
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            pruned = prunable(module) 
            if pruned:
                sparsity = getattr(module, pname+'_mask').detach().cpu().numpy().mean()
            else:
                sparsity = 1.0
                score = np.zeros(1)
            shape = param.detach().cpu().numpy().shape
            flop = flops[name][pname]
            rows.append([name, pname, sparsity, np.prod(shape), shape, flop, pruned])

    columns = ['module', 'param', 'sparsity', 'size', 'shape', 'flops', 'prunable']
    return pd.DataFrame(rows, columns=columns)

def flop(model, input_shape, device):

    total = {}
    def count_flops(name):
        def hook(module, input, output):
            flops = {}
            if isinstance(module, layers.Linear) or isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                flops['weight'] = in_features * out_features
                if module.bias is not None:
                    flops['bias'] = out_features
            if isinstance(module, layers.Conv2d) or isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = int(np.prod(module.kernel_size))
                output_size = output.size(2) * output.size(3)
                flops['weight'] = in_channels * out_channels * kernel_size * output_size
                if module.bias is not None:
                    flops['bias'] = out_channels * output_size
            if isinstance(module, layers.BatchNorm1d) or isinstance(module, nn.BatchNorm1d):
                if module.affine:
                    flops['weight'] = module.num_features
                    flops['bias'] = module.num_features
            if isinstance(module, layers.BatchNorm2d) or isinstance(module, nn.BatchNorm2d):
                output_size = output.size(2) * output.size(3)
                if module.affine:
                    flops['weight'] = module.num_features * output_size
                    flops['bias'] = module.num_features * output_size
            if isinstance(module, layers.Identity1d):
                flops['weight'] = module.num_features
            if isinstance(module, layers.Identity2d):
                output_size = output.size(2) * output.size(3)
                flops['weight'] = module.num_features * output_size
            total[name] = flops
        return hook
    
    for name, module in model.named_modules():
        module.register_forward_hook(count_flops(name))

    input = torch.ones([1] + list(input_shape)).to(device)
    model(input)

    return total


# def conservation(model, scores, batchnorm, residual):
#     r"""Summary of conservation results for a model.
#     """
#     rows = []
#     bias_flux = 0.0
#     mu = 0.0
#     for name, module in reversed(list(model.named_modules())):
#         if prunable(module, batchnorm, residual):
#             weight_flux = 0.0
#             for pname, param in module.named_parameters(recurse=False):
                
#                 # Get score
#                 score = scores[id(param)].detach().cpu().numpy()
                
#                 # Adjust batchnorm bias score for mean and variance
#                 if isinstance(module, (layers.Linear, layers.Conv2d)) and pname == "bias":
#                     bias = param.detach().cpu().numpy()
#                     score *= (bias - mu) / bias
#                     mu = 0.0
#                 if isinstance(module, (layers.BatchNorm1d, layers.BatchNorm2d)) and pname == "bias":
#                     mu = module.running_mean.detach().cpu().numpy()
                
#                 # Add flux
#                 if pname == "weight":
#                     weight_flux += score.sum()
#                 if pname == "bias":
#                     bias_flux += score.sum()
#             layer_flux = weight_flux
#             if not isinstance(module, (layers.Identity1d, layers.Identity2d)):
#                 layer_flux += bias_flux
#             rows.append([name, layer_flux])
#     columns = ['module', 'score flux']

#     return pd.DataFrame(rows, columns=columns)



from prettytable import PrettyTable
from Pruners.utils import *
def measure_node_path(model, input_shape):
    c, h, w = input_shape

    # Put input ones through the subnet and backward
    # net = copy.deepcopy(model)
    try:
        net = copynet(model)
        print('Use copynet')
    except:
        net = copy.deepcopy(model)
        print('Use copy.deepcopy')
    net = net.cpu().double()

    for p in net.parameters():
        p.data.copy_(torch.ones_like(p))

    x = torch.ones((1, c, h, w)).double()
    y = net(x)
    loss = y.sum()
    # loss
    loss.backward()

    ###########
    # EFFECTIVE PATHS
    ###########
    eff_paths = loss.item()

    # all_params = 0.
    # for name, mask in net.named_buffers():
    #     if 'weight_mask' in name and len(mask.shape) in [2,4]:
    #         all_params += torch.numel(mask)
    # all_params = all_params * 10**(-float(compression/100))
    # return all_params

    ###########
    # EFFECTIVE PARAMS
    ###########
    unpruned_params = 0.
    with torch.no_grad():
        for name, mask in net.named_buffers():
            if 'weight' in name and len(mask.shape) in [2, 4]:
                unpruned_params += mask.sum().item()

    eff_params = 0.
    masks = []
    for name, param in net.named_parameters():
        if 'weight' in name and len(param.shape) in [2, 4]:
            masks.append(None)
    with torch.no_grad():
        i = 0
        for name, param in net.named_parameters():
            # name format blocks.4.conv1.weight with shape [32,32,3,3]
            # or fc.weight with shape [10,64]
            # or blocks.6.shortcut.0.weight with shape [64,32,1,1]
            if 'weight' in name and len(param.shape) in [2, 4]:
                masks[i] = torch.where(param.grad.data != 0, 1, 0).numpy()
                eff_params += np.sum(masks[i])
                i += 1

    ###########
    # EFFECTIVE CHANNELS
    ###########
    # stat the effective neurons or channels
    
    table = PrettyTable(['Layer', '#Effective Channel'])
    table.align['Layer'] = 'c'
    df = {'Layer':[], '#Effective Channel':[]}
    i = 0   # index of stored mask
    all_n_e = 0 # all effective neurons
    layer_id = 0    # index of layer
    for name, param in net.named_parameters():
        # Iterate over all layers
        if 'weight' in name and len(param.shape) in [4]:  # Consider conv layers 
            # Ignore the shortcut layers
            if 'shortcut' in name:
                i += 1 
                continue
            
            sum_along_in_channels = np.sum(masks[i], axis=(0,2,3))
            n_e = np.where(sum_along_in_channels > 0, 1, 0).sum()
            all_n_e += n_e
            table.add_row([f'{layer_id}', n_e])
            df['Layer'].append(layer_id)
            df['#Effective Channel'].append(n_e)

            i += 1
            layer_id += 1

        if 'weight' in name and len(param.shape) in [2]:
            sum_along_in_features = np.sum(masks[i], axis=0)
            n_e = np.where(sum_along_in_features > 0, 1, 0).sum()
            all_n_e += n_e
            table.add_row([f'pre_output', n_e])
            # print('pre ', name, '\t', n_e)
            df['Layer'].append('Pre_Output')
            df['#Effective Channel'].append(n_e)

            sum_along_out_features = np.sum(masks[i], axis=1)
            n_e = np.where(sum_along_out_features > 0, 1, 0).sum()
            all_n_e += n_e
            table.add_row([f'output', n_e])
            # print(name, '\t', n_e)
            df['Layer'].append('Output')
            df['#Effective Channel'].append(n_e)
        else:
            continue

    eff_neurons = all_n_e

    data = table.get_string()
    print(data)
    # with open(f'{path}/count_effective_node.txt', 'w') as f:
    #     f.write(data)
    
    # df1 = pd.DataFrame.from_dict(df)
    # df1.to_csv(f'{path}/count_effective_node.csv', index=False)
    

    return df, eff_paths, eff_neurons, eff_params, unpruned_params


def naive_lip_l2(model, test_loader):
    model.eval()
    X = []
    Y = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, labels in test_loader:
            x = x.to(device)
            y = model(x)
            X.append(x)
            Y.append(y)
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    X = X.view(X.shape[0], -1)
    lip = -1
    for i in range(X.shape[0]):
        alpha = (X[i].unsqueeze(0) - X).norm(2, dim=-1)
        beta = (Y[i].unsqueeze(0) - Y).norm(2, dim=-1)
        sim = beta / alpha
        sim = sim.view(-1).cpu().numpy()
        sim = sim[sim!=np.Inf]
        lip = max(lip, np.nanmax(sim))
    return lip