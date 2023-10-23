
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import copy
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt 
import seaborn as sns
import types
import cvxpy as cp
import math
import pickle


from Layers.layers import *
from Utils import load
from Utils import generator

def ERK_sparsify(net, sparsity=0.9):
    print('initialize by ERK')
    density = 1 - sparsity
    erk_power_scale = 1


    total_params = 0
    for name, weight in net.named_buffers():
        if 'weight' in name and len(weight.shape) in [2,4]: 
            total_params += weight.numel()
    is_epsilon_valid = False

    dense_layers = set()
    while not is_epsilon_valid:

        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for name, mask in net.named_buffers():
            if 'weight' in name and len(mask.shape) in [2,4]:
                n_param = np.prod(mask.shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (
                                                        np.sum(mask.shape) / np.prod(mask.shape)
                                                ) ** erk_power_scale
                    divisor += raw_probabilities[name] * n_param
        epsilon = rhs / divisor
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    # print(f"Sparsity of var:{mask_name} had to be set to 0.")
                    dense_layers.add(mask_name)
        else:
            is_epsilon_valid = True

    sparsity_dict = {}
    total_nonzero = 0.0
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for name, mask in net.named_buffers():
        if 'weight' in name and len(mask.shape) in [2,4]:
            n_param = np.prod(mask.shape)
            if name in dense_layers:
                sparsity_dict[name] = 0
            else:
                probability_one = epsilon * raw_probabilities[name]
                sparsity_dict[name] = 1 - probability_one
            print(
                f"layer: {name}, shape: {mask.shape}, sparsity: {sparsity_dict[name]}"
            )
            total_nonzero += (1-sparsity_dict[name]) * mask.numel()
    print(f"Overall sparsity {1-total_nonzero / total_params}")
    return sparsity_dict
    #     mask.data.copy_((torch.rand(mask.shape) < density_dict[name]).float().data.cuda())

    #     total_nonzero += density_dict[name] * mask.numel()
    # print(f"Overall sparsity {total_nonzero / total_params}")


# This is adopted from https://github.com/VITA-Group/Random_Pruning
def ERK_plus_sparsify(net, sparsity, dense_density=1):
    print('initialize by ERK_plus')
    total_params = 0
    density = 1 - sparsity
    erk_power_scale = 1

    baseline_nonzero = 0
    for name, weight in net.named_buffers():
        if 'weight' in name and len(weight.shape) in [2,4]: 
            total_params += weight.numel()
            baseline_nonzero += weight.numel() * density

    for name, mask in net.named_buffers():
        if 'fc.weight' in name:
            total_params = total_params - mask.numel()
            density = (baseline_nonzero - mask.numel() * dense_density) / total_params
            

    is_epsilon_valid = False
    dense_layers = set()
    while not is_epsilon_valid:

        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for name, mask in net.named_buffers():
            if 'weight' in name and len(mask.shape) in [2,4]:
                n_param = np.prod(mask.shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                    rhs -= n_zeros

                else:
                    # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                    # equation above.
                    rhs += n_ones
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    if len(mask.shape) !=2 :
                        raw_probabilities[name] = (
                                                        np.sum(mask.shape) / np.prod(mask.shape)
                                                ) ** erk_power_scale
                    else:
                        raw_probabilities[name] = (
                                                            np.sum(mask.shape) / np.prod(mask.shape)
                                                    ) ** erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                    divisor += raw_probabilities[name] * n_param
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        epsilon = rhs / divisor
        # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    # print(f"Sparsity of var:{mask_name} had to be set to 0.")
                    dense_layers.add(mask_name)
        else:
            is_epsilon_valid = True

    sparsity_dict = {}
    total_nonzero = 0.0
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for name, mask in net.named_buffers():
        if 'weight' in name and len(mask.shape) in [2,4]:
            n_param = np.prod(mask.shape)
            if name in dense_layers and 'fc.weight' not in name:
                sparsity_dict[name] = 0.
            elif 'fc.weight' in name:
                sparsity_dict[name] = 1 - dense_density
            else:
                probability_one = epsilon * raw_probabilities[name]
                sparsity_dict[name] = 1 - probability_one
            print(
                f"layer: {name}, shape: {mask.shape}, sparsity: {sparsity_dict[name]}"
            )
            # self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

            total_nonzero += sparsity_dict[name] * mask.numel()


    print(f"Overall sparsity {1 - total_nonzero / total_params}")
    return sparsity_dict

def uniform_sparsify(net, sparsity):
    print('initialize by uniform+')
    total_params = 0
    sparsity_dict = {}
    n_params_dict = {}
    for name, mask in net.named_buffers():
        if 'weight' in name and len(mask.shape) in [2,4]:
            total_params += mask.numel()
            sparsity_dict[name] = sparsity
            n_params_dict[name] = mask.numel()
    total_sparse_params = total_params * (1-sparsity)
    # print(f'sparsity dict len is {len(sparsity_dict)}')
    # for k, v in sparsity_dict.items():
    #     print(f'Layer {k} with sparsity {v}')
    # exit()
    # Only remove 0.2 params of first and last layer
    if sparsity > 0.5:
        if len(sparsity_dict) == 22: # Resnet20 CIFAR10
            first_layer_name = 'conv.weight_mask'
            last_layer_name = 'fc.weight_mask'
        elif len(sparsity_dict) == 21: # Resnet18 TinyImagenet
            first_layer_name = 'conv1.0.weight_mask'
            last_layer_name = 'fc.weight_mask'
        elif len(sparsity_dict) == 17: # VGG19-gn
            first_layer_name = 'layers.0.conv.weight_mask'
            last_layer_name = 'fc.weight_mask'
        else:
            raise ValueError('Wrong net for Uniform sparsify')
        total_sparse_params = total_sparse_params - \
                            n_params_dict[first_layer_name]*0.5 - \
                            n_params_dict[last_layer_name]*0.5

        new_sparsity = total_sparse_params / \
                    (
                        total_params - \
                            n_params_dict[first_layer_name]*0.5 - \
                            n_params_dict[last_layer_name]*0.5
                    )
        for name, sparsity in sparsity_dict.items():
            sparsity_dict[name] = 1 - new_sparsity
        
        sparsity_dict[first_layer_name] = 0.5
        sparsity_dict[last_layer_name] = 0.5
            
    total_nonzero = 0
    for name, mask in net.named_buffers():
        if 'weight' in name and len(mask.shape) in [2,4]:
            print(
                f"layer: {name}, shape: {mask.shape}, sparsity: {sparsity_dict[name]}"
            )
            total_nonzero += (1-sparsity_dict[name]) * mask.numel()
    print(f"Overall sparsity {1-total_nonzero / total_params}")
    
    return sparsity_dict


def pai_sparsify(net, initializer, sparsity, loss, dataloader, device):
    # sparsity has value 0.1 means pruning 90% params
    net = net.eval()
    layer_names = []
    for name, mask in net.named_buffers():
        if 'weight' in name and len(mask.shape) in [2,4]:
            layer_names.append(name)
    pruner = load.pruner(initializer)(generator.masked_parameters(net, False, False, False))
    pruner.score(net, loss, dataloader, device)
    sparsity_dict = pruner.get_layer_sparsity_dict(sparsity, layer_names)
    return sparsity_dict





def get_intermediate_inputs(net, input_):
    
    def get_children(model: torch.nn.Module):
        # get children form model!
        children = list(model.children())
        flatt_children = []
        if children == []:
            # if model has no children; model is last child! :O
            return model
        else:
            # look for children from children... to the last child!
            for child in children:
                    if isinstance(child, nn.BatchNorm2d) or \
                        isinstance(child, nn.ReLU) or \
                        isinstance(child, nn.AdaptiveAvgPool2d):
                        continue
                    try:
                        flatt_children.extend(get_children(child))
                    except TypeError:
                        flatt_children.append(get_children(child))
            return flatt_children

    flatt_children = get_children(net)

    visualization = []

    def hook_fn(m, i, o):
        visualization.append(i)

    for layer in flatt_children:
        layer.register_forward_hook(hook_fn)
        
    # for param in net.parameters():
    #     param.data.copy_(torch.ones_like(param))

    out = net(input_)  
    
    return visualization


import time 

def optimize_layerwise(mask, inp, sparsity, alpha=0.7, 
                    beta=0.001, max_param_per_kernel=None, 
                    min_param_to_node=None,
                    init_weight=None,
                    node_constraint=False):
    start_time = time.time()
    # print('Optimizing layerwise sparse pattern')
    is_conv = False

    # Params in layer 
    n_params = int(math.ceil((1-sparsity)*mask.numel())) # This has to be integer
    
    # The value of input nodes is descirbed by P_in
    if len(mask.shape) == 4:
        C_out, C_in, kernel_size, kernel_size = mask.shape
        min_param_per_kernel = int(math.ceil(n_params/(C_in*C_out))) 
        if max_param_per_kernel is None:
            max_param_per_kernel = kernel_size*kernel_size
        # Ensure enough params to assign to valid the sparsity
        elif max_param_per_kernel < min_param_per_kernel:
            max_param_per_kernel = min_param_per_kernel
        else:   # it's oke
            pass
        
        if min_param_to_node is None:
            min_param_to_node = 1
        # Ensure the valid of eff node constraint
        elif min_param_to_node > min_param_per_kernel:    
            min_param_to_node = min_param_per_kernel
        else:   # it's oke
            pass
        
        P_in = torch.sum(inp, dim=(1,2)).numpy()
        is_conv = True
    else:
        C_out, C_in = mask.shape
        kernel_size = 1
        max_param_per_kernel = kernel_size
        min_param_to_node = 1
        # P_in = torch.sum(inp, dim=)
        if len(inp.shape) != 1:
            P_in = torch.sum(inp, dim=1).numpy()
        else:
            P_in = inp.numpy()
        if len(P_in.shape) != 1 and P_in.shape[0] != C_out:
            raise ValueError('Wrong input dimension')
    
    # Mask variable 
    M = cp.Variable((C_in, C_out), integer=True)

    scaled_M = None
    if init_weight is not None:
        if is_conv:
            mag_orders = init_weight.transpose(1,0).view(C_in, C_out, -1).abs().argsort(dim=-1, descending=True).numpy()
            init_weight = torch.sum(init_weight, dim=(2,3)).transpose(1,0).numpy()
        else:
            init_weight = init_weight.transpose(1,0).numpy()
        init_weight = np.abs(init_weight)
        # scaled_M = cp.multiply(M, init_weight)

    # Sun 
    sum_in = cp.sum(M, axis=1)
    sum_out = cp.sum(M, axis=0)
    # sum_in = cp.sum(M, axis=1) * P_in
    # sum_out = cp.sum(cp.diag(P_in)@M, axis=0)


    # If eff_node_in is small which means there is a large number of input effective node 
    inv_eff_node_in = cp.sum(cp.pos(min_param_to_node - sum_in))
    inv_eff_node_out = cp.sum(cp.pos(min_param_to_node - sum_out))

    # OPtimize nodes 
    max_nodes = C_in + C_out
    A = max_nodes - (inv_eff_node_in + inv_eff_node_out) 
    # A = A / max_nodes   # Scale to 1

    # Optimize paths
    # B = (cp.sum(P_in @ M)) / cp.sum(P_in)   # Downscale with input nodes' values
    min_out_node = int(n_params/(C_out * max_param_per_kernel))
    remainder = n_params - min_out_node * (C_out * max_param_per_kernel)
    try:
        max_path = np.sum(np.sort(P_in)[-min_out_node:] * (C_out * max_param_per_kernel)) + \
                    remainder * np.sort(P_in)[-(min_out_node+1)]
    except:
        max_path = np.sum(np.sort(P_in)[-min_out_node:] * (C_out * max_param_per_kernel))
    
    if scaled_M is not None:
        B = (cp.sum(P_in @ scaled_M)) 
        # B = (cp.sum(P_in @ scaled_M)) / np.sum(P_in)
    else:
        B = (cp.sum(P_in @ M)) / max_path
        A = A / max_nodes
    # C = (cp.sum(P_in @ M)) / max_path
    # Regulaziration
    Reg = (n_params-cp.sum(cp.pos(1-M))) / n_params     # maximize number of edges 
    # Reg = 0


    # Constraint the total activated params 

    constraint = [cp.sum(M) <= n_params, M <= max_param_per_kernel, M >= 0] 

    if node_constraint:
        constraint.append(
            cp.max(cp.sum(M, axis=0)) <= int(C_in*max_param_per_kernel**2*(1-sparsity))
        )
        constraint.append(
            cp.max(cp.sum(M, axis=1)) <= int(C_out*max_param_per_kernel**2*(1-sparsity))
        )
    # Objective function
    # alpha = 0.7
    obj =cp.Maximize(alpha * A + (1-alpha) * B + beta * Reg)

    # Init problem
    prob = cp.Problem(obj, constraint)

    # Solving
    prob.solve()
    # prob.value

    if is_conv:
        a = torch.tensor(M.value, dtype=torch.int16)
        mat = []
        for i in range(C_out):
            row = []
            for j in range(C_in):
                try:
                    r = np.zeros(kernel_size**2)
                    if init_weight is not None:
                        one_idxs = mag_orders[j,i][:a[j,i]]
                        r[one_idxs] = 1 
                    else:
                        r[:a[j,i]] = 1
                        np.random.shuffle(r)
                    row.append(r.reshape(kernel_size, kernel_size))
                except:
                    print(r)
                    print(a[j,i])
            mat.append(row)
        mat = np.array(mat)
        mask.data.copy_(torch.tensor(mat))
    else:
        mask.data.copy_(torch.tensor(M.value).transpose(1,0))

    actual_sparsity = 1 - mask.sum().item() / mask.numel()
    end_time = time.time()
    print(f'Pruning time is {end_time - start_time}')

    return mask


def fine_tune_mask(model, input_shape):
    """Change ineffective params to effective ones
    """
    print('Fine-tuning the mask')
    c, h , w = input_shape
    # Put input ones through the subnet and backward
    net = copy.deepcopy(model)
    net = net.cpu().double()

    x = torch.ones((1, c, h, w)).double()
    y = net(x)
    loss = y.sum()
    # loss
    loss.backward()

    ###########
    # EFFECTIVE PARAMS
    ###########
    eff_masks = {}
    for name, param in net.named_parameters():
        if 'weight' in name and len(param.shape) in [2, 4]:
            eff_masks[name+'_mask'] = None
    n_ineff_after = 0
    # with torch.no_grad():
    i = 0
    for name, param in net.named_parameters():
        if 'weight' in name and len(param.shape) in [2, 4]:
            eff_masks[name+'_mask'] = torch.where(param.grad.data != 0, 1, 0)
            i += 1

    for name, mask in net.named_buffers():
        if 'weight' in name and len(mask.shape) in [4]:
            c_out, c_in, k, w = mask.shape
            eff_mask = eff_masks[name]
            ineff_mask = mask - eff_mask
            n_ineff = ineff_mask.sum().item() + n_ineff_after
            if n_ineff > 1:
                print(f'Adding ones to mask of layer {name}')
                if 'shortcut' in name:
                    new_mask = eff_mask.view(c_out, c_in)
                    tmp = eff_mask.sum(dim=0).view(-1)
                    idx = torch.argsort(tmp, descending=True)
                    count = 0 
                    while n_ineff > 0:
                        curr = new_mask[:,idx[count]].sum()
                        need = c_out - curr
                        n_ineff = n_ineff - need
                        new_mask[:, idx[count]].copy_(torch.ones_like(new_mask[:, idx[count]]))
                        count += 1
                    new_mask = new_mask.view(c_out, c_in, k, w)
                    mask.data.copy_(new_mask)
                    # check if still has ineff params
                    n_ineff_after, eff_mask = check_layer_ineff_param(net, input_shape, name)
                    mask.data.copy_(eff_mask) # copy new mask
                else:
                    new_mask = eff_mask.view(-1, k, w)
                    tmp = eff_mask.sum(dim=(2,3)).view(-1)  # sum over kernels and
                    idx = torch.argsort(tmp, descending=True) # sort in desceding order
                    count = 0
                    while n_ineff > 0:  # sequentially fill full the largest kernels
                        curr = new_mask[idx[count]].sum()   
                        need = k*w - curr
                        n_ineff = n_ineff - need
                        new_mask[idx[count]].copy_(torch.ones_like(new_mask[idx[count]]))
                        count += 1
                    new_mask = new_mask.view(c_out, c_in, k, w)
                    mask.data.copy_(new_mask)
                    # check if still has ineff params
                    n_ineff_after, eff_mask = check_layer_ineff_param(net, input_shape, name)
                    mask.data.copy_(eff_mask)    # copy new mask 
        elif 'weight' in name and len(mask.shape) in [2]:
            f_out, f_in = mask.shape
            eff_mask = eff_masks[name]
            ineff_mask = mask - eff_mask
            n_ineff = ineff_mask.sum().item() + n_ineff_after
            print(f'number of ineff params in {name} is {n_ineff}')
            # tmp = eff_mask.sum(dim=0) # sum over out features
            # idx = torch.argsort(tmp, descending=True) # sort in descending order
            # count = 0
            # while n_ineff > 0:
            #     curr = 
    
    return net
                

def check_layer_ineff_param(model, input_shape, layer_name):
    """Check whether a layer still has ineff params after fine-tune or not
    If yes, move these ineff params to the next layers
    """
    print('Fine-tuning the mask')
    c, h , w = input_shape
    # Put input ones through the subnet and backward
    net = copy.deepcopy(model)
    net = net.cpu().double()

    x = torch.ones((1, c, h, w)).double()
    y = net(x)
    loss = y.sum()
    # loss
    loss.backward()

    ###########
    # EFFECTIVE PARAMS
    ###########
    eff_masks = {}
    all_n_ineff = 0
    for name, param in net.named_parameters():
        if 'weight' in name and len(param.shape) in [2, 4]:
            eff_masks[name+'_mask'] = None
    with torch.no_grad():
        i = 0
        for name, param in net.named_parameters():
            if 'weight' in name and len(param.shape) in [2, 4]:
                eff_masks[name+'_mask'] = torch.where(param.grad.data != 0, 1, 0)
                i += 1

        for name, mask in net.named_buffers():
            if name == layer_name:
                if 'weight' in name and len(mask.shape) in [4]:
                    c_out, c_in, k, w = mask.shape
                    eff_mask = eff_masks[name]
                    ineff_mask = mask - eff_mask
                    n_ineff = ineff_mask.sum().item()
                    if n_ineff > 0:
                        print(f"Layer {name} still has {n_ineff} ineffective params")
                        print(f'Move this {n_ineff} ineffective params to the next layers')
                else:
                    f_out, f_in = mask.shape
                    eff_mask = eff_masks[name]
                    ineff_mask = mask - eff_mask
                    n_ineff = ineff_mask.sum().item()
                    if n_ineff > 0:
                        print(f"Layer {name} still has {n_ineff} ineffective params")
                break
                    
    del net
    return n_ineff, eff_masks[name]


def count_ineff_param(model, input_shape):
    """Change ineffective params to effective ones
    """
    print('Fine-tuning the mask')
    c, h , w = input_shape
    # Put input ones through the subnet and backward
    net = copy.deepcopy(model)
    net = net.cpu().double()

    x = torch.ones((1, c, h, w)).double()
    y = net(x)
    loss = y.sum()
    # loss
    loss.backward()

    ###########
    # EFFECTIVE PARAMS
    ###########
    eff_masks = {}
    all_n_ineff = 0
    for name, param in net.named_parameters():
        if 'weight' in name and len(param.shape) in [2, 4]:
            eff_masks[name+'_mask'] = None
    with torch.no_grad():
        i = 0
        for name, param in net.named_parameters():
            if 'weight' in name and len(param.shape) in [2, 4]:
                eff_masks[name+'_mask'] = torch.where(param.grad.data != 0, 1, 0)
                i += 1

        for name, mask in net.named_buffers():
            if 'weight' in name and len(mask.shape) in [4]:
                c_out, c_in, k, w = mask.shape
                eff_mask = eff_masks[name]
                ineff_mask = mask - eff_mask
                n_ineff = ineff_mask.sum().item()
                print(f"Layer {name} has {n_ineff} ineffective params")
            if 'weight' in name and len(mask.shape) in [2]:
                f_out, f_in = mask.shape
                eff_mask = eff_masks[name]
                ineff_mask = mask - eff_mask
                n_ineff = ineff_mask.sum().item()
                print(f'Layer {name} has {n_ineff} ineffective params')
            all_n_ineff += n_ineff
    del net
    return all_n_ineff

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x, y=None):
        return x

class IgnoreBatchnorm2d(torch.nn.BatchNorm2d):

    def forward(self, input):
        return input

class ScaleAvgPool2d(torch.nn.AvgPool2d):
        # self.kernel_size = kernel_size

    def forward(self, input):
        scale = self.kernel_size if not isinstance(self.kernel_size, tuple) else sum(self.kernel_size)
        return scale*F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)

class ScaleAdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d):
    """
    In NAS-Bench-201 adaptive average pooling layer has output size = 1 
    scale = h * w
    """
    def forward(self, input):
        _, _, h, w = input.shape
        scale = h*w
        return scale * F.adaptive_avg_pool2d(input, self.output_size)

# batchnorms = []
# for n, m in network.named_modules():
#     if isinstance(m, torch.nn.BatchNorm2d):
#         # m.replac
#         # network.get_submodule = Identity()
#         batchnorms.append(n)

# batchnorms = [k.split('.') for k in batchnorms]
# for *parent, k in batchnorms:
#     network.get_submodule('.'.join(parent))[int(k)] = Identity()
# network

def copynet(net, batchnorm=False, avgpool=False):
    """ To compute the number of effective paths we ignore the batchnorm and re-scale adaptive pooling
    This function use to change batchnorm to IgnoreBatchNorm and
    Adaptive to Scale Adaptive which are used to compute the number of paths
    """
    cloned_net = copy.deepcopy(net)
    if not batchnorm:
        for module in cloned_net.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.forward = types.MethodType(IgnoreBatchnorm2d.forward, module)
    if not avgpool:
        for module in cloned_net.modules():
            if isinstance(module, nn.AvgPool2d):
                module.forward = types.MethodType(ScaleAvgPool2d.forward, module)
            if isinstance(module, nn.AdaptiveAvgPool2d):
                module.forward = types.MethodType(ScaleAdaptiveAvgPool2d.forward, module)
    return cloned_net

def store_mask(net, file_name):
    masks = []
    print('='*40)
    print('Storing mask')
    with torch.no_grad():
        for name, mask in net.named_buffers():
            if 'weight' in name and len(mask.shape) in [2,4]:
                masks.append(mask.detach().cpu().numpy())
        with open(file_name, 'wb') as f:
            pickle.dump(masks, f)
    print('Finish storing mask')
    print('='*40)