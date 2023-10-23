import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

def cal_n_effective_path(model, device, dataloader):
    # print('check')
    tmp_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in tmp_model.named_parameters():
            for n, m in tmp_model.named_buffers():
                if name + '_mask' == n:
                    param.copy_(torch.ones_like(param)*m)
        
        z = next(iter(dataloader))[0]
        # size = x.shape
        x = torch.ones_like(z).to(device)
        y = tmp_model(x[0:1])
        print('effective paths ', y.sum().item())


def random_add_in(model, device, p=0.01):
    for name, mask in model.named_buffers(): 
        if 'bn' in name:
            continue
        new_mask = mask.clone()
        n = (new_mask==0).sum().item()
        if n == 0: continue
        expeced_growth_probability = p
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        new_mask_ = new_mask.byte() | new_weights
        if (new_mask_!=0).sum().item() == 0:
            new_mask_ = new_mask
        # print(f'sum of non zero in new mask {name}: ', new_mask_.sum().item())
        # print(f'sum of non zero in mask {name}', mask.sum().item())
        # exit()
        
        mask.copy_(new_mask_)
        # Assign new weight to zeros
        with torch.no_grad():
            for n, param in model.named_parameters():
                if name == n + '_mask':
                    param.copy_(param.data*(torch.ones_like(new_weights.int())-new_weights.int()))

    return model

def reg_added_weights(model, added_weights):
    reg_loss = torch.tensor(0.).cuda()
    for n, p in model.named_parameters():
        for nm, m in added_weights.items():
            if nm == n + '_mask':
                reg_loss += torch.norm(p*m)
    
    return reg_loss

def random_add_in_training(model, loss, optimizer, dataloader, device, epoch, verbose, added_weights, log_interval=10, p=0.01, delta_t=1):
    model.train()
    # prev_mask = {}
    # for n, m in model.named_buffers():
    #     prev_mask[n] = m.clone()
    # print('Before add-in')
    # cal_n_effective_path(model, device, dataloader)
    # model = random_add_in(model, device, p)
    # print('After add-in')
    # cal_n_effective_path(model, device, dataloader)
    
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        reg_loss = reg_added_weights(model, added_weights)
        train_loss = loss(output, target) + 10*reg_loss
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()


        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))

    # for n, m in model.named_buffers():
    #     m.copy_(prev_mask[n])

        
    return total / len(dataloader.dataset)





def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5

def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose, wandb=None):
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5]]
    for epoch in tqdm(range(epochs)):
        # train_loss = random_add_in_training(model, loss, optimizer, train_loader, device, epoch, verbose)
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        # for n, p in model.named_parameters():
        #     print(f'norm of layer {n} is \t {torch.norm(p)}')
        # print('\nTest accuracy: \t ', accuracy1)
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        print('\nTest accuracy: \t ', accuracy1)
        if wandb is not None:
            wandb.log({
                "Test Loss": test_loss,
                "Test Accuracy": accuracy1,
                "Train Loss": train_loss 
            })
        row = [train_loss, test_loss, accuracy1, accuracy5]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    return pd.DataFrame(rows, columns=columns)




def random_train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose, args):
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5]]

    # Store the real subnetwork
    prev_mask = {}
    for n, m in model.named_buffers():
        prev_mask[n] = m.clone()
    
    for epoch in tqdm(range(epochs)):
        if epoch % args.addin_epoch == 0 and epoch < 130:
            # Restore to the real subnetwork
            for n, m in model.named_buffers():
                m.copy_(prev_mask[n])
            
            print('Before add-in')
            cal_n_effective_path(model, device, train_loader)
            model = random_add_in(model, device, 0.001)
            print('After add-in')
            cal_n_effective_path(model, device, train_loader)

            new_mask = {}
            for n, m in model.named_buffers():
                new_mask[n] = m.clone()

            added_weights = {}
            for (n1, m1), (n2, m2) in zip(prev_mask.items(), new_mask.items()):
                added_weights[n1] = torch.logical_xor(m1, m2).int().to(device)


        train_loss = random_add_in_training(model, loss, optimizer, train_loader, device, epoch, verbose, added_weights)
        # train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)

        # Restore to previous mask to test
        for n, m in model.named_buffers():
            m.copy_(prev_mask[n])

        # Check norm weight to verify whether the network learn or not
        for n, p in model.named_parameters():
            print(f'norm of layer {n} is \t {torch.norm(p)}')

        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        # Back to new mask to train
        for n, m in model.named_buffers():
            m.copy_(new_mask[n])
            
        print('\nTest accuracy: \t ', accuracy1)
        row = [train_loss, test_loss, accuracy1, accuracy5]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    return pd.DataFrame(rows, columns=columns)


