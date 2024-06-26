import os
import json
import random
import copy

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model import load_model, resume_model
from dataset import load_dataset
from arguments import advparser as parser
from train import train, test
from utils import *


dispatcher = AttrDispatcher('crt_method')


class CorrectionUnit(nn.Module):
    def __init__(self, num_filters, Di, k):
        super(CorrectionUnit, self).__init__()
        self.conv1 = nn.Conv2d(
            num_filters, Di, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(Di)
        self.conv2 = nn.Conv2d(
            Di, Di, kernel_size=k, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(Di)
        self.conv3 = nn.Conv2d(
            Di, Di, kernel_size=k, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(Di)
        self.conv4 = nn.Conv2d(
            Di, num_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        out += self.shortcut(x)
        return out


class ConcatCorrect(nn.Module):
    def __init__(self, conv_layer, indices, prune_indices, order):
        super(ConcatCorrect, self).__init__()
        self.indices = indices
        self.prune_indices = prune_indices
        self.order = order
        self.conv = conv_layer
        num_filters = len(indices)
        self.cru = CorrectionUnit(num_filters, num_filters, 3)

    def forward(self, x):
        out = self.conv(x)
        if self.prune_indices is not None and self.order == 'first_prune':
            out[:, self.prune_indices] = 0
        if self.indices is not None:
            out[:, self.indices] = self.cru(out[:, self.indices])
        if self.prune_indices is not None:
            out[:, self.prune_indices] = 0
        return out



class ReplaceCorrect(nn.Module):
    def __init__(self, conv_layer, indices, prune_indices, order):
        super(ReplaceCorrect, self).__init__()
        self.indices = indices
        self.prune_indices = prune_indices
        self.order = order
        self.conv = conv_layer
        self.cru = nn.Conv2d(
            conv_layer.in_channels,
            len(indices),
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            groups=conv_layer.groups,
            bias=False)

    def forward(self, x):
        out = self.conv(x)
        if self.order == 'first_prune':
          if self.prune_indices is not None:
            out[:, self.prune_indices] = 0
          if self.indices is not None:
            out[:, self.indices] = self.cru(x)
        elif self.order == 'first_patch':
          if self.indices is not None:
            out[:, self.indices] = self.cru(x)
          if self.prune_indices is not None:
            out[:, self.prune_indices] = 0
        elif self.prune_indices is not None or self.indices is not None:
          if self.prune_indices is not None:
            out[:, self.prune_indices] = 0
          if self.indices is not None:
            out[:, self.indices] = self.cru(x)
        return out


class NoneCorrect(nn.Module):
    def __init__(self, conv_layer, indices, prune_indices, order):
        super(NoneCorrect, self).__init__()
        self.indices = indices
        self.prune_indices = prune_indices
        self.order = order
        self.conv = conv_layer

    def forward(self, x):
        out = self.conv(x)
        if self.prune_indices is not None and self.order == 'first_prune':
            out[:, self.prune_indices] = 0
        if self.indices is not None:
            out[:, self.indices] = 0
        # if self.prune_indices is not None:
        #     out[:, self.prune_indices] = 0
        return out



def construct_model(opt, model, patch=True):
    # sus_filters = json.load(open(os.path.join(
    #     opt.output_dir, opt.dataset, opt.model, f'susp_filters_{opt.fs_method}.json'
    # ))) if opt.susp_side in ('front', 'rear') else {}

    sus_filters = json.load(open(os.path.join(
        opt.output_dir, opt.dataset, opt.model, 'susp_filters_perfloss.json'
    ))) if opt.susp_side in ('front', 'rear') else {}

    prune_filters = json.load(open(os.path.join(
        opt.output_dir, opt.dataset, opt.model, 'susp_filters_distweight.json'
    )))

    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    for layer_name in conv_names:
        module = rgetattr(model, layer_name)

        num_susp = int(module.out_channels * opt.susp_ratio)
        num_prune = int(module.out_channels * opt.prune_ratio)
        if opt.pporder=='first_patch'or opt.pporder=='first_prune':
          if opt.susp_side == 'front' and opt.prune_side=='front':
            indices = sus_filters[layer_name][opt.patch_indices][:num_susp] if num_susp > 0 else None
            prune_indices = prune_filters[layer_name][opt.prune_indices][:num_prune] if opt.prune and num_prune > 0 else None
          elif opt.susp_side == 'rear' and opt.prune_side=='rear':
            indices = sus_filters[layer_name][opt.patch_indices][-num_susp:] if num_susp > 0 else None
            prune_indices = prune_filters[layer_name][opt.prune_indices][-num_prune:] if opt.prune and num_prune > 0 else None
          elif opt.susp_side == 'front' and opt.prune_side=='rear':
            indices = sus_filters[layer_name][opt.patch_indices][:num_susp:] if num_susp > 0 else None
            prune_indices = prune_filters[layer_name][opt.prune_indices][-num_prune:] if opt.prune and num_prune > 0 else None       
          elif opt.susp_side == 'random':
            indices = random.sample(range(module.out_channels), num_susp) if num_susp > 0 else None
            prune_indices = random.sample(range(module.out_channels), num_prune) if opt.prune and num_prune > 0 else None
          else:
            raise ValueError('Invalid suspicious side')
        elif opt.pporder=='first_patch_no_intersection':
          if opt.susp_side == 'front'and opt.prune_side=='front':
            indices = sus_filters[layer_name][opt.patch_indices][:num_susp] if num_susp > 0 else None
            prune_indices = [idx for idx in prune_filters[layer_name][opt.prune_indices] if idx not in indices]
            prune_indices = prune_indices[:num_prune] if opt.prune and num_prune > 0 else None
          elif opt.susp_side == 'rear' and opt.prune_side=='rear':
            indices = sus_filters[layer_name][opt.patch_indices][-num_susp:] if num_susp > 0 else None
            prune_indices = [idx for idx in prune_filters[layer_name][opt.prune_indices] if idx not in indices]
            prune_indices = prune_indices[-num_prune:] if opt.prune and num_prune > 0 else None
          elif opt.susp_side == 'front' and opt.prune_side=='rear':
            indices = sus_filters[layer_name][opt.patch_indices][:num_susp] if num_susp > 0 else None
            prune_indices = [idx for idx in prune_filters[layer_name][opt.prune_indices] if idx not in indices]
            prune_indices = prune_indices[-num_prune:] if opt.prune and num_prune > 0 else None
          elif opt.susp_side == 'random':
            indices = random.sample(range(module.out_channels), num_susp) if num_susp > 0 else None
            # prune_indices = [idx for idx in prune_filters[layer_name][opt.prune_indices] if idx not in indices]
            prune_indices = random.sample(range(module.out_channels), num_prune) if opt.prune and num_prune > 0 else None
          else:
            raise ValueError('Invalid suspicious side')
        elif opt.pporder=='first_prune_no_intersection':
          if opt.susp_side == 'front' and opt.prune_side=='front':
            prune_indices = prune_filters[layer_name][opt.prune_indices][:num_prune] if opt.prune and num_prune > 0 else None
            indices = [idx for idx in sus_filters[layer_name][opt.patch_indices] if idx not in prune_indices]
            indices = indices[:num_susp] if num_susp > 0 else None
          elif opt.susp_side == 'rear'and opt.prune_side=='rear':
            prune_indices = prune_filters[layer_name][opt.prune_indices][-num_prune:] if opt.prune and num_prune > 0 else None
            indices = [idx for idx in sus_filters[layer_name][opt.patch_indices] if idx not in prune_indices]
            indices = indices[-num_susp:] if num_susp > 0 else None
          elif opt.susp_side == 'front'and opt.prune_side=='rear':
            prune_indices = prune_filters[layer_name][opt.prune_indices][-num_prune:] if opt.prune and num_prune > 0 else None
            indices = [idx for idx in sus_filters[layer_name][opt.patch_indices] if idx not in prune_indices]
            indices = indices[:num_susp] if num_susp > 0 else None
          elif opt.susp_side == 'random':
            indices = random.sample(range(module.out_channels), num_susp) if num_susp > 0 else None
            # prune_indices = [idx for idx in prune_filters[layer_name][opt.prune_indices] if idx not in indices]
            prune_indices = random.sample(range(module.out_channels), num_prune) if opt.prune and num_prune > 0 else None
          else:
            raise ValueError('Invalid suspicious side')

        if module.groups != 1:
            continue

        if patch is False:
            correct_module = NoneCorrect(module, indices, prune_indices, opt.pporder)
        elif opt.pt_method == 'DC':
            correct_module = ConcatCorrect(module, indices, prune_indices, opt.pporder) 
        elif 'DP' in opt.pt_method:
            # print('11')
            correct_module = ReplaceCorrect(module, indices, prune_indices, opt.pporder)
            # print('22')
        else:
            raise ValueError('Invalid correct type')
            
        rsetattr(model, layer_name, correct_module)
        # print(layer_name)
        # print(indices)
        # print(prune_indices)
    return model

def fine_tune(opt, model, device):
    _, trainloader = load_dataset(opt, split='train')
    _, valloader = load_dataset(opt, split='val')

    indices = {}
    for name, module in model.named_modules():
        if isinstance(module, ConcatCorrect) \
                or isinstance(module, ReplaceCorrect) \
                or isinstance(module, NoneCorrect):
            indices[name] = module.indices
            module.indices = None

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        # model.parameters(),  # correction unit + finetune
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    model = model.to(device)
    best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')
    for epoch in range(opt.crt_epoch):
        print('Finetune Epoch: {}'.format(epoch))
        train(model, trainloader, optimizer, criterion, device)
        acc, *_ = test(model, valloader, criterion, device)
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
    print('[info] the best finetune accuracy is {:.4f}%'.format(best_acc))
    model = model.to('cpu')

    for name, module in model.named_modules():
        if isinstance(module, ConcatCorrect) \
                or isinstance(module, ReplaceCorrect) \
                or isinstance(module, NoneCorrect):
            module.indices = indices[name]  

def extract_info(model, info_type):
    info = {}
    for n, m in model.named_modules():
        if isinstance(m, ConcatCorrect) \
            or isinstance(m, ReplaceCorrect) \
            or isinstance(m, NoneCorrect):
            if info_type == "indices":
                info[n] = m.indices
            elif info_type == "prune_indices":
                info[n] = m.prune_indices
            elif info_type == "order":
                info[n] = m.order
    return info

@dispatcher.register('patch')
def patch(opt, model, device):
    if opt.pt_method == 'DP-s':
        _, trainloader = load_dataset(opt, split='train', aug=True)
        _, valloader = load_dataset(opt, split='val', aug=True)
    elif opt.pt_method == 'DP-SS':
        trainset, _ = load_dataset(opt, split='train', aug=True)
        _, valloader = load_dataset(opt, split='val', aug=True)
    elif opt.pt_method == 'SS-DP':
        _, trainloader = load_dataset(opt, split='train')
        _, valloader = load_dataset(opt, split='val')
        model = resume_model(opt, model, state=f'sensei')
    else:
        _, trainloader = load_dataset(opt, split='train', noise=True, noise_type='random')
        _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append')
        # _, trainloader = load_dataset(opt, split='train')
        # _, valloader = load_dataset(opt, split='val')

    model = construct_model(opt, model)
   
    if opt.finetune is True:
        fine_tune(opt, model, device)

    model = model.to(device)

    for name, module in model.named_modules():
        if 'cru' in name:
            for param in module.parameters():
                param.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    sel_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        # model.parameters(),  # correction unit + finetune
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    start_epoch = -1
    if opt.resume:
        ckp = torch.load(get_model_path(opt, state=f'patch_{opt.fs_method}'))
        model.load_state_dict(ckp['net'])
        optimizer.load_state_dict(ckp['optim'])
        scheduler.load_state_dict(ckp['sched'])
        start_epoch = ckp['cepoch']
        best_acc = ckp['acc']
        for n, m in model.named_modules():
            if isinstance(m, ConcatCorrect) \
                    or isinstance(m, ReplaceCorrect) \
                    or isinstance(m, NoneCorrect):
                m.indices = ckp['indices'][n]
                m.prune_indices = ckp['prune_indices'][n]
                m.order = ckp['order'][n]
    else:
        best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')

    for epoch in range(start_epoch + 1, opt.crt_epoch):
        print('Epoch: {}'.format(epoch))
        if opt.pt_method == 'DP-SS':
            trainset.selective_augment(model, sel_criterion, opt.batch_size, device)  # type: ignore
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4
            )
        train(model, trainloader, optimizer, criterion, device)
        acc, *_ = test(model, valloader, criterion, device)
        if acc > best_acc:
            print('Saving...')
            state = {
                'cepoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc,
                'indices': extract_info(model, 'indices'),
                'prune_indices': extract_info(model, 'prune_indices'),
                'order': extract_info(model, 'order')
            }
            torch.save(state, get_model_path(opt, state=f'patch_{opt.fs_method}'))
            # torch.save(state, get_model_path(opt, state=f'patch_{opt.fs_method}_finetune'))
            # torch.save(state, get_model_path(opt, state=f'patch_{opt.fs_method}_r25_{opt.susp_side}'))
            # torch.save(state, get_model_path(opt, state=f'patch_{opt.pt_method}'))
            best_acc = acc
        scheduler.step()
    print('[info] the best retrain accuracy is {:.4f}%'.format(best_acc))


@dispatcher.register('finetune')
def finetune(opt, model, device):
    _, trainloader = load_dataset(opt, split='train', noise=True, noise_type='random')
    _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append')

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')
    for epoch in range(0, opt.crt_epoch):
        print('Epoch: {}'.format(epoch))
        train(model, trainloader, optimizer, criterion, device)
        acc, *_ = test(model, valloader, criterion, device)
        if acc > best_acc:
            print('Saving...')
            state = {
                'cepoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc
            }
            torch.save(state, get_model_path(opt, state='finetune'))
            best_acc = acc
        scheduler.step()
    print('[info] the best retrain accuracy is {:.4f}%'.format(best_acc))


@dispatcher.register('sensei')
def sensei(opt, model, device):
    trainset, _ = load_dataset(opt, split='train', aug=True)
    _, valloader = load_dataset(opt, split='val', aug=True)

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    sel_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')
    for epoch in range(0, opt.crt_epoch):
        print('Epoch: {}'.format(epoch))
        trainset.selective_augment(model, sel_criterion, opt.batch_size, device)  # type: ignore
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4
        )
        train(model, trainloader, optimizer, criterion, device)
        acc, *_ = test(model, valloader, criterion, device)
        if acc > best_acc:
            print('Saving...')
            state = {
                'cepoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc
            }
            torch.save(state, get_model_path(opt, state='sensei'))
            best_acc = acc
        scheduler.step()
    print('[info] the best retrain accuracy is {:.4f}%'.format(best_acc))


@dispatcher.register('apricot')
def apricot(opt, model, device):
    guard_folder(opt, folder='apricot')

    # trainset, trainloader = load_dataset(opt, split='train', noise=True, noise_type='random')
    # _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append')
    trainset, trainloader = load_dataset(opt, split='train', aug=True)
    _, valloader = load_dataset(opt, split='val', aug=True)

    # create rDLMs
    NUM_SUBMODELS = 20
    # SUBSET_SIZE = 10000  # switch to 1000 for stl10
    SUBMODEL_EPOCHS = 40
    subset_step = int((len(trainset) - SUBSET_SIZE) // NUM_SUBMODELS)

    for sub_idx in tqdm(range(NUM_SUBMODELS), desc='rDLMs'):
        submodel_path = get_model_path(opt, folder='apricot', state=f'sub_{sub_idx}')
        if os.path.exists(submodel_path):
            continue

        subset_indices = list(range(subset_step * sub_idx, subset_step * sub_idx + SUBSET_SIZE))
        subset = torch.utils.data.Subset(trainset, subset_indices)
        subloader = torch.utils.data.DataLoader(
            subset, batch_size=opt.batch_size, shuffle=True, num_workers=4
        )

        submodel = copy.deepcopy(model).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            submodel.parameters(),
            lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        best_acc, *_ = test(submodel, valloader, criterion, device, desc='Baseline')
        for epoch in tqdm(range(0, SUBMODEL_EPOCHS), desc='epochs'):
            train(submodel, subloader, optimizer, criterion, device)
            acc, *_ = test(submodel, valloader, criterion, device)
            if acc > best_acc:
                print('Saving...')
                state = {
                    'cepoch': epoch,
                    'net': submodel.state_dict(),
                    'optim': optimizer.state_dict(),
                    'sched': scheduler.state_dict(),
                    'acc': acc
                }
                torch.save(state, submodel_path)
                best_acc = acc
            scheduler.step()
        print('[info] the best submodel accuracy is {:.4f}%'.format(best_acc))

    # submodels
    sm_equals_path = get_model_path(opt, folder='apricot', state='sub_pred_equals')

    if not os.path.exists(sm_equals_path):
        submodel = copy.deepcopy(model).to(device).eval()
        seqloader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.batch_size, shuffle=False, num_workers=4
        )
        submodels_equals = []
        for sub_idx in tqdm(range(NUM_SUBMODELS), desc='subModelPreds'):
            submodel_path = get_model_path(opt, folder='apricot', state=f'sub_{sub_idx}')
            state = torch.load(submodel_path)
            submodel.load_state_dict(state['net'])

            equals = []
            for inputs, targets in tqdm(seqloader, desc='Batch', leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    outputs = submodel(inputs)
                _, predicted = outputs.max(1)
                eqs = predicted.eq(targets).flatten()
                equals.append(eqs)
            equals = torch.cat(equals)

            submodels_equals.append(equals)
        submodels_equals = torch.stack(submodels_equals)
        torch.save(submodels_equals, sm_equals_path)
    else:
        submodels_equals = torch.load(sm_equals_path)

    # Fixing process
    NUM_LOOP_COUNT = 3
    BATCH_SIZE = 20
    LEARNING_RATE = 0.001

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    best_acc, *_ = test(model, valloader, criterion, device)
    best_weights = copy.deepcopy(model.state_dict())

    submodel_weights = []
    for sub_idx in range(NUM_SUBMODELS):
        submodel_path = get_model_path(opt, folder='apricot', state=f'sub_{sub_idx}')
        state = torch.load(submodel_path)
        submodel_weights.append(state['net'])

    for loop_idx in range(NUM_LOOP_COUNT):
        print('Loop: {}'.format(loop_idx))
        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(range(len(trainset))),
            batch_size=BATCH_SIZE, drop_last=False
        )
        for indices in tqdm(sampler, desc='Sampler'):
            model.eval()
            base_weights = copy.deepcopy(model.state_dict())

            inputs = torch.stack([trainset[ind][0] for ind in indices]).to(device)  # type: ignore
            targets = torch.tensor([trainset[ind][1] for ind in indices]).to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                equals = predicted.eq(targets).flatten()

            for ind, equal in zip(indices, equals):
                if equal:
                    continue
                correct_submodels = [
                    submodel_weights[i] for i in range(NUM_SUBMODELS)
                    if submodels_equals[i][ind]
                ]
                if len(correct_submodels) == 0:
                    continue

                # use strategy 2
                for key in base_weights.keys():
                    if 'num_batches_tracked' in key:
                        continue
                    correct_weight = torch.mean(
                        torch.stack([m[key] for m in correct_submodels]), dim=0)
                    correct_diff = base_weights[key] - correct_weight.to(device)
                    p_corr = len(correct_submodels) / NUM_SUBMODELS  # proportion
                    base_weights[key] = base_weights[key] + LEARNING_RATE * p_corr * correct_diff

            model.load_state_dict(base_weights)
            acc, *_ = test(model, valloader, criterion, device, desc='Eval')
            if acc > best_acc:
                best_weights = copy.deepcopy(base_weights)
                print('Saving {}'.format(acc))
                torch.save({'net': best_weights}, get_model_path(opt, state='apricot'))
                best_acc = acc

            train(model, trainloader, optimizer, criterion, device)
            acc, *_ = test(model, valloader, criterion, device, desc='Eval')
            if acc > best_acc:
                best_weights = copy.deepcopy(base_weights)
                print('Saving {}'.format(acc))
                torch.save({'net': best_weights}, get_model_path(opt, state='apricot'))
                best_acc = acc

            model.load_state_dict(best_weights)

        # violate the original
        break


@dispatcher.register('robot')
def robot(opt, model, device):
    # trainset, _ = load_dataset(opt, split='train', noise=True, noise_type='random')
    # _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append')
    trainset, _ = load_dataset(opt, split='train', aug=True)
    _, valloader = load_dataset(opt, split='val', aug=True)

    model = model.to(device).eval()
    criterion = torch.nn.CrossEntropyLoss()

    # compute FOL
    ROBOT_ELLIPSIS = 0.01

    seqloader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size, shuffle=False, num_workers=4
    )
    fols = []
    for inputs, targets in tqdm(seqloader, desc='FOL'):
        cur_batch_size = targets.size(0)

        with torch.enable_grad():
            grad_inputs = inputs.clone().detach()
            grad_inputs, targets = grad_inputs.to(device), targets.to(device)
            grad_inputs.requires_grad = True
            model.zero_grad()
            outputs = model(grad_inputs)
            loss = criterion(outputs, targets)
            loss.backward()

        grads_flat = grad_inputs.grad.cpu().numpy().reshape(cur_batch_size, -1)
        grads_norm = np.linalg.norm(grads_flat, ord=1, axis=1)
        grads_diff = grads_flat - inputs.numpy().reshape(cur_batch_size, -1)
        i_fols = -1. * (grads_flat * grads_diff).sum(axis=1) + ROBOT_ELLIPSIS * grads_norm
        fols.append(i_fols)
    fols = np.concatenate(fols)

    SELECTED_NUM = 10000
    indices = np.argsort(fols)
    sel_indices = np.concatenate((indices[:SELECTED_NUM//2], indices[-SELECTED_NUM//2:]))
    print(sel_indices)
    seqloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset, sel_indices.tolist()),
        batch_size=opt.batch_size, shuffle=True, num_workers=4
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    RETRAIN_EPOCHS = 40
    best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')
    for epoch in range(0, RETRAIN_EPOCHS):
        print('Epoch: {}'.format(epoch))
        train(model, seqloader, optimizer, criterion, device)
        acc, *_ = test(model, valloader, criterion, device)
        if acc > best_acc:
            print('Saving...')
            state = {
                'cepoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc
            }
            torch.save(state, get_model_path(opt, state='robot'))
            best_acc = acc
        scheduler.step()
    print('[info] the best retrain accuracy is {:.4f}%'.format(best_acc))


@dispatcher.register('gini')
def deepgini(opt, model, device):
    # trainset, _ = load_dataset(opt, split='train', noise=True, noise_type='random')
    # _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append')
    trainset, _ = load_dataset(opt, split='train', aug=True)
    _, valloader = load_dataset(opt, split='val', aug=True)

    model = model.to(device).eval()
    criterion = torch.nn.CrossEntropyLoss()

    # compute gini index
    seqloader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size, shuffle=False, num_workers=4
    )
    ginis = []
    for inputs, targets in tqdm(seqloader, desc='Gini'):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        probs = F.softmax(outputs, dim=1)
        gini = probs.square().sum(dim=1).mul(-1.).add(1.)
        ginis.append(gini.detach().cpu())
    ginis = torch.cat(ginis)

    indices = torch.argsort(ginis, descending=True)
    train_size = int(len(trainset) * 0.1)
    sel_indices = indices[:train_size]
    seqloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset, sel_indices.tolist()),
        batch_size=opt.batch_size, shuffle=True, num_workers=4
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    RETRAIN_EPOCHS = 40
    best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')
    for epoch in range(0, RETRAIN_EPOCHS):
        print('Epoch: {}'.format(epoch))
        train(model, seqloader, optimizer, criterion, device)
        acc, *_ = test(model, valloader, criterion, device)
        if acc > best_acc:
            print('Saving...')
            state = {
                'cepoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc
            }
            torch.save(state, get_model_path(opt, state='gini'))
            best_acc = acc
        scheduler.step()
    print('[info] the best retrain accuracy is {:.4f}%'.format(best_acc))



def augmix_train(net, train_loader, optimizer, scheduler, device, no_jsd=False):
    """Train for one epoch."""
    net.train()
    loss_ema = 0.
    for images, targets in tqdm(train_loader, desc='Train'):
        optimizer.zero_grad()

        if no_jsd:
            images = images.to(device)
            targets = targets.to(device)
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
        else:
            images_all = torch.cat(images, 0).to(device)
            targets = targets.to(device)
            logits_all = net(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(
                logits_all, images[0].size(0))

            # Cross-entropy is only computed on clean images
            loss = F.cross_entropy(logits_clean, targets)

            p_clean, p_aug1, p_aug2 = F.softmax(
                logits_clean, dim=1), F.softmax(
                    logits_aug1, dim=1), F.softmax(
                        logits_aug2, dim=1)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_ema = loss_ema * 0.9 + float(loss) * 0.1

    return loss_ema


@dispatcher.register('augmix')
def augmix(opt, model, device):
    _, trainloader = load_dataset(opt, split='train', mix=True)
    # _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append')
    _, valloader = load_dataset(opt, split='val', aug=True)

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True
    )
    total_steps = opt.crt_epoch * len(trainloader)  # type: ignore
    lr_max, lr_min = 1, 1e-6 / opt.lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
    )

    best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')
    for epoch in range(0, opt.crt_epoch):
        print('Epoch: {}'.format(epoch))
        augmix_train(model, trainloader, optimizer, scheduler, device, no_jsd=False)
        acc, *_ = test(model, valloader, criterion, device)
        if acc > best_acc:
            print('Saving...')
            state = {
                'cepoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc
            }
            torch.save(state, get_model_path(opt, state='augmix'))
            best_acc = acc
    print('[info] the best retrain accuracy is {:.4f}%'.format(best_acc))


def main():
    opt = parser.parse_args()
    print(opt)

    device = torch.device(f'cuda:{opt.gpu}' if opt.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = load_model(opt, pretrained=True)
    dispatcher(opt, model, device)


if __name__ == '__main__':
    main()

