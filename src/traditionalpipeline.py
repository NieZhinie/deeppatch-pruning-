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

class PatchingCorrect(nn.Module):
    def __init__(self, conv_layer, indices, prune_indices, order):
        super(PatchingCorrect, self).__init__()
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
        
        out[:, self.indices] = self.cru(x)
        
        return out


class PruningCorrect(nn.Module):
    def __init__(self, conv_layer, indices, prune_indices, order):
        super(PruningCorrect, self).__init__()
        self.indices = indices
        self.prune_indices = prune_indices
        self.order = order
        self.conv = conv_layer

    def forward(self, x):
        out = self.conv(x)
        out[:, self.prune_indices] = 0
        
        return out


def extract_info(model, info_type):
    info = {}
    for n, m in model.named_modules():
        if isinstance(m, PruningCorrect) \
            or isinstance(m, PatchingCorrect):
            if info_type == "indices":
                info[n] = m.indices
            elif info_type == "prune_indices":
                info[n] = m.prune_indices
            elif info_type == "order":
                info[n] = m.order
    return info



def traditional_finetune(opt, model, device):
    print('running traditional finetuning')
    _, trainloader = load_dataset(opt, split='train')
    _, valloader = load_dataset(opt, split='val')

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')
    for epoch in range(0, opt.trad_finetune_epoch):
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
    return model

def traditional_pruning(opt, model, device):
    print('running traditional pruning')
    prune_filters = json.load(open(os.path.join(
        opt.output_dir, opt.dataset, opt.model, 'susp_filters_distweight.json'
    )))

    sus_filters = json.load(open(os.path.join(
        opt.output_dir, opt.dataset, opt.model, 'susp_filters_perfloss.json'
    ))) if opt.susp_side in ('front', 'rear') else {}

    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
  # print(conv_names)
    for layer_name in conv_names:
        module = rgetattr(model, layer_name)

        num_susp = int(module.out_channels * opt.susp_ratio)
        num_prune = int(module.out_channels * opt.prune_ratio)
             
        if opt.prune_side=='front':
          prune_indices = prune_filters[layer_name][opt.prune_indices][:num_prune] if opt.prune and num_prune > 0 else None
          indices = [idx for idx in sus_filters[layer_name][opt.patch_indices] if idx not in prune_indices]
          indices = indices[:num_susp] if num_susp > 0 else None
        
        elif opt.prune_side=='rear':
            prune_indices = prune_filters[layer_name][opt.prune_indices][-num_prune:] if opt.prune and num_prune > 0 else None
            indices = [idx for idx in sus_filters[layer_name][opt.patch_indices] if idx not in prune_indices]
            indices = indices[:num_susp] if num_susp > 0 else None
        else:
            raise ValueError('Invalid suspicious side')

        if module.groups != 1:
            continue

  
        correct_module = PruningCorrect(module, indices, prune_indices, opt.pporder)            
        rsetattr(model, layer_name, correct_module)

        
    model=traditional_finetune(opt, model, device)
    return model

@dispatcher.register('traditionalpipeline')
def traditionalpipeline(opt, model, device):
  
  model=traditional_pruning(opt, model, device)
  

  prune_filters = json.load(open(os.path.join(
        opt.output_dir, opt.dataset, opt.model, 'susp_filters_distweight.json'
    )))

  sus_filters = json.load(open(os.path.join(
        opt.output_dir, opt.dataset, opt.model, 'susp_filters_perfloss.json'
    ))) if opt.susp_side in ('front', 'rear') else {}

  conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
  # print(conv_names)
  for layer_name in conv_names:
        
        # if layer_name.endswith('.conv'):
        #   layer_name_original = layer_name[:-5]
        # else:
        #   layer_name_original =layer_name
        module = rgetattr(model, layer_name)

        if layer_name.endswith('.conv'):
          print("1111")
          layer_name = layer_name[:-5]
         # 定义新的名称，这里简单地在原有名称前加上"custom_"
          module._get_name = lambda: layer_name

        num_susp = int(module.out_channels * opt.susp_ratio)
        num_prune = int(module.out_channels * opt.prune_ratio)
             
        if opt.prune_side=='front':
          prune_indices = prune_filters[layer_name][opt.prune_indices][:num_prune] if opt.prune and num_prune > 0 else None
          indices = [idx for idx in sus_filters[layer_name][opt.patch_indices] if idx not in prune_indices]
          indices = indices[:num_susp] if num_susp > 0 else None
        
        elif opt.prune_side=='rear':
            prune_indices = prune_filters[layer_name][opt.prune_indices][-num_prune:] if opt.prune and num_prune > 0 else None
            indices = [idx for idx in sus_filters[layer_name][opt.patch_indices] if idx not in prune_indices]
            indices = indices[:num_susp] if num_susp > 0 else None
        else:
            raise ValueError('Invalid suspicious side')

        if module.groups != 1:
            continue

  
        correct_module = PatchingCorrect(module, indices, prune_indices, opt.pporder)            
        rsetattr(model, layer_name, correct_module)
        # print(layer_name)
        # print(indices)
        # print(prune_indices)
  _, trainloader = load_dataset(opt, split='train', noise=True, noise_type='random')
  _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append')
        # _, trainloader = load_dataset(opt, split='train')
        # _, valloader = load_dataset(opt, split='val')
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
            if isinstance(m, PatchingCorrect) \
                    or isinstance(m, PruningCorrect):
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
                # 'indices': extract_info(model, 'indices'),
                # 'prune_indices': extract_info(model, 'prune_indices'),
                # 'order': extract_info(model, 'order')
            }
            torch.save(state, get_model_path(opt, state=f'patch_{opt.fs_method}'))
            best_acc = acc
        scheduler.step()
  print('[info] the best retrain accuracy is {:.4f}%'.format(best_acc))

def main():
    opt = parser.parse_args()
    print(opt)

    device = torch.device(f'cuda:{opt.gpu}' if opt.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = load_model(opt, pretrained=True)
    dispatcher(opt, model, device)

if __name__ == '__main__':
    main()
