'''
This script implements training of a model on steel surface dataset
Args: 
  model --> vgg or resnet
  out_dir --> directory to save results (log file, weights, ...)
  use_aug --> whether to use Augmentation (as described in Sec. 4.3 of our paper)
'''

####
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json

from modules import VGG16, resnet18

from train_utils import train_loop
from utils import set_seed, get_file_paths
from data_utils import image_dataset, transform_test, transform_train, transform_val, transform_aug

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='vgg', help='type of model you want to train [vgg or resnet]')
parser.add_argument('--out_dir', type=str, default='vgg', help='where your results shall be saved')
parser.add_argument('--use_aug', type=str, default='False', help='Whether to use augmentations')


args = parser.parse_args()
out_dir = args.out_dir
model_name = args.model
use_aug = args.use_aug

if use_aug.lower() == 'true':
    use_aug = True
else:
    use_aug = False
print(use_aug)


device = 'cuda'
seed = 42
set_seed(seed)

train_files = get_file_paths()
if model_name.lower() == 'vgg':
    model = VGG16()
    model_type = 'vgg16'
    step_train_files = []
    for _, elem in train_files.items():
        step_train_files.extend(elem[:138])
    train_set = image_dataset(image_files=step_train_files, 
                                transforms=transform_train)
elif model_name.lower() == 'resnet':
    model = resnet18()
    model_type = 'resnet18'
    step_train_files = []
    for _, elem in train_files.items():
        step_train_files.extend(elem[138:])
    train_set = image_dataset(image_files=step_train_files, 
                                transforms=transform_train)

model = model.to(device)
model.freeze_weights()
# === 4) Loss & Optimizer === 
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

if use_aug:
    aug_set = image_dataset(image_files=step_train_files, 
                            transforms=transform_aug)
    aug_loader = DataLoader(aug_set, batch_size=16, shuffle=True)
else:
    aug_loader = None
val_set = image_dataset(ds_path='MetalSurfaceDefectsData/valid', 
                        transforms=transform_val)
test_set = image_dataset(ds_path='MetalSurfaceDefectsData/test', 
                        transforms=transform_test)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
modeling_stats = {}
run_file = f'{out_dir}/100_ims/run_1'
os.makedirs(run_file, exist_ok=True)
train_loop(train_loader=train_loader, 
           aug_loader = aug_loader,
        val_loader = val_loader, 
        log_file = f'{run_file}/train_stats_final.txt', 
        model = model, 
        criterion = criterion, 
        optimizer = optimizer)

if use_aug == True:
    aug = 'aug'
else:
    aug = ''
torch.save(model.state_dict(), f'./weights/{model_type}_split_ds_{aug}.pth')
            
