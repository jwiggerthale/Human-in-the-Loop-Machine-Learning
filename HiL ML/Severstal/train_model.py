'''
This script implements training of models
EfficientNet-B0 as well as ResNet is trained
Each model is pretrained on 50% of data from balanced dataset
Afterwards fine tuning on unbalanced dataset
Optionally use augmentation
Models are stored in output dir for appropriate seed
Call with args: 
    out_dir --> where to store (default: outputs)
    seed --> seed (default 42)
    use_aug --> whether to use augmentations (default 'True')
'''

import os
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import json

from modules import VGG16, resnet18, VIT, efficientnet_b0

from train_utils import train_loop
from utils import set_seed, get_file_paths
from data_utils import image_dataset, image_dataset_pd, transform_test, transform_train, transform_val, transform_aug

import argparse


parser = argparse.ArgumentParser()


parser.add_argument('--out_dir', type=str, default='outputs', help='where your results shall be saved')
parser.add_argument('--seed', type=int, default=42, help='seed to use')
parser.add_argument('--use_aug', type=str, default='True', help='whether to use advanced augmentation with gamma and brightness')
parser.add_argument('--device', type=str, default='cuda', help='device to use')


args = parser.parse_args()
out_dir = args.out_dir
seed = args.seed
use_aug = args.use_aug
device = args.device
if use_aug.lower() == 'true':
    use_aug = True
else:
    use_aug = False
    

set_seed(seed)

'''
pretraining with balanced dataset
'''

train_files = pd.read_csv('train_files_large.csv')
val_files = pd.read_csv('val_files.csv')
test_files = pd.read_csv('test_files.csv')


train_files = train_files.sample(frac=1).reset_index(drop=True)
len_train = len(train_files)
split = int(len_train/2)
train_files_rn = train_files.iloc[:split, :]
train_files_eff = train_files.iloc[split:, :]

train_set_rn = image_dataset_pd(image_files=train_files_rn, 
                            transforms=transform_train)
train_set_eff = image_dataset_pd(image_files=train_files_eff, 
                            transforms=transform_train)
aug_set_rn = image_dataset_pd(image_files=train_files_rn ,
                            transforms=transform_aug)
aug_set_eff = image_dataset_pd(image_files=train_files_eff, 
                            transforms=transform_aug)

val_set = image_dataset_pd(image_files=val_files, 
                            transforms=transform_val)


test_set = image_dataset_pd(image_files=test_files, 
                            transforms=transform_test)

train_loader_rn = DataLoader(train_set_rn, batch_size=16, shuffle=True)
aug_loader_rn = DataLoader(aug_set_rn, batch_size=16, shuffle=True)
train_loader_eff = DataLoader(train_set_eff, batch_size=64, shuffle=True)
aug_loader_eff = DataLoader(aug_set_eff, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


model = resnet18()
model_type = 'resnet18'
model.to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, min_lr=1e-6)


modeling_stats = {}
run_file = f'{out_dir}/resnet_pre/seed_{seed}'
os.makedirs(run_file, exist_ok=True)
train_loop(train_loader=train_loader_rn,
        val_loader = val_loader, 
        log_file = f'{run_file}/train_stats.txt', 
        model = model, 
        criterion = criterion, 
        optimizer = optimizer, 
        #scheduler = scheduler, 
        model_type = model_type, 
        aug_loader= aug_loader_rn,
        num_epochs=200)

    


del model
torch.cuda.empty_cache()


model = efficientnet_b0()
model_type = 'effnet'
model = model.to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, min_lr=1e-6)


modeling_stats = {}
run_file = f'{out_dir}/effnet_pre/seed_{seed}'
os.makedirs(run_file, exist_ok=True)
train_loop(train_loader=train_loader_eff,
        val_loader = val_loader, 
        log_file = f'{run_file}/train_stats.txt', 
        model = model, 
        criterion = criterion, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        model_type = model_type, 
        aug_loader= aug_loader_eff,
        num_epochs=200)


del model
torch.cuda.empty_cache()


'''
Fine tuning on unbalanced datase
'''     

with open('train_ims.json', 'r') as in_file:
    train_files = json.load(in_file)

#train_files = pd.read_csv('train_files.csv')

with open('val_ims.json', 'r') as in_file:
    val_files = json.load(in_file)


with open('test_ims.json', 'r') as in_file:
    test_files = json.load(in_file)



train_set = image_dataset(image_files=train_files, 
                            transforms=transform_train)
aug_set = image_dataset(image_files=train_files, 
                            transforms=transform_aug)

val_set = image_dataset(image_files=val_files, 
                            transforms=transform_val)


test_set = image_dataset(image_files=test_files, 
                            transforms=transform_test)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
aug_loader = DataLoader(aug_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


# Resnet
model = resnet18()
model_type = 'resnet18'
model.load_state_dict(torch.load(f'{out_dir}/resnet_pre/seed_{seed}/model_acc.pth'))
model.to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, min_lr=1e-6)


modeling_stats = {}
run_file = f'{out_dir}/resnet_fine/seed_{seed}'
os.makedirs(run_file, exist_ok=True)
train_loop(train_loader=train_loader,
        val_loader = val_loader, 
        log_file = f'{run_file}/train_stats.txt', 
        model = model, 
        criterion = criterion, 
        optimizer = optimizer, 
        #scheduler = scheduler, 
        model_type = model_type, 
        aug_loader= aug_loader,
        num_epochs=200)

del model
torch.cuda.empty_cache()

# Effnet
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
aug_loader = DataLoader(aug_set, batch_size=64, shuffle=True)

model = efficientnet_b0()
model_type = 'effnet'
model = model.to(device)
model.load_state_dict(torch.load(f'{out_dir}/effnet_pre/seed_{seed}/model_acc.pth'))

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, min_lr=1e-6)


modeling_stats = {}
run_file = f'{out_dir}/effnet_fine/seed_{seed}'
os.makedirs(run_file, exist_ok=True)
train_loop(train_loader=train_loader,
        val_loader = val_loader, 
        log_file = f'{run_file}/train_stats.txt', 
        model = model, 
        criterion = criterion, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        model_type = model_type, 
        aug_loader= aug_loader,
        num_epochs=200)
