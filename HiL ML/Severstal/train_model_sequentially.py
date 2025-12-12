'''
This script implements iterative training of a model for data collection process
please refer to Sec. 3.3. and 4.1 for details on the process
in the script, 
    - a model is trained sequentially on a dataset that is getting larger and larger
    - for each enlargement of the dataset, it is measured how many images had been classified correctly by the model 
    - test is repeated for different step sizes and initial dataset sizes
    - statistics are saved to csv file 
arguments: 
    model --> model type tp be trained 
    out_dir --> where to store
    seed --> seed to use
'''
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
import json
import datetime

from modules import VGG16, resnet18, efficientnet_b0

from train_utils import train_loop, get_stats
from utils import set_seed, get_file_paths
from data_utils import image_dataset, transform_test, transform_train, transform_val

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='resnet', help='type of model you want to train [vgg or resnet]')
parser.add_argument('--out_dir', type=str, default='./outputs/data_collection', help='where your results shall be saved')
parser.add_argument('--seed', type=int, default=42, help='where your results shall be saved')


args = parser.parse_args()
out_dir = args.out_dir
model_name = args.model
seed = args.seed

device = 'cuda'
set_seed(seed)

runtime_stats = []


with open('train_ims_balanced.json', 'r') as in_file:
    train_files = json.load(in_file)

with open('val_ims_balanced.json', 'r') as in_file:
    val_files = json.load(in_file)




class_counts = {i: 0 for i in range(5)}
for file, cat in train_files.items():
        class_counts[cat] += 1


for initial_train_samples in [5, 10, 15, 20]:
    for step_size in [5, 10, 15]:
        start = datetime.datetime.now()
        if model_name.lower() == 'vgg':
            model = VGG16()
        elif model_name.lower() == 'resnet':
            model = resnet18()
        elif model_name.lower() == 'efficentnet':
            model = efficientnet_b0()
        else:
            print(f'No valid model type provided')
            break
        model = model.to(device)
        #model.freeze_weights()
        criterion = nn.CrossEntropyLoss() 
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, min_lr=1e-6)
        step_train_files = {}
        class_counts = {i: 0 for i in range(5)}
        for file, cat in train_files.items():
            if class_counts[cat] < initial_train_samples:
                step_train_files[file] = cat
                class_counts[cat] += 1
        train_set = image_dataset(image_files=step_train_files, 
                                transforms=transform_train)
        val_set = image_dataset(val_files, 
                                transforms=transform_val)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
        modeling_stats = {}
        run_file = f'{out_dir}/{seed}/{initial_train_samples}/{step_size}'
        os.makedirs(run_file, exist_ok=True)
        train_loop(train_loader=train_loader, 
                val_loader = val_loader, 
                log_dir = run_file, 
                model = model, 
                early_stopping=5,
                criterion = criterion, 
                optimizer = optimizer, 
                model_type=model_name, 
                num_samples = initial_train_samples, 
                scheduler=scheduler)
        last_train_samples = initial_train_samples
        counter = 0
        for train_samples in np.arange(initial_train_samples + step_size, 195 + step_size, step_size):
            if train_samples > 195:
                train_samples = 195
            new_files = {}
            class_counts = {i: 0 for i in range(5)}
            for file, cat in train_files.items():
                if class_counts[cat] < train_samples:
                    class_counts[cat] += 1
                    if class_counts[cat] > last_train_samples:
                        new_files[file] = cat
            print(len(new_files))
            last_train_samples = train_samples
            new_set = image_dataset(image_files=new_files, 
                                    transforms=transform_test)
            new_loader = DataLoader(new_set, batch_size=16, shuffle=False)
            val_acc, val_loss, manually_labeled = get_stats(new_loader, 
                                                            model = model, 
                                                            criterion = criterion)
            modeling_stats[str(train_samples)] = {'acc': str(val_acc), 
                                            'loss': str(val_loss), 
                                            'effort': str(manually_labeled)}
            step_train_files = {}
            class_counts = {i: 0 for i in range(5)}
            for file, cat in train_files.items():
                if class_counts[cat] < train_samples:
                    class_counts[cat] += 1
                    step_train_files[file] = cat
            train_set = image_dataset(image_files=step_train_files, 
                                transforms=transform_train)
            train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, min_lr=1e-6)
            train_loop(train_loader=train_loader, 
                val_loader = val_loader, 
                log_dir = run_file, 
                num_epochs=20, 
                early_stopping=3, 
                model = model, 
                criterion = criterion, 
                optimizer = optimizer, 
                model_type=model_name, 
                num_samples = train_samples, 
                scheduler=scheduler)
        end = datetime.datetime.now()
        delta = end - start 
        runtime_stats.append([initial_train_samples, step_size, delta])
        with open(f'{out_dir}/modeling_stats_{model_name}_{seed}_{initial_train_samples}_{step_size}.json', 'w', encoding='utf-8') as out_file:
            json.dump(modeling_stats, out_file)



pd.DataFrame(runtime_stats, columns = ['initial_samples', 'steps_size', 'time']).to_csv(f'{out_dir}/RuntimeStats_{model_name}_seed_{seed}.csv', index = False)
            
