'''
This script implements a training logic as described in our Sec. 3.3 of our paper
Model is trained iteratively on increasing amount of images
Stats are recorded
'''


####
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
import datetime

from modules import VGG16, resnet18

from train_utils import train_loop, get_stats
from utils import set_seed, get_file_paths
from data_utils import image_dataset, transform_test, transform_train, transform_val

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='vgg', help='type of model you want to train [vgg or resnet]')
parser.add_argument('--out_dir', type=str, default='vgg', help='where your results shall be saved')


args = parser.parse_args()
out_dir = args.out_dir
model_name = args.model


device = 'cuda'
seed = 42
set_seed(seed)

runtime_stats = []

for initial_train_samples in [5, 10, 15, 20]:
    for step_size in [2, 4, 6, 8, 10]:
        start = datetime.datetime.now()
        if model_name.lower() == 'vgg':
            model = VGG16()
        elif model_name.lower() == 'resnet':
            model = resnet18()
        else:
            print(f'No valid model type provided')
            break
        model = model.to(device)
        model.freeze_weights()
        criterion = nn.CrossEntropyLoss() 
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        train_files = get_file_paths()
        step_train_files = []
        for _, elem in train_files.items():
            step_train_files.extend(elem[:initial_train_samples])
        train_set = image_dataset(image_files=step_train_files, 
                                transforms=transform_train)
        val_set = image_dataset(ds_path='MetalSurfaceDefectsData/valid', 
                                transforms=transform_val)
        test_set = image_dataset(ds_path='MetalSurfaceDefectsData/test', 
                                transforms=transform_test)
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
        modeling_stats = {}
        run_file = f'{out_dir}/{initial_train_samples}/{step_size}'
        os.makedirs(run_file, exist_ok=True)
        train_loop(train_loader=train_loader, 
                val_loader = val_loader, 
                log_file = f'{run_file}/train_stats_{initial_train_samples}_samples.txt', 
                model = model, 
                early_stopping=5,
                criterion = criterion, 
                optimizer = optimizer)
        #torch.save(model.state_dict(), f'{run_file}/model_{initial_train_samples}_samples.pth')
        last_train_samples = initial_train_samples
        counter = 0
        for train_samples in np.arange(initial_train_samples + step_size, 276 + step_size, step_size):
            if train_samples > 276:
                train_samples = 276
            new_files = []
            for _, elem in train_files.items():
                new_files.extend(elem[last_train_samples:train_samples])
            new_set = image_dataset(image_files=new_files, 
                                    transforms=transform_test)
            new_loader = DataLoader(new_set, batch_size=16, shuffle=False)
            val_acc, val_loss, manually_labeled = get_stats(new_loader, 
                                                            model = model, 
                                                            criterion = criterion)
            modeling_stats[str(train_samples)] = {'acc': str(val_acc), 
                                            'loss': str(val_loss), 
                                            'effort': str(manually_labeled)}
            step_train_files = []
            for _, elem in train_files.items():
                step_train_files.extend(elem[:train_samples])
            train_set = image_dataset(image_files=step_train_files, 
                                transforms=transform_train)
            train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
            train_loop(train_loader=train_loader, 
                val_loader = val_loader, 
                log_file = f'{run_file}/train_stats_{train_samples}_samples.txt', 
                num_epochs=20, 
                early_stopping=3, 
                model = model, 
                criterion = criterion, 
                optimizer = optimizer)
            #torch.save(model.state_dict(), f'{run_file}/model_{train_samples}_samples.pth')
        end = datetime.datetime.now()
        delta = end - start 
        runtime_stats.append([initial_train_samples, step_size, delta])
        with open(f'{run_file}/modeling_stats.json', 'w', encoding='utf-8') as out_file:
            json.dump(modeling_stats, out_file)



pd.DataFrame(runtime_stats, columns = ['initial_samples', 'steps_size', 'time']).to_csv(f'RuntimeStats_{model_name}.csv', index = False)
torch.save(model.state_dict(), f'./weights/support_model_{model_name}.pth')
            
