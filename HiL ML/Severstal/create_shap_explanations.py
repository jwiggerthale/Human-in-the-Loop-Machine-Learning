'''
This script creates explantions with for model with shap
LOT OF MEMORY REQUIRED
'''
import os
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import shap

from modules import VGG16, resnet18, VIT, efficientnet_b0

from train_utils import train_loop
from utils import set_seed, get_file_paths
from data_utils import image_dataset, image_dataset_pd, transform_test, transform_train, transform_val, transform_aug

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


train_files = pd.read_csv('train_files_large.csv')
val_files = pd.read_csv('val_files.csv')
test_files = pd.read_csv('test_files.csv')


train_set = image_dataset_pd(image_files=train_files, 
                            transforms=transform_train)
aug_set = image_dataset_pd(image_files=train_files, 
                            transforms=transform_aug)

val_set = image_dataset_pd(image_files=val_files, 
                            transforms=transform_val)
test_set = image_dataset_pd(image_files=test_files, 
                            transforms=transform_test)


if model_name.lower() == 'resnet':
    model = resnet18()
    model_type = 'resnet18'
    model.load_state_dict(torch.load('./weights/resnet_acc_tuned.pth')) 
elif model_name.lower() == 'efficientnet':
    model = efficientnet_b0()
    model_type = 'effnet'
    model.load_state_dict(torch.load('./weights/effnet_acc_tuned.pth'))

model = model.to(device)
model.eval()    

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
aug_loader = DataLoader(aug_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False)



train_ims, train_labels, train_paths = next(iter(train_loader))
train_ims = train_ims.to(device)
train_ims = train_ims.clone().detach().requires_grad_(True)




explainer = shap.GradientExplainer(model, train_ims)

num_samples = 10
counter = 0
for test_ims, test_labels, test_paths in iter(test_loader):
    test_ims = test_ims.to(device)
    shap_values, indexes = explainer.shap_values(
        test_ims,
        ranked_outputs=1,        
        output_rank_order="max")

    # Get shap values in numpy array
    shap_array = shap_values.squeeze(-1)               
    shap_numpy = np.transpose(shap_array, (0, 2, 3, 1))

    # Get images in numpy array
    test_np = test_ims.detach().cpu().numpy()
    test_np = np.transpose(test_np, (0, 2, 3, 1))  # (N, H, W, C)

    # denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    test_np_denorm = std * test_np + mean
    test_np_denorm = np.clip(test_np_denorm, 0, 1)

    import matplotlib.pyplot as plt

    # get predictions for and plot everything
    with torch.no_grad():
        logits = model(test_ims)
        probs = torch.softmax(logits, dim=1)
        pred_cls = probs.argmax(dim=1).cpu().numpy()


    shap.image_plot(
        [shap_numpy],          # Liste von Arrays, eins pro Output
        test_np_denorm,        # Originalbilder
        labels=pred_cls
    )
    plt.savefig(f'.outputs/ims/SHAP_Explanations/{counter}.png')
    counter += 1
    if counter > num_samples:
        break

