'''
this script creates saliency maps
image files are taken from STATS_FILE.json
  --> direct link to uncertainty allows to partition saliency maps
saliency maps are created and stored in out dir (subfolder for images with high and low uncertainty)
'''

import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torch
import numpy as np
import json
import pandas as pd
import os

from data_utils import image_dataset, image_dataset_pd, transform_test, transform_val, label_to_num
from utils import set_seed
from modules import VGG16, resnet18, efficientnet_b0



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


out_dir = './outputs'
if not os.path.isdir(f'{out_dir}/good_saliencies'):
    os.makedirs(f'{out_dir}/good_saliencies', exist_ok=True)
if not os.path.isdir(f'{out_dir}/bad_saliencies'):
    os.makedirs(f'{out_dir}/bad_saliencies', exist_ok=True)

device = 'cuda'
seed = 42
set_seed(seed)


fp = 'model_validation.json'
with open(fp, 'r', encoding = 'utf-8') as in_file:
    stats = json.load(in_file)


bad_ims = {}
good_ims = {}
wrong_preds = 0
for fp, stat in stats.items():
    found = False
    if stat['deviation'] == '1':
        found  = True
    if stat['eff_uncertain'] == '1':
        found = True
    if stat['rn_uncertain'] == '1':
        found = True
    if (stat['eff_correct'] == '0' or stat['rn_correct'] == '0'):
        bad_ims[fp] = stat['label']
    if found == False:
        good_ims[fp] = stat['label']

    

good_set = image_dataset(image_files=good_ims, 
                         transforms=transform_test)
good_loader = DataLoader(good_set, batch_size=1, shuffle=False)
bad_set = image_dataset(image_files=bad_ims, 
                         transforms=transform_test)
bad_loader = DataLoader(bad_set, batch_size=1, shuffle=False)


model = efficientnet_b0().to('cuda')
model.load_state_dict(torch.load('./weights/effnet_acc_tuned.pth'))

rn_model = resnet18().to('cuda')
rn_model.load_state_dict(torch.load('./weights/resnet_acc_tuned.pth'))
test_files = pd.read_csv('test_files.csv')


i = 0
for ims, labels, paths in iter(good_loader):
    ims = ims.to('cuda')
    i += 1
    if(i>=200):
        break
    ims.requires_grad_(True)
    out = model.forward(ims)
    ims.requires_grad = True
    out = model.forward(ims)
    one_hot_output = torch.zeros(out.size()).to('cuda')
    one_hot_output[0][out[0].argmax()] = 1
    preds = torch.exp(out)
    preds = preds.max(dim=1)[1]
    out.backward(gradient=one_hot_output, retain_graph=True)  
    saliency, _ = torch.max(ims.grad.data.abs(), dim=1) 
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.02})
    img = ims[0].detach().cpu().permute(1,2,0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  
    img = np.clip(img, 0, 1)
    fp =paths[0]
    axes[1].imshow(np.array(img), cmap = 'gray')
    axes[1].set_title('Original Image')
    axes[1].axis('off')
    im0 = axes[0].imshow(np.flipud(saliency[0].cpu().detach().numpy()), cmap = 'hot')
    axes[0].set_title('Saliency Map for EffieientNet Model')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.086, pad=0.04, orientation = 'horizontal')
    torch.cuda.empty_cache()
    ims.grad.zero_() 
    ims.requires_grad_(True)
    out = rn_model.forward(ims)
    classes = [x.argmax(dim = 0) for x in out]
    rn_model.zero_grad()
    out = rn_model.forward(ims) 
    one_hot_output = torch.zeros(out.size()).to('cuda')
    one_hot_output[0][out[0].argmax()] = 1
    preds = torch.exp(out)
    preds = preds.max(dim=1)[1]
    out.backward(gradient=one_hot_output, retain_graph=True) 
    saliency, _ = torch.max(ims.grad.data.abs(), dim=1)  
    im2 = axes[2].imshow(np.flipud(saliency[0].cpu().detach().numpy()), cmap = 'hot')
    axes[2].set_title('Saliency Map for ResNet Model')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.086, pad=0.04, orientation = 'horizontal')
    plt.tight_layout()
    fig.suptitle(f'Image of class {labels.detach().numpy()[0]} and saliency map', y = 1.0, size = 30)
    plt.savefig(f"./sample_saliencies/SaliencyMap_class{labels.detach().numpy()[0]}_{fp.split('/')[-1].split('.')[0]}.png", bbox_inches='tight')
    torch.cuda.empty_cache()
    plt.close()

counter = 0
for ims, labels, paths in iter(bad_loader):
    counter += 1
    ims = ims.to('cuda')
    if(counter >= 200):
        break
    ims.requires_grad_(True)
    out = model.forward(ims)
    classes = [x.argmax(dim = 0) for x in out]
    model.zero_grad()
    out = model.forward(ims)  
    one_hot_output = torch.zeros(out.size()).to('cuda')
    one_hot_output[0][out[0].argmax()] = 1
    preds = torch.exp(out)
    preds = preds.max(dim=1)[1]
    out.backward(gradient=one_hot_output, retain_graph=True)  
    saliency, _ = torch.max(ims.grad.data.abs(), dim=1)  
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.02})
    img = ims[0].detach().cpu().permute(1,2,0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    fp =paths[0]
    axes[1].imshow(np.array(img), cmap = 'gray')
    axes[1].set_title('Original Image')
    axes[1].axis('off')
    im0 = axes[0].imshow(np.fliplr(saliency[0].cpu().detach().numpy()), cmap = 'hot')
    axes[0].set_title('Saliency Map for EffieientNet Model')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.086, pad=0.04, orientation = 'horizontal')
    torch.cuda.empty_cache()
    ims.grad.zero_() 
    ims.requires_grad_(True)
    out = rn_model.forward(ims)
    classes = [x.argmax(dim = 0) for x in out]
    rn_model.zero_grad()
    out = rn_model.forward(ims)  
    one_hot_output = torch.zeros(out.size()).to('cuda')
    one_hot_output[0][out[0].argmax()] = 1
    preds = torch.exp(out)
    preds = preds.max(dim=1)[1]
    out.backward(gradient=one_hot_output, retain_graph=True)  
    saliency, _ = torch.max(ims.grad.data.abs(), dim=1) 
    im2 = axes[2].imshow(np.fliplr(saliency[0].cpu().detach().numpy()), cmap = 'hot')
    axes[2].set_title('Saliency Map for ResNet Model')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.086, pad=0.04, orientation = 'horizontal')
    plt.tight_layout()
    fig.suptitle(f'Image of class {num_to_label[labels.detach().numpy()[0]]} and saliency map', y = 1.0, size = 30)
    plt.savefig(f"{out_dir}/bad_saliencies/SaliencyMap_class{labels.detach().numpy()[0]}_{fp.split('/')[-1].split('.')[0]}.png", bbox_inches='tight')
    torch.cuda.empty_cache()
    plt.close()
