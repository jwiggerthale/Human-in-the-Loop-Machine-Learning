''''
This script implements calculation of model uncertianty when incrementally increasing the share of pixels in cluded in the image
For more details see Sec. 4.2 (Fig. 6) of our paper
'''

import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torch
import json
import numpy as np


from data_utils import image_dataset, label_to_num, transform_test
from train_utils import mc_predict
from utils import set_seed, get_file_paths
from modules import VGG16, resnet18



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



device = 'cuda'
seed = 42
set_seed(seed)

num_to_label = {value: key for key, value in label_to_num.items()}


model_name = 'vgg'
train_files = get_file_paths()
if model_name.lower() == 'vgg':
    model = VGG16().to('cuda')
    model.load_state_dict(torch.load('./weights/vgg_model.pth'))
    model_type = 'vgg16'
    step_train_files = []
    for _, elem in train_files.items():
        step_train_files.extend(elem[:138])
    train_set = image_dataset(image_files=step_train_files, 
                                transforms=transform_test)
elif model_name.lower() == 'resnet':
    model = resnet18().to('cuda')
    model.load_state_dict(torch.load('./weights/resnet_model.pth'))
    model_type = 'resnet18'
    step_train_files = []
    for _, elem in train_files.items():
        step_train_files.extend(elem[138:])
    train_set = image_dataset(image_files=step_train_files, 
                                transforms=transform_test)
    


train_loader = DataLoader(train_set, batch_size=1, shuffle=False)



confidences = {}
for share in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]:
    model_uncs = []
    i = 0
    for ims, labels, paths in iter(train_loader):
        ims = ims.to('cuda')
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
        saliency_flat = saliency.flatten()
        k = int(share/100 * saliency_flat.numel())
        threshold_value = torch.topk(saliency_flat, k).values.min()
        top_5_percent_mask = saliency >= threshold_value
        highlighted_ims = ims.clone()
        for c in range(3):
            highlighted_ims[0][c][~top_5_percent_mask[0]] = 0
        _, unc = mc_predict(model = model, 
                        x = highlighted_ims)
        model_uncs.append(unc.item())
    confidences[share] = np.mean(model_uncs)
        


with open(f'ModelUncsShare_vgg.json', 'w', encoding='utf-8') as out_file:
    json.dump(confidences, out_file)
    
