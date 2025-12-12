'''
This script implements creation of an uncertainty curve as described in Sec. 4.2 of our elabortaion
Share of pixels available for classification is increased gradually and uncertainty is calculated
Results are stored in a .json file that can be used for plotting
'''

import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torch
import json
import numpy as np
import pandas as pd


from data_utils import image_dataset_pd, transform_test
from train_utils import mc_predict
from utils import set_seed
from modules import efficientnet_b0, resnet18



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



device = 'cuda'
seed = 42
set_seed(seed)



model_name = 'effnet'
if model_name.lower() == 'effnet':
    model = efficientnet_b0().to('cuda')
    model.load_state_dict(torch.load(f'./outputs/effnet_fine/seed_{seed}model_acc.pth'))



elif model_name.lower() == 'resnet':
    model = resnet18().to('cuda')
    model.load_state_dict(torch.load(f'./outputs/resnet_fine/seed_{seed}model_acc.pth'))

model.eval()                        
out_name = f'./outputs/reliability/uncertainty_curve_seed_{seed}_{model_name}.json'
if not os.path.isdir('./outputs/reliability'):
  os.makedirs('./outputs/reliability', exist_ok = True)

model.eval()

test_files = pd.read_csv('test_files.csv')

test_set = image_dataset_pd(image_files=test_files, 
                            transforms=transform_test)


test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

confidences = {}
for share in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]:
    model_uncs = []
    for ims, labels, paths in iter(test_loader):
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
        top_share_percent_mask = saliency >= threshold_value
        highlighted_ims = ims.clone()
        for c in range(3): 
            highlighted_ims[0][c][~top_share_percent_mask[0]] = 0
        _, unc = mc_predict(model = model, 
                        x = highlighted_ims)
        model_uncs.append(unc.item())
    confidences[share] = np.mean(model_uncs)
        


with open(out_file, 'w', encoding='utf-8') as out_file:
    json.dump(confidences, out_file)
    
