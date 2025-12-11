from torch.utils.data import DataLoader
import torch
import json
import pandas as pd


from data_utils import image_dataset, image_dataset_pd, transform_test, transform_train, transform_val
from utils import set_seed, get_file_paths
from modules import VGG16, resnet18, efficientnet_b0
from train_utils import mc_predict

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'
seed = 42
set_seed(seed)

val_files = pd.read_csv('val_files.csv')
val_set = image_dataset_pd(image_files=val_files, 
                            transforms=transform_val)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
model_name = 'resnet'

if model_name.lower() == 'resnet':
    model = resnet18().to('cuda')
    model.load_state_dict(torch.load('./weights/resnet_acc_tuned.pth'))

elif model_name.lower() == 'effnet':
    model = efficientnet_b0().to('cuda')
    model.set_dropout(p = 0.2)
    model.load_state_dict(torch.load('./weights/effnet_acc_tuned.pth'))

else:
    print(f'No valid model type provided')

model.eval()

uncertainties = {'correct': {'ims': [], 'labels': [], 'preds': [], 'uncertainties': []},
                 'wrong': {'ims': [], 'labels': [], 'preds': [], 'uncertainties': []}
                 }
for x, y, paths in val_loader:
    x = x.to('cuda')
    y = y.to('cuda')
    pred_cls, un = mc_predict(model = model, 
                          x = x)
    correct = pred_cls == y
    for i, elem in enumerate(pred_cls.reshape(-1)):
        label = y[i].item()
        fp = paths[i]
        unc = un[i].sum().item()
        if correct[i].int().item() == 1:
            uncertainties['correct']['ims'].append(fp)
            uncertainties['correct']['labels'].append(label)
            uncertainties['correct']['preds'].append(elem.item())
            uncertainties['correct']['uncertainties'].append(unc)
        else:
            uncertainties['wrong']['ims'].append(fp)
            uncertainties['wrong']['labels'].append(label)
            uncertainties['wrong']['preds'].append(elem.item())
            uncertainties['wrong']['uncertainties'].append(unc)
    del x
    del y
    del pred_cls
    del un
    torch.cuda.empty_cache()

with open(f'./outputs/uncertainties/{model_name}_uncertainties.json', 'w', encoding = 'utf-8') as out_file:
  json.dump(uncertainties, out_file)
            
            
