'''
This script serves to get the uncertainties from MC dropout for defining uncertainty thresholds
No arguments, key variables have to be defined in script
  model_name --> resnet or vgg
  out_file_name --> name of json file to store results
'''

from torch.utils.data import DataLoader
import torch
import json


from data_utils import image_dataset, transform_test
from utils import set_seed, get_file_paths
from modules import VGG16, resnet18
from train_utils import mc_predict

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



model_name = 'resnet'
out_file_name = f'{model_name}_uncertainties.json'

device = 'cuda'
seed = 42
set_seed(seed)


test_set = image_dataset(ds_path='MetalSurfaceDefectsData/test', 
                        transforms=transform_test)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
val_set = image_dataset(ds_path='MetalSurfaceDefectsData/valid', 
                        transforms=transform_test)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

train_files = get_file_paths()
if model_name.lower() == 'vgg':
    model = VGG16().to('cuda')
    model.load_state_dict(torch.load('./weights/vgg16_split_ds_aug.pth'))
    step_train_files = []
    for _, elem in train_files.items():
        step_train_files.extend(elem[138:])
    train_set = image_dataset(image_files=step_train_files, 
                                transforms=transform_test)    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=False)
elif model_name.lower() == 'resnet':
    model = resnet18().to('cuda')
    model.load_state_dict(torch.load('./weights/resnet18_split_ds_aug.pth'))
    step_train_files = []
    for _, elem in train_files.items():
        step_train_files.extend(elem[:138])
    train_set = image_dataset(image_files=step_train_files, 
                                transforms=transform_test)    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=False)
else:
    print(f'No valid model type provided')


uncertainties = {'correct': {'ims': [], 'labels': [], 'preds': [], 'uncertainties': []},
                 'wrong': {'ims': [], 'labels': [], 'preds': [], 'uncertainties': []}
                 }
for x, y, paths in test_loader:
    x = x.to('cuda')
    y = y.to('cuda')
    pred_cls, un = mc_predict(model = model, 
                          x = x)
    #pred_cls = pred.argmax(dim = 1)
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

for x, y, paths in val_loader:
    x = x.to('cuda')
    y = y.to('cuda')
    pred_cls, un = mc_predict(model = model, 
                          x = x)
    #pred_cls = pred.argmax(dim = 1)
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

for x, y, paths in train_loader:
    x = x.to('cuda')
    y = y.to('cuda')
    pred_cls, un = mc_predict(model = model, 
                          x = x)
    #pred_cls = pred.argmax(dim = 1)
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
    



with open(out_file_name, 'w', encoding = 'utf-8') as out_file:
  json.dump(uncertainties, out_file)
            
            




