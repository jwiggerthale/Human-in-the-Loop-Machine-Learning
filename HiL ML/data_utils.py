'''
This script contains everything required to create a data loader for metal surface dataset
'''


from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
from PIL import Image
import numpy as np
import random


label_to_num = {'Scratches': 0, 
                'Rolled': 1, 
                'Pitted': 2, 
                'Patches': 3, 
                'Inclusion': 4, 
                'Crazing': 5}


class RandomGradient:
    def __init__(self, p=0.5, max_alpha=0.5):
        self.p = p
        self.max_alpha = max_alpha

    def __call__(self, img_tensor):
        if random.random() >= self.p:
            return img_tensor
        C, H, W = img_tensor.shape
        min_val = img_tensor.min()
        max_val = img_tensor.max()
        a = random.uniform(0.0, self.max_alpha)
        # Horizontal oder vertikal
        if random.random() < 0.5:
            # Horizontal
            if random.random() < 0.5:
                ramp = torch.linspace(1.0 - a, 1.0 + a, steps=W).view(1, 1, W).expand(1, H, W)
            else:
                ramp = torch.linspace(1.0 + a, 1.0 - a, steps=W).view(1, 1, W).expand(1, H, W)
        else:
            # Vertikal
            if random.random() < 0.5:
                ramp = torch.linspace(1.0 - a, 1.0 + a, steps=H).view(1, H, 1).expand(1, H, W)
            else:
                ramp = torch.linspace(1.0 + a, 1.0 - a, steps=H).view(1, H, 1).expand(1, H, W)
        out = img_tensor * ramp
        return torch.clamp(out, min_val, max_val)



transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.095, 0.095, 0.095], [0.149, 0.149, 0.149])
    #transforms.Normalize([0.09936217326743935, 0.09936217326743935, 0.09936217326743935],[0.15983977644971742, 0.15983977644971742, 0.15983977644971742])
    ])
transform_aug = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    RandomGradient(p = 1.1),
    transforms.Normalize([0.095, 0.095, 0.095], [0.149, 0.149, 0.149])
    #transforms.Normalize([0.09936217326743935, 0.09936217326743935, 0.09936217326743935],[0.15983977644971742, 0.15983977644971742, 0.15983977644971742])
    ])
transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.095, 0.095, 0.095], [0.149, 0.149, 0.149])
    #transforms.Normalize([0.09936217326743935, 0.09936217326743935, 0.09936217326743935],[0.15983977644971742, 0.15983977644971742, 0.15983977644971742])
    ])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.095, 0.095, 0.095], [0.149, 0.149, 0.149])
    #transforms.Normalize([0.09936217326743935, 0.09936217326743935, 0.09936217326743935],[0.15983977644971742, 0.15983977644971742, 0.15983977644971742])
    ])



class image_dataset(Dataset):
    def __init__(self, 
                 ds_path: str = 'MetalSurfaceDefectsData/train', 
                 image_files: list = None,
                 transforms: transforms.Compose = None):
        super().__init__()
        data_dirs = os.listdir(ds_path)
        assert os.path.isdir(ds_path)
        files = {}
        dirs = [ds_path]
        if image_files == None:
            while len(dirs) > 0:
                d = dirs[0]
                for f in os.listdir(d):
                    name = f'{d}/{f}'
                    if os.path.isfile(name):
                        if not d in files:
                            files[d] = []
                        files[d].append(name)
                    else:
                        dirs.append(name)
                dirs.remove(d)
            self.files = []
            for _, file_names in files.items():
                self.files.extend(file_names)
        else:
            self.files = image_files
        self.transforms = transforms
    def __len__(self):
        return(len(self.files))
    def __getitem__(self, index):
        fp = self.files[index]
        im = Image.open(fp).convert("L")
        # Convert to RGB
        im = Image.merge("RGB", (im, im, im))
        im = np.array(im)
        label = fp.split('/')[-2]
        label = label_to_num[label]
        if self.transforms != None:
            im = self.transforms(im)
        return im, label, fp
            
