'''
This script implements reation of saliency maps for selected images
'''

import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torch
from PIL import Image
import numpy as np


from data_utils import image_dataset, transform_test, transform_val, label_to_num
from utils import set_seed
from modules import VGG16, resnet18



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



device = 'cuda'
seed = 42
set_seed(seed)

num_to_label = {value: key for key, value in label_to_num.items()}

# define which ims you want to use in .txt-file or just list in scrip
with open('good_ims.txt', 'r', encoding = 'utf-8') as in_file:
    good_im_files = in_file.readlines()



good_im_files = [elem.replace('\n', '') for elem in good_im_files]

test_set = image_dataset(image_files=good_im_files, 
                         transforms=transform_test)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


i = 0
for ims, labels, paths in iter(test_loader):
    model = VGG16().to('cuda')
    model.load_state_dict(torch.load('./weights/vgg_model.pth'))
    ims = ims.to('cuda')
    i += 1
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
    fig, axes = plt.subplots(1, 3, figsize=(10, 15))
    fp =paths[0]
    im = Image.open(fp)
    axes[1].imshow(np.array(im), cmap = 'gray')
    axes[1].set_title('Original Image')
    axes[1].axis('off')
    axes[0].imshow(saliency[0].cpu().detach().numpy(), cmap='hot')
    axes[0].set_title('Saliency Map for VGG Model')
    axes[0].axis('off')
    del model
    torch.cuda.empty_cache()
    ims.grad.zero_() 
    model = resnet18().to('cuda')
    model.load_state_dict(torch.load('./weights/resnet_model.pth'))
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
    axes[2].imshow(saliency[0].cpu().detach().numpy(), cmap='hot')
    axes[2].set_title('Saliency Map for ResNet Model')
    axes[2].axis('off')
    plt.tight_layout()
    fig.suptitle(f'Image of class {num_to_label[labels.detach().numpy()[0]]} and saliency map', y = 1.0, size = 30)
    plt.savefig(f"./good_saliencies/SaliencyMap_{fp.split('/')[-1].split('.')[0]}.png")
    del model
    torch.cuda.empty_cache()
    plt.close()


