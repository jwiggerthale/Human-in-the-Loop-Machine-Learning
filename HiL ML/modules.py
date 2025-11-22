'''
This script implements base models for image recognition
Each model is based on the implementation from torchvision and employs a targeted classification head/ fc-layer
Initialize model with number of classes in your use case
Optionally: use pretrained weights
  --> if so: make sure, you provide the correct path to model weights
'''

import torch
import torch.nn as nn
from torchvision import models





class VGG16(nn.Module):
    def __init__(self, 
                num_classes: int = 6, 
                pretrained: bool = True, 
                model_weights: str = './weights/vgg16_pretrained.pth'):
        super().__init__()
        model = models.vgg16()
        if pretrained == True:
          model.load_state_dict(torch.load(model_weights))
        self.feature_extractor = model.features
        self.pooler = model.avgpool
        self.clf = model.classifier
        self.clf[6] = nn.Linear(4096, num_classes)
    # freeze feature extractor
    def freeze_weights(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
    def forward(self, 
                x: torch.tensor):
        pred = self.feature_extractor(x)
        pred = self.pooler(pred)
        pred = pred.reshape(-1, 25088)
        pred = self.clf(pred)
        return pred



'''
custom, flexible class for resnet18 model with dropout
'''
class resnet18(nn.Module):
    def __init__(self, 
                num_classes: int = 6, 
                dropout_rate: float = 0.5, 
                pretrained: bool = True, 
                model_weights: str = './weights/resnet18_pretrained.pth'):
        super().__init__()
        self.model = models.resnet18()
        if pretrained == True:
          self.model.load_state_dict(torch.load(model_weights))
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(num_features, num_classes)
    # freeze feature extractor
    def freeze_weights(self, 
                       layers: list = [1,2,3]):
        if 1 in layers:
            for p in self.model.layer1.parameters():
                p.requires_grad = False
        if 2 in layers:
            for p in self.model.layer2.parameters():
                p.requires_grad = False
        if 3 in layers:
            for p in self.model.layer3.parameters():
                p.requires_grad = False
        if 4 in layers:
            for p in self.model.layer4.parameters():
                p.requires_grad = False
    def forward(self, 
                x: torch.tensor):
        pred = self.model(x)
        pred = self.dropout(pred)
        pred = self.fc(pred)
        return pred
