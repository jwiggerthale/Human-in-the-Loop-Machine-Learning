import torch
import torch.nn as nn
from torchvision import models


'''
This class implements a vgg 16 model 
initialize with num_classes
model uses pretrained weights from weights path if not specified oterwise
make sure, weight file exists
'''
class VGG16(nn.Module):
    def __init__(self, 
                num_classes: int = 5, 
                pretrained: bool = True, 
                weights: str = '/data/Models/image_recognition/vgg16_pretrained.pth'):
        super().__init__()
        model = models.vgg16()
        if pretrained == True:
          model.load_state_dict(torch.load(weights))
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
This class implements an efficientnet b0 model 
initialize with num_classes
model uses pretrained weights from weights path if not specified oterwise
make sure, weight file exists
'''
class efficientnet_b0(nn.Module):
    def __init__(self, 
                num_classes: int = 5, 
                pretrained: bool = True, 
                weights: str = '/data/Models/image_recognition/efficientnet_b0.pth'):
        super().__init__()
        model = models.efficientnet_b0(pretrained = False)
        if pretrained == True:
          model.load_state_dict(torch.load(weights))
        self.model = model
        self.model.classifier = nn.Sequential(nn.Dropout(0.2), 
                                              nn.Linear(1280, num_classes))
    # freeze feature extractor
    def freeze_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True
    
    # freeze certain layers
    def freeze_layers(self, 
                      num_frozen_blocks: int = 9):
        for i in range(num_frozen_blocks):
            for param in self.model.encoder.layers[i].parameters():
                param.requires_grad = False

    '''
    Function to enable dropout (for MC dropout)
    '''
    def set_dropout(self, 
                    p=0.2):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = p

    def forward(self, 
                x: torch.tensor):
        pred = self.model(x)
        return pred

'''
This class implements an VIT model 
initialize with num_classes
model uses pretrained weights from weights path if not specified oterwise
make sure, weight file exists
'''
class VIT(nn.Module):
    def __init__(self, 
                num_classes: int = 5, 
                pretrained: bool = True, 
                weights: str = '/data/Models/image_recognition/vit.pth'):
        super().__init__()
        model = models.vit_b_16()
        if pretrained == True:
          model.load_state_dict(torch.load(weights))
        self.model = model
        self.model.heads = nn.Linear(768, num_classes)
    # freeze feature extractor
    def freeze_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.heads.parameters():
            param.requires_grad = True
    
    # freeze certain layers
    def freeze_layers(self, 
                      num_frozen_blocks: int = 9):
        for i in range(num_frozen_blocks):
            for param in self.model.encoder.layers[i].parameters():
                param.requires_grad = False

    def set_dropout(self, 
                    p=0.2):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = p

    def forward(self, 
                x: torch.tensor):
        pred = self.model(x)
        return pred


'''
This class implements an ResNet18 model 
initialize with num_classes
model uses pretrained weights from weights path if not specified oterwise
make sure, weight file exists
'''
class resnet18(nn.Module):
    def __init__(self, 
                num_classes: int = 5, 
                dropout_rate: float = 0.5, 
                pretrained: bool = True, 
                weights: str = '/data/Models/image_recognition/resnet18_pretrained.pth'):
        super().__init__()
        self.model = models.resnet18()
        if pretrained == True:
          self.model.load_state_dict(torch.load(weights))
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
        pred = self.model.layer1(x)
        pred = self.dropout(pred)
        pred = self.model.layer2(pred)
        pred = self.dropout(pred)
        pred = self.model.layer3(pred)
        pred = self.dropout(pred)
        pred = self.model.layer4(pred)
        pred = self.dropout(pred)
        pred = self.fc(pred)
        return pred
