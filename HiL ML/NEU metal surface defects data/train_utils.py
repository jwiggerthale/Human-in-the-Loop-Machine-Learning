'''
The script implements basic training logic but also tools to evaluate models
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import log



def get_stats(data_loader: DataLoader, 
              model: nn.Module, 
              criterion: nn.Module, 
              device: str = 'cuda'):
        val_loss = 0.0
        val_acc = 0.0
        manually_labeled = 0
        for x, y, _ in iter(data_loader):
            loss, acc, wrong = val_step(x = x, 
                                        y = y, 
                                        criterion = criterion,
                                        model=model,
                                        device = device 
                                        )
            val_loss += loss
            val_acc += acc
            manually_labeled += wrong
        val_loss /= len(data_loader)
        val_acc /= len(data_loader)
        return val_acc.item(), val_loss.item(), manually_labeled.item()

def train_step(x: torch.tensor,
               y: torch.tensor, 
               device: str,
               model: nn.Module, 
               criterion: nn.Module, 
               optimizer: nn.Module):
    model.train()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()



'''
function to make prediction with model
call with: 
    x: torch.tensor --> image you want to predict on
    model: nn.Module --> model to be applied
returns: 
    preds: torch.tensor --> predicted classes
    logits: torch.tensor --> raw model output
'''
def predict(x: torch.tensor, 
            model: nn.Module):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)
    return preds, logits



@torch.no_grad()
def mc_predict(model: nn.Module, 
               x: torch.Tensor, 
               num_samples: int = 20):
    # 1) set model to eval (BN, ... stays frozen)
    was_training = model.training
    model.eval()
    # 2) enable dropout
    def _enable_dropout(m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
    model.apply(_enable_dropout)
    # 3) forward passes
    probs = []
    for _ in range(num_samples):
        logits = model(x)                      
        probs.append(torch.softmax(logits, dim=-1))
    probs = torch.stack(probs, dim=0)         # [T, B, C]
    # 4) aggregate and get uncertainty
    mean_probs = probs.mean(dim=0)             # [B, C]
    pred_cls = mean_probs.argmax(dim=1)
    un = probs.std(dim = 0)
    # (optional) sinnvollere Unsicherheiten als "std-sum":
    # PrÃ¤diktive Entropie
    entropy = -(mean_probs.clamp_min(1e-8) * mean_probs.clamp_min(1e-8).log()).sum(dim=1)
    # 5) Urspr. Train/Eval-Zustand wiederherstellen (optional)
    if was_training:
        model.train()
    return pred_cls, entropy

'''
def mc_predict(model: nn.Module, 
               x: torch.tensor, 
               num_samples: int = 20):
    model.train()
    preds = []
    for _ in range(num_samples):
        pred = model(x)
        preds.append(torch.softmax(pred, dim=-1))
    preds = torch.stack(preds, dim=0)
    mean_pred = preds.mean(dim=0)                    
    uncertainty = preds.std(dim = 0)
    pred_cls = pred.argmax(dim = 1)
    return pred_cls, uncertainty
'''


def val_step(x: torch.tensor, 
             y: torch.tensor, 
             model: nn.Module,
             criterion: nn.Module, 
             device: str = 'cuda'):
    x, y = x.to(device), y.to(device)
    preds, probs = predict(model = model, 
                           x = x)
    loss = criterion(probs, y)
    acc = (preds == y).sum()/len(x)
    wrong = len(x) - (preds == y).sum()
    return loss, acc, wrong



def train_loop(train_loader: DataLoader, 
               val_loader: DataLoader, 
               model: nn.Module, 
               criterion: nn.Module, 
               optimizer: nn.Module,
               aug_loader: DataLoader = None,
               num_epochs: int = 100, 
               early_stopping: int = 20, 
               log_file: str = 'log.txt', 
               device: str = 'cuda'):
    best_acc = 0.0
    best_loss = np.inf
    counter = 0
    for epoch in range(num_epochs):
        counter += 1
        running_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        for x, y, _ in iter(train_loader):
            loss = train_step(x = x, 
                              y = y, 
                              model = model, 
                              criterion = criterion, 
                              optimizer = optimizer, 
                              device = device)
            running_loss += loss
        running_loss /= len(train_loader)
        if aug_loader is not None:
            running_loss *= len(train_loader)
            for x, y, _ in iter(aug_loader):
                loss = train_step(x = x, 
                                y = y, 
                                model = model, 
                                criterion = criterion, 
                                optimizer = optimizer, 
                                device = device)
                running_loss += loss
            running_loss /= (len(train_loader) + len(aug_loader))
        log(f'training in epoch {epoch +1 } completed: ; loss: {running_loss}', 
            file = log_file)
        for x, y, _ in iter(val_loader):
            loss, acc, _ = val_step(x = x, 
                                    y = y, 
                                    model = model, 
                                    criterion = criterion, 
                                    device = device)
            val_loss += loss
            val_acc += acc
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        log(f'validation in epoch {epoch +1 } completed: acc: {val_acc}; loss: {val_loss}', 
            file=log_file)
        if val_acc > best_acc:
            best_acc = val_acc
            #print(f'new best model in epoch {epoch + 1}: accuracy: {val_acc}')
            counter = 0
            #torch.save(model.state_dict(), 'VGGModel_Acc.pth')
        elif val_loss < best_loss:
            best_loss = val_loss
            #print(f'new best model in epoch {epoch +1 }: val loss: {val_loss}')
            counter = 0
            #torch.save(model.state_dict(), 'VGGModel_Loss.pth')
        elif(counter > early_stopping):
            #print(f'No improvement in {early_stopping} epochs; Training interrupted in epoch {epoch+1}')
            torch.cuda.empty_cache()
            break
    torch.cuda.empty_cache()


   

