import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import log
import os



'''
function to get stats of model 
mainly defined for facilitating evaluation of data generation process
call with 
    data_loader --> data loader for data
    model --> model to be applied
    criterion --> criterion to measure performance
    device --> device to use
returns 
    val_acc --> accuracy on all samples from data loader
    val_loss --> avg loss on all samples from data_loader
    manually_labeled --> number of wrongly classified images(that have to be labeled manually during data collection)
'''
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


'''
function to perform train step
automatically called in train_loop
performs forward and backward pass
returns loss
'''
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


'''
function to perfrom mc dropout
sets model to eval and then activates training mode in dropout layers
performs num_samples forward passes and calculates entropy
call with: 
    model --> model to use 
    x --> input to model
    num_samples --> number of samples for MC dropout
returns:
    pred_cls --> predicted class(es)
    entropy --> measure for uncertainty
'''
@torch.no_grad()
def mc_predict(model: nn.Module, 
               x: torch.Tensor, 
               num_samples: int = 40):
    was_training = model.training
    model.eval()
    def _enable_dropout(m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
    model.apply(_enable_dropout)
    probs = []
    for _ in range(num_samples):
        logits = model(x)                      
        probs.append(torch.softmax(logits, dim=-1))
    probs = torch.stack(probs, dim=0)         # [T, B, C]
    mean_probs = probs.mean(dim=0)             # [B, C]
    pred_cls = mean_probs.argmax(dim=1)
    entropy = -(mean_probs.clamp_min(1e-8) * mean_probs.clamp_min(1e-8).log()).sum(dim=1)
    if was_training:
        model.train()
    return pred_cls, entropy

'''
function to perform validation step
automatically called in train_loop
returns 
    loss --> loss on samples
    acc --> acc on samples
    wrong --> number of wrongly classified samples
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


'''
function that orchestrates training
best model with regard to loss and accuracy is stored in log_dir
best accuracy and loss is written in {log_dir}/train_stats.txt
call with: 
               train_loader: DataLoader --> data loader for train data 
               val_loader: DataLoader --> data loader for train data 
               model: nn.Module--> model to be trained
               criterion: nn.Module --> loss function
               optimizer: nn.Module --> optimizer
               scheduler: nn.Module = None --> scheduler
               aug_loader: DataLoader = None --> data loader for train data with augmentation 
               num_epochs: int = 100
               early_stopping: int = 20 --> when to interrupt training if no improvement  
               log_dir: str = 'log' --> directory to log
               device: str = 'cuda'
               logging: bool = False --> whether to write performance after every single epoch
'''
def train_loop(train_loader: DataLoader, 
               val_loader: DataLoader, 
               model: nn.Module, 
               criterion: nn.Module, 
               optimizer: nn.Module,
               scheduler: nn.Module = None,
               aug_loader: DataLoader = None,
               num_epochs: int = 100, 
               early_stopping: int = 20, 
               log_dir: str = 'log', 
               device: str = 'cuda', 
               logging: bool = False):
    os.makedirs(log_dir, exist_ok = True)
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
            running_loss *= len(train_loader) # multiply with len of train loader again to get total loss
            for x, y, _ in iter(aug_loader):
                loss = train_step(x = x, 
                                y = y, 
                                model = model, 
                                criterion = criterion, 
                                optimizer = optimizer, 
                                device = device)
                running_loss += loss
            running_loss /= (len(train_loader) + len(aug_loader)) # divide by len of train and aug loader
        if logging == True:
            log(f'training in epoch {epoch +1 } completed: ; loss: {running_loss}', 
                file = f'{log_dir}/train_stats.txt')
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
        if scheduler is not None: 
            scheduler.step(val_loss)
        if logging == True:
            log(f'validation in epoch {epoch +1 } completed: acc: {val_acc}; loss: {val_loss}', 
                file=f'{log_dir}/train_stats.txt')
        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), f'{log_dir}/model_acc.pth')
        elif val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f'{log_dir}/model_loss.pth')
        elif(counter > early_stopping):
            torch.cuda.empty_cache()
            log(f'model training in epoch {epoch +1 } completed; best acc: {best_acc}; loss: {best_loss}', 
                file=f'{log_dir}/train_stats.txt')
            break
    log(f'model training in epoch {epoch +1 } completed; best acc: {best_acc}; loss: {best_loss}', 
                file=f'{log_dir}/train_stats.txt')
    torch.cuda.empty_cache()


   

