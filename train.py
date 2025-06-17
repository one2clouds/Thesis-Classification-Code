import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import torch 
from tqdm import tqdm 
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pathlib
import matplotlib.pyplot as plt
from glob import glob 
import pandas as pd 
import os
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from get_data_v2 import get_dataloader_v2
from metrics import data_imbalance_check
from efficient_net import Efficient_Net 
from res_net import Res_Net

import argparse
import wandb
import warnings
warnings.filterwarnings('ignore')



device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_needed_metrics(labels, predicted): 
    assert isinstance(labels, list)
    assert isinstance(predicted, list)
    
    accuracy = balanced_accuracy_score(labels, predicted)
    precision = precision_score(labels, predicted, zero_division=0.0, average='weighted')
    recall = recall_score(labels, predicted, zero_division=0.0, average='weighted')
    f1 = f1_score(labels, predicted, zero_division=0.0, average='weighted')
    
    return accuracy, precision, recall, f1



def train(model, train_loader, val_loader, loss_func, optimizer, num_epochs):
    best_val_f1 = -100
    best_val_loss = 1000

    patience = 30
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        acc_train_epoch, precision_train_epoch, recall_train_epoch, f1_train_epoch  = [], [], [], []
        for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch}/{num_epochs}', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            train_loss = loss_func(outputs, labels)
            train_loss.backward()
            optimizer.step()
            
        
            _, predicted_train = torch.max(outputs, 1)

            # print(f"Labels : {labels}")
            # print(f"Predicted : {predicted_train}")
            

            acc_batch_train, precision_batch_train, recall_batch_train, f1_batch_train = get_needed_metrics(labels.cpu().detach().tolist(), predicted_train.cpu().detach().tolist())
        
            acc_train_epoch.append(acc_batch_train)
            precision_train_epoch.append(precision_batch_train)
            recall_train_epoch.append(recall_batch_train)
            f1_train_epoch.append(f1_batch_train)
        
        # Validating the model
        model.eval()
        acc_val_epoch, precision_val_epoch, recall_val_epoch, f1_val_epoch  = [], [], [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Testing {epoch}/{num_epochs}', unit='batch'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = loss_func(outputs, labels)
                _, predicted_val = torch.max(outputs, 1)

                acc_batch_val, precision_batch_val, recall_batch_val, f1_batch_val = get_needed_metrics(labels.cpu().detach().tolist(), predicted_val.cpu().detach().tolist())

                acc_val_epoch.append(acc_batch_val)
                precision_val_epoch.append(precision_batch_val)
                recall_val_epoch.append(recall_batch_val)
                f1_val_epoch.append(f1_batch_val)
                
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {train_loss.item():.4f}, '
            f'Train Accuracy: {torch.tensor(acc_train_epoch).mean():.2f}%, '
            f'Train Precision: {torch.tensor(precision_train_epoch).mean():.2f}%, '
            f'Train Recall: {torch.tensor(recall_train_epoch).mean() * 100:.2f}%, '
            f'Train F1: {torch.tensor(f1_train_epoch).mean() * 100:.2f}%, '

            f'Val Loss: {val_loss.item():.4f}, '
            f'Val Accuracy: {torch.tensor(acc_val_epoch).mean() * 100:.2f}%, '
            f'Val Precision: {torch.tensor(precision_val_epoch).mean() * 100:.2f}%, '
            f'Val Recall: {torch.tensor(recall_val_epoch).mean() * 100:.2f}%, '
            f'Val F1: {torch.tensor(f1_val_epoch).mean() * 100:.2f}%')

        
        wandb.log({"train/train_loss": train_loss.item(), 
                   "train/train_balanced_accuracy":torch.tensor(acc_train_epoch).mean(), 
                   "train/train_precision":torch.tensor(precision_train_epoch).mean(), 
                   "train/train_recall":torch.tensor(recall_train_epoch).mean(),
                   "train/train_f1":torch.tensor(f1_train_epoch).mean(),                   
                   })
        
        wandb.log({"epoch": epoch + 1})

        wandb.log({"val/val_loss": val_loss.item(), 
                   "val/val_balanced_accuracy":torch.tensor(acc_val_epoch).mean(), 
                   "val/val_precision":torch.tensor(precision_val_epoch).mean(), 
                   "val/val_recall":torch.tensor(recall_val_epoch).mean(),
                   "val/val_f1":torch.tensor(f1_val_epoch).mean()
                   })
        
        
        if torch.tensor(f1_val_epoch).mean() > best_val_f1:
            best_val_f1 = torch.tensor(f1_val_epoch).mean()
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }
            torch.save(checkpoint, "model_best_val_f1.pth")
            print(f"Best f1 model saved, Recall - {torch.tensor(f1_val_epoch).mean()} ")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }
            torch.save(checkpoint, "model_least_val_loss.pth")
            print(f"Least val loss model saved, val_loss - {val_loss.item()} ")

        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }
        torch.save(checkpoint, "model_recent.pth")


        if val_loss.item() > best_val_loss:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }
            torch.save(checkpoint, "last_epoch_model_ckpt.pth")
            print(f"Model saved")
            exit(0)
        
    return model, epoch, optimizer, train_loss


if __name__ == '__main__':
    wandb.init(
        project="Steel Scrap Classification", 
        config={
            "epochs":2000, 
            "batch_size":256, 
            "lr": 1e-4
        }
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--valid_path', type=str, required=True)
    args = parser.parse_args()
    
    # data_imbalance_check(args.train_path)
    # train_loader = get_dataloader(args.train_path, get_path=False, batch_size=wandb.config.batch_size, shuffle=True)
    # valid_loader = get_dataloader(args.valid_path, get_path=False, batch_size=wandb.config.batch_size, shuffle=False)

    train_loader = get_dataloader_v2(args.train_path, batch_size=wandb.config.batch_size, shuffle=True)
    valid_loader = get_dataloader_v2(args.valid_path, batch_size=wandb.config.batch_size, shuffle=False)

    root=pathlib.Path(args.train_path)
    classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

    # print(classes) # ['BACKGROUND', 'E1', 'E2', 'E3', 'E40', 'E5H', 'E6', 'E8', 'EHRB']

    if args.model == 'efficient_net':
        model = Efficient_Net(classes=classes)
    elif args.model == 'resnet':
        model = Res_Net(classes=classes)
    else:
        print('model not found')
        exit()

    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr)
    num_epochs = wandb.config.epochs

    model, epoch, optimizer, train_loss = train(model, train_loader, valid_loader, loss_func, optimizer, num_epochs)
    


# python train.py --model efficient_net --train_path '/home/shirshak/Thesis_Data/DOES/TRAIN/' --valid_path '/home/shirshak/Thesis_Data/DOES/TEST_tiles/'