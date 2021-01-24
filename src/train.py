import numpy as np
import pandas as pd 

from dataset.leafdisease import CLDDataset
from model.resnet import CustomResNext50 , CustomResNext18
import torch.utils.data as data
import torchvision.transforms as transforms 
import torch

from utils.args import parse_args
import torch.optim as optim 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os 
from utils.utils import save_checkpoint, load_checkpoint


def train(args, train_loader, valid_loader):


    model = CustomResNext50().to(args.device)

    pos_weight = torch.tensor([19.68, 9.77, 8.96, 1.63 , 8.30])

    # loss_f = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(args.device)

    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), args.lr)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=15, 
                            eta_min=args.eta_min, last_epoch=-1)
    best_acc = 0

    if os.path.exists(args.checkpoints):
        load_checkpoint(args.model_path, model, optimizer)
        print('Load model Successfully')

    for epoch in range(1 ,args.epochs + 1):

        train_loss , valid_loss = 0, 0 
        
        print(f" Epoch {epoch}")

        model.train()

        correct = 0

        for idx , (imgs, label) in enumerate(train_loader):
            imgs, lbls = imgs.to(args.device), label.to(args.device)

            lbls = lbls.long()
            preds = model(imgs)            
            pred = preds.max(1, keepdim=True)[1] 
            # print(pred)

            loss = loss_f(preds, lbls)
            train_loss += loss
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            correct += pred.eq(lbls.view_as(pred)).sum().item()


            print('\t Train Epoch: {} [ {}/{} ({:.0f}%) ] \t Loss: {:.4f}'.format(
                    epoch, idx * len(imgs), len(train_loader.dataset),
                    100. * idx / len(train_loader), loss.item()), end= '            \r')
            if idx > 10 :
                break
        print()

        train_acc = correct / len(train_loader.dataset)
        
        model.eval()


        correct = 0
        with torch.no_grad():
            for idx , (imgs, label) in enumerate(valid_loader):
                imgs, lbls = imgs.to(args.device), label.to(args.device)

                lbls = lbls.long()
                preds = model(imgs)            
                pred = preds.max(1, keepdim=True)[1]
                loss = loss_f(preds, lbls)
                
                valid_loss += loss

                correct += pred.eq(lbls.view_as(pred)).sum().item()
                
                print('\t Val Epoch: {} [ {}/{} ({:.0f}%) ] Loss: {:.2f}'.format(
                    epoch, idx * len(imgs), len(valid_loader.dataset),
                    100. * idx / len(valid_loader), loss.item()), end= '  \r')
        
        print()

        valid_loss /= len(valid_loader.dataset)
        acc = correct / len(valid_loader.dataset)

        print('\t Train Loss:{:.4f} , \t val Loss:{:.4f} \t Train_acc:{:.4f}, \tVal_acc:{:.4f} \n'.format(train_loss, valid_loss, train_acc , acc))

        if acc > best_acc :

            if not os.path.exists(args.checkpoints):
                os.makedirs(args.checkpoints)
                print('Make File Successfully')
            best_acc = acc

            save_checkpoint(f'{args.checkpoints}/bestmodel_50.pth' , model, optimizer)




if __name__=='__main__':

    args = parse_args()
    
    torch.manual_seed(2020)

    df = pd.read_csv(args.train_csv)
    dataset = CLDDataset(df, dirs=args.train_dirs, mode = 'train')
    
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size
    
    train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size])


    train_loader = DataLoader(train_set,
                              batch_size=args.bsize,
                              num_workers=6,
                              shuffle=True)

    valid_loader = DataLoader(valid_set,
                              batch_size=args.bsize,
                              num_workers=6)

    train(args, train_loader, valid_loader)