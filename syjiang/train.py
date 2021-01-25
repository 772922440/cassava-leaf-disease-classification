import numpy as np
import pandas as pd 

from dataset.leafdisease import CLDDataset
from model.resnet import CustomResNext50 , CustomResNext18, EfficientNet , SelectBackbone
import torch.utils.data as data
import torchvision.transforms as transforms 
import torch

from utils.args import parse_args
import torch.optim as optim 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os 
from utils.utils import save_checkpoint, load_checkpoint, save_results
import pickle
from efficientnet_pytorch import EfficientNet



def train(args, train_loader, valid_loader):


    model = SelectBackbone(args.backbone)
    model = model.to(args.device)

    if args.pretrained:
        load_checkpoint(args.model_path, model, optimizer)
        print('Load model Successfully')
    else:
        print('No the pretrained model, sorry!')

    pos_weight = torch.tensor([19.68, 9.77, 8.96, 1.63 , 8.30])

    loss_f = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), args.lr)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=args.epochs // 5, 
                            eta_min=args.eta_min, last_epoch=-1)

    best_acc = 0


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

            loss = loss_f(preds, lbls)
            train_loss += loss
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            correct += pred.eq(lbls.view_as(pred)).sum().item()


            print('\t Train Epoch: {} [ {}/{} ({:.0f}%) ] \t Loss: {:.4f}'.format(
                    epoch, idx * len(imgs), len(train_loader.dataset),
                    100. * idx / len(train_loader), loss.item()), end= '            \r')

        print()

        train_acc = correct / len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)
        
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
        save_results(epoch, train_loss, valid_loss, train_acc , acc, './results/', args.save_filename)

        if acc > best_acc :

            if not os.path.exists(args.checkpoints):
                os.makedirs(args.checkpoints)
                print('Make File Successfully')
            best_acc = acc
            save_checkpoint(f'{args.checkpoints}/{args.backbone}.pth' , model, optimizer)




if __name__=='__main__':

    args = parse_args()
    
    torch.manual_seed(2020)
    np.random.seed(2020)

    df = pd.read_csv(args.train_csv)
    
    # Split into train df and test df and save pickle

    if args.SplitFlag:

        dirs = list(df.image_id)
        np.random.shuffle(dirs)
        split = int(len(dirs) * 0.9)
        train_dirs, valid_dirs = dirs[:split], dirs[split:]
    
        with open(args.train_set_pickle, "wb") as f:
            pickle.dump(train_dirs, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(args.valid_set_pickle, "wb") as f:
            pickle.dump(valid_dirs, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("\t[Info] dump train valid set")
    else :

        with open(args.train_set_pickle, 'rb') as f:
            train_dirs = pickle.load(f)
        with open(args.valid_set_pickle, 'rb') as f:
            valid_dirs = pickle.load(f)


    df_set_index = df.set_index('image_id', drop = True)    

    df_train , df_val = df_set_index.loc[train_dirs].reset_index(), df_set_index.loc[valid_dirs].reset_index()

    train_dataset = CLDDataset(df_train, 
                    dirs=args.train_dirs, 
                    mode = 'train',
                    DataAugmentationStrong = args.DataAugmentationStrong)

    valid_dataset = CLDDataset(df_val, 
                    dirs=args.train_dirs, 
                    mode = 'valid',
                    DataAugmentationStrong = args.DataAugmentationStrong)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.bsize,
                              num_workers=6,
                              shuffle=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.bsize,
                              num_workers=6)

    train(args, train_loader, valid_loader)