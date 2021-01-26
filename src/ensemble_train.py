from os.path import join
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import warnings 
warnings.filterwarnings('ignore')

# our codes
from utils import utils, torch_utils, cls_loss, optim
from dataset import leafdisease as ld
from model import get_backbone

# read config
config = utils.read_all_config()
utils.mkdir(config.model_base_path)
torch_utils.seed_torch(seed=config.seed)

print(config)


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = utils.AverageMeter()
    data_time =  utils.AverageMeter()
    losses =  utils.AverageMeter()
    scores =  utils.AverageMeter()

    if config.amp:
        scaler = GradScaler()

    # switch to train mode
    model.train()
    preds = []
    preds_labels = []
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # forward
        if config.amp:
            with autocast():
                y_preds = model(images)
                loss = criterion(y_preds, labels)
        else:
            y_preds = model(images)
            loss = criterion(y_preds, labels)

        # record accuracy
        preds.append(y_preds.softmax(dim=-1).to('cpu'))
        preds_labels.append(labels.to('cpu'))
        # record loss
        losses.update(loss.item(), batch_size)

        if config.amp:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % config.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  #'LR: {lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=utils.timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   #lr=scheduler.get_lr()[0],
                   ))
    preds = torch.cat(preds, dim=0)
    preds_labels = torch.cat(preds_labels, dim=0)
    return losses.avg, preds, preds_labels


def valid_fn(valid_loader, model, criterion, device):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    scores = utils.AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.softmax(dim=-1).to('cpu'))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % config.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=utils.timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    predictions = torch.cat(preds, dim=0)
    return losses.avg, predictions

def main():
    # train k fold
    if 'k' not in config or config.k < 0 or config.k >= config.k_folds:
        print("please input correct k fold.(ensemble_train.py name=ensemble_train k=xxx)")
        return 1

    # output dir
    utils.mkdir(join(config.model_base_path, config.backbone))

    # init model
    transform_train, transform_valid = ld.get_albu_transform(config.transform, config)
    model = get_backbone(config.backbone, config).to(device=config.device)

    # optimizer
    optimizer = optim.get_optimizer(config.optimizer, config, model.parameters())
    config.T_max = config.epochs
    scheduler = torch_utils.get_scheduler(config.scheduler, config, optimizer)
    criterion = cls_loss.get_criterion(config.criterion, config)
    print(f'Criterion: {criterion}')

    # split train valid
    train = pd.read_csv(join(config.data_base_path, str(config.k_folds) + 'folds.csv'))
    train['filepath'] = train.image_id.apply(lambda x: join(config.data_base_path, config.train_images, f'{x}'))
    valid = train[train.fold == config.k].reset_index(drop=True)
    train = train[train.fold != config.k].reset_index(drop=True)

    # loader
    train_dataset = ld.CLDDataset(train, 'train', transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=config.num_workers, pin_memory=True, drop_last=False)

    valid_dataset = ld.CLDDataset(valid, 'valid', transform=transform_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True, drop_last=False)

    # train epochs
    best_score = 0.
    best_epoch = 0
    for epoch in range(config.epochs):
        start_time = time.time()
        # train
        avg_loss, train_preds, train_labels = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, config.device)
        train_score = accuracy_score(train_labels, train_preds.argmax(dim=-1))

        # eval
        avg_val_loss, val_preds = valid_fn(valid_loader, model, criterion, config.device)
        valid_labels = valid[config.target_col].values
        val_score = accuracy_score(valid_labels, val_preds.argmax(dim=-1))

        # scheduler
        torch_utils.scheduler_step(scheduler, avg_val_loss)
        
        # log
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        print(f'Epoch {epoch+1} - train accuracy: {train_score} eval accuracy: {val_score}')

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch+1
            print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save(model.state_dict(), 
                join(config.model_base_path, config.backbone, f'fold{config.k}_best.pth'))

    print(f'Best Epoch: {best_epoch} Best Score: {best_score:.4f}')

# run
main()