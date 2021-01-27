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
import torch.nn.init as init

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

class EnsembleWeight(nn.Module):
    def __init__(self, model_size = 5, target_size = 5):
        super().__init__()
        self.w = nn.Parameter(torch.empty(model_size, target_size))
        init.constant_(self.w, 1e-4)

    def forward(self, x): 
        # b, model, cls
        x = torch.sum(torch.softmax(self.w, dim=0) * x, dim=1)
        x /= x.sum(dim=-1, keepdim=True).detach()
        return x


def train_fn(train_loader, weight_model, criterion, optimizer, epoch, scheduler, device):
    batch_time = utils.AverageMeter()
    data_time =  utils.AverageMeter()
    losses =  utils.AverageMeter()
    scores =  utils.AverageMeter()

    # switch to train mode
    weight_model.train()
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

        # backbone forward
        probs = []
        with torch.no_grad():
            for m in config.model_list:
                backbone = m['backbone']
                model = get_backbone(backbone, config).to(device=config.device)
                model.eval()

                for filename in m['filename']:
                    model_path = join(config.model_base_path, backbone, filename)
                    model.load_state_dict(torch.load(model_path))
                    prob = model(images).softmax(dim=-1) # b, cls
                    probs.append(prob)
        probs = torch.stack(probs, dim=1) # b, models, cls

        # mixer forward
        y_preds = weight_model(probs)
        loss = criterion(y_preds, labels)

        # record accuracy
        preds.append(y_preds.softmax(dim=-1).to('cpu'))
        preds_labels.append(labels.to('cpu'))
        # record loss
        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(weight_model.parameters(), config.max_grad_norm)
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
            print(torch.softmax(weight_model.w, dim=0))
    preds = torch.cat(preds, dim=0)
    preds_labels = torch.cat(preds_labels, dim=0)
    return losses.avg, preds, preds_labels


def main():
    # output dir
    utils.mkdir(join(config.model_base_path, 'ensemble_weight'))

    # init model
    transform_train = ld.get_albu_transform(config.transform, config)[0]
    model_size = sum(len(m['filename']) for m in config.model_list)
    weight_model = EnsembleWeight(model_size, config.target_size).to(device=config.device)

    # optimizer
    optimizer = optim.get_optimizer(config.optimizer, config, weight_model.parameters())
    config.T_max = config.epochs
    scheduler = torch_utils.get_scheduler(config.scheduler, config, optimizer)
    criterion = cls_loss.get_criterion(config.criterion, config)
    print(f'Criterion: {criterion}')

    # train data
    train = pd.read_csv(join(config.data_base_path, config.train_csv))
    train['filepath'] = train.image_id.apply(lambda x: join(config.data_base_path, config.train_images, f'{x}'))

    # loader
    train_dataset = ld.CLDDataset(train, 'train', transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=config.num_workers, pin_memory=True, drop_last=False)

    # train epochs
    best_train_score = 0.
    best_epoch = 0
    best_weight = torch.softmax(weight_model.w, dim=0)
    for epoch in range(config.epochs):
        start_time = time.time()
        # train
        avg_loss, train_preds, train_labels = train_fn(train_loader, weight_model, criterion, optimizer, epoch, scheduler, config.device)
        train_score = accuracy_score(train_labels, train_preds.argmax(dim=-1))

        # scheduler
        torch_utils.scheduler_step(scheduler, None)
        
        # log
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}, train accuracy: {train_score},  time: {elapsed:.0f}s')

        if train_score > best_train_score:
            best_train_score = train_score
            best_epoch = epoch
            best_weight = torch.softmax(weight_model.w, dim=0)
            
            print(f'Epoch {best_epoch+1} - Save bese train accuracy: {best_train_score}')
            print(best_weight)
            torch.save(weight_model.state_dict(), 
                join(config.model_base_path, 'ensemble_weight', f'best.pth'))

    print(f'Epoch {best_epoch+1} - Best train accuracy: {best_train_score}')
    print(best_weight)

# run
main()