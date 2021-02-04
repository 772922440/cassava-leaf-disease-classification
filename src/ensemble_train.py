from os.path import join
import time
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

import warnings

warnings.filterwarnings('ignore')

try:
    from apex import amp
    apex_support = True
except:
    apex_support = False 


try:
    # from pytorch_toolbelt.inference import tta
    import ttach as tta
    tta_support = True
except:
    tta_support = False

# our codes
from utils import utils, torch_utils, cls_loss, optim, dist
from dataset import leafdisease as ld

from model import get_backbone

# read config
config = utils.read_all_config()
utils.mkdir(config.model_base_path)
torch_utils.seed_torch(seed=config.seed)

def CutMix(images, labels, beta):

    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(images.size()[0]).to(config.device)


    target_a = labels
    target_b = labels[rand_index]
    bbx1, bby1, bbx2, bby2 = utils.rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

    return images, target_a, target_b ,lam


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = utils.AverageMeter()
    data_time =  utils.AverageMeter()
    losses =  utils.AverageMeter()
    scores =  utils.AverageMeter()

    if config.amp:
        scaler = GradScaler()

    if config.distance_loss:
        distance_loss = cls_loss.EuclieanDistanceLoss()
        distance_loss_avg =  utils.AverageMeter()

    # switch to train mode
    model.train()
    preds = []
    preds_labels = []
    start = end = time.time()

    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # forward
        if config.amp:
            assert not config.distance_loss
            with autocast():
                y_preds = model(images)
                loss = criterion(y_preds, labels)

        elif config.cutmix:
            images , target_a, target_b , lam = CutMix(images, labels, config.beta)
            y_preds = model(images)
            loss = criterion(y_preds, target_a) * lam + criterion(y_preds, target_b) * (1. - lam)

        else:
            if config.distance_loss:
                y_preds, embedings = model(images)
                loss = criterion(y_preds, labels)
                dis_loss, batch1 = distance_loss(embedings, labels)
                if batch1:
                    distance_loss_avg.update(dis_loss.item(), batch1)

                loss = loss + config.distance_loss * dis_loss
            else:
                y_preds = model(images)
                loss = criterion(y_preds, labels)

        # record accuracy
        preds.append(y_preds.softmax(dim=-1).to('cpu'))
        preds_labels.append(labels.to('cpu'))
        # record loss
        losses.update(loss.item(), batch_size)

        # enlarge batch size
        loss /= config.accumulated_gradient

        if config.amp:
            assert config.accumulated_gradient == 1
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        elif apex_support and config.apex:
            assert config.accumulated_gradient == 1
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                # scaled_loss.step(optimizer)
            optimizer.step()
        else:
            if step % config.accumulated_gradient == 0:
                optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if step % config.accumulated_gradient == config.accumulated_gradient - 1:
                optimizer.step()

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
            if config.distance_loss:
                print('Distance Loss: {distance_loss_avg.val:.4f}({distance_loss_avg.avg:.4f})'
                    .format(distance_loss_avg=distance_loss_avg))

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
    preds_labels = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            if config.TTA and tta_support:

                assert not config.cosine_loss
                tta_train_trans, tta_valid_trans =  ld.get_albu_transform('valid_tta', config)
                tta_model = tta.ClassificationTTAWrapper(model, tta_valid_trans)
                y_preds = tta_model(images)
            
            else:
                if config.distance_loss:
                    y_preds, _ = model(images)
                else:
                    y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.softmax(dim=-1).to('cpu'))
        preds_labels.append(labels.to('cpu'))
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
    preds_labels = torch.cat(preds_labels, dim=0)
    return losses.avg, predictions, preds_labels

def main(local_rank=0, world_size=1):
    if local_rank == 0:
        print(config)

    # train k fold
    if 'k' not in config or config.k < 0 or config.k >= config.k_folds:
        print("please input correct k fold.(ensemble_train.py name=ensemble_train k=xxx)")
        return 1

    # output dir
    model_save_path = join(config.model_base_path, config.backbone + config.model_suffix)
    utils.mkdir(model_save_path)

    # init dist
    if config.DDP:
        dist.setup(local_rank, world_size, config.port)
        config.batch_size //= world_size

    # enlarge batch size
    config.batch_size //= config.accumulated_gradient

    # init model
    model = get_backbone(config.backbone, config).to(device=config.device)

    if config.DDP:
        print(f"Use DPP, You have {torch.cuda.device_count} GPUs")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # optimizer
    optimizer = optim.get_optimizer(config.optimizer, config, model.parameters())

    # apex
    if apex_support and config.apex:
        print("\t[Info] Use fp16_precision")
        model, optimizer = amp.initialize(model, optimizer,
            opt_level='O1', keep_batchnorm_fp32=True, verbosity=0)

    if config.TTA and tta_support:
        print("Use TTA")

    config.T_max = config.epochs
    scheduler = torch_utils.get_scheduler(config.scheduler, config, optimizer)
    criterion = cls_loss.get_criterion(config.criterion, config)
    print(f'Criterion: {criterion}')

    # split train valid
    train = pd.read_csv(join(config.data_base_path, config.k_folds_csv))
    train['filepath'] = train.image_id.apply(lambda x: join(config.data_base_path, config.train_images, f'{x}'))
    valid = train[train.fold == config.k].reset_index(drop=True)
    train = train[train.fold != config.k].reset_index(drop=True)

    # loader
    transform_train, transform_valid = ld.get_albu_transform(config.transform, config)

    train_dataset = ld.CLDDataset(train, 'train', transform=transform_train)
    valid_dataset = ld.CLDDataset(valid, 'valid', transform=transform_valid)

    if config.DDP:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, rank=local_rank)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, 
            num_workers=config.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, 
            num_workers=config.num_workers, pin_memory=True, drop_last=False, 
            sampler=DistributedSampler(valid_dataset, shuffle=False, rank=local_rank))
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, drop_last=False)
    

    # train epochs
    best_score = 0.
    best_train_score = 0.
    best_epoch = 0
    best_confusion_matrix = []
    print('############### Begin Train ###################')
    for epoch in range(config.epochs):
        start_time = time.time()
        # train
        avg_loss, train_preds, train_labels = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, config.device)
        train_score = accuracy_score(train_labels, train_preds.argmax(dim=-1))

        # eval

        avg_val_loss, val_preds, val_labels = valid_fn(valid_loader, model, criterion, config.device)
        val_score = accuracy_score(val_labels, val_preds.argmax(dim=-1))
        matrix = confusion_matrix(val_labels, val_preds.argmax(dim=-1))

        # sync scores
        if config.DDP:
            avg_loss = dist.all_reduce_scalar(avg_loss, config.device, world_size, mean=True)
            train_score = dist.all_reduce_scalar(train_score, config.device, world_size, mean=True)
            val_score = dist.all_reduce_scalar(val_score, config.device, world_size, mean=True)
            avg_val_loss = dist.all_reduce_scalar(avg_val_loss, config.device,  world_size, mean=True)
            matrix = dist.all_reduce_array(matrix, config.device)

            # for shuffle
            train_sampler.set_epoch(epoch + 1)


        # scheduler
        torch_utils.scheduler_step(scheduler, avg_val_loss)
        
        if local_rank == 0:
            # log
            elapsed = time.time() - start_time
            print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
            print(f'Epoch {epoch+1} - train accuracy: {train_score} eval accuracy: {val_score}')
            print(f'Best Epoch: {best_epoch}, Train Score {best_train_score:.4f}:, Best Score: {best_score:.4f}' + "\n")

            # TODO: maybe need to be deleted
            # we have saved last record to model file path
            if config.save_filename:
                utils.save_results(epoch+1, avg_loss.item(), avg_val_loss.item(), train_score , val_score, './results/', config.save_filename)

            if val_score > best_score:
                best_score = val_score
                best_train_score = train_score
                best_epoch = epoch+1
                best_confusion_matrix = matrix
                
                if config.norm_confusion_matrix:
                    best_confusion_matrix = best_confusion_matrix.astype('float') \
                                / np.sum(best_confusion_matrix, axis=1, keepdims=True).astype('float')

                print(f'Epoch {epoch+1} - Train Score {best_train_score:.4f}:, Save Best Score: {best_score:.4f}')
                print(best_confusion_matrix)
                torch.save(model.state_dict(), join(model_save_path, f'fold{config.k}_best.pth'))


    if local_rank == 0:
        # print final log
        print(config)
        print(f'Best Epoch: {best_epoch}, Train Score {best_train_score:.4f}:, Best Score: {best_score:.4f}')
        print(best_confusion_matrix)

        # write final log
        with open(join(model_save_path, f'fold{config.k}_log.txt'), 'w') as f:
            f.write(str(config) + "\n")
            f.write(f'Best Epoch: {best_epoch}, Train Score {best_train_score:.4f}:, Best Score: {best_score:.4f}' + "\n")
            f.write(str(best_confusion_matrix) + "\n")

    if config.DDP:
        dist.cleanup()


if __name__ == "__main__":
    if config.DDP:
        dist.spawn(main)
    else:
        main()
