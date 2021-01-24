import sys
import os
import math
import time
import random
import shutil
import albumentations
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
from scipy.special import softmax
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

import warnings 
warnings.filterwarnings('ignore')

# our codes
from utils import utils, torch_utils
from dataset import leafdisease as ld
from model import resnet


# read config
config = utils.read_all_config()
logger = utils.get_logger(config)
utils.mkdir(config.output_dir)
torch_utils.seed_torch(seed=config.seed)

@contextmanager
def timer(name):
    t0 = time.time()
    logger.info(f'[{name}] start')
    yield
    logger.info(f'[{name}] done in {time.time() - t0:.0f} s.')

# transform
transforms = A.Compose([
    A.CenterCrop(config.image_size, config.image_size, p=1),
    A.Resize(config.image_size, config.image_size),
    A.Normalize(),
    ToTensorV2()
])


# test data
test = pd.read_csv(config.test_csv)
test['filepath'] = test.image_id.apply(lambda x: os.path.join(config.test_images, f'{x}'))
test_dataset = ld.CLDDataset(test, 'test', transform=transforms)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

# load model
model = resnet.CustomResNext50(target_size=config.target_size)


def inference(model, states, test_loader, device):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs

# ====================================================
# Helper functions for efficientnet
# ====================================================
def inference_func(test_loader):
    model.eval()
    bar = tqdm(test_loader)

    LOGITS = []
    PREDS = []
    
    with torch.no_grad():
        for batch_idx, images in enumerate(bar):
            x = images.to(device)
            logits = model(x)
            LOGITS.append(logits.cpu())
            PREDS += [torch.softmax(logits, 1).detach().cpu()]
        PREDS = torch.cat(PREDS).cpu().numpy()
        LOGITS = torch.cat(LOGITS).cpu().numpy()
    return PREDS

def tta_inference_func(test_loader):
    model.eval()
    bar = tqdm(test_loader)
    PREDS = []
    LOGITS = []

    with torch.no_grad():
        for batch_idx, images in enumerate(bar):
            x = images.to(device)
            x = torch.stack([x,x.flip(-1),x.flip(-2),x.flip(-1,-2),
            x.transpose(-1,-2),x.transpose(-1,-2).flip(-1),
            x.transpose(-1,-2).flip(-2),x.transpose(-1,-2).flip(-1,-2)],0)
            x = x.view(-1, 3, image_size, image_size)
            logits = model(x)
            logits = logits.view(BATCH_SIZE, 8, -1).mean(1)
            PREDS += [torch.softmax(logits, 1).detach().cpu()]
            LOGITS.append(logits.cpu())

        PREDS = torch.cat(PREDS).cpu().numpy()
        
    return PREDS


# ====================================================
# inference
# ====================================================

#for Resnext
model = CustomResNext(CFG.model_name, pretrained=False)
#model = enet_v2(enet_type[i], out_dim=5)
states = [load_state(MODEL_DIR+f'{CFG.model_name}_fold{fold}.pth') for fold in CFG.trn_fold]
test_dataset = TestDataset(test, transform=get_transforms(data='valid'))
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, 
                         num_workers=CFG.num_workers, pin_memory=True)
predictions = inference(model, states, test_loader, device)

#for Efficientnet
test_preds = []
for i in range(len(enet_type)):
    model = enet_v2(enet_type[i], out_dim=5)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path[i]))
    test_preds += [tta_inference_func(test_loader_efficient)]

# submission
pred = 0.5*predictions + 0.5*np.mean(test_preds, axis=0)
test['label'] = softmax(pred).argmax(1)
test[['image_id', 'label']].to_csv(OUTPUT_DIR+'submission.csv', index=False)
test.head()