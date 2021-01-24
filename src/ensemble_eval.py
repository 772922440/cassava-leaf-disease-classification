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
from model import REGISTRY as MODEL_FACTORY

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


def inference(model_list, test_loader, device):
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []

        # load models
        for m in range(config.model_list):
            backbone = m['backbone']
            model = MODEL_FACTORY[backbone](target_size=config.target_size)
            model.eval()

            for filename in m['filename']:
                model_path = os.path.join(config.model_base_path, backbone, filename)
                model.load_state_dict(torch.load(model_path))

                with torch.no_grad():
                    y_preds = model(images)
                avg_preds.append(y_preds.softmax(1).to('cpu'))

        # simple mean weights
        avg_preds = torch.mean(avg_preds, dim=0)
        probs.append(avg_preds)

    probs = torch.cat(probs, dim=0)
    return probs

# predict
probs = inference(config.model_list, test_loader, config.device)
test['label'] = torch.argmax(probs, dim=-1).numpy()
test[['image_id', 'label']].to_csv(os.path.join(config.output_dir, 'submission.csv'), index=False)
test.head()