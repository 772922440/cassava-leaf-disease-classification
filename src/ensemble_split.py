import sys
import os
from os.path import join
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
torch_utils.seed_torch(seed=config.seed)

train = pd.read_csv(join(config.data_base_path, config.train_csv))

folds = train.copy()
Fold = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[config.target_col])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
print(folds.groupby(['fold', config.target_col]).size())

folds[['image_id', 'label', 'fold']] \
    .to_csv(join(config.data_base_path, str(config.k_folds) + 'folds.csv'), index=False)