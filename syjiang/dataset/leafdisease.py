import os
import glob 
import torch
import pickle

import numpy as np
# import pandas as pd
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import cv2
from PIL import Image, ImageOps


import albumentations as A
from albumentations.pytorch import ToTensorV2



# @staticmethod 
def get_transform(ch=1):
    train_trans = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomRotation(60, fill=(0,)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                    transforms.ColorJitter(0.1,0.1,0.1,0)
                ],p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    test_trans = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225], )
            ])
    return train_trans, test_trans


def GetTrainStrongTransforms():

    transform = A.Compose([
                A.Resize(512,512),
                A.RandomRotate90(),
                A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                ),
                A.Flip(),
                A.RandomCrop(width=512, height=512),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(),
                    A.GaussNoise(),
                    ], p=0.2),
                
                A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=.1),
                    A.IAAPiecewiseAffine(p=0.3),
                    ], p=0.2),

                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.IAASharpen(),
                    A.IAAEmboss(),
                    A.RandomBrightnessContrast(),            
                    ], p=0.3),
                A.HueSaturationValue(p=0.3),
                ToTensorV2(),
                ])



    # transform = A.Compose([
    #         A.RandomRotate90(),
    #         A.Flip(),
    #         A.Transpose(),

            # A.OneOf([
            #     A.IAAAdditiveGaussianNoise(),
            #     A.GaussNoise(),
            # ], p=0.2),
            # A.OneOf([
            #     A.MotionBlur(p=.2),
            #     A.MedianBlur(blur_limit=3, p=0.1),
            #     A.Blur(blur_limit=3, p=0.1),
            # ], p=0.2),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            # A.OneOf([
            #     A.OpticalDistortion(p=0.3),
            #     A.GridDistortion(p=.1),
            #     A.IAAPiecewiseAffine(p=0.3),
            # ], p=0.2),
            # A.OneOf([
            #     A.CLAHE(clip_limit=2),
            #     A.IAASharpen(),
            #     A.IAAEmboss(),
            #     A.RandomBrightnessContrast(),            
            # ], p=0.3),
            # A.HueSaturationValue(p=0.3),
            # ToTensorV2(p=1.0),
        # ])

    return transform


def GetValidStrongTransforms():
    transform = A.Compose([
        A.Resize(512,512),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
        ])
    return transform





class CLDDataset(Dataset):
    def __init__(self, df, dirs ,mode, DataAugmentationStrong):

        self.train_trans , self.valid_trans = get_transform(ch=3)
        self.strong_aug_train = GetTrainStrongTransforms()
        self.strong_aug_valid = GetValidStrongTransforms()
        self.df = df
        self.dirs = dirs
        self.mode = mode
        self.strongdata = DataAugmentationStrong

    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, index):
        
        row = self.df.loc[index]

        image = cv2.imread(os.path.join(self.dirs, row.image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = image.astype(np.float32)
        # image = image.transpose(2,0,1)
        # print(image.shape)

        if self.mode == 'train':

            if self.strongdata:
                image = self.strong_aug_train(image = image)['image']
                return image.float(), torch.tensor(row.label).float()
            else:
                image = Image.fromarray(image)
                image = self.train_trans(image)
                
                return image,   torch.tensor(row.label).float()

        # print(image.shape)
            
            # image = image.permute(2,0,1)
            # image = self.train_trans(image)
        if self.mode == 'valid':

            if self.strongdata:
                image = self.strong_aug_valid(image = image)['image']
                return image.float(), torch.tensor(row.label).float()
            else:
                image = Image.fromarray(image)
                image = self.valid_trans(image) # is tensor

                return image,   torch.tensor(row.label).float()


# class DatasetWarpper(object):
#     """docstring for  DataWarpper"""
#     def __init__(self, path, valid_size):
#         super( DataWarpper, self).__init__()
#         self.path = path
#         self.valid_size = valid_size

#     def get_dataloaders(self):
        
#         return train_loader, valid_loader

        