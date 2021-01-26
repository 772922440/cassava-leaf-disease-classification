import torch
import torchvision.transforms as transforms 
from torch.utils.data import Dataset

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import PIL


def get_transform(image_size):
    train_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.RandomRotation(60, fill=(0,)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                    transforms.ColorJitter(0.1,0.1,0.1,0)
                ],p=0.4),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225], )
            ])
    return train_trans, test_trans

def get_albu_transform(image_size):
    train_trans = A.Compose([
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
    test_trans = A.Compose([
        A.Resize(512,512),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
        ])
    return train_trans, test_trans


class CLDDataset(Dataset):
    def __init__(self, df, mode, transform=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

        if isinstance(transform ,torchvision.transforms.transforms.Compose):
            self.transform_type = 'torch'
        else:
            self.transform_type = 'albu'
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.loc[index]
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # H W C
        
        # tranform
        if self.transform_type == "torch":
            image = self.transform(image)
        else:
            image = self.transform(image=image)['image']
        
        if self.mode == 'test':
            return torch.tensor(image).float()
        else:
            return torch.tensor(image).float(), torch.tensor(row.label).long()

        