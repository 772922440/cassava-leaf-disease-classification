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

        