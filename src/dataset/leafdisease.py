from albumentations.augmentations.transforms import RandomCrop, RandomResizedCrop, Resize
import torch
import torchvision.transforms as transforms 
from torch.utils.data import Dataset

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import PIL
import timm

try:
    # from pytorch_toolbelt.inference import tta
    import ttach as tta
    tta_support = True
except:
    tta_support = False


def get_albu_transform(transform, config):
    # default test_trans
    test_trans = A.Compose([
        A.Resize(config.image_size,config.image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),

        ToTensorV2(),
    ])
    
    # train
    if transform == "strong_fix":
        train_trans =  A.Compose([
                A.Compose([
                    A.RandomRotate90(p=0.5),
                    A.Flip(p=0.5),
                    A.RandomResizedCrop(width=config.image_size, height=config.image_size, scale=(0.2, 1.0), p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                    A.IAAAdditiveGaussianNoise(p=0.2),
                    A.OneOf([
                        A.CLAHE(clip_limit=2),
                        A.IAASharpen(),
                        A.RandomBrightnessContrast(),            
                        ], p=0.3),
                    ], p=config.p),
                A.Resize(config.image_size,config.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
                ])
    elif transform == "strong_fix2":
        train_trans =  A.Compose([
                A.Compose([
                    A.RandomRotate90(p=0.5),
                    A.Flip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
                    A.RandomResizedCrop(width=config.image_size, height=config.image_size, scale=(0.2, 1.0), p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                ], p=config.p),
                A.Resize(config.image_size,config.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
                ])
    elif transform == "strong_fix3" or transform == "light": 
        train_trans =  A.Compose([
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=0.2, 
                    sat_shift_limit=0.2, 
                    val_shift_limit=0.2, 
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1,0.1), 
                    contrast_limit=(-0.1, 0.1), 
                    p=0.5
                ),
                A.RandomResizedCrop(config.image_size, config.image_size, scale=(0.2, 1), p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
                ])
    elif transform == "heavy": 
        train_trans =  A.Compose([
            A.Resize(config.image_size, config.image_size),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
            A.RGBShift(r_shift_limit=20, 
                g_shift_limit=20, 
                b_shift_limit=20, 
                always_apply=False, 
                p=0.5
            ),
            A.ChannelDropout(
                channel_drop_range=(1, 1), 
                fill_value=0, 
                always_apply=False, 
                p=0.5
            ),
            A.GridDistortion(num_steps=5, 
                distort_limit=0.3, 
                interpolation=1, 
                border_mode=4, 
                value=None, 
                mask_value=None, 
                always_apply=False, 
                p=0.5
            ),
            A.CoarseDropout(p=0.5),
            A.Cutout(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
            ])
    elif transform == "auto_augment_transform": 
        train_trans =  timm.data.auto_augment.auto_augment_transform('originalr', None)
    else:
        raise "transform error"

    return train_trans, test_trans




def get_torch_transform(image_size):
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

        
