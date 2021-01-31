import torch
import torchvision.transforms as transforms 
from torch.utils.data import Dataset

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import PIL

try:
    # from pytorch_toolbelt.inference import tta
    import ttach as tta
    tta_support = True
except:
    tta_support = False



def get_albu_transform(transform, config):

    if transform == 'valid_tta':
        test_trans = A.Compose([
            A.RandomCrop(config.image_size,config.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:

        test_trans = A.Compose([
            A.Resize(config.image_size,config.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]),

    if transform == "torchvision":
        train_trans, test_trans = get_torch_transform(config.image_size)
    elif transform == "strong":
        train_trans = A.Compose([
                A.Compose([
                    A.RandomRotate90(),
                    A.Flip(),
                    A.RandomCrop(width=config.image_size, height=config.image_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                   A.OneOf([
                       A.MotionBlur(p=.2),
                       A.MedianBlur(blur_limit=3, p=0.1),
                       A.Blur(blur_limit=3, p=0.1),
                       ], p=0.2),
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
                    ], p=config.p),
                A.Resize(config.image_size,config.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
                ])
    elif transform == "strong_fix":
        train_trans =  A.Compose([
                A.Compose([
                    A.RandomRotate90(),
                    A.Flip(),
                    A.RandomCrop(width=config.image_size, height=config.image_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.OneOf([
                        A.IAAAdditiveGaussianNoise(),
                        ], p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
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
                # 旋转平移
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),

                # 光照色彩/直方图均衡化
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                A.CLAHE(p=0.2),
                
                # 空间扭曲/局部屏蔽
                A.OpticalDistortion(p=0.2),
                A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, always_apply=False, p=0.2),

                # 归一化
                A.Resize(config.image_size,config.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
                ])

        test_trans = A.Compose([
            # 直方图均衡化
            A.CLAHE(p=1), 

            # 归一化
            A.Resize(config.image_size,config.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
            ])    
    elif transform == "valid_tta" and tta_support:
        train_trans = tta.Compose([
                        tta.HorizontalFlip(),
                        tta.VerticalFlip(),
                        #tta.FiveCrops(crop_height=config.image_size, crop_width=config.image_size),
                        # tta.Rotate90(angles=[0, 90, 180, 270]),
                    ])
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

        
