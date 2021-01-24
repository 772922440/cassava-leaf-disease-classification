import os
import glob 
import torch
import pickle

# import numpy as np
# import pandas as pd
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from PIL import Image, ImageOps


# @staticmethod 
def get_transform(ch=1):
    train_trans = transforms.Compose([
            transforms.Resize(600 , 800),
            transforms.RandomRotation(60, fill=(0,)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                    transforms.ColorJitter(0.1,0.1,0.1,0)
                ],p=0.4),
            #GaussianBlur(kernel_size=int(0.01 * img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*ch, std=[0.5]*ch)
        ])
    test_trans = transforms.Compose([
            # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*ch, std=[0.5]*ch)
        ])
    return train_trans, test_trans



class CLDDataset(Dataset):
    def __init__(self, df, dirs ,mode):

        self.train_trans , self.valid_trans = get_transform(ch=3)
        self.df = df
        self.dirs = dirs
        self.mode = mode

    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, index):
        
        row = self.df.loc[index]

        # print(os.path.join(self.dirs, row.image_id))
        # image = cv2.imread(os.path.join(self.dirs, row.image_id) ) 
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.open(os.path.join(self.dirs, row.image_id) )
        if self.mode == 'train':
            image = self.train_trans(image)
        if self.mode == 'valid':
            image = self.valid_trans(image)

        # print(image.shape)

        return image,  torch.tensor(row.label).float()







# class DatasetWarpper(object):
#     """docstring for  DataWarpper"""
#     def __init__(self, path, valid_size):
#         super( DataWarpper, self).__init__()
#         self.path = path
#         self.valid_size = valid_size

#     def get_dataloaders(self):
        
#         return train_loader, valid_loader

        