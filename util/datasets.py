import os
from PIL import Image
import numpy as np
import random 
import pickle

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets

class pretrain_dataset(Dataset):
    def __init__(self, root='/workspace/mmdetection/data/coco/train2017',  transform=None, args=None):
        self.root = root
        self.args = args
        self.img_names_list = os.listdir(root)
        self.transform = transform

    def __getitem__(self, idx):
        
        img_path  = os.path.join(self.root, self.img_names_list[idx])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.img_names_list)

    def _get_names_and_dirs(self):
        img_names_list = os.listdir(root)

        return img_names_list, img_dir
    
