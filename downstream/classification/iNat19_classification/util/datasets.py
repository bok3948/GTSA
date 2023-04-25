# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from PIL import Image

import torch
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import json
from torch.utils.data import Dataset

class iNat19_Dataset(Dataset):
    def __init__(self, root=None, split='train', transform=None):
        self.root = root
        self.split = split
        
        #load json 
        if self.split == 'train':
            json_file = os.path.join(self.root, self.split+"2019.json")
            
        else:
            json_file = os.path.join(self.root, 'val'+"2019.json")
            
        all_annos = json.load(open(json_file, 'r'))

        self.annos = all_annos['annotations']
        self.images = all_annos['images']
        self.transform = transform
        self.num_classes = len(all_annos["categories"])
        
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        assert self.images[idx]["id"] == self.annos[idx]["id"]
        label = self.annos[idx]["category_id"]
        label = torch.tensor(label).long()
        img_path = os.path.join(self.root, self.images[idx]["file_name"])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label




def build_iNat_dataset(is_train, args):
    transform = build_transform(is_train, args)
    dataset = iNat19_Dataset(root=args.data_path, split='train' if is_train else 'val', transform=transform)

    print(dataset)

    return dataset
    

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
