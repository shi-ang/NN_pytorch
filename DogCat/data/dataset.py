# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:24:14 2019

@author: shian
"""

import os
from PIL import Image
import torch.utils.data as data
from torchvision import transforms as T

class Dogcat(data.Dataset):
    def __init__(self, root, transforms = None, train = True, test = False):
        # Obtain the directory of all images
        self.root = root
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        
        self.test = test
        self.train = train
        
        # imgs is shuffle. Sort imgs based on the files' name
        # test1: data/test1/8973.jpg    find 8973 and sort
        # train: data/train/cat.10004.jpg   find 10004 and sort
        if self.test:
            imgs = sorted(imgs, key = lambda x: x.split('.')[-2].split('/')[-1])
        else:
            imgs = sorted(imgs, key = lambda x: x.split('.')[-2])
        
        # Divide the imgs set into trainset, validation set, and test set
        if self.test:
            self.imgs = imgs
        elif self.train:
            self.imgs = imgs[:int(0.7*len(imgs))]
        else:
            self.imgs = imgs[int(0.7*len(imgs)):]
            
        if transforms is None:
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
            self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize])
        else:
            self.transforms = transforms
            
    def __getitem__(self, index):
        img_path = self.imgs[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        if self.test:
            # Don't have label data of test set, will return the index number of their files' name
            label = img_path.split('.')[-2].split('/')[-1]
        else:    
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        return data, label
    
    def __len__(self):
        return len(self.imgs)
        
"""
root = './test1/'
imgs = [os.path.join(root, img) for img in os.listdir(root)]
print(imgs[12000].split('.')[-2].split('/')[-1])
#label = 1 if 'dog' in img_path.split('/')[-1] else 0
"""
