# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:27:51 2019

@author: shian
"""
import torch
from data.dataset import Dogcat
from config import DefaultConfig
import models
from utils import Visualizer
from torch.utils.data import DataLoader
import meter

def help():
    """
    python file.py help
    print the information about this function
    """
    
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)
    
opt = DefaultConfig()
    
trainset = Dogcat(root = './data/train/', train = True)
testset = Dogcat(opt.test_data_root, train = False, test = True)


help()


def train(**kwargs):
    # Change configuration
    opt.parse(kwargs)
    # Create a new visualizer
    vis = Visualizer(opt.env)
    
    # Use string to aquire the desired model
    # Equals to model = models.AlexNet()
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.to(opt.device)
    model.train()
        
    # Data
    trainset = Dogcat(opt.train_data_root, train = True, test = False)
    valset = Dogcat(opt.train_data_root, train = False, test = False)
    trainset_loader = DataLoader(trainset, opt.batch_size, shuffle = True, num_workers = opt.num_workers)
    valset_loader = DataLoader(valset, opt.batch_size, shuffle = False, num_workers = opt.num_workers)
    
    # Criterion(loss function) and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay = opt.weight_decay)
    
    for ep in range(opt.max_epoch):
        
        for batch_idx, (data, label) in enumerate(trainset_loader):
            if opt.use_gpu:
                data, label = data.to(opt.device), label.to(opt.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
        