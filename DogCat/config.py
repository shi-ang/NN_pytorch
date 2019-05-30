# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:37:20 2019

@author: shian
"""
import warnings
import torch

class DefaultConfig(object):
    env = 'default' # visdom environment
    vis_port = 8097 # visdom port, default 8097, could be non-implied
    model = 'AlexNet' # The model selected from ./models, name must be same with the name in ./models/__init__/py
    
    train_data_root = './data/train/' # Train data root
    test_data_root = './data/test1/' # Test data root
    load_model_path = None # Load pretrained model, None means don't load
    
    batch_size = 128 
    use_gpu = True
    num_workers = 4 # How many parallel workers for loading data
    print_freq = 20 # print log every N batch
    
    max_epoch = 10 # How many times will the training set be trained
    lr = 0.01 # Learning rate
    lr_decay = 0.95 # When val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # Hyperparameter, coefficient of L2 penalty C
    
    device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')
    
    def parse(self, kwargs):
        """
        Update attributes based on dictionary kwargs

        Parameters
        ----------
        kwargs : TYPE [dict]
            DESCRIPTION. {'max_epoch':5, 'lr' = 0.001}

        Returns
        -------
        None.

        """
        
        """
        hasattr(object, name): determine whether this object has attribut name, return bool
        getattr(object, name): get the value of attribut name in object
        setattr(object, name, values): set attribut name in object to values, if the attribute name does not exist, create the attribute then set the values  
        """
        for k, v in kwargs.item():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self, k, v)
        
        print('user config:')
        # Print all the attribut starts with '__'
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
        