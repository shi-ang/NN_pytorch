#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:26:22 2019

@author: shiang
"""

import torch
import torch.nn as nn
import time

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(print(self))
    
    """
    Provide two method (load, save) for all the subclass
    """
    def load(self, path):
        # load_state_dict takes a dictionary object, not a path.
        # self.load_state_dict(path) is wrong
        self.load_state_dict(torch.load(path))
        
    def save(self, name = None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            # strftime convert data and time to string
            # Y - year, m - month, d -date, H - hour, M - minute, S - second
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        print('Model saved to %s' %name)