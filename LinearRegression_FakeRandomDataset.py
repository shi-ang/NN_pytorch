# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:10:38 2019

@author: shian
"""

import torch
from matplotlib import pyplot as plt
from IPython import display

torch.manual_seed(123)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_fake_data(length = 10):
    x = torch.rand(length, 1, device = device) *5
    y = x *2 + 3 + torch.randn(length, 1, device = device) * 0.1
    return x, y

data, label = get_fake_data(length = 100)
plt.scatter(data.squeeze().cpu().numpy(), label.squeeze().cpu().numpy())

w = torch.rand(1, 1).to(device)
b = torch.zeros(1, 1).to(device)
print(w.item())
print(b.item())

log_interval = 50
lr = 0.0001
for ep in range(5000):
    #forward
    predict = data.mm(w) + b.expand_as(label)
    #calculate loss
    loss = 0.5 * (predict - label) ** 2
    loss = loss.mean()
    #backward propogation
    dloss = 1
    
    dpredict = dloss* (predict - label)
    dw = data.t().mm(dpredict)
    db = dpredict.sum()
    
    #update parameters
    w.sub_(lr*dw)
    b.sub_(lr*db)
    
    if ep % log_interval == (0):
        print('Epoch: {}, {}/{}({}%), weight: {}, bias: {}'.format(
                ep, ep*len(data), 500*len(data), ep/500, w.item(), b.item()))
