# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 20:28:28 2019

@author: brand
"""

import torch 
import torch.nn as nn
from model1 import predictor

model = predictor(17, 11)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

checkpoint = torch.load('/ModelState.pt')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

for p in model.parameters():
    print(p.data)