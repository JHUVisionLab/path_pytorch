from __future__ import print_function, division
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
from torch.utils.data import random_split
from torchvision import transforms, utils, models


import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # useful stateless functions

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 
                               16, 
                               5, 
                               stride=1, 
                               padding=2,
                               bias=True)
        self.conv2 = nn.Conv2d(16, 
                               8, 
                               3, 
                               stride=1, 
                               padding=1,
                               bias=True)
        # assign layer objects to class attributes
        self.fc1 = nn.Linear(input_size*input_size*8, hidden_size)
        # nn.init package contains convenient initialization methods
        # http://pytorch.org/docs/master/nn.html#torch-nn-init 
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
    
    def forward(self, x):
        # forward always defines connectivity
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = flatten(x)
        scores = self.fc2(F.relu(self.fc1(x)))
        return scores

def resnet50(num_classes):
  model = models.resnet50(pretrained=True)
  num_ftrs = model.fc.in_features
    #I recommend training with these layers unfrozen for a couple of epochs after the initial frozen training
  for param in model.parameters():
      param.requires_grad = False
  model.fc = torch.nn.Linear(num_ftrs, len(num_classes))
  return model

