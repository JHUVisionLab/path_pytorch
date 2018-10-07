from __future__ import print_function, division
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
from torch.utils.data import random_split
from torchvision import transforms, utils, models
from resnet import ResNet, ResNet_Tiling, ResNet_Tiling_2fc, ResNet_2fc
# from resnet import resnet50_fc, resnet50_tiling_1fc, resnet50_tiling_2fc

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


def resnet50_train(num_classes):
  model = resnet50_fc(pretrained=True, num_classes = 4)
  return model

def resnet50_train_tiling(num_classes=4, num_res = 3):
  model = resnet50_tiling_1fc(pretrained=True, num_classes = 4, num_res = num_res)
  return model

def resnet50_train_tiling2(num_classes=4, num_res = 3):
  model = resnet50_tiling_2fc(pretrained=True, num_classes = 4, num_res = num_res)
  return model

def resnet18(pretrained=False, **kwargs):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False, **kwargs):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  
  return model


def resnet50_fc(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet_2fc(Bottleneck, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict = False)

    ### Freeze base model of resnet
  for param in model.parameters():
    param.requires_grad = False

    ### Set fc layers to be trainabale
  model.fc1.weight.requires_grad = True
  model.fc1.bias.requires_grad = True
  model.fc2.weight.requires_grad = True
  model.fc2.bias.requires_grad = True
  return model

def resnet50_tiling_2fc(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet_Tiling(Bottleneck, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict = False)

  ### Freeze base model of resnet
  for param in model.parameters():
    param.requires_grad = False

  # ct = 0
  # for child in model_ft.children():
  #   ct += 1
  #   if ct < 9:
 #        for param in child.parameters():
 #            param.requires_grad = False

  ### Set fc layers to be trainabale
  model.fc1.weight.requires_grad = True
  model.fc1.bias.requires_grad = True
  model.fc2.weight.requires_grad = True
  model.fc2.bias.requires_grad = True
  return model

def resnet50_tiling_1fc(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet_Tiling(Bottleneck, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict = False)

  ### Freeze base model of resnet
  for param in model.parameters():
    param.requires_grad = False

  # ct = 0
  # for child in model.children():
  #   ct += 1
  #   if ct < 9:
 #        for param in child.parameters():
 #            param.requires_grad = False

  ### Set fc layers to be trainabale
  model.fc1.weight.requires_grad = True
  model.fc1.bias.requires_grad = True
  return model


def resnet101(pretrained=False, **kwargs):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False, **kwargs):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model
