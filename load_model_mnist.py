# -*- coding: utf-8 -*-
"""
Created on Thu May 30 08:41:50 2024

@author: Piput Setyono
"""

import torch.nn as nn
import torchvision.models as models

class MnistMobileNetV3Small(nn.Module):
  def __init__(self, in_channels=1, target_class=10):
    super(MnistMobileNetV3Small, self).__init__()

    # Load a pretrained model from torchvision.models in Pytorch
    self.model = models.mobilenet_v3_small(pretrained=True, progress=True)

    # Change the input layer to take Grayscale image if in_channel = 1. 
    if in_channels == 1:
        self.model.features[0][0]= nn.Conv2d(in_channels, 16, kernel_size=[3,3], stride=[2,2], padding=[1,1], bias=False)
    
    # Change the output layer to output target_class
    num_ftrs = self.model.classifier[3].in_features
    self.model.classifier[3] = nn.Linear(num_ftrs, target_class)

  def forward(self, x):
    return self.model(x)

class MnistMobileNetV3Large(nn.Module):
  def __init__(self, in_channels=1, target_class=10):
    super(MnistMobileNetV3Large, self).__init__()

    # Load a pretrained model from torchvision.models in Pytorch
    self.model = models.mobilenet_v3_large(pretrained=True, progress=True)

    # Change the input layer to take Grayscale image if in_channel = 1. 
    if in_channels == 1:
        self.model.features[0][0]= nn.Conv2d(in_channels, 16, kernel_size=[3,3], stride=[2,2], padding=[1,1], bias=False)
    
    # Change the output layer to output target_class
    num_ftrs = self.model.classifier[3].in_features
    self.model.classifier[3] = nn.Linear(num_ftrs, target_class)

  def forward(self, x):
    return self.model(x)

class MnistMobileNetV2(nn.Module):
  def __init__(self, in_channels=1, target_class=10):
    super(MnistMobileNetV2, self).__init__()

    # Load a pretrained model from torchvision.models in Pytorch
    self.model = models.mobilenet_v2(pretrained=True, progress=True)

    # Change the input layer to take Grayscale image if in_channel = 1. 
    if in_channels == 1:
        self.model.features[0][0]= nn.Conv2d(in_channels, 32, kernel_size=[3,3], stride=[2,2], padding=[1,1], bias=False)
    
    # Change the output layer to output 10 classes instead of 1000 classes
    num_ftrs = self.model.classifier[1].in_features
    self.model.classifier[1] = nn.Linear(num_ftrs, target_class)

  def forward(self, x):
    return self.model(x)

class MnistResNet50(nn.Module):
  def __init__(self, in_channels=1, target_class=10):
    super(MnistResNet50, self).__init__()

    # Load a pretrained model from torchvision.models in Pytorch
    self.model = models.resnet50(pretrained=True, progress=True)

    # Change the input layer to take Grayscale image if in_channel = 1. 
    if in_channels == 1:
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    # Change the output layer to output 10 classes instead of 1000 classes
    num_ftrs = self.model.fc.in_features
    self.model.fc = nn.Linear(num_ftrs, target_class)

  def forward(self, x):
    return self.model(x)
