#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : torch_test.py
#   Author      : YunYang1994
#   Created date: 2019-07-30 11:39:32
#   Description :
#
#================================================================


import torch
import torchvision
import torchvision.transforms as transforms
from torch_model import torch_model

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False,
                                          transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)
# Prepare model
torch_model = torch_model.to(device)
state_dict = torch.load("./model.pth")
torch_model.load_state_dict(state_dict)

# Test the model
torch_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = torch_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('=> Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
