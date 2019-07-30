#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : convert_weight.py
#   Author      : YunYang1994
#   Created date: 2019-07-30 11:59:13
#   Description :
#
#================================================================

import torch
import numpy as np
from torch_model import ConvNet

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Prepare model
model = ConvNet().to(device)
state_dict = torch.load("./model.pth")
model.load_state_dict(state_dict)

# Parsing Layers
children = model.children()
Sequential_1 = next(children)
Layers = Sequential_1.children()

conv_layer = next(Layers)
print(conv_layer)
bn_layer = next(Layers)
print(bn_layer)

Linear_layer = next(children)
print(Linear_layer)

# Parsing weights
conv_weight = conv_layer.weight.detach().numpy()
conv_weight = np.transpose(conv_weight, [2, 3, 1, 0])
bias_weight = conv_layer.bias.detach().numpy()

gama = bn_layer.weight.detach().numpy()
beta = bn_layer.bias.detach().numpy()
running_mean = bn_layer.running_mean.detach().numpy()
running_var = bn_layer.running_var.detach().numpy()

linear_weight = Linear_layer.weight.detach().numpy()
linear_weight = np.transpose(linear_weight, [1, 0])
linear_bias = Linear_layer.bias.detach().numpy()

# Saving weights as ,npy file
weights_dict = {"conv":[conv_weight, bias_weight],
               "bn":[gama, beta, running_mean, running_var],
               "linear":[linear_weight, linear_bias]}
np.save("model.npy", weights_dict)



