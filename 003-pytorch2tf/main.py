#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : main.py
#   Author      : YunYang1994
#   Created date: 2019-07-30 14:01:56
#   Description :
#
#================================================================

import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms

# from tf_model import tf_model
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense
from torch_model import torch_model

# tensorflow model
tf_model = tf.keras.Sequential()
tf_model.add(Conv2D(16, 3, 1, padding="same", input_shape=(28, 28, 1)))
tf_model.add(BatchNormalization())
tf_model.add(ReLU())
tf_model.add(Flatten())
tf_model.add(Dense(10))

# Parsing layers
tf_conv_layer = tf_model.layers[0]
tf_bn_layer = tf_model.layers[1]
tf_linear_layer = tf_model.layers[-1]

# pytorch model
state_dict = torch.load("./model.pth")
torch_model.load_state_dict(state_dict)

children = torch_model.children()
Sequential_1 = next(children)
Layers = Sequential_1.children()

torch_conv_layer = next(Layers)
print(torch_conv_layer)
torch_bn_layer = next(Layers)
print(torch_bn_layer)

torch_Linear_layer = next(children)
print(torch_Linear_layer)

# Parsing weights
conv_weight = torch_conv_layer.weight.detach().numpy()
conv_weight = np.transpose(conv_weight, [2, 3, 1, 0])
bias_weight = torch_conv_layer.bias.detach().numpy()

conv_layer_weights = [conv_weight, bias_weight]

gama = torch_bn_layer.weight.detach().numpy()
beta = torch_bn_layer.bias.detach().numpy()
running_mean = torch_bn_layer.running_mean.detach().numpy()
running_var = torch_bn_layer.running_var.detach().numpy()

bn_layer_weights = [gama, beta, running_mean, running_var]

linear_weight = torch_Linear_layer.weight.detach().numpy()
linear_weight = np.transpose(linear_weight, [1, 0])
linear_bias = torch_Linear_layer.bias.detach().numpy()

linear_layer_weights = [linear_weight, linear_bias]

# # Loading weights
# weights_dict = np.load("model.npy", encoding='latin1').item()
# conv_layer_weights = weights_dict["conv"]
# bn_layer_weights = weights_dict["bn"]
# linear_layer_weights = weights_dict["linear"]

# Assigning weights
tf_conv_layer.set_weights(conv_layer_weights)
tf_bn_layer.set_weights(bn_layer_weights)
tf_linear_layer.set_weights(linear_layer_weights)


test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False,
                                          transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=True)
dataset = iter(test_loader)
torch_image, torch_label = next(dataset)
tf_image, tf_label = np.transpose(torch_image.numpy(), [0, 2, 3, 1]), torch_label.numpy()

with torch.no_grad():
    torch_model.eval()
    torch_output = torch_model(torch_image).numpy()

tf_output = tf_model(tf_image).numpy()

print(torch_output)
print(tf_output)

print("label : %d, torch : %d, tf : % d" %(tf_label, np.argmax(torch_output), np.argmax(tf_output)))

