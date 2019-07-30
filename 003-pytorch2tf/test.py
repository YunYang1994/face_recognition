#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-07-29 20:24:48
#   Description :
#
#================================================================

import cv2
import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf

image = np.arange(224*224*3).reshape([224, 224, 3])
image = image / 255.
image = np.expand_dims(image, 0).astype(np.float32)

tf_image = image
torch_image = np.transpose(image, [0, 3, 1, 2])
torch_image = torch.Tensor(torch_image)

bn = nn.BatchNorm2d(32)
torch.nn.init.normal_(bn.running_mean)
torch.nn.init.constant_(bn.running_var, 100)

# Define pytorch model
model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True),
        bn,
        nn.LeakyReLU(0.1, inplace=True))

with torch.no_grad():
    model.eval()
    torch_result = model(torch_image)
    torch_result = np.transpose(torch_result, [0, 2, 3, 1])

# Parsing weights ...
def parsing_weights(model):
    layers = model.children()
    conv_layer = next(layers)
    conv_weight = conv_layer.weight.detach().numpy()
    bias_weight = conv_layer.bias.detach().numpy()
    conv_weight = np.transpose(conv_weight, [2, 3, 1, 0])
    bn_layer = next(layers)

    print(conv_layer, bn_layer)

    gama = bn_layer.weight.detach().numpy()
    beta = bn_layer.bias.detach().numpy()
    running_mean = bn_layer.running_mean.detach().numpy()
    running_var = bn_layer.running_var.detach().numpy()

    conv_weight = [conv_weight, bias_weight]
    bn_weight = [gama, beta, running_mean, running_var]
    return conv_weight, bn_weight

# Saving model
torch.save(model.state_dict(), 'test.pth')
state_dict = torch.load("./test.pth")
model.load_state_dict(state_dict)
conv_weight, bn_weight = parsing_weights(model)

# Define tf_model
input_layer = tf.keras.layers.Input([None, None, 3])
conv = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=True)(input_layer)
bn = tf.keras.layers.BatchNormalization()(conv)
y = tf.nn.leaky_relu(bn, alpha=0.1)
tf_model = tf.keras.Model(input_layer, y)

# Assign torch weights to tf_model
tf_model.layers[1].set_weights(conv_weight)
tf_model.layers[2].set_weights(bn_weight)

# Print result
tf_result = tf_model(tf_image)

print(torch_result[0][0][0])
print(tf_result[0][0][0])


