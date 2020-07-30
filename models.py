#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : models.py
#   Author      : YunYang1994
#   Created date: 2020-03-22 00:37:56
#   Description :
#
#================================================================

import cv2
import numpy as np
import tensorflow as tf
from utils import normalize

"""
reference from
    https://github.com/deepinsight/insightface/blob/master/src/symbols/fmobilefacenet.py
"""

def load_weights(weight_file, model):
    weights_dict = np.load(weight_file, allow_pickle=True).item()
    layer_names = weights_dict.keys()

    for layer_name in layer_names:
        if layer_name in ["mulscalar0_second", "minusscalar0_second"]:
            continue

        layer = model.get_layer(name=layer_name)
        weights = weights_dict[layer_name]

        if "batchnorm" in layer_name:
            gamma = weights_dict[layer_name]['scale']
            beta  = weights_dict[layer_name]['bias']
            mean  = weights_dict[layer_name]['mean']
            var   = weights_dict[layer_name]['var']
            layer.set_weights([gamma, beta, mean, var])

        elif "relu" in layer_name:
            gamma = weights["gamma"]
            gamma = gamma[np.newaxis, np.newaxis, :]
            layer.set_weights([gamma])
        elif 'pre_fc1' in layer_name:
            layer.set_weights([weights["weights"], weights["bias"]])

        elif 'fc1' in layer_name:
            beta  = weights_dict[layer_name]['bias']
            mean  = weights_dict[layer_name]['mean']
            var   = weights_dict[layer_name]['var']
            gamma = np.ones_like(var)
            layer.set_weights([gamma, beta, mean, var])

        else:
            layer.set_weights([weights["weights"]])

    return None

def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=0, num_group=1, name=None, suffix=''):
    out = tf.pad(data, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

    if num_group == 1:
        out = tf.keras.layers.Conv2D(num_filter, kernel, stride, use_bias=False,
                name='%s%s_conv2d' %(name, suffix))(out)
    else:
        depth_multiplier = num_filter // num_group
        out = tf.keras.layers.DepthwiseConv2D(kernel, stride, use_bias=False,
                depth_multiplier=depth_multiplier, name="%s%s_conv2d" %(name, suffix))(out)

    out = tf.keras.layers.BatchNormalization(epsilon=0.0010000000474974513, momentum=0.9,
                name="%s%s_batchnorm" %(name, suffix))(out)
    return out

def conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=1, num_group=1, name=None, suffix=''):
    out = Linear(data, num_filter, kernel, stride, pad, num_group, name, suffix)
    out = tf.keras.layers.PReLU(shared_axes=[1,2], name='%s%s_relu' %(name, suffix))(out)
    return out

def DResidual(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=1, num_group=1, name=None, suffix=''):
    conv_sep = conv(data, num_group, kernel=(1, 1), stride=(1, 1), pad=0, num_group=1, name='%s%s_conv_sep' %(name, suffix))
    conv_dw  = conv(conv_sep, num_group, kernel, stride, pad, num_group=num_group, name='%s%s_conv_dw' %(name, suffix))
    proj = Linear(conv_dw, num_out, kernel=(1, 1), stride=(1, 1), pad=0, num_group=1, name='%s%s_conv_proj' %(name, suffix))
    return proj

def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=1, num_group=1, name=None, suffix=''):
    identity=data
    for i in range(num_block):
    	shortcut = identity
    	out = DResidual(identity, num_out, kernel, stride, pad, num_group, name='%s%s_block' %(name, suffix), suffix='%d'%i)
    	identity = out + shortcut
    return identity

def get_model(image_w, image_h, pretrained=False):

    data = tf.keras.layers.Input(shape=(image_h, image_w, 3))
    norm_data      = tf.keras.layers.Lambda(lambda x: (x-127.5)*0.0078125)(data)

    conv_1         = conv(norm_data,   num_filter=64, kernel=(3, 3), pad=1, stride=(2, 2), name="conv_1")
    conv_2_dw      = conv(conv_1,       num_group=64, num_filter=64, kernel=(3, 3), pad=1, stride=(1, 1), name="conv_2_dw")

    conv_23        = DResidual(conv_2_dw,           num_out=64,  kernel=(3, 3), stride=(2, 2), pad=1, num_group=128, name="dconv_23")
    conv_3         = Residual(conv_23, num_block=4, num_out=64,  kernel=(3, 3), stride=(1, 1), pad=1, num_group=128, name="res_3")
    conv_34        = DResidual(conv_3,              num_out=128, kernel=(3, 3), stride=(2, 2), pad=1, num_group=256, name="dconv_34")
    conv_4         = Residual(conv_34, num_block=6, num_out=128, kernel=(3, 3), stride=(1, 1), pad=1, num_group=256, name="res_4")
    conv_45        = DResidual(conv_4,              num_out=128, kernel=(3, 3), stride=(2, 2), pad=1, num_group=512, name="dconv_45")
    conv_5         = Residual(conv_45, num_block=2, num_out=128, kernel=(3, 3), stride=(1, 1), pad=1, num_group=256, name="res_5")

    conv_6_sep     = conv(conv_5, num_filter=512, kernel=(1, 1), pad=0, stride=(1, 1), name="conv_6sep")
    conv_6_dw      = Linear(conv_6_sep, num_filter=512, kernel=(7, 7), stride=(1, 1), pad=0, num_group=512, name="conv_6dw7_7")
    conv_6_flatten = tf.keras.layers.Flatten()(conv_6_dw)
    conv_6_fc      = tf.keras.layers.Dense(128, name='pre_fc1')(conv_6_flatten)
    fc1            = tf.keras.layers.BatchNormalization(epsilon=1.9999999494757503e-05, momentum=0.9, name='fc1')(conv_6_fc)

    model = tf.keras.Model(inputs=data, outputs=fc1)
    if pretrained:
        load_weights("./models/mobilefacenet.npy", model)
    return model

class MobileFaceNet(object):
    def __init__(self, image_w=112, image_h=112, pretrained=True):
        self.image_w = image_w
        self.image_h = image_h
        self.model = get_model(image_w, image_h, pretrained)

    def __call__(self, image):
        image = cv2.resize(image, (self.image_w, self.image_h))
        image = np.expand_dims(image, 0).astype(np.float32)
        embedding = self.model.predict_on_batch(image)
        return normalize(embedding)


class IResnet(object):
    def __init__(self, image_w=112, image_h=112, tflite_model=None):
        self.image_w = image_w
        self.image_h = image_h

        self.interpreter = tf.lite.Interpreter(model_path=tflite_model)
        self.interpreter.allocate_tensors()

        # 获取输入和输出张量。
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, image):
        image = cv2.resize(image, (self.image_w, self.image_h))
        image = np.expand_dims(image, 0).astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()

        embedding = self.interpreter.get_tensor(self.output_details[0]['index'])
        return normalize(embedding)






