#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : convert_to_tflite.py
#   Author      : YunYang1994
#   Created date: 2020-02-15 14:47:28
#   Description :
#
#================================================================

import sys
sys.path.append("../")

import numpy as np
import tensorflow as tf
from mtcnn import PNet, RNet, ONet

def load_weights(model, weights_file):
    weights_dict = np.load(weights_file, encoding='latin1').item()
    for layer_name in weights_dict.keys():
        layer = model.get_layer(layer_name)
        if "conv" in layer_name:
            layer.set_weights([weights_dict[layer_name]["weights"], weights_dict[layer_name]["biases"]])
        else:
            prelu_weight = weights_dict[layer_name]['alpha']
            try:
                layer.set_weights([prelu_weight])
            except:
                layer.set_weights([prelu_weight[np.newaxis, np.newaxis, :]])
    return True

pnet, rnet, onet = PNet(), RNet(), ONet()
pnet(tf.ones(shape=[1,  12,  12, 3]))
rnet(tf.ones(shape=[1,  24,  24 ,3]))
onet(tf.ones(shape=[1,  48,  48, 3]))
load_weights(pnet, "./det1.npy"), load_weights(rnet, "./det2.npy"), load_weights(onet, "./det3.npy")

pnet.predict(tf.ones(shape=[1,  12,  12, 3]))
pnet_converter = tf.lite.TFLiteConverter.from_keras_model(pnet)
pnet_converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
with open("pnet.tflite", "wb") as f:
    pnet_tflite_model = pnet_converter.convert()
    f.write(pnet_tflite_model)

rnet.predict(tf.ones(shape=[1,  24,  24, 3]))
rnet_converter = tf.lite.TFLiteConverter.from_keras_model(rnet)
rnet_converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
with open("rnet.tflite", "wb") as f:
    rnet_tflite_model = rnet_converter.convert()
    f.write(rnet_tflite_model)

onet.predict(tf.ones(shape=[1,  48,  48, 3]))
onet_converter = tf.lite.TFLiteConverter.from_keras_model(onet)
onet_converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
with open("onet.tflite", "wb") as f:
    onet_tflite_model = onet_converter.convert()
    f.write(onet_tflite_model)

