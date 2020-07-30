#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : IResnet.py
#   Author      : YunYang1994
#   Created date: 2020-03-21 11:37:32
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf

__weights_dict = dict()

is_train = False

def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes', allow_pickle=True).item()

    return weights_dict


def KitModel(weight_file = None):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    minusscalar0_second = tf.constant(__weights_dict['minusscalar0_second']['value'], dtype=tf.float32, name='minusscalar0_second')
    mulscalar0_second = tf.constant(__weights_dict['mulscalar0_second']['value'], dtype=tf.float32, name='mulscalar0_second')
    data            = tf.placeholder(tf.float32, shape = (None, 112, 112, 3), name = 'data')
    minusscalar0    = data - minusscalar0_second
    mulscalar0      = minusscalar0 * mulscalar0_second
    conv0_pad       = tf.pad(mulscalar0, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv0           = convolution(conv0_pad, group=1, strides=[1, 1], padding='VALID', name='conv0')
    bn0             = batch_normalization(conv0, variance_epsilon=1.9999999494757503e-05, name='bn0')
    relu0           = prelu(bn0, name='relu0')
    stage1_unit1_bn1 = batch_normalization(relu0, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_bn1')
    stage1_unit1_conv1sc = convolution(relu0, group=1, strides=[2, 2], padding='VALID', name='stage1_unit1_conv1sc')
    stage1_unit1_conv1_pad = tf.pad(stage1_unit1_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit1_conv1 = convolution(stage1_unit1_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage1_unit1_conv1')
    stage1_unit1_sc = batch_normalization(stage1_unit1_conv1sc, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_sc')
    stage1_unit1_bn2 = batch_normalization(stage1_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_bn2')
    stage1_unit1_relu1 = prelu(stage1_unit1_bn2, name='stage1_unit1_relu1')
    stage1_unit1_conv2_pad = tf.pad(stage1_unit1_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit1_conv2 = convolution(stage1_unit1_conv2_pad, group=1, strides=[2, 2], padding='VALID', name='stage1_unit1_conv2')
    stage1_unit1_bn3 = batch_normalization(stage1_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_bn3')
    plus0           = stage1_unit1_bn3 + stage1_unit1_sc
    stage1_unit2_bn1 = batch_normalization(plus0, variance_epsilon=1.9999999494757503e-05, name='stage1_unit2_bn1')
    stage1_unit2_conv1_pad = tf.pad(stage1_unit2_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit2_conv1 = convolution(stage1_unit2_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage1_unit2_conv1')
    stage1_unit2_bn2 = batch_normalization(stage1_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit2_bn2')
    stage1_unit2_relu1 = prelu(stage1_unit2_bn2, name='stage1_unit2_relu1')
    stage1_unit2_conv2_pad = tf.pad(stage1_unit2_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit2_conv2 = convolution(stage1_unit2_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage1_unit2_conv2')
    stage1_unit2_bn3 = batch_normalization(stage1_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage1_unit2_bn3')
    plus1           = stage1_unit2_bn3 + plus0
    stage1_unit3_bn1 = batch_normalization(plus1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit3_bn1')
    stage1_unit3_conv1_pad = tf.pad(stage1_unit3_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit3_conv1 = convolution(stage1_unit3_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage1_unit3_conv1')
    stage1_unit3_bn2 = batch_normalization(stage1_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit3_bn2')
    stage1_unit3_relu1 = prelu(stage1_unit3_bn2, name='stage1_unit3_relu1')
    stage1_unit3_conv2_pad = tf.pad(stage1_unit3_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage1_unit3_conv2 = convolution(stage1_unit3_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage1_unit3_conv2')
    stage1_unit3_bn3 = batch_normalization(stage1_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage1_unit3_bn3')
    plus2           = stage1_unit3_bn3 + plus1
    stage2_unit1_bn1 = batch_normalization(plus2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_bn1')
    stage2_unit1_conv1sc = convolution(plus2, group=1, strides=[2, 2], padding='VALID', name='stage2_unit1_conv1sc')
    stage2_unit1_conv1_pad = tf.pad(stage2_unit1_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit1_conv1 = convolution(stage2_unit1_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit1_conv1')
    stage2_unit1_sc = batch_normalization(stage2_unit1_conv1sc, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_sc')
    stage2_unit1_bn2 = batch_normalization(stage2_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_bn2')
    stage2_unit1_relu1 = prelu(stage2_unit1_bn2, name='stage2_unit1_relu1')
    stage2_unit1_conv2_pad = tf.pad(stage2_unit1_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit1_conv2 = convolution(stage2_unit1_conv2_pad, group=1, strides=[2, 2], padding='VALID', name='stage2_unit1_conv2')
    stage2_unit1_bn3 = batch_normalization(stage2_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_bn3')
    plus3           = stage2_unit1_bn3 + stage2_unit1_sc
    stage2_unit2_bn1 = batch_normalization(plus3, variance_epsilon=1.9999999494757503e-05, name='stage2_unit2_bn1')
    stage2_unit2_conv1_pad = tf.pad(stage2_unit2_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit2_conv1 = convolution(stage2_unit2_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit2_conv1')
    stage2_unit2_bn2 = batch_normalization(stage2_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit2_bn2')
    stage2_unit2_relu1 = prelu(stage2_unit2_bn2, name='stage2_unit2_relu1')
    stage2_unit2_conv2_pad = tf.pad(stage2_unit2_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit2_conv2 = convolution(stage2_unit2_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit2_conv2')
    stage2_unit2_bn3 = batch_normalization(stage2_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit2_bn3')
    plus4           = stage2_unit2_bn3 + plus3
    stage2_unit3_bn1 = batch_normalization(plus4, variance_epsilon=1.9999999494757503e-05, name='stage2_unit3_bn1')
    stage2_unit3_conv1_pad = tf.pad(stage2_unit3_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit3_conv1 = convolution(stage2_unit3_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit3_conv1')
    stage2_unit3_bn2 = batch_normalization(stage2_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit3_bn2')
    stage2_unit3_relu1 = prelu(stage2_unit3_bn2, name='stage2_unit3_relu1')
    stage2_unit3_conv2_pad = tf.pad(stage2_unit3_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit3_conv2 = convolution(stage2_unit3_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit3_conv2')
    stage2_unit3_bn3 = batch_normalization(stage2_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit3_bn3')
    plus5           = stage2_unit3_bn3 + plus4
    stage2_unit4_bn1 = batch_normalization(plus5, variance_epsilon=1.9999999494757503e-05, name='stage2_unit4_bn1')
    stage2_unit4_conv1_pad = tf.pad(stage2_unit4_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit4_conv1 = convolution(stage2_unit4_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit4_conv1')
    stage2_unit4_bn2 = batch_normalization(stage2_unit4_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit4_bn2')
    stage2_unit4_relu1 = prelu(stage2_unit4_bn2, name='stage2_unit4_relu1')
    stage2_unit4_conv2_pad = tf.pad(stage2_unit4_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit4_conv2 = convolution(stage2_unit4_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit4_conv2')
    stage2_unit4_bn3 = batch_normalization(stage2_unit4_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit4_bn3')
    plus6           = stage2_unit4_bn3 + plus5
    stage2_unit5_bn1 = batch_normalization(plus6, variance_epsilon=1.9999999494757503e-05, name='stage2_unit5_bn1')
    stage2_unit5_conv1_pad = tf.pad(stage2_unit5_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit5_conv1 = convolution(stage2_unit5_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit5_conv1')
    stage2_unit5_bn2 = batch_normalization(stage2_unit5_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit5_bn2')
    stage2_unit5_relu1 = prelu(stage2_unit5_bn2, name='stage2_unit5_relu1')
    stage2_unit5_conv2_pad = tf.pad(stage2_unit5_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit5_conv2 = convolution(stage2_unit5_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit5_conv2')
    stage2_unit5_bn3 = batch_normalization(stage2_unit5_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit5_bn3')
    plus7           = stage2_unit5_bn3 + plus6
    stage2_unit6_bn1 = batch_normalization(plus7, variance_epsilon=1.9999999494757503e-05, name='stage2_unit6_bn1')
    stage2_unit6_conv1_pad = tf.pad(stage2_unit6_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit6_conv1 = convolution(stage2_unit6_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit6_conv1')
    stage2_unit6_bn2 = batch_normalization(stage2_unit6_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit6_bn2')
    stage2_unit6_relu1 = prelu(stage2_unit6_bn2, name='stage2_unit6_relu1')
    stage2_unit6_conv2_pad = tf.pad(stage2_unit6_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit6_conv2 = convolution(stage2_unit6_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit6_conv2')
    stage2_unit6_bn3 = batch_normalization(stage2_unit6_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit6_bn3')
    plus8           = stage2_unit6_bn3 + plus7
    stage2_unit7_bn1 = batch_normalization(plus8, variance_epsilon=1.9999999494757503e-05, name='stage2_unit7_bn1')
    stage2_unit7_conv1_pad = tf.pad(stage2_unit7_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit7_conv1 = convolution(stage2_unit7_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit7_conv1')
    stage2_unit7_bn2 = batch_normalization(stage2_unit7_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit7_bn2')
    stage2_unit7_relu1 = prelu(stage2_unit7_bn2, name='stage2_unit7_relu1')
    stage2_unit7_conv2_pad = tf.pad(stage2_unit7_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit7_conv2 = convolution(stage2_unit7_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit7_conv2')
    stage2_unit7_bn3 = batch_normalization(stage2_unit7_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit7_bn3')
    plus9           = stage2_unit7_bn3 + plus8
    stage2_unit8_bn1 = batch_normalization(plus9, variance_epsilon=1.9999999494757503e-05, name='stage2_unit8_bn1')
    stage2_unit8_conv1_pad = tf.pad(stage2_unit8_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit8_conv1 = convolution(stage2_unit8_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit8_conv1')
    stage2_unit8_bn2 = batch_normalization(stage2_unit8_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit8_bn2')
    stage2_unit8_relu1 = prelu(stage2_unit8_bn2, name='stage2_unit8_relu1')
    stage2_unit8_conv2_pad = tf.pad(stage2_unit8_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit8_conv2 = convolution(stage2_unit8_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit8_conv2')
    stage2_unit8_bn3 = batch_normalization(stage2_unit8_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit8_bn3')
    plus10          = stage2_unit8_bn3 + plus9
    stage2_unit9_bn1 = batch_normalization(plus10, variance_epsilon=1.9999999494757503e-05, name='stage2_unit9_bn1')
    stage2_unit9_conv1_pad = tf.pad(stage2_unit9_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit9_conv1 = convolution(stage2_unit9_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit9_conv1')
    stage2_unit9_bn2 = batch_normalization(stage2_unit9_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit9_bn2')
    stage2_unit9_relu1 = prelu(stage2_unit9_bn2, name='stage2_unit9_relu1')
    stage2_unit9_conv2_pad = tf.pad(stage2_unit9_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit9_conv2 = convolution(stage2_unit9_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit9_conv2')
    stage2_unit9_bn3 = batch_normalization(stage2_unit9_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit9_bn3')
    plus11          = stage2_unit9_bn3 + plus10
    stage2_unit10_bn1 = batch_normalization(plus11, variance_epsilon=1.9999999494757503e-05, name='stage2_unit10_bn1')
    stage2_unit10_conv1_pad = tf.pad(stage2_unit10_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit10_conv1 = convolution(stage2_unit10_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit10_conv1')
    stage2_unit10_bn2 = batch_normalization(stage2_unit10_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit10_bn2')
    stage2_unit10_relu1 = prelu(stage2_unit10_bn2, name='stage2_unit10_relu1')
    stage2_unit10_conv2_pad = tf.pad(stage2_unit10_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit10_conv2 = convolution(stage2_unit10_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit10_conv2')
    stage2_unit10_bn3 = batch_normalization(stage2_unit10_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit10_bn3')
    plus12          = stage2_unit10_bn3 + plus11
    stage2_unit11_bn1 = batch_normalization(plus12, variance_epsilon=1.9999999494757503e-05, name='stage2_unit11_bn1')
    stage2_unit11_conv1_pad = tf.pad(stage2_unit11_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit11_conv1 = convolution(stage2_unit11_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit11_conv1')
    stage2_unit11_bn2 = batch_normalization(stage2_unit11_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit11_bn2')
    stage2_unit11_relu1 = prelu(stage2_unit11_bn2, name='stage2_unit11_relu1')
    stage2_unit11_conv2_pad = tf.pad(stage2_unit11_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit11_conv2 = convolution(stage2_unit11_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit11_conv2')
    stage2_unit11_bn3 = batch_normalization(stage2_unit11_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit11_bn3')
    plus13          = stage2_unit11_bn3 + plus12
    stage2_unit12_bn1 = batch_normalization(plus13, variance_epsilon=1.9999999494757503e-05, name='stage2_unit12_bn1')
    stage2_unit12_conv1_pad = tf.pad(stage2_unit12_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit12_conv1 = convolution(stage2_unit12_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit12_conv1')
    stage2_unit12_bn2 = batch_normalization(stage2_unit12_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit12_bn2')
    stage2_unit12_relu1 = prelu(stage2_unit12_bn2, name='stage2_unit12_relu1')
    stage2_unit12_conv2_pad = tf.pad(stage2_unit12_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit12_conv2 = convolution(stage2_unit12_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit12_conv2')
    stage2_unit12_bn3 = batch_normalization(stage2_unit12_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit12_bn3')
    plus14          = stage2_unit12_bn3 + plus13
    stage2_unit13_bn1 = batch_normalization(plus14, variance_epsilon=1.9999999494757503e-05, name='stage2_unit13_bn1')
    stage2_unit13_conv1_pad = tf.pad(stage2_unit13_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit13_conv1 = convolution(stage2_unit13_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit13_conv1')
    stage2_unit13_bn2 = batch_normalization(stage2_unit13_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit13_bn2')
    stage2_unit13_relu1 = prelu(stage2_unit13_bn2, name='stage2_unit13_relu1')
    stage2_unit13_conv2_pad = tf.pad(stage2_unit13_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage2_unit13_conv2 = convolution(stage2_unit13_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage2_unit13_conv2')
    stage2_unit13_bn3 = batch_normalization(stage2_unit13_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit13_bn3')
    plus15          = stage2_unit13_bn3 + plus14
    stage3_unit1_bn1 = batch_normalization(plus15, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_bn1')
    stage3_unit1_conv1sc = convolution(plus15, group=1, strides=[2, 2], padding='VALID', name='stage3_unit1_conv1sc')
    stage3_unit1_conv1_pad = tf.pad(stage3_unit1_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit1_conv1 = convolution(stage3_unit1_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit1_conv1')
    stage3_unit1_sc = batch_normalization(stage3_unit1_conv1sc, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_sc')
    stage3_unit1_bn2 = batch_normalization(stage3_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_bn2')
    stage3_unit1_relu1 = prelu(stage3_unit1_bn2, name='stage3_unit1_relu1')
    stage3_unit1_conv2_pad = tf.pad(stage3_unit1_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit1_conv2 = convolution(stage3_unit1_conv2_pad, group=1, strides=[2, 2], padding='VALID', name='stage3_unit1_conv2')
    stage3_unit1_bn3 = batch_normalization(stage3_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_bn3')
    plus16          = stage3_unit1_bn3 + stage3_unit1_sc
    stage3_unit2_bn1 = batch_normalization(plus16, variance_epsilon=1.9999999494757503e-05, name='stage3_unit2_bn1')
    stage3_unit2_conv1_pad = tf.pad(stage3_unit2_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit2_conv1 = convolution(stage3_unit2_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit2_conv1')
    stage3_unit2_bn2 = batch_normalization(stage3_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit2_bn2')
    stage3_unit2_relu1 = prelu(stage3_unit2_bn2, name='stage3_unit2_relu1')
    stage3_unit2_conv2_pad = tf.pad(stage3_unit2_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit2_conv2 = convolution(stage3_unit2_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit2_conv2')
    stage3_unit2_bn3 = batch_normalization(stage3_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit2_bn3')
    plus17          = stage3_unit2_bn3 + plus16
    stage3_unit3_bn1 = batch_normalization(plus17, variance_epsilon=1.9999999494757503e-05, name='stage3_unit3_bn1')
    stage3_unit3_conv1_pad = tf.pad(stage3_unit3_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit3_conv1 = convolution(stage3_unit3_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit3_conv1')
    stage3_unit3_bn2 = batch_normalization(stage3_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit3_bn2')
    stage3_unit3_relu1 = prelu(stage3_unit3_bn2, name='stage3_unit3_relu1')
    stage3_unit3_conv2_pad = tf.pad(stage3_unit3_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit3_conv2 = convolution(stage3_unit3_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit3_conv2')
    stage3_unit3_bn3 = batch_normalization(stage3_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit3_bn3')
    plus18          = stage3_unit3_bn3 + plus17
    stage3_unit4_bn1 = batch_normalization(plus18, variance_epsilon=1.9999999494757503e-05, name='stage3_unit4_bn1')
    stage3_unit4_conv1_pad = tf.pad(stage3_unit4_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit4_conv1 = convolution(stage3_unit4_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit4_conv1')
    stage3_unit4_bn2 = batch_normalization(stage3_unit4_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit4_bn2')
    stage3_unit4_relu1 = prelu(stage3_unit4_bn2, name='stage3_unit4_relu1')
    stage3_unit4_conv2_pad = tf.pad(stage3_unit4_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit4_conv2 = convolution(stage3_unit4_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit4_conv2')
    stage3_unit4_bn3 = batch_normalization(stage3_unit4_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit4_bn3')
    plus19          = stage3_unit4_bn3 + plus18
    stage3_unit5_bn1 = batch_normalization(plus19, variance_epsilon=1.9999999494757503e-05, name='stage3_unit5_bn1')
    stage3_unit5_conv1_pad = tf.pad(stage3_unit5_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit5_conv1 = convolution(stage3_unit5_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit5_conv1')
    stage3_unit5_bn2 = batch_normalization(stage3_unit5_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit5_bn2')
    stage3_unit5_relu1 = prelu(stage3_unit5_bn2, name='stage3_unit5_relu1')
    stage3_unit5_conv2_pad = tf.pad(stage3_unit5_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit5_conv2 = convolution(stage3_unit5_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit5_conv2')
    stage3_unit5_bn3 = batch_normalization(stage3_unit5_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit5_bn3')
    plus20          = stage3_unit5_bn3 + plus19
    stage3_unit6_bn1 = batch_normalization(plus20, variance_epsilon=1.9999999494757503e-05, name='stage3_unit6_bn1')
    stage3_unit6_conv1_pad = tf.pad(stage3_unit6_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit6_conv1 = convolution(stage3_unit6_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit6_conv1')
    stage3_unit6_bn2 = batch_normalization(stage3_unit6_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit6_bn2')
    stage3_unit6_relu1 = prelu(stage3_unit6_bn2, name='stage3_unit6_relu1')
    stage3_unit6_conv2_pad = tf.pad(stage3_unit6_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit6_conv2 = convolution(stage3_unit6_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit6_conv2')
    stage3_unit6_bn3 = batch_normalization(stage3_unit6_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit6_bn3')
    plus21          = stage3_unit6_bn3 + plus20
    stage3_unit7_bn1 = batch_normalization(plus21, variance_epsilon=1.9999999494757503e-05, name='stage3_unit7_bn1')
    stage3_unit7_conv1_pad = tf.pad(stage3_unit7_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit7_conv1 = convolution(stage3_unit7_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit7_conv1')
    stage3_unit7_bn2 = batch_normalization(stage3_unit7_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit7_bn2')
    stage3_unit7_relu1 = prelu(stage3_unit7_bn2, name='stage3_unit7_relu1')
    stage3_unit7_conv2_pad = tf.pad(stage3_unit7_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit7_conv2 = convolution(stage3_unit7_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit7_conv2')
    stage3_unit7_bn3 = batch_normalization(stage3_unit7_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit7_bn3')
    plus22          = stage3_unit7_bn3 + plus21
    stage3_unit8_bn1 = batch_normalization(plus22, variance_epsilon=1.9999999494757503e-05, name='stage3_unit8_bn1')
    stage3_unit8_conv1_pad = tf.pad(stage3_unit8_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit8_conv1 = convolution(stage3_unit8_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit8_conv1')
    stage3_unit8_bn2 = batch_normalization(stage3_unit8_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit8_bn2')
    stage3_unit8_relu1 = prelu(stage3_unit8_bn2, name='stage3_unit8_relu1')
    stage3_unit8_conv2_pad = tf.pad(stage3_unit8_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit8_conv2 = convolution(stage3_unit8_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit8_conv2')
    stage3_unit8_bn3 = batch_normalization(stage3_unit8_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit8_bn3')
    plus23          = stage3_unit8_bn3 + plus22
    stage3_unit9_bn1 = batch_normalization(plus23, variance_epsilon=1.9999999494757503e-05, name='stage3_unit9_bn1')
    stage3_unit9_conv1_pad = tf.pad(stage3_unit9_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit9_conv1 = convolution(stage3_unit9_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit9_conv1')
    stage3_unit9_bn2 = batch_normalization(stage3_unit9_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit9_bn2')
    stage3_unit9_relu1 = prelu(stage3_unit9_bn2, name='stage3_unit9_relu1')
    stage3_unit9_conv2_pad = tf.pad(stage3_unit9_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit9_conv2 = convolution(stage3_unit9_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit9_conv2')
    stage3_unit9_bn3 = batch_normalization(stage3_unit9_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit9_bn3')
    plus24          = stage3_unit9_bn3 + plus23
    stage3_unit10_bn1 = batch_normalization(plus24, variance_epsilon=1.9999999494757503e-05, name='stage3_unit10_bn1')
    stage3_unit10_conv1_pad = tf.pad(stage3_unit10_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit10_conv1 = convolution(stage3_unit10_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit10_conv1')
    stage3_unit10_bn2 = batch_normalization(stage3_unit10_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit10_bn2')
    stage3_unit10_relu1 = prelu(stage3_unit10_bn2, name='stage3_unit10_relu1')
    stage3_unit10_conv2_pad = tf.pad(stage3_unit10_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit10_conv2 = convolution(stage3_unit10_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit10_conv2')
    stage3_unit10_bn3 = batch_normalization(stage3_unit10_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit10_bn3')
    plus25          = stage3_unit10_bn3 + plus24
    stage3_unit11_bn1 = batch_normalization(plus25, variance_epsilon=1.9999999494757503e-05, name='stage3_unit11_bn1')
    stage3_unit11_conv1_pad = tf.pad(stage3_unit11_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit11_conv1 = convolution(stage3_unit11_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit11_conv1')
    stage3_unit11_bn2 = batch_normalization(stage3_unit11_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit11_bn2')
    stage3_unit11_relu1 = prelu(stage3_unit11_bn2, name='stage3_unit11_relu1')
    stage3_unit11_conv2_pad = tf.pad(stage3_unit11_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit11_conv2 = convolution(stage3_unit11_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit11_conv2')
    stage3_unit11_bn3 = batch_normalization(stage3_unit11_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit11_bn3')
    plus26          = stage3_unit11_bn3 + plus25
    stage3_unit12_bn1 = batch_normalization(plus26, variance_epsilon=1.9999999494757503e-05, name='stage3_unit12_bn1')
    stage3_unit12_conv1_pad = tf.pad(stage3_unit12_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit12_conv1 = convolution(stage3_unit12_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit12_conv1')
    stage3_unit12_bn2 = batch_normalization(stage3_unit12_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit12_bn2')
    stage3_unit12_relu1 = prelu(stage3_unit12_bn2, name='stage3_unit12_relu1')
    stage3_unit12_conv2_pad = tf.pad(stage3_unit12_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit12_conv2 = convolution(stage3_unit12_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit12_conv2')
    stage3_unit12_bn3 = batch_normalization(stage3_unit12_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit12_bn3')
    plus27          = stage3_unit12_bn3 + plus26
    stage3_unit13_bn1 = batch_normalization(plus27, variance_epsilon=1.9999999494757503e-05, name='stage3_unit13_bn1')
    stage3_unit13_conv1_pad = tf.pad(stage3_unit13_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit13_conv1 = convolution(stage3_unit13_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit13_conv1')
    stage3_unit13_bn2 = batch_normalization(stage3_unit13_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit13_bn2')
    stage3_unit13_relu1 = prelu(stage3_unit13_bn2, name='stage3_unit13_relu1')
    stage3_unit13_conv2_pad = tf.pad(stage3_unit13_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit13_conv2 = convolution(stage3_unit13_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit13_conv2')
    stage3_unit13_bn3 = batch_normalization(stage3_unit13_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit13_bn3')
    plus28          = stage3_unit13_bn3 + plus27
    stage3_unit14_bn1 = batch_normalization(plus28, variance_epsilon=1.9999999494757503e-05, name='stage3_unit14_bn1')
    stage3_unit14_conv1_pad = tf.pad(stage3_unit14_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit14_conv1 = convolution(stage3_unit14_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit14_conv1')
    stage3_unit14_bn2 = batch_normalization(stage3_unit14_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit14_bn2')
    stage3_unit14_relu1 = prelu(stage3_unit14_bn2, name='stage3_unit14_relu1')
    stage3_unit14_conv2_pad = tf.pad(stage3_unit14_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit14_conv2 = convolution(stage3_unit14_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit14_conv2')
    stage3_unit14_bn3 = batch_normalization(stage3_unit14_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit14_bn3')
    plus29          = stage3_unit14_bn3 + plus28
    stage3_unit15_bn1 = batch_normalization(plus29, variance_epsilon=1.9999999494757503e-05, name='stage3_unit15_bn1')
    stage3_unit15_conv1_pad = tf.pad(stage3_unit15_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit15_conv1 = convolution(stage3_unit15_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit15_conv1')
    stage3_unit15_bn2 = batch_normalization(stage3_unit15_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit15_bn2')
    stage3_unit15_relu1 = prelu(stage3_unit15_bn2, name='stage3_unit15_relu1')
    stage3_unit15_conv2_pad = tf.pad(stage3_unit15_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit15_conv2 = convolution(stage3_unit15_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit15_conv2')
    stage3_unit15_bn3 = batch_normalization(stage3_unit15_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit15_bn3')
    plus30          = stage3_unit15_bn3 + plus29
    stage3_unit16_bn1 = batch_normalization(plus30, variance_epsilon=1.9999999494757503e-05, name='stage3_unit16_bn1')
    stage3_unit16_conv1_pad = tf.pad(stage3_unit16_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit16_conv1 = convolution(stage3_unit16_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit16_conv1')
    stage3_unit16_bn2 = batch_normalization(stage3_unit16_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit16_bn2')
    stage3_unit16_relu1 = prelu(stage3_unit16_bn2, name='stage3_unit16_relu1')
    stage3_unit16_conv2_pad = tf.pad(stage3_unit16_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit16_conv2 = convolution(stage3_unit16_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit16_conv2')
    stage3_unit16_bn3 = batch_normalization(stage3_unit16_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit16_bn3')
    plus31          = stage3_unit16_bn3 + plus30
    stage3_unit17_bn1 = batch_normalization(plus31, variance_epsilon=1.9999999494757503e-05, name='stage3_unit17_bn1')
    stage3_unit17_conv1_pad = tf.pad(stage3_unit17_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit17_conv1 = convolution(stage3_unit17_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit17_conv1')
    stage3_unit17_bn2 = batch_normalization(stage3_unit17_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit17_bn2')
    stage3_unit17_relu1 = prelu(stage3_unit17_bn2, name='stage3_unit17_relu1')
    stage3_unit17_conv2_pad = tf.pad(stage3_unit17_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit17_conv2 = convolution(stage3_unit17_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit17_conv2')
    stage3_unit17_bn3 = batch_normalization(stage3_unit17_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit17_bn3')
    plus32          = stage3_unit17_bn3 + plus31
    stage3_unit18_bn1 = batch_normalization(plus32, variance_epsilon=1.9999999494757503e-05, name='stage3_unit18_bn1')
    stage3_unit18_conv1_pad = tf.pad(stage3_unit18_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit18_conv1 = convolution(stage3_unit18_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit18_conv1')
    stage3_unit18_bn2 = batch_normalization(stage3_unit18_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit18_bn2')
    stage3_unit18_relu1 = prelu(stage3_unit18_bn2, name='stage3_unit18_relu1')
    stage3_unit18_conv2_pad = tf.pad(stage3_unit18_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit18_conv2 = convolution(stage3_unit18_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit18_conv2')
    stage3_unit18_bn3 = batch_normalization(stage3_unit18_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit18_bn3')
    plus33          = stage3_unit18_bn3 + plus32
    stage3_unit19_bn1 = batch_normalization(plus33, variance_epsilon=1.9999999494757503e-05, name='stage3_unit19_bn1')
    stage3_unit19_conv1_pad = tf.pad(stage3_unit19_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit19_conv1 = convolution(stage3_unit19_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit19_conv1')
    stage3_unit19_bn2 = batch_normalization(stage3_unit19_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit19_bn2')
    stage3_unit19_relu1 = prelu(stage3_unit19_bn2, name='stage3_unit19_relu1')
    stage3_unit19_conv2_pad = tf.pad(stage3_unit19_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit19_conv2 = convolution(stage3_unit19_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit19_conv2')
    stage3_unit19_bn3 = batch_normalization(stage3_unit19_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit19_bn3')
    plus34          = stage3_unit19_bn3 + plus33
    stage3_unit20_bn1 = batch_normalization(plus34, variance_epsilon=1.9999999494757503e-05, name='stage3_unit20_bn1')
    stage3_unit20_conv1_pad = tf.pad(stage3_unit20_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit20_conv1 = convolution(stage3_unit20_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit20_conv1')
    stage3_unit20_bn2 = batch_normalization(stage3_unit20_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit20_bn2')
    stage3_unit20_relu1 = prelu(stage3_unit20_bn2, name='stage3_unit20_relu1')
    stage3_unit20_conv2_pad = tf.pad(stage3_unit20_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit20_conv2 = convolution(stage3_unit20_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit20_conv2')
    stage3_unit20_bn3 = batch_normalization(stage3_unit20_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit20_bn3')
    plus35          = stage3_unit20_bn3 + plus34
    stage3_unit21_bn1 = batch_normalization(plus35, variance_epsilon=1.9999999494757503e-05, name='stage3_unit21_bn1')
    stage3_unit21_conv1_pad = tf.pad(stage3_unit21_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit21_conv1 = convolution(stage3_unit21_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit21_conv1')
    stage3_unit21_bn2 = batch_normalization(stage3_unit21_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit21_bn2')
    stage3_unit21_relu1 = prelu(stage3_unit21_bn2, name='stage3_unit21_relu1')
    stage3_unit21_conv2_pad = tf.pad(stage3_unit21_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit21_conv2 = convolution(stage3_unit21_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit21_conv2')
    stage3_unit21_bn3 = batch_normalization(stage3_unit21_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit21_bn3')
    plus36          = stage3_unit21_bn3 + plus35
    stage3_unit22_bn1 = batch_normalization(plus36, variance_epsilon=1.9999999494757503e-05, name='stage3_unit22_bn1')
    stage3_unit22_conv1_pad = tf.pad(stage3_unit22_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit22_conv1 = convolution(stage3_unit22_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit22_conv1')
    stage3_unit22_bn2 = batch_normalization(stage3_unit22_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit22_bn2')
    stage3_unit22_relu1 = prelu(stage3_unit22_bn2, name='stage3_unit22_relu1')
    stage3_unit22_conv2_pad = tf.pad(stage3_unit22_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit22_conv2 = convolution(stage3_unit22_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit22_conv2')
    stage3_unit22_bn3 = batch_normalization(stage3_unit22_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit22_bn3')
    plus37          = stage3_unit22_bn3 + plus36
    stage3_unit23_bn1 = batch_normalization(plus37, variance_epsilon=1.9999999494757503e-05, name='stage3_unit23_bn1')
    stage3_unit23_conv1_pad = tf.pad(stage3_unit23_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit23_conv1 = convolution(stage3_unit23_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit23_conv1')
    stage3_unit23_bn2 = batch_normalization(stage3_unit23_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit23_bn2')
    stage3_unit23_relu1 = prelu(stage3_unit23_bn2, name='stage3_unit23_relu1')
    stage3_unit23_conv2_pad = tf.pad(stage3_unit23_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit23_conv2 = convolution(stage3_unit23_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit23_conv2')
    stage3_unit23_bn3 = batch_normalization(stage3_unit23_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit23_bn3')
    plus38          = stage3_unit23_bn3 + plus37
    stage3_unit24_bn1 = batch_normalization(plus38, variance_epsilon=1.9999999494757503e-05, name='stage3_unit24_bn1')
    stage3_unit24_conv1_pad = tf.pad(stage3_unit24_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit24_conv1 = convolution(stage3_unit24_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit24_conv1')
    stage3_unit24_bn2 = batch_normalization(stage3_unit24_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit24_bn2')
    stage3_unit24_relu1 = prelu(stage3_unit24_bn2, name='stage3_unit24_relu1')
    stage3_unit24_conv2_pad = tf.pad(stage3_unit24_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit24_conv2 = convolution(stage3_unit24_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit24_conv2')
    stage3_unit24_bn3 = batch_normalization(stage3_unit24_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit24_bn3')
    plus39          = stage3_unit24_bn3 + plus38
    stage3_unit25_bn1 = batch_normalization(plus39, variance_epsilon=1.9999999494757503e-05, name='stage3_unit25_bn1')
    stage3_unit25_conv1_pad = tf.pad(stage3_unit25_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit25_conv1 = convolution(stage3_unit25_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit25_conv1')
    stage3_unit25_bn2 = batch_normalization(stage3_unit25_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit25_bn2')
    stage3_unit25_relu1 = prelu(stage3_unit25_bn2, name='stage3_unit25_relu1')
    stage3_unit25_conv2_pad = tf.pad(stage3_unit25_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit25_conv2 = convolution(stage3_unit25_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit25_conv2')
    stage3_unit25_bn3 = batch_normalization(stage3_unit25_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit25_bn3')
    plus40          = stage3_unit25_bn3 + plus39
    stage3_unit26_bn1 = batch_normalization(plus40, variance_epsilon=1.9999999494757503e-05, name='stage3_unit26_bn1')
    stage3_unit26_conv1_pad = tf.pad(stage3_unit26_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit26_conv1 = convolution(stage3_unit26_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit26_conv1')
    stage3_unit26_bn2 = batch_normalization(stage3_unit26_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit26_bn2')
    stage3_unit26_relu1 = prelu(stage3_unit26_bn2, name='stage3_unit26_relu1')
    stage3_unit26_conv2_pad = tf.pad(stage3_unit26_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit26_conv2 = convolution(stage3_unit26_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit26_conv2')
    stage3_unit26_bn3 = batch_normalization(stage3_unit26_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit26_bn3')
    plus41          = stage3_unit26_bn3 + plus40
    stage3_unit27_bn1 = batch_normalization(plus41, variance_epsilon=1.9999999494757503e-05, name='stage3_unit27_bn1')
    stage3_unit27_conv1_pad = tf.pad(stage3_unit27_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit27_conv1 = convolution(stage3_unit27_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit27_conv1')
    stage3_unit27_bn2 = batch_normalization(stage3_unit27_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit27_bn2')
    stage3_unit27_relu1 = prelu(stage3_unit27_bn2, name='stage3_unit27_relu1')
    stage3_unit27_conv2_pad = tf.pad(stage3_unit27_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit27_conv2 = convolution(stage3_unit27_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit27_conv2')
    stage3_unit27_bn3 = batch_normalization(stage3_unit27_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit27_bn3')
    plus42          = stage3_unit27_bn3 + plus41
    stage3_unit28_bn1 = batch_normalization(plus42, variance_epsilon=1.9999999494757503e-05, name='stage3_unit28_bn1')
    stage3_unit28_conv1_pad = tf.pad(stage3_unit28_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit28_conv1 = convolution(stage3_unit28_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit28_conv1')
    stage3_unit28_bn2 = batch_normalization(stage3_unit28_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit28_bn2')
    stage3_unit28_relu1 = prelu(stage3_unit28_bn2, name='stage3_unit28_relu1')
    stage3_unit28_conv2_pad = tf.pad(stage3_unit28_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit28_conv2 = convolution(stage3_unit28_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit28_conv2')
    stage3_unit28_bn3 = batch_normalization(stage3_unit28_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit28_bn3')
    plus43          = stage3_unit28_bn3 + plus42
    stage3_unit29_bn1 = batch_normalization(plus43, variance_epsilon=1.9999999494757503e-05, name='stage3_unit29_bn1')
    stage3_unit29_conv1_pad = tf.pad(stage3_unit29_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit29_conv1 = convolution(stage3_unit29_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit29_conv1')
    stage3_unit29_bn2 = batch_normalization(stage3_unit29_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit29_bn2')
    stage3_unit29_relu1 = prelu(stage3_unit29_bn2, name='stage3_unit29_relu1')
    stage3_unit29_conv2_pad = tf.pad(stage3_unit29_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit29_conv2 = convolution(stage3_unit29_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit29_conv2')
    stage3_unit29_bn3 = batch_normalization(stage3_unit29_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit29_bn3')
    plus44          = stage3_unit29_bn3 + plus43
    stage3_unit30_bn1 = batch_normalization(plus44, variance_epsilon=1.9999999494757503e-05, name='stage3_unit30_bn1')
    stage3_unit30_conv1_pad = tf.pad(stage3_unit30_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit30_conv1 = convolution(stage3_unit30_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit30_conv1')
    stage3_unit30_bn2 = batch_normalization(stage3_unit30_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit30_bn2')
    stage3_unit30_relu1 = prelu(stage3_unit30_bn2, name='stage3_unit30_relu1')
    stage3_unit30_conv2_pad = tf.pad(stage3_unit30_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage3_unit30_conv2 = convolution(stage3_unit30_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage3_unit30_conv2')
    stage3_unit30_bn3 = batch_normalization(stage3_unit30_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit30_bn3')
    plus45          = stage3_unit30_bn3 + plus44
    stage4_unit1_bn1 = batch_normalization(plus45, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_bn1')
    stage4_unit1_conv1sc = convolution(plus45, group=1, strides=[2, 2], padding='VALID', name='stage4_unit1_conv1sc')
    stage4_unit1_conv1_pad = tf.pad(stage4_unit1_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit1_conv1 = convolution(stage4_unit1_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage4_unit1_conv1')
    stage4_unit1_sc = batch_normalization(stage4_unit1_conv1sc, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_sc')
    stage4_unit1_bn2 = batch_normalization(stage4_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_bn2')
    stage4_unit1_relu1 = prelu(stage4_unit1_bn2, name='stage4_unit1_relu1')
    stage4_unit1_conv2_pad = tf.pad(stage4_unit1_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit1_conv2 = convolution(stage4_unit1_conv2_pad, group=1, strides=[2, 2], padding='VALID', name='stage4_unit1_conv2')
    stage4_unit1_bn3 = batch_normalization(stage4_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_bn3')
    plus46          = stage4_unit1_bn3 + stage4_unit1_sc
    stage4_unit2_bn1 = batch_normalization(plus46, variance_epsilon=1.9999999494757503e-05, name='stage4_unit2_bn1')
    stage4_unit2_conv1_pad = tf.pad(stage4_unit2_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit2_conv1 = convolution(stage4_unit2_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage4_unit2_conv1')
    stage4_unit2_bn2 = batch_normalization(stage4_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage4_unit2_bn2')
    stage4_unit2_relu1 = prelu(stage4_unit2_bn2, name='stage4_unit2_relu1')
    stage4_unit2_conv2_pad = tf.pad(stage4_unit2_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit2_conv2 = convolution(stage4_unit2_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage4_unit2_conv2')
    stage4_unit2_bn3 = batch_normalization(stage4_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage4_unit2_bn3')
    plus47          = stage4_unit2_bn3 + plus46
    stage4_unit3_bn1 = batch_normalization(plus47, variance_epsilon=1.9999999494757503e-05, name='stage4_unit3_bn1')
    stage4_unit3_conv1_pad = tf.pad(stage4_unit3_bn1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit3_conv1 = convolution(stage4_unit3_conv1_pad, group=1, strides=[1, 1], padding='VALID', name='stage4_unit3_conv1')
    stage4_unit3_bn2 = batch_normalization(stage4_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage4_unit3_bn2')
    stage4_unit3_relu1 = prelu(stage4_unit3_bn2, name='stage4_unit3_relu1')
    stage4_unit3_conv2_pad = tf.pad(stage4_unit3_relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    stage4_unit3_conv2 = convolution(stage4_unit3_conv2_pad, group=1, strides=[1, 1], padding='VALID', name='stage4_unit3_conv2')
    stage4_unit3_bn3 = batch_normalization(stage4_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage4_unit3_bn3')
    plus48          = stage4_unit3_bn3 + plus47
    bn1             = batch_normalization(plus48, variance_epsilon=1.9999999494757503e-05, name='bn1')
    pre_fc1_flatten = tf.contrib.layers.flatten(bn1)
    pre_fc1         = tf.layers.dense(pre_fc1_flatten, 512, kernel_initializer = tf.constant_initializer(__weights_dict['pre_fc1']['weights']), bias_initializer = tf.constant_initializer(__weights_dict['pre_fc1']['bias']), use_bias = True)
    fc1             = batch_normalization(pre_fc1, variance_epsilon=1.9999999494757503e-05, name='fc1')
    return data, fc1


def prelu(input, name):
    gamma = tf.Variable(__weights_dict[name]['gamma'], name=name + "_gamma", trainable=is_train)
    return tf.maximum(0.0, input) + gamma * tf.minimum(0.0, input)


def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, name=name, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, name=name, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer

def batch_normalization(input, name, **kwargs):
    mean = tf.Variable(__weights_dict[name]['mean'], name = name + "_mean", trainable = is_train)
    variance = tf.Variable(__weights_dict[name]['var'], name = name + "_var", trainable = is_train)
    offset = tf.Variable(__weights_dict[name]['bias'], name = name + "_bias", trainable = is_train) if 'bias' in __weights_dict[name] else None
    scale = tf.Variable(__weights_dict[name]['scale'], name = name + "_scale", trainable = is_train) if 'scale' in __weights_dict[name] else None
    return tf.nn.batch_normalization(input, mean, variance, offset, scale, name = name, **kwargs)

model = KitModel("./weights.npy")
data = np.ones([1, 112, 112, 3])
sess = tf.Session()

sess.run(tf.global_variables_initializer())

result = sess.run(model[1], feed_dict={model[0]:data})

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = ["fc1/add_1"])

with tf.gfile.GFile("IResnet.pb", "wb") as f:
    f.write(converted_graph_def.SerializeToString())


