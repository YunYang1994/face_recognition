#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-08-21 14:02:21
#   Description :
#
#================================================================

import tensorflow as tf

x = tf.constant(1., shape=[1,2,2,1])
w = tf.constant(1., shape=[2,2,1,1])
y = tf.nn.conv2d_transpose(x, w, output_shape=[1,4,4,1], strides=[1,2,2,1])
print(y.shape)



