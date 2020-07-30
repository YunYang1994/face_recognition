#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : mobileFaceNet.py
#   Author      : YunYang1994
#   Created date: 2020-02-23 16:10:08
#   Description :
#
#================================================================


import cv2
import mxnet as mx
import numpy as np
from utils import normalize
from easydict import EasyDict as edict


class MobileFaceNet(object):
    def __init__(self, model_path="./models/model", gpu_id=None):
        self.model = edict()
        self.model.ctx = mx.gpu(gpu_id) if gpu_id else mx.cpu()
        self.model.sym, self.model.arg_params, self.model.aux_params = mx.model.load_checkpoint(model_path, 0000)
        self.model.arg_params, self.model.aux_params = self.ch_dev(self.model.arg_params,
                                                                           self.model.aux_params, self.model.ctx)
        self.all_layers = self.model.sym.get_internals()
        self.model.sym = self.all_layers['fc1_output']

    def ch_dev(self, arg_params, aux_params, ctx):
        new_args = dict()
        new_auxs = dict()
        for k, v in arg_params.items():
            new_args[k] = v.as_in_context(ctx)
        for k, v in aux_params.items():
            new_auxs[k] = v.as_in_context(ctx)
        return new_args, new_auxs


    def __call__(self, image):
        """
        image channels order: RGB
        """
        image = cv2.resize(image, (112, 112))
        image = np.transpose(image, (2,0,1))
        image = np.expand_dims(image, 0)

        self.model.arg_params["data"] = mx.nd.array(image, self.model.ctx)
        self.model.arg_params["softmax_label"] = mx.nd.empty((1,), self.model.ctx)
        exe = self.model.sym.bind(self.model.ctx, self.model.arg_params ,
                                              args_grad=None, grad_req="null", aux_states=self.model.aux_params)
        exe.forward(is_train=False)
        embedding = exe.outputs[0].asnumpy()
        return normalize(embedding)


