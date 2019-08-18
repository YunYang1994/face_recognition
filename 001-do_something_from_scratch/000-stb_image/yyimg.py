#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yyimg.py
#   Author      : YunYang1994
#   Created date: 2019-06-20 15:10:49
#   Description :
#
#================================================================

from ctypes import *


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

lib = CDLL("libyyimg.so", RTLD_GLOBAL)

load_image_lib = lib.load_image
load_image_lib.argtypes = [c_char_p]
load_image_lib.restype = IMAGE


def load_image(f):
    return load_image_lib(f.encode('ascii'))

save_image_lib = lib.save_image
save_image_lib.argtypes = [IMAGE, c_char_p]


def save_image(im, f):
    return save_image_lib(im, f.encode('ascii'))



