#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-06-20 15:14:58
#   Description :
#
#================================================================

import yyimg

image_path = "../data/dog.jpg"
image = yyimg.load_image(image_path)
yyimg.save_image(image, "result")


