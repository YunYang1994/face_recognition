/*================================================================
*   Copyright (C) 2019 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : image_basics.h
*   Author      : YunYang1994
*   Created date: 2019-06-21 12:28:17
*   Description :
*
*===============================================================*/

#ifndef IMAGE_BASICS_H
#define IMAGE_BASICS_H

#include "image.h"

float get_pixel(image im, int x, int y, int c);
float bilinear_interpolate(image im, float x, float y, int c);
float nn_interpolate(image im, float x, float y, int c);
void set_pixel(image im, int x, int y, int c, float v);

image rgb2gray(image im);
image thresh_binary(image im, float thresh);
image bilinear_resize(image im, int w, int h);
image nn_resize(image im, int w, int h);

#endif


