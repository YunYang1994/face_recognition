/*================================================================
*   Copyright (C) 2019 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : image_basics.c
*   Author      : YunYang1994
*   Created date: 2019-06-21 12:30:33
*   Description :
*
*===============================================================*/

#include <math.h>
#include <assert.h>

#include "image_basics.h"

float get_pixel(image im, int x, int y, int c)
{
    if(x >= im.w) x = im.w - 1;
    if(y >= im.h) y = im.h - 1;
    if(x < 0) x = 0;
    if(y < 0) y = 0;
    assert(c >= 0 && c < im.c);
    return im.data[x + im.w * y + im.w * im.h * c];
}

void set_pixel(image im, int x, int y, int c, float v)
{
    assert(c >= 0 && c < im.c);
    if(x >= 0 && x < im.w && y >= 0 && y <= im.h){
        im.data[x + im.w * y + im.w * im.h * c] = v;
    }
}

image rgb2gray(image im){
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
    for(int i=0; i<im.w; i++){
        for(int j=0; j<im.h; j++){
            int idx = i + im.w * j;
            gray.data[idx] = 0.299 * im.data[idx] +
                0.587 * im.data[idx + im.w*im.h*1] + 0.114 * im.data[ idx + im.w*im.h*2];
        }
    }
    return gray;
}

image thresh_binary(image im, float thresh){
    assert(im.c == 1);
    image binary = make_image(im.w, im.h, 1);
    for(int i=0; i<im.w; i++){
        for(int j=0; j<im.h; j++){
            int idx = i + im.w*j;
            if(im.data[idx] > thresh){
                binary.data[idx] = 1.;
            }else{
                binary.data[idx] = 0.;
            }
        }
    }
    return binary;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    int lx = (int) floor(x);
    int ly = (int) floor(y);

    float Q11 = get_pixel(im, lx, ly, c);       // Q11的坐标为[lx, ly]，    得到它像素
    float Q12 = get_pixel(im, lx, ly+1, c);     // Q12的坐标为[lx, ly+1]，  得到它像素
    float Q21 = get_pixel(im, lx+1, ly, c);     // Q21的坐标为[lx+1, ly]，  得到它像素
    float Q22 = get_pixel(im, lx+1, ly+1, c);   // Q22的坐标为[lx+1, ly+1]，得到它像素

    float R1  = (x - lx)*Q21 + (lx+1-x)*Q11;    // R1 的坐标为[x, ly],   插值得到它像素
    float R2  = (x - lx)*Q22 + (lx+1-x)*Q12;    // R2 的坐标为[x, ly+1], 插值得到它像素

    float P  = (y - ly)*R2 + (ly+1-y)*R1;       // P 的坐标为[x, y], 插值得到它像素
    return P;
}

image bilinear_resize(image im, int w, int h)
{
    image r = make_image(w, h, im.c);
    float xscale = (float)im.w/w;
    float yscale = (float)im.h/h;
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                float y = (j+.5)*yscale - .5;
                float x = (i+.5)*xscale - .5;
                float val = bilinear_interpolate(im, x, y, k);
                set_pixel(r, i, j, k, val);
            }
        }
    }
    return r;
}

float nn_interpolate(image im, float x, float y, int c)
{
    int lx = (int) round(x);
    int ly = (int) round(y);
    float P = get_pixel(im, lx, ly, c);
    return P;
}

image nn_resize(image im, int w, int h)
{
    image r = make_image(w, h, im.c);
    float xscale = (float)im.w/w;
    float yscale = (float)im.h/h;
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                float y = (j+.5)*yscale - .5;
                float x = (i+.5)*xscale - .5;
                float val = nn_interpolate(im, x, y, k);
                set_pixel(r, i, j, k, val);
            }
        }
    }
    return r;
}






