/*================================================================
*   Copyright (C) 2019 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : image.c
*   Author      : YunYang1994
*   Created date: 2019-06-20 13:00:23
*   Description :
*
*===============================================================*/


#include <stdio.h>
#include <stdlib.h>

#include "image.h"


image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0; /* 没有初始化的指针可能指向任何一处内存，所以最好用0初始化*/
    out.h = h;    /* 定义图片的高度*/
    out.w = w;    /* 定义图片的宽度*/
    out.c = c;    /* 定义图片的颜色通道*/
    return out;   /* 返回image结构体*/
}


image make_image(int w, int h, int c)
{
    /*根据w, h, c来创建一个image结构体并初始化*/
    image out = make_empty_image(w,h,c);
    /*为图片的像素矩阵分配内存空间, 返回一个指向float类型的指针*/
    out.data = (float *)calloc(h*w*c, sizeof(float));
    return out;
}


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

image load_image_stb(const char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    // Function stbi_load
    // Standard parameters:
    //    int *x                 -- outputs image width in pixels
    //    int *y                 -- outputs image height in pixels
    //    int *channels_in_file  -- outputs # of image components in image file
    //    int desired_channels   -- if non-zero, # of image components requested in result
    //
    // The return value from an image loader is an 'unsigned char *' which points
    // to the pixel data, or NULL on an allocation failure or if the image is
    // corrupt or invalid. The pixel data consists of *y scanlines of *x pixels,
    // with each pixel consisting of N interleaved 8-bit components; the first
    // pixel pointed to is top-left-most in the image. There is no padding between
    // image scanlines or between pixels, regardless of format. The number of
    // components N is 'desired_channels' if desired_channels is non-zero, or
    // *channels_in_file otherwise. If desired_channels is non-zero,
    // *channels_in_file has the number of components that _would_ have been
    // output otherwise. E.g. if you set desired_channels to 4, you will always
    // get RGBA output, but you can check *channels_in_file to see if it's trivially
    // opaque because e.g. there were only 3 channels in the source image.
    //
    // An output image with N components has the following components interleaved
    // in this order in each pixel:
    //
    //     N=#comp     components
    //       1           grey
    //       2           grey, alpha
    //       3           red, green, blue
    //       4           red, green, blue, alpha
    //
    // If image loading fails for any reason, the return value will be NULL,
    // and *x, *y, *channels_in_file will be unchanged. The function
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n",
            filename, stbi_failure_reason());
        exit(0);
    }
    if (channels) c = channels;
    int i,j,k;
    image im = make_image(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                /* printf("the pixel in the coordinates of (%d,%d,%d) is %d\n", i, j, k, data[src_index]); */
                im.data[dst_index] = (float)data[src_index]/255.; /*将每个像素scale到0～1之间*/
            }
        }
    }
    //We don't like alpha channels, so discard it !
    if(im.c == 4) im.c = 3;
    free(data); /* 释放掉stbi_load读取图片所占用的内存空间*/
    return im;
}


image load_image(const char *filename)
{
    image out = load_image_stb(filename, 0);
    return out;
}


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


void save_image_stb(image im, const char *name)
{
    char buff[256];
    sprintf(buff, "%s.jpg", name);
    unsigned char *data = (unsigned char *)calloc(im.w*im.h*im.c, sizeof(char));
    int i,k;
    for(k = 0; k < im.c; ++k){
        for(i = 0; i < im.w*im.h; ++i){
            data[i*im.c+k] = (unsigned char) roundf((255*im.data[i + k*im.w*im.h]));
        }
    }
    int success = stbi_write_jpg(buff, im.w, im.h, im.c, data, 100);
    free(data);
    if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
}

void save_image(image im, const char *name)
{
    save_image_stb(im, name);
}

void free_image(image im)
{
    free(im.data);
}

