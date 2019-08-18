/*================================================================
*   Copyright (C) 2019 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : test.c
*   Author      : YunYang1994
*   Created date: 2019-06-20 10:00:35
*   Description :
*
*===============================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image.h"
#include "image_basics.h"

void test_get_pixel(image im){
    for(int i=0; i<im.w; i++){
        for(int j=0; j<im.h; j++){
            for(int k=0; k<im.c; k++){
                int pixel = 255 * get_pixel(im, i, j, k);
                printf("pixel in [%d, %d, %d] is %d\n", i, j, k, pixel);
            }
        }
    }
}

void test_set_pixel(image im){
    printf("removing R channels ...\n");
    for(int i=0; i<im.w; i++){
        for(int j=0; j<im.h; j++){
            set_pixel(im, i, j, 0, 0.); /* 0-> R channels; 1-> G channels; 2-> B channels*/
        }
    }
    save_image(im, "result");
}

void test_rgb2gray(image im){
    image gray = rgb2gray(im);
    save_image(gray, "result");
}

void test_thresh_binary(image im, float thresh){
    image gray = rgb2gray(im);
    image binary = thresh_binary(gray, thresh);
    save_image(binary, "result");
}

void test_bilinear_resize(image im){
    image resize_image = bilinear_resize(im, 416, 416);
    save_image(resize_image, "result");
}


void test_nn_resize(image im){
    image resize_image = nn_resize(im, 416, 416);
    save_image(resize_image, "result");
}

int main(int argc, char **argv)
{
    if(argc < 3){
        fprintf(stderr, "usage: %s <function> <image_path>\n", argv[0]);
        return 0;
    }

    image im = load_image(argv[2]);

    if(0 == strcmp(argv[1], "get_pixel")){
        test_get_pixel(im);
    }else if(0 == strcmp(argv[1], "set_pixel")){
        test_set_pixel(im);
    }else if(0 == strcmp(argv[1], "rgb2gray")){
        test_rgb2gray(im);
    }else if(0 == strcmp(argv[1], "thresh_binary")){
        test_thresh_binary(im, 0.5);
    }else if(0 == strcmp(argv[1], "bilinear_resize")){
        test_bilinear_resize(im);
    }else if(0 == strcmp(argv[1], "nn_resize")){
        test_nn_resize(im);
    }

    free_image(im);
    return 0;
}
