/*================================================================
*   Copyright (C) 2019 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : image.h
*   Author      : YunYang1994
*   Created date: 2019-06-20 10:02:16
*   Description :
*
*===============================================================*/

#ifndef IMAGE_H
#define IMAGE_H

typedef struct{
    int w,h,c;
    float *data;
} image;


// Loading and saving
image make_image(int w, int h, int c);
image load_image(const char *filename);
void save_image(image im, const char *name);
void free_image(image im);

#endif
