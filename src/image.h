/*================================================================
*   Copyright (C) 2020 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : image.h
*   Author      : YunYang1994
*   Created date: 2020-04-11 17:27:11
*   Description :
*
*===============================================================*/

#include <iostream>

class Image{
    public:
        float *data;
        int rows;
        int cols;
        int channels;
        int size;

        Image(int h, int w, int c);
        ~Image();
        Image(const Image &im);   // copy constructor
};

Image imread(char *filename, int channels);


