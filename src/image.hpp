/*================================================================
*   Copyright (C) 2020 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : image.hpp
*   Author      : YunYang1994
*   Created date: 2020-04-11 17:27:11
*   Description :
*
*===============================================================*/

#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <string>
#include <iostream>

class Image{
    public:
        float *data;
        int rows;
        int cols;
        int channels;
        int size;

        Image(int h, int w, int c);             // 构造函数
        ~Image();                               // 析构函数
        Image(const Image &im);                 // 拷贝构造函数

        float &at(int y, int x, int z) const;   // 按照 [H, W, C] 顺序索引像素值
        Image copy();
};

Image imread(std::string filename, int channels);
void imwrite(std::string filename, Image im);

#endif
