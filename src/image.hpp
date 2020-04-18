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

        Image();
        Image(int h, int w, int c);                // 构造函数
        ~Image();                                  // 析构函数
        Image(const Image &other);                 // 拷贝构造函数
        Image& operator=(const Image &other);      // 赋值函数

        float &at(int y, int x, int z) const;      // 按照 [H, W, C] 顺序索引像素值
        Image copy();

        Image gray();                              // 转灰度图函数
        Image resize(int w, int h);                // 图像的 resize 操作，双线性插值
        Image resize(float factor);
};

Image imread(std::string filename, int channels);  // 读取图片
void imwrite(std::string filename, Image im);      // 写出图片

#endif
