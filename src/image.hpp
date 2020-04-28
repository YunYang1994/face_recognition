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

#include <cstring>
#include <assert.h>
#include <iostream>

class Image{
    public:
        float *data;
        int rows;
        int cols;
        int channels;
        int size;

        Image(){                                    // 默认构造函数
            rows = 1;
            cols = 1;
            channels = 3;
            size = 3;

            data = (float *)calloc(size, sizeof(float));
        };

        Image(int h, int w, int c){                 // 构造函数
            rows = h;
            cols = w;
            channels = c;
            size = h * w * c;

            data = (float *)calloc(size, sizeof(float));        // calloc 在动态分配完内存后，自动初始化该内存空间为零
        };

        ~Image(){                                   // 析构函数
            free(data);                             // 释放空间
            data = NULL;
        };

        Image(const Image &other){                  // 拷贝构造函数
            this->rows = other.rows;
            this->cols = other.cols;
            this->size = other.size;
            this->channels = other.channels;

            this->data = (float *)calloc(other.size, sizeof(float));        // 重新申请一块内存
            memcpy(this->data, other.data, other.size * sizeof(float));     // 将数据拷贝过来
        };

        Image& operator=(const Image &other){       // 赋值函数
            if(&other != this){
                this->rows = other.rows;
                this->cols = other.cols;
                this->size = other.size;
                this->channels = other.channels;

                free(this->data);                           // 必须释放原有的内存，然后再重新申请一块内存

                this->data = (float *)calloc(other.size, sizeof(float));
                memcpy(this->data, other.data, other.size * sizeof(float));
            }
            return *this;
        };

        float &at(int y, int x, int z) const{               // 按照 [H, W, C] 顺序索引像素值
            assert(x < cols && y < rows && z < channels);
            return data[x + y*cols + z*rows*cols];
        };

        Image copy();

        Image gray();                              // 转灰度图函数
        Image resize(int w, int h);                // 图像的 resize 操作，最近邻插值
        Image resize(float scale);
};

Image imread(std::string filename, int channels);  // 读取图片
void imwrite(std::string filename, Image im);      // 写出图片

#endif


