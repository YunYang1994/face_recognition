/*================================================================
*   Copyright (C) 2020 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : image.cpp
*   Author      : YunYang1994
*   Created date: 2020-04-11 18:15:20
*   Description :
*
*===============================================================*/

#include <stdio.h>
#include <cmath>
#include <string>
#include "image.hpp"
#include "stb_image.h"


Image Image::gray(){                                  // 彩色图转灰度图，三个颜色通道求平均即可
    if(channels == 1) return *this;
    Image im(rows, cols, 1);
    for(int i=0; i<rows; i++)
        for(int j=0; j<cols; j++)
            im.at(i,j,0) = (*this).at(i, j, 0) / 3.f + (*this).at(i, j, 1) / 3.f + (*this).at(i, j, 2) / 3.f;
    return im;
}


Image Image::resize(int w, int h){                    // 最近邻插值函数
    assert(w>0 & h>0);

    float scale_h = (float)(rows-1) / (h-1);
    float scale_w = (float)(cols-1) / (w-1);

    Image im(h, w, channels);

    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            for(int k=0; k<channels; k++){
                int ori = round(i * scale_h);
                int orj = round(j * scale_w);
                im.at(i, j, k) = (*this).at(ori, orj, k);
            }
        }
    }
    return im;
}


Image Image::resize(float factor){
    int w = cols * factor;
    int h = rows * factor;
    Image im = this->resize(w, h);
    return im;
}


Image Image::copy(){
    Image im(rows, cols, channels);
    memcpy(im.data, data, im.size * sizeof(float));
    return im;
}

                                                                   // 利用 stb_image.h 和 stb_image_write.h 来读写图片

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image imread(std::string filename, int channels){
    int w, h, c;

    unsigned char *data = stbi_load(filename.c_str(), &w, &h, &c, channels);
    if(!data){
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename.c_str(), stbi_failure_reason());
        exit(0);
    }

    int i, j, k;
    Image im(h, w, c);

    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index];
            }
        }
    }

    free(data);
    data = NULL;

    return im;
}

void imwrite(std::string filename, Image im){
    std::string f = filename.substr(filename.length() - 3, filename.length());
    unsigned char *data = (unsigned char *)calloc(im.size, sizeof(char));

    int i,k;
    for(k = 0; k < im.channels; ++k){
        for(i = 0; i < im.cols*im.rows; ++i){
            data[i*im.channels+k] = (unsigned char) im.data[i + k*im.cols*im.rows];
        }
    }

    int success = 0;
    if(f == "png")       success = stbi_write_png(filename.c_str(), im.cols, im.rows, im.channels, data, im.cols*im.channels);
    else if (f == "jpg") success = stbi_write_jpg(filename.c_str(), im.cols, im.rows, im.channels, data, 80);
    else if (f == "bmp") success = stbi_write_bmp(filename.c_str(), im.cols, im.rows, im.channels, data);
    else if (f == "tga") success = stbi_write_tga(filename.c_str(), im.cols, im.rows, im.channels, data);

    free(data);
    if(!success) std::cerr << "Failed to write image " << filename.c_str() << std::endl;

}

