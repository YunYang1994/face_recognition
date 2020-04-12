/*================================================================
*   Copyright (C) 2020 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : example01.cpp
*   Author      : YunYang1994
*   Created date: 2020-04-11 18:21:49
*   Description :
*
*===============================================================*/

#include <string>
#include <iostream>
#include "yynet.hpp"

void print_image(Image im)
{
    int i, j, k;
    for(i =0 ; i < im.channels; ++i){
        for(j =0 ; j < im.rows; ++j){
            for(k = 0; k < im.cols; ++k){
                printf("%.2lf, ", im.data[i*im.rows*im.cols + j*im.cols + k]);
                if(k > 30) break;
            }
            printf("\n");
            if(j > 30) break;
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv){
    if(argc != 2){
        std::cerr << "useage: " << argv[0] << " <image_path>" << std::endl;
        exit(0);
    }

    std::string image_path = argv[1];
    Image im = imread(argv[1], 1);
    imwrite("test.png", im);

    Image km = im.copy();

    im.at(0,1,0) = 255;
    Image jm(im);

    print_image(jm);
    print_image(km);

    return 0;
}
