/*================================================================
*   Copyright (C) 2020 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : example02.cpp
*   Author      : YunYang1994
*   Created date: 2020-04-18 01:19:49
*   Description :
*
*===============================================================*/

#include <string>
#include <iostream>
#include <stdlib.h>
#include "yynet.hpp"


int main(int argc, char **argv){
    if(argc != 3){
        std::cerr << "useage: " << argv[0] << " <image_path>" << " <channels>" << std::endl;
        exit(0);
    }

    Image im = imread(argv[1], atoi(argv[2]));
    Image jm = im.resize(416, 416);

    Image gm = im.gray();

    imwrite("color.png", im);
    imwrite("gray.png",  gm);

    return 0;
}
