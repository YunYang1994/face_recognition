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
#include <stdlib.h>
#include "yynet.hpp"


int main(int argc, char **argv){
    if(argc != 3){
        std::cerr << "useage: " << argv[0] << " <image_path>" << " <channels>" << std::endl;
        exit(0);
    }

    Image im = imread(argv[1], atoi(argv[2]));
    imwrite("result.png", im);

    return 0;
}
