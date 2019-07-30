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

#include "image.h"

int main(int argc, char **argv)
{
    if(argc < 2){
        printf("%s\n", "No image input!");
    }
    else{
        image im = load_image(argv[1]);
        save_image(im, "result");
        free_image(im);
    }

    return 0;

}
