其实我一开始听说要用`stb_image`来读写图片的时候是拒绝的，因为不是已经有`OpenCV`了吗？后来才知道，它完全不需要安装任何依赖项！只要你有`stb_image.h`和`stb_image_write.h`这两个头文件，你就可以随时随地读写图片!

## stb_image
官方链接：[https://github.com/nothings/stb](https://github.com/nothings/stb)

|library|description|
|---|:---:|
|[stb_image.h](https://github.com/nothings/stb/blob/master/stb_image.h)|image loading/decoding from file/memory: JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC|
|[stb_image_write.h](https://github.com/nothings/stb/blob/master/stb_image_write.h)|image writing to disk: PNG, TGA, BMP|

## 图像基础
如下图所示：图片有RGB三个颜色通道，图片的宽和高分别用`x`与`y`表示，通道深度用`k`表示，所以这些图片的像素是一个三维矩阵，但其实在c语言中是作为一维矩阵存储的，即将三维矩阵在`x`轴上展平，变成一维数组，方便索引。

![image](https://user-images.githubusercontent.com/30433053/62187369-c18c2200-b39b-11e9-8bdc-35378a9174d7.png)

```c
for(k = 0; k < c; ++k){                 // 在每个颜色通道上索引；
    for(y = 0; y < h; ++y){             // 在图片高度上索引；
        for(x = 0; x < w; ++x){         // 在图片宽度上索引；
	    int idx = x + w*y + w*h*k;  // 打印每个像素值；
	    printf("the pixel in the coordinates of (%d,%d,%d) is %d\n", x,y,k,data[idx]; 
	}
    }
}
```
![image](https://user-images.githubusercontent.com/30433053/62187392-cea91100-b39b-11e9-89b0-7c183d923269.png)



## 代码思路
首先创建一个`image`的结构体，该结构体的成员分别包含图片的长、宽、颜色通道和`float`类型的指针【用于访问每个像素】，接着根据图片的像素数目申请一段连续的内存，最后利用`save_image_stb`函数将每个像素值rescal到0～1范围内并塞进该内存空间里。
![image](https://user-images.githubusercontent.com/30433053/62187338-a8837100-b39b-11e9-8eca-9d8f389f7c46.png)
## 代码运行
```bashrc
$ make                         // 编译代码
$ ./yyimg ../data/dog.jpg      // C++ 接口
$ python test.py               // python 接口
```




