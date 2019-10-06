其实我一开始听说要用stb_image来读写图片的时候是拒绝的，因为不是已经有OpenCV了吗？后来才知道，它完全不需要安装任何依赖项！只要你有stb_image.h和stb_image_write.h这两个头文件，你就可以随时随地读写图片!

## stb_image
--------------------
官方链接：[https://github.com/nothings/stb](https://github.com/nothings/stb)

| library | description |
|---|:---:|
|[stb_image.h](https://github.com/nothings/stb/blob/master/stb_image.h)|image loading/decoding from file/memory: JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC|
|[stb_image_write.h](https://github.com/nothings/stb/blob/master/stb_image_write.h)|image writing to disk: PNG, TGA, BMP|

## useage

```c
int w, h, c;
unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
```
上述的 data 是一个指针，它指向类型为 unsigned char 的图片像素值。图片有RGB三个颜色通道，其像素数据是一个三维矩阵 (RGB)，但其实在 data 里是作为一维矩阵存储的，即将三维矩阵在x轴上展平，变成一维数组，方便索引。

![image](https://user-images.githubusercontent.com/30433053/62187369-c18c2200-b39b-11e9-8bdc-35378a9174d7.png)

![image](https://user-images.githubusercontent.com/30433053/62187392-cea91100-b39b-11e9-89b0-7c183d923269.png)

例如，在 image.c 中针对图片里每一个像素做 rescale 操作的话是这个样子：

```c
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index]/255.; /*将每个像素scale到0～1之间*/
            }
        }
    }
```

