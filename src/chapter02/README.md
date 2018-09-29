# 第 2 章 CUDA 编程模型


目录
* [2.1. 内存管理](#21-内存管理)

[TOC]

在CUDA编程模型中，会包含多个CPU和GPU，每个GPU和CPU的内存都由一条PCI-Express总线隔开。为了清楚地指明不同的物理内存空间， 我们需要约定

- **主机: CPU及内存， 其变量以h_为前缀**

- **设备: GPU及内存， 其变量以d_为前缀**

现在，重要的是应学会如何为主机和设备分配内存以及在CPU和GPU之间拷贝共享数据。

### 2.1 内存管理
**CUDA编程模型假设系统是由一个主机(CPU)和设备(GPU)组成的， 而且各自拥有独立的内存，并且核函数实在设备上运行的**.为了使你拥有充分的控制权并且使系统达到最佳性能，CUDA运行时负责分配与释放设备内存，并且在主机内存与设备内存之间传输数据。

下面，我们将通过一个简单的两个数组相加的例子来学习如何在主机和设备之间进行数据传输。
####  2.1.1 在CPU上运算

```cpp
========================== 代码清单 2-1 sumArraysOnHost.c ==========================
// sumArraysOnHost.c
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
// 在主机上执行相加运算
void sumArraysOnHost(float *A, float *B, float *C, const int N){
    for (int idx=0; idx<N; idx++){
        C[idx] = A[idx] + B[idx];
    }
}
// 初始化数据值
void initialData(float *ip, int size){
    // generate different seed for random number
    time_t t;
    srand((unsigned int) time(&t));

    for (int i=0; i<size; i++){
        ip[i] = (float)(rand() & 0xFF)/10.0f;
    }
}
// 打印数组
void print(float *array, const int N){
    for (int idx=0; idx<N; idx++){
        printf(" %f", array[idx]);
    }
    printf("\n");
}

int main(){
    int nElem = 4;
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);
    print(h_A, nElem);
    print(h_B, nElem);

    sumArraysOnHost(h_A, h_B, h_C, nElem);
    print(h_C, nElem);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
```
这是一个纯C语言编写的程序，请见[`sumArraysOnHost.c`](https://github.com/YunYang1994/cuda-tutorial/blob/master/src/chapter02/sumArraysOnHost.c)，可以使用像下面这样用`nvcc`进行编译
```bash
$ nvcc -Xcompiler -std=c99 sumArraysOnHost.c -o sumCPU
$ ./sumCPU
 18.900000 22.400000 19.600000 22.700001
 18.900000 22.400000 19.600000 22.700001
 37.799999 44.799999 39.200001 45.400002
```
在上述命令行中，需要注意:

> `-Xcompiler`是指用于指定命令行选项是指向C编译器

> `-std=99` 指的是将按照C99标准进行编译

####  2.1.2 在GPU上运算
现在， 你可以在GPU上修改代码来进行数组加法运算， 用`cudaMalloc`在GPU上申请内存。
```bashrc
float *d_A, *d_B, *d_C;             //先定义数值指针，用来存放地址
cudaMalloc((float**)&d_A, nBytes);  //申请n个字节的内存后，返回地址
cudaMalloc((float**)&d_B, nBytes);
cudaMalloc((float**)&d_C, nBytes);
```
然后使用`cudaMemcpy`函数把数据从**主机内存拷贝到GPU的全局内存中**，参数`cudaMemcpyHostToDevice`指定了数据的拷贝方向。
```bashrc
cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
```
当数据被转移到**GPU的全局内存后**，主机段调用核函数在GPU上进行数组求和运算。**一旦内核被调用，控制权立刻被传回主机，这样的话，内核与
主机的是异步进行。**
```bashrc
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N){...}
```
当内核在GPU上完成了对所有数组元素的处理后，其结果将通过```cudaMemcpy```函数复制回到CPU内存中去。
```bashrc
cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost); # cudaMemcpyDeviceToHost, GPU-->CPU
```
最后，一定别忘了调用`cudaFree`函数来释放GPU的内存。
```bashrc
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
```
关于在GPU上进行数组相加运算，详细代码请见[`sumArraysOnGPU.cu`](https://github.com/YunYang1994/cuda-tutorial/blob/master/src/chapter02/sumArraysOnGPU.cu)，现在使用以下命令来编译和执行
```bashrc
$ nvcc -arch sm_20 sumArraysOnGPU.cu -o sumGPU
$ ./sumGPU
malloc memory on Host
initialize data on Host
 3.900000 19.000000 18.700001 7.900000
 3.900000 19.000000 18.700001 7.900000
malloc memory on GPU
copying inputs from Host to Device
copying output from Device to Host
Caculating On GPU
 7.800000 38.000000 37.400002 15.800000
```

### 2.1 编写核函数

核函数是在设备端执行的代码，它的调用形式表现为
```bashrc
kernel_name <<<block, thread>>>(argument list);
```
`block`是指网格维度，表示启动块的数目；`thread`表示的是块的维度，也就是每个块中线程的数目。每个线程的坐标表里以`blockIdx`和`threadIdx`来表示,因此我们可以得到总线程数量为`block*thread`。例如，在下图中有4096个线程块，因此网格维度gridDim=4096；每个块中有256个线程，因此块维度blockDim=256，因此一共有4096*256个线程。
<img src="image/cuda-memory-structure.jpg" alt="logo" height="200" align="center" />













