# 第 1 章 CUDA 编程模型

[TOC]

在CUDA编程模型中，会包含多个CPU和GPU，每个GPU和CPU的内存都由一条PCI-Express总线隔开。为了清楚地指明不同的物理内存空间， 我们需要约定

- **主机: CPU及内存， 其变量以h_为前缀**

- **设备: GPU及内存， 其变量以d_为前缀**

现在，重要的是应学会如何为主机和设备分配内存以及在CPU和GPU之间拷贝共享数据。

###  内存管理
**CUDA编程模型假设系统是由一个主机(CPU)和设备(GPU)组成的， 而且各自拥有独立的内存，并且核函数实在设备上运行的**.为了使你拥有充分的控制权并
且使系统达到最佳性能，CUDA运行时负责分配与释放设备内存，并且在主机内存与设备内存之间传输数据。

下面，我们将通过一个简单的两个数组相加的例子来学习如何在主机和设备之间进行数据传输。
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
