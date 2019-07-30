
# 第 1 章 了解CUDA编程

从计算的角度来看，并行计算可以被定义为同时使用多个计算资源来执行并发计算， 大的问题可以被分解成很多小问题，然后在不同计算资源上分别并行处理这些小问题。
CUDA是一种通用的异构并行计算平台和编程模型，你可以利用CUDA平台像在CPU上那样使用GPU来进行计算。

[TOC]

**编译环境**：本代码将使用`nvcc`编译器来编译，你可以使用以下命令来检查CUDA是否正确安装:

```bash
$ which nvcc
/usr/local/cuda-8.0/bin/nvcc  # cuda-8.0 版本
```

###  用GPU输出 Hello World
不妨先写一个cuda C程序，命名为`helloFromGPU`，用它来输出字符串 “Hello World from GPU！” 
```cpp
========================== 代码清单 1-1 Hello World from GPU (hello.cu) ==========================
// hello.cu
#include <stdio.h>

__global__ void helloFromGPU (void) 
{
    printf("Hello World from GPU!\n");
}

int main(void)
{
    // hello from cpu
    printf("Hello World from CPU!\n");

    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}
```
在linux终端下使用以下命令进行编译[`hello.cu`](https://github.com/YunYang1994/CodeFun/blob/master/004-cuda_tutorial/chapter01/hello.cu)，然后执行程序得到
```bash
$ nvcc -arch sm_20 hello.cu -o hello
$ ./hello
Hello World from CPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
```
在上面的代码中，`cudaDeviceReset`表示重置当前线程所关联过的当前设备的所有资源；修饰符`__global__`告诉编译器这是一个内核函数，它将从CPU中调用，然后在GPU上执行，在CPU上通过下面的代码启动内核函数

```bash
helloFromGPU <<<1, 10>>>();
```

> 三重尖号意味着从主线程到端代码的调用。1和10分别表示有1个块区域和10个线程，后续会作相关介绍。


###  CUDA 编程结构

一个典型的 CUDA 编程结构应该包括下面5个主要的步骤：

- **1. 分配GPU内存**

- **2. 从CPU内存中拷贝数据到GPU内存中去**

- **3. 调用CUDA 内核函数来完成程序指定的运算**

- **4. 将数据从GPU中 拷回CPU内存**

- **5. 释放GPU 内存空间**

在上述代码中， 你只看到了第三步: 调用内核。

下一章: [**矩阵求和运算**](https://github.com/YunYang1994/CodeFun/tree/master/004-cuda_tutorial/chapter02)

