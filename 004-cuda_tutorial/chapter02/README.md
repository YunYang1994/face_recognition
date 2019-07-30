# 第 2 章 矩阵求和运算


目录
* [2.1. 内存管理](#21-内存管理)
* [2.2. 编写核函数](#22-编写核函数)
* [2.3. 矩阵求和运算](#23-矩阵求和运算)

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
这是一个纯C语言编写的程序，请见[`sumArraysOnHost.c`](https://github.com/YunYang1994/CodeFun/blob/master/004-cuda_tutorial/chapter02/sumArraysOnHost.c)，可以使用像下面这样用`nvcc`进行编译
```bash
$ nvcc -Xcompiler -std=c99 sumArraysOnHost.c -o sumArraysOnHost
$ ./sumArraysOnHost
 22.400000 24.799999 20.000000 14.200000
 22.400000 24.799999 20.000000 14.200000
 44.799999 49.599998 40.000000 28.400000
```
在上述命令行中，需要注意:

> `-Xcompiler`是指用于指定命令行选项是指向C编译器

> `-std=c99` 指的是将按照C99标准进行编译

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
关于在GPU上进行数组相加运算，详细代码请见[`sumArraysOnGPU.cu`](https://github.com/YunYang1994/cuda-tutorial/blob/master/chapter02/sumArraysOnGPU.cu)，现在使用以下命令来编译和执行
```bashrc
$ nvcc -arch=sm_20 sumArraysOnGPU.cu -o sumArraysOnGPU
$ ./sumArraysOnGPU
malloc memory on Host
initialize data on Host
 8.900000 23.000000 14.100000 12.200000
 8.900000 23.000000 14.100000 12.200000
malloc memory on GPU
copying inputs from Host to Device
copying output from Device to Host
Caculating On GPU
 17.799999 46.000000 28.200001 24.400000
```

### 2.2 编写核函数

不妨先来介绍下核函数、块和线程的概念。核函数是在设备端执行的代码，它描述的是在GPU上运行计算的任务，它的调用形式具体表现为
```bashrc
kernel_name <<<block, thread>>>(argument list);
```
`argument list`是形参，`block`是指网格维度，表示启动块的数目；`thread`表示的是块的维度，也就是每个块中线程的数目。每个线程的坐标表里以`blockIdx`和`threadIdx`来表示,因此我们可以得到总线程数量为`block*thread`。例如，在下图中有4096个线程块，因此网格维度gridDim=4096；每个块中有256个线程，因此块维度blockDim=256，因此一共有4096*256个线程。

<div align=center><img src="https://github.com/YunYang1994/CodeFun/blob/master/004-cuda_tutorial/image/block-thread.jpg" alt="logo" height="200"></div>

当核函数被调用时，许多不同的CUDA线程并行执行同一个计算任务，以下用`__global`声明定义核函数:
```bashrc
__global__ void kernel_name(argument list); // 核函数必须要有一个void返回类型
```
下表总结了 CUDA C 程序中函数类型的限定符。函数限定符将指定一个函数在主机上执行还是在设备上执行，以及可被主机调用还是被设备调用。

| 限定符  | 执行 | 调用 | 备注 |
| ---------- | -----------| ---------- | -----------|
| `__global__`   | 在设备端执行   | 可从主机、设备端调用 | 必须有一个void返回类型 |
| `__device__`   | 在设备端执行   | 仅能从设备端调用 |
| `__host__`  | 在主机端执行 | 仅能从主机端上调用 | 可以省略不写 |

考虑一个简单的例子，将两个大小为6为向量**A**和**B**相加为例。由于每个元素相加过程不存在相关性，现在使考虑使用两个块，每个块包含3个线程来计算该过程。因此来说，**每个线程的计算就是每个元素的相加过程**。在代码[`sumArraysOnGPU.cu`](https://github.com/YunYang1994/CodeFun/blob/master/004-cuda_tutorial/chapter02/sumArraysOnGPU.cu)的基础上，我们需要

#### 1. 定义块和线程
```cpp
dim3 block(2);
dim3 thread(3);
```
#### 2. 定义核函数
在这里，每个线程都将调用同一个核函数。因此可以考虑基于给定块索引和线程索引来计算全局数据访问的唯一索引:
```cpp
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx]; // 检查是否越界
    printf("%f + %f = %f Caculated On GPU: block %d thread %d\n", 
             A[idx], B[idx], C[idx], blockIdx.x, threadIdx.x);
}
```
#### 3. 执行和编译
完整代码见[`sumArraysOnGPU1.cu`](https://github.com/YunYang1994/CodeFun/tree/master/004-cuda_tutorial/chapter02/sumArraysOnGPU1.cu)，最终通过以下命令编译执行，得到
```bashrc
$  nvcc -arch=sm_20 sumArraysOnGPU1.cu -o sumArraysOnGPU1
$ ./sumArraysOnGPU1
向量 A: 20.400000 25.299999 1.000000 12.300000 17.700001 18.299999
向量 B: 20.400000 25.299999 1.000000 12.300000 17.700001 18.299999
向量 C 的每个元素计算过程:
20.400000 + 20.400000 = 40.799999 Caculated On GPU: block 0 thread 0
25.299999 + 25.299999 = 50.599998 Caculated On GPU: block 0 thread 1
1.000000 + 1.000000 = 2.000000 Caculated On GPU: block 0 thread 2
12.300000 + 12.300000 = 24.600000 Caculated On GPU: block 1 thread 0
17.700001 + 17.700001 = 35.400002 Caculated On GPU: block 1 thread 1
18.299999 + 18.299999 = 36.599998 Caculated On GPU: block 1 thread 2
```
### 2.3 矩阵求和运算

####  2.3.1 矩阵索引
在一个二维矩阵加法的核函数中，一个线程通常被分配一个数据元素来处理。首先要完成的任务是如何使用块和线程索引从全局内存中访问指定的数据。一般来说，其步骤如下

- **第一步，可以用以下公式把线程和块索引映射到矩阵坐标上，称为坐标索引:**
```bashrc
ix = threadIdx.x + blockIdx.x * blockDim.x
iy = threadIdx.y + blockIdx.y * blockDim.y

坐标索引: (ix, iy)
```

![image](../image/thread-index.png)


- **第二步，可以用以下公式把矩阵坐标映射到全局内存的索引/存储单元上，称为全局索引:**
```bashrc
idx = iy*nx + ix // nx 表示在x维度上元素个数, 对于(4,4)矩阵而言, nx=4

全局索引: idx
```

例如,对于维度为(4,4)的矩阵而言, 
```bashrc
                                0,  1,  2,  3,
                                4,  5,  6,  7,
                                8,  9,  10, 11,
                                12, 13, 14, 15,
                                
那么，对于元素'9' --> 坐标索引 (1,2)， 全局内存索引为 idx = 2*4 + 1 = 9
```
再比如，对于一个(6,8)维度的矩阵而言，假如划分为6个块，每个块有8个线程，那么就如下图所示

![image](../image/block-index.png)

从上图中也可以看出: `block`也有两个维度，分别是`blockIdx.x`和`blockIdx.y`，即`block`的索引坐标表示为(`blockIdx.x`,`blockIdx.y`)。类似地，在每个`block`中，线程`thread`也有两个维度。

####  2.3.2 求和运算
在本小节中，我们将对二维矩阵的求和运算作并行处理。由于矩阵是二维的，不妨考虑将使用一个二维网格和二维块来编写一个矩阵加法的核函数。

```cpp
__global__ void sumMatrixOnGPU2D(int *MatA, int *MatB, int *MatC, int nx, int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;    // 矩阵的存储都是通过一维数组的形式存储，因此我们需要计算全局索引位置
    
    if(ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}
```

然后我们通过以下命令编译和执行文件[`sumMatrixOnGPU-2D-grid-2D-block.cu`](https://github.com/YunYang1994/CodeFun/blob/master/004-cuda_tutorial/chapter02/sumMatrixOnGPU-2D-grid-2D-block.cu)
```bashrc
$ nvcc -arch=sm_20 sumMatrixOnGPU-2D-grid-2D-block.cu -o sumMatrixOnGPU-2D-grid-2D-block
$ ./sumMatrixOnGPU-2D-grid-2D-block

Matrix: (4.4)
  3  7  9  8
  7 10 12  9
 17 10 12 18
 12 22 17 21


Matrix: (4.4)
  0  7  4  9
  5 13 13 16
 10  9 12 14
 19 18 23 17

sumMatrixOnHost elapsed 0.000001 sec
sumMatrixOnGPU2D <<<(1, 1), (32, 32)>>> elapsed 0.000021 sec

Matrix: (4.4)
  3 14 13 17
 12 23 25 25
 27 19 24 32
 31 40 40 38

Arrays match!
```
当我们把矩阵的维度渐渐增大时，GPU和CPU的运行时间差异就很明显了

| 矩阵维度 | 4 x 4 | 16 x 16 | 256 x 256 | 512 x 512 | 1024 x 1024 | 2048 x 2048 | 8192 x 8192 |
| ---------- | -----------| ---------- | -----------| ---------- | -----------| ---------- | -----------|
| `CPU`   | 0.000001 s  | 0.000004 s | 0.000206 s | 0.000797 s | 0.003192 s | 0.012736 s | 0.051285 s |
| `GPU`   | 0.000021 s  | 0.000021 s | 0.000113 s | 0.000449 s | 0.000923 s | 0.001989 s | 0.006241 s |

上一章: [**了解CUDA编程**](https://github.com/YunYang1994/CodeFun/blob/master/004-cuda_tutorial/chapter01/README.md)<br>
下一章: [**矩阵相乘运算**](https://github.com/YunYang1994/CodeFun/blob/master/004-cuda_tutorial/chapter03/README.md)






