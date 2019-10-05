#include <stdio.h>

// __global__ 告诉编译器这个函数将会从cpu中调用，然后在gpu上运行
__global__ void helloFromGPU (void) 
{
    printf("Hello World from GPU!\n");
}

int main(void)
{
    // hello from cpu
    printf("Hello World from CPU!\n");

    helloFromGPU <<<1, 10>>>();
    // <<<1, 10>> 表示从主线程到设备端代码调用10个线程
    cudaDeviceReset();
    return 0;
}
