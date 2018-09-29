#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void){
    printf("threadIdx: (%d, %d, %d) "
            "blockIdx: (%d, %d, %d) "
            "blockDim: (%d, %d, %d) "
            "gridDim:  (%d, %d, %d)\n",
            threadIdx.x, threadIdx.y, threadIdx.z,
            blockIdx.x , blockIdx.y, blockIdx.z,
            blockDim.x , blockDim.y, blockDim.z,
            gridDim.x, gridDim.y, gridDim.z);
}

int main()
{
    // define total data element
    int nElem = 6; // 一共6个元素，每个元素由一个线程来控制

    // define grid and block structure
    dim3 block (3); // 每个网格3个线程
    dim3 grid ((nElem+block.x-1)/block.x);  // 2个网格
    /* dim3 grid(2); */

    // check grid and block dimension from host side // 查看主机端的块变量
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

    // check grid and block dimension from device side
    checkIndex <<<grid, block>>> ();
    // reset device before you leave
    cudaDeviceReset();

    return 0;
}





