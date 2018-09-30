#include "../common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void initialInt(int *ip, int size){
    for (int i=0; i<size; i++){
        ip[i] = i + rand()%10;
    }
}

void printMatrix(int *C, const int nx, const int ny){
    int *ic = C;
    printf("\nMatrix: (%d.%d)\n", nx, ny);
    for (int iy=0; iy<ny; iy++){
        for (int ix=0; ix<nx; ix++){
            printf("%3d", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void sumMatrixOnGPU2D(int *MatA, int *MatB, int *MatC, int nx, int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    
    if(ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

void sumMatrixOnHost(int *MatA, int *MatB, int *MatC, int nx, int ny){
    int *ia = MatA;
    int *ib = MatB;
    int *ic = MatC;

    for (int iy=0; iy<ny; iy++){
        for (int ix=0; ix<nx; ix++){
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; ib += nx; ic += nx;
    }
}

bool checkResult(int *MatC, int *h_C, int nxy){
    for (int i=0; i<nxy; i++){
        if (MatC[i] != h_C[i]){
            printf("Matc[%d]: %d != h_C[%d]: %d\n", i, MatC[i], i, h_C[i]);
            return false;
        }
    }
    return true;
}


int main(){

    /* int nx = 1<<13; */
    /* int ny = 1<<13; */
    int nx = 10240;
    int ny = 1024;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    int *h_A, *h_B, *h_C, *h_MatC;
    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    h_C = (int *)malloc(nBytes);
    h_MatC = (int *)malloc(nBytes);

    initialInt(h_A, nxy);
    /* printMatrix(h_A, nx, ny); */
    initialInt(h_B, nxy);
    /* printMatrix(h_B, nx, ny); */

    memset(h_C, 0, nBytes);
    memset(h_MatC, 0, nBytes);

    double iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, h_C, nx, ny);
    double iElaps = cpuSecond() - iStart;
    /* printMatrix(h_C, nx, ny); */
    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);
    

    int *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);
    
    // transfer data from host to Device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    iStart = cpuSecond();
    sumMatrixOnGPU2D <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d, %d), (%d, %d)>>> elapsed %f sec\n",
            grid.x, grid.y, block.x, block.y, iElaps);

    cudaMemcpy(h_MatC, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    if (checkResult(h_MatC, h_C, nxy)){
        printf("Arrays match!\n");
    }
    else{
        printf("Arrays don't match!\n");
    }
    
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_MatA); cudaFree(d_MatB); cudaFree(d_MatC);

    cudaDeviceReset();
    return 0;
}

