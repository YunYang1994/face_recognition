#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N){

    printf("Caculating On GPU\n");
    for (int idx=0; idx<N; idx++){
        C[idx] = A[idx] + B[idx];
    }
}


void initialData(float *ip, int size){
    // generate different seed for random number
    time_t t;
    srand((unsigned int) time(&t));

    for (int i=0; i<size; i++){
        ip[i] = (float)(rand() & 0xFF)/10.0f;
    }
}


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

    printf("malloc memory on Host\n");
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    printf("initialize data on Host\n");
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    print(h_A, nElem);
    print(h_B, nElem);

    printf("malloc memory on GPU\n");
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);
    
    printf("copying inputs from Host to Device\n");
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    sumArraysOnGPU <<<1, 1>>>(d_A, d_B, d_C, nElem); // 异步计算
    printf("copying output from Device to Host\n");
    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);
    print(h_C, nElem);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}



