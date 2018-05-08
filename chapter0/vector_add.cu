#include "vector_add.h"

__global__ void add(int *a, int *b, int *c){
    int tid = blockIdx.x;
    if(tid<N){
        c[tid] = a[tid] +b[tid];
    }
}

void kernel_wrapper(int *a, int *b, int *c){
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**) &dev_a, N*sizeof(int));
    cudaMalloc((void**) &dev_b, N*sizeof(int));
    cudaMalloc((void**) &dev_c, N*sizeof(int));

    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice);

    add<<<N,1>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);    
    cudaFree(dev_b);    
    cudaFree(dev_c);   
}