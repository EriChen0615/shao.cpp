#include "helper_cuda.h"
#include <cuda_runtime.h>

__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

extern "C" int cuda_add(float *h_a, float *h_b, float *h_c, int n) {
    float *d_a, *d_b, *d_c;

    checkCudaErrors(cudaMalloc((void **)&d_a, n * sizeof(float)) );
    checkCudaErrors(cudaMalloc((void **)&d_b, n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_c, n * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate grid and block dimensions
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    getLastCudaError("In cuda_add: add_kernel failed");

    checkCudaErrors(cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
    return 0;
}