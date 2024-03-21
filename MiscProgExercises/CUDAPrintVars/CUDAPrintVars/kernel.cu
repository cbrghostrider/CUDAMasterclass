
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

// CUDA Kernel for the first programming exercise in the course.
// Programming Exercise 1.
__global__ void printKernel() {
    printf("threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; \t blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d; \t gridDim.x=%d, gridDim.y=%d, gridDim.z=%d\n",
        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z);
}

int main() {
    int nx = 4, ny = 4, nz = 4;
    dim3 block(2, 2, 2);
    dim3 grid(nx/block.x, ny/block.y, nz/block.z);

    printKernel << <grid, block >> > ();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    
    return 0;
}

