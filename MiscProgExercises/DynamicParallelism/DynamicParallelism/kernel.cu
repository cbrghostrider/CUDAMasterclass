
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


#define mallocCheck(ptr) do { mallocAssert(ptr, __FILE__, __LINE__); } while(0)

void mallocAssert(void* ptr, const char* file, int line) {
    if (ptr == nullptr) {
        fprintf(stderr, "[%s: %d]: Malloc failed! Aborting!\n", file, line);
        exit(1);
    }
}

#define gpuErrCheck(val) do { gpuAssert(val, __FILE__, __LINE__); } while (0)

void gpuAssert(cudaError val, const char* file, int line) {
    if (val != cudaSuccess) {
        fprintf(stderr, "[%s: %d]: CUDA Error encountered: %s\n", file, line, cudaGetErrorString(val));
        exit(val);
    }
}

__global__ void dyn_parallel_v1(int size, int depth, int from_threadIdx, int from_blockIdx) {
    printf("dyn_parallel_v1: [ threadIdx.x: %d; blockIdx.x: %d; ] [ from_threadIdx.x: %d; from_blockIdx.x: %d; ] [gridDim.x: %d ; blockDim.x: %d] size = %d; depth = %d\n", 
            threadIdx.x, blockIdx.x, from_threadIdx, from_blockIdx, gridDim.x, blockDim.x, size, depth);

    if (size == 1) {
        return;
    }

    if (threadIdx.x == 0) {
        dyn_parallel_v1 << < 1, size / 2 >> > (size/2, depth+1, threadIdx.x, blockIdx.x);
    }
}

__global__ void dyn_parallel_v2(int size, int depth, int from_threadIdx, int from_blockIdx) {
    printf("dyn_parallel_v2: [ threadIdx.x: %d; blockIdx.x: %d; ] [ from_threadIdx.x: %d; from_blockIdx.x: %d; ] [gridDim.x: %d ; blockDim.x: %d] size = %d; depth = %d\n",
        threadIdx.x, blockIdx.x, from_threadIdx, from_blockIdx, gridDim.x, blockDim.x, size, depth);

    if (size == 1) {
        return;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dyn_parallel_v2 << < 2, size / 2 >> > (size / 2, depth + 1, threadIdx.x, blockIdx.x);
    }
}


int main() {

    dyn_parallel_v1 << <2, 8>> > (8, 0, -1, -1);
    gpuErrCheck(cudaDeviceSynchronize());

    printf("-----------------------------------------------------------------------------------------------------------------\n");

    dyn_parallel_v2 << <2, 8 >> > (8, 0, -1, -1);
    gpuErrCheck(cudaDeviceSynchronize());

    gpuErrCheck(cudaDeviceReset());

    return 0;
}


