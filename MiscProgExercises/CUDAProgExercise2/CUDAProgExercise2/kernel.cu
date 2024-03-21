
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


__global__ void cuda_kernel(int* input) {
    const int block_size = blockDim.x * blockDim.y * blockDim.z;

    // Derive the unique tid within a block.
    int b_layer_offset = blockDim.x * blockDim.y * threadIdx.z;
    int b_row_offset = blockDim.x * threadIdx.y;
    int tid = threadIdx.x + b_row_offset + b_layer_offset;

    // Derive the starting location of the block.
    int layer_offset = (gridDim.x * gridDim.y * blockIdx.z) * block_size;
    int row_offset = (gridDim.x * blockIdx.y) * block_size;
    int offset = blockIdx.x * block_size;
    int start_block = offset + row_offset + layer_offset;

    // Derive the unique gid.
    int gid = start_block + tid;

    printf("Thread: [%d, %d, %d]; \tBlock: [%d, %d, %d]; \ttid=%d; \tgid=%d; \tvalue=%d\n", 
            threadIdx.x, threadIdx.y, threadIdx.z,
            blockIdx.x, blockIdx.y, blockIdx.z,
            tid, gid, input[gid]);
}

int main() {
    int nx = 4, ny = 4, nz = 4;
    dim3 block(2, 2, 2);
    dim3 grid(nx/block.x, ny/block.y, nz/block.z);

    const int num_elems = 64;
    const int num_bytes = sizeof(int) * num_elems;

    // Allocate the host memory and initialize.
    int* h_input = (int*)malloc(num_bytes);
    time_t t;
    srand((unsigned)time(&t));
    printf("Input Array = {");
    for (int i = 0; i < num_elems; i++) {
        h_input[i] = (int)(rand() & 0xff);
        printf("%d, ", h_input[i]);
    }
    printf("\b\b}\n");

    // Copy to the device memory.
    int* d_input = NULL;
    cudaMalloc((void**)&d_input, num_bytes);
    cudaMemcpy(d_input, h_input, num_bytes, cudaMemcpyHostToDevice);

    // Run the CUDA kernel.
    cuda_kernel<<<grid, block>>>(d_input);
    cudaDeviceSynchronize();

    // Free the host and device memories.
    cudaFree(d_input);
    free(h_input);

    cudaDeviceReset();
    return 0;
}

