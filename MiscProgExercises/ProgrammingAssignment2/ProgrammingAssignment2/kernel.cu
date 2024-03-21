
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_ELEMS (4194304)
#define NUM_BYTES (sizeof(int) * NUM_ELEMS)

#define mallocCheck(ptr) do { mallocAssert(ptr, __FILE__, __LINE__); } while(0)

void mallocAssert(void* ptr, const char *file, int line) {
  if (ptr == nullptr) {
    fprintf(stderr, "[%s: %d]: Malloc failed! Aborting!\n", file, line);
    exit(1);
  }
}

#define gpuErrCheck(val) do { gpuAssert(val, __FILE__, __LINE__); } while (0)

void gpuAssert(cudaError val, const char *file, int line) {
  if (val != cudaSuccess) {
    fprintf(stderr, "[%s: %d]: CUDA Error encountered: %s\n", file, line, cudaGetErrorString(val));
    exit(val);
  }
}

void rand_init_array(int* input, int size) {
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++) {
		input[i] = (int)(rand() & 0xff);
	}
}

bool validate(int* cpu_results, int* gpu_results, int size) {
	for (int i = 0; i < size; i++) {
		if (cpu_results[i] != gpu_results[i]) {
			fprintf(stderr, "Mismatch on element %d: CPU=%d, GPU=%d\n", i, cpu_results[i], gpu_results[i]);
			return false;
		}
	}
	return true;
}

double time_delta(clock_t start, clock_t end) {
	return ((double)(end - start)/(double)(CLOCKS_PER_SEC));
}

void sum_arr3_cpu(int* result, int* a, int* b, int* c, int size) {
	for (int i = 0; i < size; i++) {
		result[i] = a[i] + b[i] + c[i];
	}
}

__global__ void sum_arr3_gpu(int* result, int* a, int* b, int* c, int size) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid < size) {
		result[gid] = a[gid] + b[gid] + c[gid];
	}
}

void run_block(int block_size) {
	printf("Running for Block Size: %d, and NUM_ELEMS: %d\n", block_size, NUM_ELEMS);

	dim3 block(block_size);
	dim3 grid(NUM_ELEMS / block_size);

	// Allocate and initialize host arrays.
	int* a, * b, * c, * cpu_results, * gpu_results;
	a = (int*)malloc(NUM_BYTES); mallocCheck(a);
	b = (int*)malloc(NUM_BYTES); mallocCheck(a);
	c = (int*)malloc(NUM_BYTES); mallocCheck(a);
	cpu_results = (int*)malloc(NUM_BYTES); mallocCheck(cpu_results);
	gpu_results = (int*)malloc(NUM_BYTES); mallocCheck(gpu_results);

	rand_init_array(a, NUM_ELEMS);
	rand_init_array(b, NUM_ELEMS);
	rand_init_array(c, NUM_ELEMS);

	// Compute results for verification on the CPU, and clock it.
	clock_t cpu_start, cpu_end;
	cpu_start = clock();
	sum_arr3_cpu(cpu_results, a, b, c, NUM_ELEMS);
	cpu_end = clock();

	// Allocate and initialize the device memory.
	int* d_a, * d_b, * d_c, * d_gpu_results;
	gpuErrCheck(cudaMalloc((void**)&d_a, NUM_BYTES));
	gpuErrCheck(cudaMalloc((void**)&d_b, NUM_BYTES));
	gpuErrCheck(cudaMalloc((void**)&d_c, NUM_BYTES));
	gpuErrCheck(cudaMalloc((void**)&d_gpu_results, NUM_BYTES));

	clock_t htod_start, htod_end;
	htod_start = clock();
	gpuErrCheck(cudaMemcpy(d_a, a, NUM_BYTES, cudaMemcpyHostToDevice));
	gpuErrCheck(cudaMemcpy(d_b, b, NUM_BYTES, cudaMemcpyHostToDevice));
	gpuErrCheck(cudaMemcpy(d_c, c, NUM_BYTES, cudaMemcpyHostToDevice));
	htod_end = clock();

	// Run the cuda kernel.
	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	sum_arr3_gpu << <grid, block >> > (d_gpu_results, d_a, d_b, d_c, NUM_ELEMS);
	cudaDeviceSynchronize();
	gpu_end = clock();

	// Copy the results to host memory.
	clock_t dtoh_start, dtoh_end;
	dtoh_start = clock();
	gpuErrCheck(cudaMemcpy(gpu_results, d_gpu_results, NUM_BYTES, cudaMemcpyDeviceToHost));
	dtoh_end = clock();

	// Validate that the answers are the same.
	if (!validate(cpu_results, gpu_results, NUM_ELEMS)) {
		fprintf(stderr, "CPU and GPU results do not match... aborting!\n");
		exit(2);
	}

	// Show the performance comparisons:	
	printf("GPU results successfully validated against CPU results!\n");
	printf("GPU vs CPU comparison is as follows:\n");
	printf("CPU computation time: %10.6f\n", time_delta(cpu_start, cpu_end));
	printf("GPU computation time: %10.6f\n", time_delta(gpu_start, gpu_end));
	printf("Host to Device mem transfer time: %10.6f\n", time_delta(htod_start, htod_end));
	printf("Device to Host mem transfer time: %10.6f\n", time_delta(dtoh_start, dtoh_end));
	printf("\n--------------------------------------------------------------------------------------------\n\n");

	// Free all the state.
	gpuErrCheck(cudaFree(d_a));
	gpuErrCheck(cudaFree(d_b));
	gpuErrCheck(cudaFree(d_c));
	gpuErrCheck(cudaFree(d_gpu_results));

	free(a);
	free(b);
	free(c);
	free(cpu_results);
	free(gpu_results);

}

int main() {
	
	run_block(64); 
	run_block(128);
	run_block(256);
	run_block(512);

	cudaDeviceReset();
	return 0;
}



