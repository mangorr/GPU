#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// ssh ym2360@access.cims.nyu.edu
// ssh cuda4
// scp factorize.cu ym2360@access.cims.nyu.edu:/home/ym2360/gpu/lab3 
// nvcc -o ff factorize.cu -lm
// time ./ff 10000

__global__ void calculatePrime (unsigned int *, unsigned int);

int main (int argc, char ** argv) {
    if (argc != 2) {
		printf("Wrong number of arguments. Exit!\n");
		exit(1);
	}

    // clock_t start, end;
    // start = clock();

    unsigned int N = atoi(argv[1]);
    size_t size = sizeof(unsigned int) * (N - 1); // Ignore 1

    unsigned int * cprimes = (unsigned int*) malloc(size);
    unsigned int * gPrimes;

    cudaMalloc(&gPrimes, size);
    cudaMemset(gPrimes, 0, size);
    
    // end = clock();
    // printf("Time taken by the memory part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);


    // start = clock();

    dim3 grid(100); // Can be changed
    dim3 block(1000);

    calculatePrime<<<grid, block>>>(gPrimes, N);

    cudaMemcpy(cprimes, gPrimes, size, cudaMemcpyDeviceToHost); // Get the result of GPU

    // end = clock();
    // printf("Time taken by the GPU part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    // start = clock();

    unsigned int i = 0;
    while (i < N - 1 && N != 1) {
        if (cprimes[i] == 0 && N % (i + 2) == 0) {
            printf("%u ", i + 2);
            N /= (i + 2);
            i = 0;
        } else {
            ++i;
        }
    }
    printf("\n");

    cudaFree(gPrimes);
    free(cprimes);

    // end = clock();
    // printf("Time taken by the later CPU part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    return 0;
}

// To label each number as prime number or not
__global__ void calculatePrime (unsigned int * gPrimes, unsigned int N) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x + 2;
    
    unsigned int index = 2 * threadId;

    while (index <= N) {
        gPrimes[index - 2] = 1;
        index += threadId;
    }
}