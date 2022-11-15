// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda.h>
#include <curand.h>

// includes, project
#include <helper_cuda.h> // helper functions for cuda

// CUDA global constants
__constant__ int a, b, c, thread_i;

// Kernel routine
__global__ void square_avg(float *d_i, float *d_o)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int first_index = index * thread_i;
    float result = 0;
    for (int iter = first_index; iter < first_index + thread_i; iter++)
    {
        result += a * d_i[iter] * d_i[iter] + b * d_i[iter] + c;
    }
    d_o[index] = result / thread_i;
}

int main(int argc, const char **argv)
{
    const int h_a = 3.0;
    const int h_b = 5.0;
    const int h_c = 4.0;

    const int h_io = 640000;
    const int h_co = 6400;
    const int h_thread_i = h_io / h_co;

    // Allocate memory on host and device
    checkCudaErrors(cudaMemcpyToSymbol(a, &h_a, sizeof(h_a)));
    checkCudaErrors(cudaMemcpyToSymbol(b, &h_b, sizeof(h_b)));
    checkCudaErrors(cudaMemcpyToSymbol(c, &h_c, sizeof(h_c)));
    checkCudaErrors(cudaMemcpyToSymbol(thread_i, &h_thread_i, sizeof(h_thread_i)));

    float *d_in, *d_out, *h_out;
    checkCudaErrors(cudaMalloc((void **)&d_in, sizeof(float) * h_io));
    checkCudaErrors(cudaMalloc((void **)&d_out, sizeof(float) * h_co));
    h_out = (float *)malloc(sizeof(float) * h_co);

    // Random number generation
    curandGenerator_t random_generator;
    checkCudaErrors(curandCreateGenerator(&random_generator, CURAND_RNG_PSEUDO_DEFAULT));
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(random_generator, 1234ULL));
    checkCudaErrors(curandGenerateNormal(random_generator, d_in, h_io, 0.0f, 1.0f));

    square_avg<<<200, 32>>>(d_in, d_out);

    checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(float) * h_co, cudaMemcpyDeviceToHost));

    float out_sum = 0;
    for (int i = 0; i < h_co; i++)
    {
        out_sum += h_out[i];
    }

    printf("Mean:%f\n\n", out_sum / static_cast<float>(h_co));

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
    free(h_out);
}