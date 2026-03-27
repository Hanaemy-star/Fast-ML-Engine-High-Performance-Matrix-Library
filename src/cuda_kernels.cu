#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void matmul_kernel(const double* A, const double* B, double* C, int M, int K, int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_matmul_kernel(const double* A, const double* B, double* C, int M, int K, int N) {
    double *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)K * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, A, (size_t)M * K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, (size_t)K * N * sizeof(double), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);

    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}