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

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    const int TILE_SIZE = 16;

    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
            s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);

    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void launch_matmul_shared_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);

    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_tiled_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}