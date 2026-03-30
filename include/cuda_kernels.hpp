#pragma once

void launch_matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N);

void launch_matmul_shared_kernel(const float* A, const float* B, float* C, int M, int K, int N);