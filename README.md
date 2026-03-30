# Fast-ML-Engine: High-Performance Matrix Operations Library

A C++/CUDA library demonstrating deep optimization of linear algebra kernels. Developed as a high-performance backend for machine learning workloads, achieving a **150x+ speedup** over naive implementations.

##  Performance Evolution (FP32 Precision)

###  CPU Benchmarks (1,000 x 1,000 Matrix)

| Stage | Optimization Strategy | Time (sec) | GFLOPS | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Baseline** (Naive C++) | 310.0s | 0.006 | 1x |
| 2 | **Cache-Friendly** (IKJ Order) | 2.5s | 0.800 | 124x |
| 3 | **Vectorized** (AVX2 + FMA) | 1.3s | 1.538 | 238x |
| 4 | **Parallel** (OpenMP + SIMD) | 0.3s | 6.667 | 1,033x |

---

###  GPU Benchmarks (10,000 x 10,000 Matrix)

| Stage | Optimization Strategy | Time (sec) | GFLOPS | Speedup (vs Stage 4) |
| :--- | :--- | :--- | :--- | :--- |
| 5 | **GPU Naive** (CUDA Kernel) | 3.0s* | 667.0 | ~100x |
| 6 | **GPU Tiled** (Shared Memory) | **2.0s*** | **1,000.0+** | **150x+** |

*\*Note: GPU timing includes memory transfer (HtoD / DtoH) for 2.4GB of data.*

##  Key Features & Optimizations

### 1. CPU Optimization (AVX2 / OpenMP)
* **Data Locality:** Reordered loops to IKJ pattern to minimize cache misses.
* **SIMD Vectorization:** Explicit use of Intel Intrinsics (`__m256`, `_mm256_fmadd_ps`) to process 8 floats per cycle.
* **Multi-threading:** Work-sharing using OpenMP with optimized grain size.

### 2. GPU Optimization (CUDA)
* **Memory Tiling:** Implemented **Shared Memory Tiling** to reduce Global Memory transactions. Each block of threads cooperatively loads data into L1-speed shared memory.
* **Occupancy Tuning:** Optimized `TILE_SIZE` and block dimensions (16x16) for maximum hardware utilization.
* **Synchronization:** Precise use of `__syncthreads()` to prevent race conditions during tiled loading.

##  Project Structure
* `Tensor.cpp`: Core logic, SIMD/OpenMP implementations.
* `cuda_kernels.cu`: CUDA kernels (Naive & Tiled).
* `autograd`: (From previous version) Reverse-mode automatic differentiation.

##  Requirements
* NVIDIA GPU (Compute Capability 7.5+)
* CUDA Toolkit 13.2+
* C++20 Compiler (GCC/Clang/MSVC)
* OpenMP support
