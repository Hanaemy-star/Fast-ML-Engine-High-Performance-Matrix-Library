# Fast-ML-Engine-High-Performance-Matrix-Library
## Performance Evolution

The core of this engine is a highly optimized Matrix Multiplication (GEMM) implementation. Through several stages of low-level optimization, the execution time for a **1000x1000** double-precision matrix was reduced from **310 seconds** to **0.3 seconds** (~1000x speedup).

### Optimization Benchmark (1000x1000 Matrix)

| Optimization Stage | Time (sec) | GFLOPS | Speedup |
| :--- | :--- | :--- | :--- |
| **Baseline** (Naive O(n³) traversal) | 310.0 s | 0.006 | 1x |
| **IKJ Order** (Cache-friendly access) | 2.5 s | 0.8 | 124x |
| **AVX2 + SIMD** (Vectorization) | 1.3 s | 1.5 | 238x |
| **SIMD + Tiling** (L1 Cache Locality) | 1.2 s | 1.6 | 258x |
| **OpenMP** (Multithreading) | 0.3 s | 6.6 | **1033x** |

### Key Technical Improvements

* **Cache Locality (Loop Tiling):** Implemented a block-based approach ($32 \times 32$ tiles) to ensure data stays in the L1/L2 cache, drastically reducing expensive RAM fetches.
* **SIMD Vectorization:** Leveraged Intel AVX2 intrinsics (`_mm256_fmadd_pd`) to perform 4 double-precision Fused Multiply-Add operations per CPU cycle.
* **Parallel Computing:** Scaled computation across all available CPU cores using OpenMP with a thread-safe work-sharing scheduler.
* **Memory Sympathy:** Achieved a ~1000x increase in throughput by aligning software logic with modern CPU microarchitecture.

> "Achieved a 1000x speedup on CPU through deep cache optimization, manual vectorization, and multi-core scaling."
