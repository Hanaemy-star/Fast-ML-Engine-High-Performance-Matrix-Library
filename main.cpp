#include "tensor.hpp"
#include "cuda_kernels.hpp"
#include <chrono>

int main() {
    size_t size = 10000;

    auto m1 = std::make_shared<Tensor>(std::vector<size_t>{size, size}, 1.1);
    auto m2 = std::make_shared<Tensor>(std::vector<size_t>{size, size}, 1.1);

    std::cout << "Starting benchmark for " << size << "x" << size << " matrix..." << std::endl;

    std::cout << "Matmul on GPU without shared memory" << std::endl;
    auto start1 = std::chrono::high_resolution_clock::now();

    auto result1 = std::make_shared<Tensor>(std::vector<size_t>{size, size}, 0.0);

    launch_matmul_kernel(m1->get_data().data(), m2->get_data().data(), result1->get_data().data(), size, size, size);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds1 = end1 - start1;

    std::cout << "Time: " << elapsed_seconds1.count() << " seconds" << std::endl;
    std::cout << "GFLOPS: " << (2.0 * size * size * size / (elapsed_seconds1.count() * 1e9)) << std::endl;

    std::cout << std::endl;

    std::cout << "Matmul on GPU with shared memory" << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();

    auto result2 = std::make_shared<Tensor>(std::vector<size_t>{size, size}, 0.0);

    launch_matmul_shared_kernel(m1->get_data().data(), m2->get_data().data(), result2->get_data().data(), size, size, size);

    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds2 = end2 - start2;

    std::cout << "Time: " << elapsed_seconds2.count() << " seconds" << std::endl;
    std::cout << "GFLOPS: " << (2.0 * size * size * size / (elapsed_seconds2.count() * 1e9)) << std::endl;

    return 0;
}
