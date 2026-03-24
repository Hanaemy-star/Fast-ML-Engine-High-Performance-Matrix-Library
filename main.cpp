#include "tensor.hpp"

#include <chrono>

int main() {
    size_t size = 1000;

    auto m1 = std::make_shared<Tensor>(std::vector<size_t>{size, size}, 1.1);
    auto m2 = std::make_shared<Tensor>(std::vector<size_t>{size, size}, 1.1);

    std::cout << "Starting benchmark for " << size << "x" << size << " matrix..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    auto result = m1 * m2;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Time: " << elapsed_seconds.count() << " seconds" << std::endl;
    std::cout << "GFLOPS: " << (2.0 * size * size * size / (elapsed_seconds.count() * 1e9)) << std::endl;

    return 0;
}
