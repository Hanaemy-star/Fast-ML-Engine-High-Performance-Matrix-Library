#include "tensor.hpp"

#include <chrono>

int main() {

    auto matrix1 = std::make_shared<Tensor>(std::vector<size_t>{1000, 1000}, 2.0);
    auto matrix2 = std::make_shared<Tensor>(std::vector<size_t>{1000, 1000}, 2.0);

    auto start = std::chrono::system_clock::now();
    auto result = matrix1 * matrix2;
    auto end = std::chrono::system_clock::now();
    
    std::cout << end - start << "\n";

    return 0;
}
