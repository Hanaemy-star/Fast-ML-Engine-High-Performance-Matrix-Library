#include "tensor.hpp"
#include <numeric>
#include <algorithm>
#include <immintrin.h>

Tensor::Tensor(std::vector<size_t> shape, float initial_value, bool requires_grad) : shape(shape) , requires_grad(requires_grad) {
    if (requires_grad) {
        grad = std::make_shared<Tensor>(shape, 0.0, false);
    }
    size_t total_size = calculate_size(shape);
    data = std::vector<float>(total_size, initial_value);
}

size_t Tensor::calculate_size(const std::vector<size_t>& s) {
    if (s.empty()) return 0;
    size_t res = 1;
    for (auto dim : s) res *= dim;
    return res;
}

float& Tensor::operator()(const std::vector<size_t>& indices) {
    size_t flat_index = 0;
    size_t strides = 1;
    for (int i = indices.size() - 1; i >= 0; i--) {
        flat_index += indices[i] * strides;
        strides *= shape[i];
    }
    return data[flat_index];
}

float Tensor::operator()(const std::vector<size_t>& indices) const {
    size_t flat_index = 0;
    size_t strides = 1;
    for (int i = indices.size() - 1; i >= 0; i--) {
        flat_index += indices[i] * strides;
        strides *= shape[i];
    }
    return data[flat_index];
}

void Tensor::reshape(std::vector<size_t> nshape) {
    if (calculate_size(nshape) != data.size()) {
        throw std::invalid_argument("New shape must have the same total number of elements");
    }
    shape = nshape;
}

void Tensor::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}

void Tensor::zero_grad() {
    grad->fill(0.0);
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (shape != other.shape) {
        throw std::invalid_argument("Shapes must match for in-place addition");
    }
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

std::vector<float>& Tensor::get_data() {
    return this->data;
}

std::shared_ptr<Tensor> Tensor::add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape != b->shape) throw std::invalid_argument("Shapes must match");

    bool req_grad = a->requires_grad || b->requires_grad;
    auto result = std::make_shared<Tensor>(a->shape, 0.0, req_grad);

    for (size_t i = 0; i < a->data.size(); i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    if (req_grad) {
        result->prev = {a, b};

        result->_backward = [a, b, result]() {
            if (a->requires_grad) {
                for (size_t i = 0; i < a->grad->data.size(); i++) {
                    a->grad->data[i] += result->grad->data[i];
                }
            }
            if (b->requires_grad) {
                for (size_t i = 0; i < b->grad->data.size(); i++) {
                    b->grad->data[i] += result->grad->data[i];
                }
            }
        };
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::matrixmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::shared_ptr<Tensor> result = a->matmul(b);

    bool req_grad = a->requires_grad || b->requires_grad;

    if (req_grad) {
        result->requires_grad = true;
        result->grad = std::make_shared<Tensor>(result->shape, 0.0, false);

        result->prev = {a, b};

        result->_backward = [a, b, result]() {
            if (a->requires_grad) {
                auto d_a = result->grad->matmul(b->transpose());
                *(a->grad) += *d_a;
            }
            if (b->requires_grad) {
                auto d_b = a->transpose()->matmul(result->grad);
                *(b->grad) += *d_b;
            }
        };
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::transpose() const {
    if (this->shape.size() != 2) throw std::invalid_argument("Size must be 2");

    std::vector<size_t> new_shape = this->shape;
    std::reverse(new_shape.begin(), new_shape.end());

    auto result = std::make_shared<Tensor>(new_shape, 0.0,
                                             this->requires_grad);

    for (size_t i = 0; i < this->shape[0]; i++) {
        for (size_t j = 0; j < this->shape[1]; j++) {
            (*result)({j, i}) = (*this)({i, j});
        }
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other) const {
    if (shape.size() != 2 || other->shape.size() != 2)
        throw std::invalid_argument("matmul is for 2D tensors only");
    if (shape[1] != other->shape[0])
        throw std::invalid_argument("Inner dimensions must match");

    size_t M = shape[0];
    size_t K = shape[1];
    size_t N = other->shape[1];
    auto d = {M, N};
    auto result = std::make_shared<Tensor>(d, 0.0);

    const float* a_ptr = this->data.data();
    const float* b_ptr = other->data.data();
    float* res_ptr = result->data.data();

    const int BS = 32;
    #pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < M; ii += BS) {
        size_t i_end = std::min(ii + BS, (long long)M);

        for (size_t kk = 0; kk < K; kk += BS) {
            size_t k_end = std::min(kk + BS, K);

            for (size_t jj = 0; jj < N; jj += BS) {
                size_t j_end = std::min(jj + BS, N);

                for (size_t i = ii; i < i_end; i++) {
                    for (size_t k = kk; k < k_end; k++) {
                        float a_val = a_ptr[i * K + k];
                        __m256 va = _mm256_set1_ps(a_val);

                        size_t j = jj;
                        for (; j + 3 < j_end; j += 4) {
                            __m256 vb = _mm256_loadu_ps(b_ptr + (k * N + j));
                            __m256 vr = _mm256_loadu_ps(res_ptr + (i * N + j));

                            vr = _mm256_fmadd_ps(va, vb, vr);

                            _mm256_storeu_ps(res_ptr + (i * N + j), vr);
                        }
                        for (; j < j_end; j++) {
                            res_ptr[i * N + j] += a_val * b_ptr[k * N + j];
                        }
                    }
                }
            }
        }
    }
    return result;
}

void Tensor::print() const {
    if (shape.size() != 2) {
        std::cout << "Printing for N-dim not implemented yet\n";
        return;
    }
    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            std::cout << (*this)({i, j}) << "\t";
        }
        std::cout << std::endl;
    }
}

Tensor& Tensor::apply_(std::function<float(float)> func) {
    for (float& val : data) {
        val = func(val);
    }
    return *this;
}

std::shared_ptr<Tensor> Tensor::apply(std::function<float(float)> func) const {
    auto result = std::make_shared<Tensor>(*this);
    for (float& val : result->data) {
        val = func(val);
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::leaky_relu() {
    auto result = std::make_shared<Tensor>(this->shape, 0.0, this->requires_grad);

    for (size_t i = 0; i < data.size(); i++) {
        result->data[i] = (data[i] > 0.0) ? data[i] : 0.01 * data[i];
    }

    result->prev = { shared_from_this() };

    if (this->requires_grad) {
        auto input_ptr = shared_from_this();

        result->_backward = [input_ptr, result]() {
            for (size_t i = 0; i < input_ptr->data.size(); i++) {
                if (input_ptr->data[i] > 0.0) {
                    input_ptr->grad->data[i] += result->grad->data[i];
                } else {
                    input_ptr->grad->data[i] += 0.01 * result->grad->data[i];
                }
            }
        };
    }
    return result;
}

Tensor Tensor::operator*(const float& scalar) const {
    Tensor result = *this;
    return result.apply_([scalar](float val) {return val * scalar;});
}

std::shared_ptr<Tensor> Tensor::get_grad() const {
    return this->grad;
}

void Tensor::build_topo(std::shared_ptr<Tensor> curr,
                    std::vector<std::shared_ptr<Tensor>>& topo,
                    std::unordered_set<Tensor*>& visited) {

    if (visited.find(curr.get()) != visited.end()) {
        return;
    }

    visited.insert(curr.get());

    for (auto& parent : curr->prev) {
        build_topo(parent, topo, visited);
    }
    topo.push_back(curr);
}

void Tensor::backward() {
    std::vector<std::shared_ptr<Tensor>> topo;
    std::unordered_set<Tensor*> visited;
    build_topo(shared_from_this(), topo, visited);

    if (this->grad) {
        this->grad->fill(1.0);
    }

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->_backward) {
            (*it)->_backward();
        }
    }
}

std::shared_ptr<Tensor> Tensor::mse_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target) {
    size_t size = pred->data.size();
    float sum_diff = 0.0;
    for (size_t i = 0; i < size; i++) {
        float diff = pred->data[i] - target->data[i];
        sum_diff += diff * diff;
    }
    float mse = sum_diff / size;

    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(std::vector<size_t>{1}, mse, pred->requires_grad);

    result->prev = {pred};
    if (pred->requires_grad) {
        result->_backward = [pred, target, result, size]() {
            float upstream_grad = result->grad->data[0];

            for (size_t i = 0; i < size; i++) {
                pred->grad->data[i] += (2.0 / size) * (pred->data[i] - target->data[i]) * upstream_grad;
            }
        };
    }
    return result;
}

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    return Tensor::add(a, b);
}

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    return Tensor::matrixmul(a, b);
}