#pragma once

#include <omp.h>
#include <iostream>
#include <vector>
#include <functional>
#include <memory>
#include <unordered_set>

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    std::vector<float> data;
    std::vector<size_t> shape;
    std::shared_ptr<Tensor> grad;
    std::vector<std::shared_ptr<Tensor>> prev;
    bool requires_grad;

    size_t calculate_size(const std::vector<size_t>& s);
    void zero_grad();

public:

    std::function<void()> _backward;

    static std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

    static std::shared_ptr<Tensor> matrixmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

    static std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);

    Tensor(std::vector<size_t> shape, float initial_value = 0.0, bool requires_grad = false);

    float& operator()(const std::vector<size_t>& indices);

    float operator()(const std::vector<size_t>& indices) const;

    void reshape(std::vector<size_t> nshape);

    void fill(float value);

    std::shared_ptr<Tensor> get_grad() const;

    std::vector<float>& get_data();

    Tensor& operator+=(const Tensor& other);

    std::shared_ptr<Tensor> transpose() const;

    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> other) const;

    void print() const;

    Tensor& apply_(std::function<float(float)> func);

    std::shared_ptr<Tensor> apply(std::function<float(float)> func) const;

    std::shared_ptr<Tensor> leaky_relu();

    Tensor operator*(const float& scalar) const;



    void build_topo(std::shared_ptr<Tensor> curr,
                    std::vector<std::shared_ptr<Tensor>>& topo,
                    std::unordered_set<Tensor*>& visited);

    void backward();
};

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);