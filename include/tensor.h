#pragma once

#include <cassert>
#include <utility>
#include <vector>
template<typename T>
class Tensor{
public:
    Tensor(std::vector<T> data, std::vector<int> shape) :  data(std::move(data)), shape(std::move(shape)) {
        initJump();
    }

    Tensor() = default;

    void load(std::vector<T> data, std::vector<int> shape) {
        this->data = std::move(data);
        this->shape = std::move(shape);
        initJump();
    }

    void load(std::vector<T> data) {
        this->data = std::move(data);
    }

    void reshape(std::vector<int> shape) {
        this->shape = std::move(shape);
        initJump();
    }

    template<typename... Args>
    T& at(Args&&... args) {
        std::vector<int> idxs{args...};
        assert(idxs.size() == dim);
        int idx = 0;
        for (int i = 0; i < dim; i++) {
            assert(idxs[i] >= 0 && idxs[i] < shape[i]);
            idx += jump[i] * idxs[i];
        }
        return data[idx];
    }


private:
    void initJump() {
        dim = shape.size();
        jump.resize(dim);
        jump.back() = 1;
        for (int i = dim - 2; i >= 0; i--) {
            jump[i] = jump[i + 1] * shape[i + 1];
        }
    }

    int dim;
    std::vector<T> data;
    std::vector<int> shape;
    std::vector<int> jump;
};