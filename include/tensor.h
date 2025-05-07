#pragma once

#include <cassert>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#define DIM_MAX 4

template<typename T>
class Tensor{
public:
    Tensor(std::vector<T> data, std::vector<int> shape) :  data(std::move(data)), shape(std::move(shape)) {
        initShape();
    }

    Tensor() = default;

    Tensor(Tensor&& oth) :  data(std::move(oth.data)), shape(std::move(oth.shape)), jump(std::move(oth.jump)) {
    }

    Tensor<T>& operator=(Tensor<T>&& oth) {
        data = std::move(oth.data);
        shape = std::move(oth.shape);
        jump = std::move(oth.jump);
        return *this;
    }

    void load(std::vector<T> data, std::vector<int> shape) {
        this->data = std::move(data);
        this->shape = std::move(shape);
        initShape();
    }

    void asShape(std::vector<int> shape) {
        this->shape = std::move(shape);
        int size = 1;
        for (int i : this->shape)
            size *= i;
        data.clear();
        data.resize(size, 0);
        initShape();
    }

    template<typename... Args>
    T& at(Args&&... args) {
        return at({args...});
    }

    template<typename... Args>
    T at(Args&&... args) const {
        return at({args...});
    }

    T& at(std::vector<int> idxs) {
        while (idxs.size() < DIM_MAX) {
            idxs.insert(idxs.begin(), 0);
        }
        int idx = 0;
        for (int i = 0; i < DIM_MAX; i++) {
            assert(idxs[i] >= 0 && idxs[i] < shape[i]);
            idx += jump[i] * idxs[i];
        }
        return data[idx];
    }
    T at(std::vector<int> idxs) const {
        while (idxs.size() < DIM_MAX) {
            idxs.insert(idxs.begin(), 0);
        }
        int idx = 0;
        for (int i = 0; i < DIM_MAX; i++) {
            assert(idxs[i] >= 0 && idxs[i] < shape[i]);
            idx += jump[i] * idxs[i];
        }
        return data[idx];
    }

    T& atOffset(int idx) {
        return data[idx];
    }

    Tensor<T> operator*(const Tensor<T>& oth) {
        assert(shape[DIM_MAX - 1] == oth.shape[DIM_MAX - 2]);
        assert(checkBroadCastValid(oth, DIM_MAX - 2));

        int k = shape[DIM_MAX - 2], u = shape[DIM_MAX - 1], v = oth.shape[DIM_MAX - 1];
        std::vector<int> new_shape{std::max(shape[0], oth.shape[0]), std::max(shape[1], oth.shape[1]), k, v};
        Tensor<T> res;
        res.asShape(new_shape);
        for (int dim0 = 0; dim0 < new_shape[0]; dim0++) {
            for (int dim1 = 0; dim1 < new_shape[1]; dim1++) {
                for (int dim2 = 0; dim2 < k; dim2++) {
                    for (int dim3 = 0; dim3 < v; dim3++) {
                        T tmp = 0;
                        for (int i = 0; i < u; i++) {
                            tmp += at(dim0 % shape[0], dim1 % shape[1], dim2, i) * oth.at(dim0 % oth.shape[0], dim1 % oth.shape[1], i, dim3);
                        }
                        res.at(dim0, dim1, dim2, dim3) = tmp;
                    }
                }
            }
        }
        
        return res;
    }

    Tensor<T> operator+(const Tensor<T>& oth) {
        assert(checkBroadCastValid(oth, DIM_MAX));
        std::vector<int> new_shape;
        for (int dim = 0; dim < DIM_MAX; dim++) {
            new_shape.push_back(std::max(shape[dim], oth.shape[dim]));
        }
        Tensor<T> res;
        res.asShape(new_shape);
        for (int dim0 = 0; dim0 < new_shape[0]; dim0++) {
            for (int dim1 = 0; dim1 < new_shape[1]; dim1++) {
                for (int dim2 = 0; dim2 < new_shape[2]; dim2++) {
                    for (int dim3 = 0; dim3 < new_shape[3]; dim3++) {
                        res.at(dim0, dim1, dim2, dim3) = at(dim0 % shape[0], dim1 % shape[1], dim2 % shape[2], dim3 % shape[3]) +
                         oth.at(dim0 % oth.shape[0], dim1 % oth.shape[1], dim2 % oth.shape[2], dim3 % oth.shape[3]);
                    }
                }
            }
        }
        return res;
    }


    void initShape() {
        while (shape.size() < DIM_MAX) {
            shape.insert(shape.begin(), 1);
        }
        jump.resize(DIM_MAX);
        jump.back() = 1;
        for (int i = DIM_MAX - 2; i >= 0; i--) {
            jump[i] = jump[i + 1] * shape[i + 1];
        }
    }

    bool checkBroadCastValid(const Tensor<T>& oth, int checkDim) {
        for (int idx = 0; idx < checkDim; idx++) {
            int max_shape = std::max(shape[idx], oth.shape[idx]);
            int min_shape = std::min(shape[idx], oth.shape[idx]);
            if (max_shape / min_shape * min_shape != max_shape) {
                return false;
            }
        }
        return true;
    }

    std::vector<T> data;
    std::vector<int> shape;
    std::vector<int> jump;
};