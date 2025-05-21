#pragma once
#include "module.h"
#include "tensor.h"
#include <cmath>
#include <vector>

template<typename T>
struct SingleHeadAttention{
    SingleHeadAttention(bool mask) : mask(mask) {}

    Tensor<T> forward(const Tensor<T>& q) {
        auto x = q.matMulTranspos(*k_cache);
        x /= std::sqrt(q.shape.back());
        if (mask) {
            Tensor<T>::forEachDim(x.shape, [&](std::vector<int> dim) {
                if (dim[x.DIM_MAX - 1] > dim[x.DIM_MAX - 2]) {
                    x.at(dim) = -1e9;
                }
            }, 1);
            mask = false;
        }
        x = softmax(x);
        x = x.matMul(*v_cache);
        return x;
    }

    Tensor<T> *k_cache, *v_cache;


    bool mask;
};