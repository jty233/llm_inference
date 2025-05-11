#pragma once
#include "module.h"
#include "tensor.h"
#include <cmath>
#include <vector>

template<typename T>
struct SingleHeadAttention{
    SingleHeadAttention(bool mask) : mask(mask) {}

    Tensor<T> forward(const Tensor<T>& q,const Tensor<T>& k,const Tensor<T>& v) {
        auto x = q.matMulTranspos(k);
        x /= std::sqrt(q.shape.back());
        if (mask) {
            Tensor<T>::forEachDim(x.shape, [&](std::vector<int> dim) {
                if (dim[x.DIM_MAX - 1] > dim[x.DIM_MAX - 2]) {
                    x.at(dim) = -1e9;
                }
            }, 1);
        }
        x = softmax(x);
        x = x.matMul(v);
        return x;
    }

    bool mask;
};