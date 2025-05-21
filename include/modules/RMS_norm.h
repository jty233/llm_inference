#pragma once
#include "tensor.h"

template<typename T>
struct RMSNorm{
    Tensor<T> forward(const Tensor<T>& input) {
        Tensor<T> mean = input.elementWiseMul(input).mean().sqrt(eps);
        return input / mean * w;
    }

    Tensor<T> w;
    double eps = 1e-6;
};