#pragma once
#include "tensor.h"

template<typename T>
struct LayerNorm{
    Tensor<T> forward(const Tensor<T>& input) {
        return input.elementWiseMul(w) + b;
    }

    Tensor<T> w, b;
};