#pragma once
#include "tensor.h"

template<typename T>
struct Linear{
    Tensor<T> forward(const Tensor<T>& input) {
        return input.matMulTranspos(w) + b;
    }

    Tensor<T> w, b;
};