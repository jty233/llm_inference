#pragma once
#include "tensor.h"
#include <cmath>

template<typename T>
Tensor<T> gelu(const Tensor<T>& input) {
    Tensor<T> res = input;
    for (auto& x : res.data) {
        const float sqrt_2_over_pi = std::sqrt(2.0f / std::acos(-1));
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        x = 0.5f * x * (1.0f + std::tanh(inner));
    }
}


template<typename T>
Tensor<T> relu(const Tensor<T>& input) {
    Tensor<T> res = input;
    for (auto& x : res.data) {
        if (x < 0) {
            x = 0;
        }
    }
}



template<typename T>
Tensor<T> softmax(const Tensor<T>& input) {
    
    
}