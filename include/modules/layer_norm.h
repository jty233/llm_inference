#pragma once
#include "tensor.h"
#include "model_parse.h"

template<typename T>
struct LayerNorm{
    void init(ModelParse& parser, const std::string& name) {
        w = parser.getTensor(name + ".weight");
        b = parser.getTensor(name + ".bias");
    }
    Tensor<T> forward(const Tensor<T>& input) {
        Tensor<T> mean = input.mean();
        Tensor<T> var = input - mean;
        var = var.elementWiseMul(var);
        var = var.mean();
        var = var.sqrt();
        Tensor<T> x_hat = (input - mean) / var;
        return w.elementWiseMul(x_hat) + b;
    }

    Tensor<T> w, b;
};