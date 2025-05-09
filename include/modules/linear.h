#pragma once
#include "model_parse.h"
#include "tensor.h"
#include <string>

template<typename T>
struct Linear{
    void init(ModelParse& parser, const std::string& name, bool normal_order = false) {
        w = parser.getTensor(name + ".weight");
        b = parser.getTensor(name + ".bias");
        if (normal_order) {
            w = w.transpose();
        }
    }
    Tensor<T> forward(const Tensor<T>& input) {
        return input.matMulTranspos(w) + b;
    }

    Tensor<T> w, b;
};