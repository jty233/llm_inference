#pragma once
#include "model_parse.h"
#include "tensor.h"
#include <string>

template<typename T>
struct Linear{
    void init(ModelParse& parser, const std::string& name, bool normal_order = false) {
        w = parser.getTensor(name + ".weight");
        b = parser.getTensor(name + ".bias");
        this->normal_order = normal_order;
    }
    Tensor<T> forward(const Tensor<T>& input) {
        if (normal_order) {
            return input.matMul(w) + b;
        }
        return input.matMulTranspos(w) + b;
    }

    Tensor<T> w, b;
    bool normal_order;
};