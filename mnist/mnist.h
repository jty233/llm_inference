#pragma once
#include "model_parse.h"
#include "module.h"
#include "modules/linear.h"
#include "tensor.h"
#include <string>
class Mnist {
public:
    Mnist(const std::string& path) : parser(path) {
        fc1.w = parser.getTensor("fc1.weight").transpose();
        fc1.b = parser.getTensor("fc1.bias");

        fc2.w = parser.getTensor("fc2.weight").transpose();
        fc2.b = parser.getTensor("fc2.bias");
    }

    Tensor<float> forward(const Tensor<float>& input) {
        auto x = fc1.forward(input);
        x = relu(x);
        x = fc2.forward(x);
        x = softmax(x);
        return x;
    }

private:
    ModelParse parser;
    Linear<float> fc1, fc2;
    
};