#pragma once
#include "model_parse.h"
#include "module.h"
#include "modules/linear.h"
#include "modules/multi_head_attention.h"
#include "tensor.h"
#include <string>
class MnistMha {
public:
    MnistMha(const std::string& path) : parser(path), mha(8, 512, false) {
        embedding.init(parser, "embedding");
        classifier0.init(parser, "classifier.0");
        classifier2.init(parser, "classifier.2");
        mha.in_proj.w = parser.getTensor("mha.in_proj_weight");
        mha.in_proj.b = parser.getTensor("mha.in_proj_bias");
        mha.out_proj.init(parser, "mha.out_proj");
    }
    Tensor<float> forward(const Tensor<float>& input) {
        auto x = embedding.forward(input);
        x = mha.forward(x);
        x = classifier0.forward(x);
        x = relu(x);
        x = classifier2.forward(x);
        return softmax(x);
    }
private:
    Linear<float> embedding, classifier0, classifier2;
    MultiHeadAttention<float> mha;
    ModelParse parser;
};