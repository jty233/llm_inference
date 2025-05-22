#pragma once
#include "grouped_head_attention.h"
#include "mlp.h"
#include "modules/RMS_norm.h"
#include "tensor.h"
template<typename T>
struct Qwen3Block {

    Tensor<T> forward(const Tensor<T>& input) {
        Tensor<T> x = attention.forward(post_attention_layernorm.forward(input)) + input;
        return mlp.forward(input_layernorm.forward(x)) + x;
    }

    RMSNorm<T> input_layernorm, post_attention_layernorm;
    GroupedHeadAttention<T> attention;
    MLP<T> mlp;
};