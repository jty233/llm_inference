#pragma once
#include "model_parse.h"
#include "module.h"
#include "modules/layer_norm.h"
#include "modules/linear.h"
#include "multi_head_attention.h"
#include "tensor.h"
#include "time_calc.h"
#include <string>
struct GPT2Block {
    GPT2Block(ModelParse& parser, const std::string& name, int num_head, int hidden_dim) : mha(num_head, hidden_dim, true) {
        ln1.init(parser, name + "ln_1");
        ln2.init(parser, name + "ln_2");
        mha.in_proj.init(parser, name + "attn.c_attn", true);
        mha.out_proj.init(parser, name + "attn.c_proj", true);
        fc.init(parser, name + "mlp.c_fc", true);
        proj.init(parser, name + "mlp.c_proj", true);

    }

    Tensor<float> forward(const Tensor<float>& input) {
        TimeCalcGuard g("gpt2 block");
        Tensor<float> x = ln1.forward(input);
        x = mha.forward(x);
        x = x + input;
        Tensor<float> out = ln2.forward(x);
        out = fc.forward(out);
        out = gelu(out);
        out = proj.forward(out);
        return out + x;
    }

    LayerNorm<float> ln1, ln2;
    MultiHeadAttention<float> mha;
    Linear<float> fc, proj;
};