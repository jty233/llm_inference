#pragma once
#include "model_parse.h"
#include "modules/RMS_norm.h"
#include "modules/linear.h"
#include "qwen3_block.h"
#include "tensor.h"
#include <string>
#include <vector>
struct  Qwen3 {
    Qwen3(const std::string& path) : parser(path) {
        blocks.reserve(block_num);
        for (int i = 0; i < block_num; i++) {
            blocks.emplace_back(num_attention_head, num_kv_head, hidden_dim, true);
        }
        int i = 0;
        lm_head.init(parser, "lm_head", false, true);
        embed_tokens = parser.getTensor("model.embed_tokens.weight");
        for (auto& block : blocks) {
            std::string name = "model.layers." + std::to_string(i) + ".";
            block.input_layernorm.w = parser.getTensor(name + "input_layernorm.weight");
            block.post_attention_layernorm.w = parser.getTensor(name + "post_attention_layernorm.weight");

            block.mlp.down.init(parser, name + "mlp.down_proj", false, true);
            block.mlp.gate.init(parser, name + "mlp.gate_proj", false, true);
            block.mlp.up.init(parser, name + "mlp.up_proj", false, true);

            block.attention.k_proj.init(parser, name + "self_attn.k_proj", false, true);
            block.attention.v_proj.init(parser, name + "self_attn.v_proj", false, true);
            block.attention.q_proj.init(parser, name + "self_attn.q_proj", false, true);
            block.attention.out_proj.init(parser, name + "self_attn.o_proj", false, true);

            block.attention.k_norm.w = parser.getTensor(name + "self_attn.k_norm.weight");
            block.attention.q_norm.w = parser.getTensor(name + "self_attn.q_norm.weight");

            i++;
        }
        norm.w = parser.getTensor("model.norm.weight");
    }

    Tensor<float> forward(const std::vector<int>& id) {
        Tensor<float> x;
        x.asShape({(int)id.size(), hidden_dim});
        for (int i = 0; i < id.size(); i++) {
            for (int j = 0; j < hidden_dim; j++) {
                x.at(i, j) = embed_tokens.at(id[i], j);
            }
        }

        for (auto& block : blocks) {
            x = block.forward(x);
        }
        
        Tensor<float> out = x.slice({{id.size() - 1, id.size()}, {0, hidden_dim}});
        out = norm.forward(out);
        return lm_head.forward(out);
    }


    int block_num = 28;
    int num_attention_head = 16;
    int num_kv_head = 8;
    int hidden_dim = 1024;
    std::vector<Qwen3Block<float>> blocks;
    Linear<float> lm_head;
    RMSNorm<float> norm;
    Tensor<float> embed_tokens;
    ModelParse parser;

};