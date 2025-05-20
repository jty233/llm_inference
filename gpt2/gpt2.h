#pragma once
#include "gpt2_block.h"
#include "model_parse.h"
#include "modules/layer_norm.h"
#include "tensor.h"
#include "time_calc.h"
#include <string>
#include <vector>

class GPT2 {
public:
    GPT2(const std::string& model_path) : parser(model_path) {
        TimeCalcGuard g("gpt2 load");
        wte = parser.getTensor("wte.weight");
        wpe = parser.getTensor("wpe.weight");
        ln_f.init(parser, "ln_f");
        for (int i = 0; i < num_blocks; i++) {
            blocks.emplace_back(parser, "h." + std::to_string(i) + ".", num_head, hidden_dim);
        }
    }

    Tensor<float> forward(const std::vector<int>& id) {
        TimeCalcGuard g("gpt2 forward");
        Tensor<float> x;
        x.asShape({(int)id.size(), hidden_dim});
        for (int i = 0; i < id.size(); i++) {
            for (int j = 0; j < hidden_dim; j++) {
                x.at(i, j) = wte.at(id[i], j) + wpe.at(i + f_token_size, j);
            }
        }
        f_token_size += id.size();

        for (auto& block : blocks) {
            x = block.forward(x);
        }
        x = ln_f.forward(x);
        Tensor<float> out;
        out.asShape({hidden_dim});
        for (int i = 0; i < hidden_dim; i++) {
            out.at(i) = x.at(int(id.size() - 1), i);
        }
        return out.matMulTranspos(wte);
    }

private:

    ModelParse parser;
    std::vector<GPT2Block> blocks;
    Tensor<float> wte, wpe;
    LayerNorm<float> ln_f;

    int num_blocks = 12;
    int num_head = 12;
    int hidden_dim = 768;

    int f_token_size = 0;

};