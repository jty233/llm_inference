#pragma once
#include "gpt2_block.h"
#include "model_parse.h"
#include "tensor.h"
#include <string>
#include <vector>

class GPT2 {
public:
    GPT2(const std::string& model_path) : parser(model_path) {
        for (int i = 0; i < num_blocks; i++) {
            blocks.emplace_back(parser, "h." + std::to_string(i), num_head, hidden_dim);
        }
    }

    Tensor<float> forward(Tensor<float> input) {
        for (auto& block : blocks) {
            input = block.forward(input);
        }
        return input;
    }

private:

    ModelParse parser;
    std::vector<GPT2Block> blocks;

    int num_blocks = 12;
    int num_head = 12;
    int hidden_dim = 768;

};