#pragma once
#include "model_parse.h"
#include <string>

class GPT2 {
public:
    GPT2(const std::string& model_path) : parser(model_path) {

    }

private:

    ModelParse parser;
};