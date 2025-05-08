#pragma once
#include "json.hpp"
#include "tensor.h"
#include <cassert>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>

class ModelParse {

public:
    ModelParse(const std::string& path) : fin(path, std::ios::binary) {
        parse();
        std::ofstream fout(path + ".json");
        fout << model_json.dump(4);
    }

    Tensor<float> getTensor(const std::string& name);


private:

    template<typename T>
    void readArg(T&& arg) {
        fin.read(reinterpret_cast<char*>(&arg), sizeof(arg));
    }

    template<typename T>
    void read(T* p, int len) {
        fin.read(reinterpret_cast<char*>(p), len);
    }
    void parse();

    nlohmann::json model_json;
    std::ifstream fin;
    long long json_length;
};