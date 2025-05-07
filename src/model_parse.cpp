#include "model_parse.h"
#include "tensor.h"

using namespace std;

void ModelParse::parse() {
    readArg(json_length);
    std::string json_str;
    json_str.resize(json_length);
    read(json_str.data(), json_length);
    model_json = nlohmann::json::parse(json_str);
    
}

Tensor<float> ModelParse::getTensor(const std::string& name) {
    auto info = model_json[name];
    assert(info["dtype"] == "F32");
    std::vector<int> shape = info["shape"];
    std::vector<long long> offset = info["data_offsets"];
    int len = offset[1] - offset[0];
    fin.seekg(offset[0] + 8 + json_length);
    std::vector<float> val(len);
    read(val.data(), len);
    return {std::move(val), std::move(shape)};
}

