#include "model_parse.h"
#include "tensor.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

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
    std::vector<int> shape = info["shape"];
    std::vector<long long> offset = info["data_offsets"];
    int len = offset[1] - offset[0];
    fin.seekg(offset[0] + 8 + json_length, ios::beg);
    if (info["dtype"] == "F32") {
        std::vector<float> val(len / sizeof(float));
        read(val.data(), len);
        return {std::move(val), std::move(shape)};
    }
    else if (info["dtype"] == "BF16") {
        int num = len / 2;
        std::vector<uint16_t> val(num);
        read(val.data(), len);
        vector<float> real_val(num);
        for (int i = 0; i < num; i++) {
            uint32_t temp = static_cast<uint32_t>(val[i]) << 16;
            memcpy(real_val.data() + i, &temp, 4);
        }
        return {std::move(real_val), std::move(shape)};
    }
    else {
        throw runtime_error("name error in model parse");
    }
}

