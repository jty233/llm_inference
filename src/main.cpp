#include <iostream>
#include <vector>
#include "model_parse.h"
#include "tensor.h"
using namespace std;
int main()
{
    ModelParse model;
    model.initModelPath("../model.safetensors");
    model.parse();
    auto tensor = model.getTensor("h.0.attn.c_attn.weight");
    cout << tensor.at(333, 555) << endl;

    Tensor<float> t;
    t.load({
        1.,2.,3.,
        4.,5.,6.,

        7.,8.,9.,
        10.,11.,12.,

        11.,21.,31.,
        41.,51.,61.,
    }, {3, 2, 3});
    cout << t.at(2,1,2) << endl;

    vector<float> v{1,2,3,4,5,6};
    Tensor<float> test(std::move(v),{2,3});
    cout << v.size() << endl;
}