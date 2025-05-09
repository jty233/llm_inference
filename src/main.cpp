#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>
#include "mnist_test.h"
#include "model_parse.h"
#include "tensor.h"
#include "modules/multi_head_attention.h"
using namespace std;
int main()
{
    // ios::sync_with_stdio(false);
    // cout.tie(0);
    // mnist_test();
    ModelParse parser("../model.safetensors");
    MultiHeadAttention<float> head(parser.getTensor("h.0.attn.c_attn.weight"), parser.getTensor("h.0.attn.c_attn.bias"), 12, 768);
    Tensor<float> input;
    input.asShape({3, 768});
    input = head.forward(input);
    // cout << input;

}