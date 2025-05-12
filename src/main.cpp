#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "gpt2.h"
#include "gpt2_tokenizer.h"
#include "module.h"
#include "tensor.h"
#include <random>
using namespace std;
const int top_k = 40;
const int logits_size = 50257;
const int eot_token = 50256;
std::random_device rd;
std::mt19937 gen(rd());
int main()
{
    // mnist_mha_test();
    // ModelParse parser("../model.safetensors");
    // MultiHeadAttention<float> head(parser, "h.0.attn.c_attn", 12, 768);
    // Tensor<float> input;
    // input.asShape({3, 768});
    // input = head.forward(input);
    // cout << input;

    GPT2 gpt2("../model.safetensors");
    GPT2Tokenizer tokenizer("../vocab.json");
    string input;
    getline(cin,input);
    vector<int> tokens = tokenizer.encode(input);
    for (auto id : tokens) {
        cout << tokenizer.id2Str[id];
    }
    while (true) {
        auto res = gpt2.forward(tokens);
        vector<pair<double,int>> logits_probs;
        for (int i = 0; i < res.shape.back(); i++) {
            logits_probs.emplace_back(-res.at(i), i);
        }
        partial_sort(logits_probs.begin(), logits_probs.begin() + top_k, logits_probs.end());
        Tensor<float> probs;
        probs.asShape({top_k});
        for (int i = 0; i < top_k; i++) {
            probs.at(i) = -logits_probs[i].first;
        }
        probs = softmax(probs);
        std::vector<double> distribution(logits_size, 0.0);
        for (int j = 0; j < top_k; j++)
        {
            distribution[logits_probs[j].second] = probs.at(j);
        }
        std::discrete_distribution dist(distribution.begin(), distribution.end());
        int maxi = dist(gen);
        if (maxi == eot_token) {
            std::cout << "<|endoftext|>\n";
            break;
        }
        cout << tokenizer.decode(maxi);
        tokens.push_back(maxi);
    }


}