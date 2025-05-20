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
#include "time_calc.h"
#include "mnist_test.h"
#include <random>
using namespace std;
const int top_k = 40;
const int logits_size = 50257;
const int eot_token = 50256;
std::random_device rd;
std::mt19937 gen(rd());
int main()
{
    mnist_test();
    mnist_mha_test();
    // Tensor<float> a({
    //     1, 2, 3,
    //     4, 5, 6
    // }, {2, 3});
    // Tensor<float> b({
    //     7, 8, 9,
    //     10, 11, 12,
    //     13, 14, 15, 16
    // }, {2, 5});
    // auto res = a.concat(b, 0);
    // cout << res << endl;
    // mnist_mha_test();
    // ModelParse parser("../model.safetensors");
    // MultiHeadAttention<float> head(parser, "h.0.attn.c_attn", 12, 768);
    // Tensor<float> input;
    // input.asShape({3, 768});
    // input = head.forward(input);
    // cout << input;

    GPT2 gpt2("../model/gpt2/model.safetensors");
    GPT2Tokenizer tokenizer("../model/gpt2/vocab.json");
    string input;
    getline(cin,input);
    vector<int> tokens = tokenizer.encode(input);
    cout << "token nums:" << tokens.size() << endl;
    for (auto id : tokens) {
        cout << tokenizer.id2Str[id];
    }
    int token_sum = 0;
    startTimeCalc("main");
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
        // tokens.push_back(maxi);
        tokens = {maxi};
        token_sum++;
    }
    cout << "tokens per sec:" << token_sum / getTimeCalcSec();

}