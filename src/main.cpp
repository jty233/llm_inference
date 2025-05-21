#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "gpt2/gpt2.h"
#include "gpt2/gpt2_tokenizer.h"
#include "module.h"
#include "tensor.h"
#include "time_calc.h"
#include "mnist/mnist_test.h"
#include <random>
using namespace std;
const int top_k = 40;
const int logits_size = 50257;
const int eot_token = 50256;
std::random_device rd;
std::mt19937 gen(rd());
int main()
{
    ModelParse parser("../model/qwen3/model.safetensors");
    return 0;

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