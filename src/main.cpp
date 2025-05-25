#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "qwen3/qwen3.h"
#include "module.h"
#include "tensor.h"
#include "time_calc.h"
#include <random>
#include "tokenizer.h"
using namespace std;
const int top_k = 40;
const int logits_size = 151936;
const int eot_token = 151645;
std::random_device rd;
std::mt19937 gen(rd());
int main()
{
    Qwen3 qwen3("../model/qwen3/model.safetensors");
    Tokenizer tokenizer;
    tokenizer.init("../model/qwen3/merges.txt", "../model/qwen3/vocab.json");

    string input = "你是什么模型";
    // // getline(cin,input);
    vector<int> token_input = {151644,872,198,151645,198};
    vector<int> tokens = tokenizer.encode(input);
    token_input.insert(token_input.end(), tokens.begin(), tokens.end());
    token_input.insert(token_input.end(), {151644,77091, 198, 151667, 198, 151668});
    

    // for (auto id : tokens) {
    //     cout << id << ',';
    // }
    // cout << endl;
    cout << "token nums:" << token_input.size() << endl;
    // cout << tokenizer.decode(token_input);
    int token_sum = 0;
    startTimeCalc("main");
    while (true) {
        auto res = qwen3.forward(token_input);
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
        else if (maxi == 151667) {
            std::cout << "<think>";
        }
        else if (maxi == 151668) {
            cout << "</think>";
        }
        else {
            cout << tokenizer.decode(maxi);
        }
        // cout << maxi << ' ';
        // token_input.push_back(maxi);
        token_input = {maxi};
        token_sum++;
    }
    cout << "tokens per sec:" << token_sum / getTimeCalcSec();

}