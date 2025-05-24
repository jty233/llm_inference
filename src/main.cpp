#include <algorithm>
#include <string>
#include <vector>
#include "qwen3/qwen3.h"
#include "qwen3/qwen3_tokenizer.h"
#include "module.h"
#include "tensor.h"
#include "time_calc.h"
#include <random>
using namespace std;
const int top_k = 10;
const int logits_size = 151936;
const int eot_token = 151645;
std::random_device rd;
std::mt19937 gen(rd());
int main()
{
    // Tensor<float> t({
    //     0,1,2,3,
    //     4,5,6,7,
    //     8,9,10,11
    // }, { 2, 6});
    // t = apply_RoPE(t, 1e6, 0);
    // cout << t;
    // t = apply_RoPE(t, 1e6, 2);
    // cout << t;
    // return 0;

    // GPT2 gpt2("../model/gpt2/model.safetensors");
    // GPT2Tokenizer tokenizer("../model/gpt2/vocab.json");
    Qwen3 qwen3("../model/qwen3/model.safetensors");
    Qwen3Tokenizer tokenizer("../model/qwen3/vocab.json");

    string input = "this part";
    // getline(cin,input);
    vector<int> tokens = tokenizer.encode(input);
    for (auto id : tokens) {
        cout << id << ',';
    }
    cout << endl;
    cout << "token nums:" << tokens.size() << endl;
    for (auto id : tokens) {
        cout << tokenizer.id2Str[id];
    }
    // vector<int> tokens{  574};
    int token_sum = 0;
    startTimeCalc("main");
    while (true) {
        auto res = qwen3.forward(tokens);
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
        // cout << maxi << ' ';
        // tokens.push_back(maxi);
        tokens = {949};
        token_sum++;
    }
    cout << "tokens per sec:" << token_sum / getTimeCalcSec();

}