#pragma once
#include "modules/linear.h"
#include "single_head_attention.h"
#include "tensor.h"
#include <cassert>
#include <vector>

template<typename T>
struct MultiHeadAttention{
    MultiHeadAttention(int num_head, int hidden_dim, bool mask) : num_head(num_head), heads(num_head, mask), x(num_head), hidden_dim(hidden_dim) {
        head_dim = hidden_dim / num_head;
    }

    Tensor<T> forward(const Tensor<T>& input) {
        int d_model = input.shape.back();
        assert(d_model % num_head == 0);
        auto qkv = in_proj.forward(input);
        for (int i = 0; i < num_head; i++) {
            int off_beg = i * head_dim;
            int off_end = off_beg + head_dim;
            x[i] = heads[i].forward(qkv.slice({{off_beg, off_end}}), qkv.slice({{hidden_dim + off_beg, hidden_dim + off_end}}), qkv.slice({{hidden_dim * 2 + off_beg, hidden_dim * 2 + off_end}}));
        }
        return out_proj.forward(Tensor<T>::concat(x));
    }

    std::vector<SingleHeadAttention<T>> heads;
    std::vector<Tensor<T>> x;
    Linear<T> in_proj, out_proj;
    int num_head;
    int hidden_dim;
    int head_dim;
};