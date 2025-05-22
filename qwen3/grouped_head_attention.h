#pragma once
#include "modules/linear.h"
#include "single_head_attention.h"
#include "tensor.h"
#include <cassert>
#include <cmath>
#include <vector>
#include "modules/RMS_norm.h"

template<typename T>
struct GroupedHeadAttention{
    GroupedHeadAttention(int num_attention_head, int num_kv_head, int hidden_dim, bool mask) : num_attention_head(num_attention_head), num_kv_head(num_kv_head),
         heads(num_attention_head, mask), k_cache(num_kv_head), v_cache(num_kv_head), x(num_attention_head), hidden_dim(hidden_dim) {
        assert(num_attention_head % num_kv_head == 0);
        per_group_num = num_attention_head / num_kv_head;
        for (int i = 0; i < num_attention_head; i++) {
            heads[i].k_cache = &k_cache[i / per_group_num];
            heads[i].v_cache = &v_cache[i / per_group_num];
        }
    }

    Tensor<T> forward(const Tensor<T>& input) {
        auto q = q_proj.forward(input);
        auto k = k_proj.forward(input);
        auto v = v_proj.forward(input);
        int d_model = q.shape.back();
        assert(d_model % num_attention_head == 0);
        int head_dim = d_model / num_attention_head;
        for (int i = 0; i < num_kv_head; i++) {
            std::vector<int> dim_range({i * head_dim, (i + 1) * head_dim});
            k_cache[i] = k_cache[i].concat(apply_RoPE(k_norm.forward(k.slice({dim_range}))), 1);
            v_cache[i] = v_cache[i].concat(v.slice({dim_range}), 1);
        }

        for (int i = 0; i < num_attention_head; i++) {
            std::vector<int> dim_range({i * head_dim, (i + 1) * head_dim});
            x[i] = heads[i].forward(apply_RoPE(q_norm.forward(q.slice({dim_range}))));
        }
        f_token_num += input.shape[Tensor<T>::DIM_MAX - 2];
        return out_proj.forward(Tensor<T>::concat_vec(x));
    }

    Tensor<T> apply_RoPE(const Tensor<T>& input) {
        Tensor<T> res;
        auto shape = res.shape;
        res.asShape(shape);
        int d = shape.back();
        shape.pop_back();
        Tensor<T>::forEachDim(shape, [&] (std::vector<int> dim) {
            int m = dim.back() + f_token_num;
            dim.push_back(0);
            int off = Tensor<T>::idxs2Offset(dim);
            for (int i = 0; i < d; i += 2) {
                double theta = std::pow(rope_base, -2. * (i / 2) / d); // NOLINT
                res.data[off + i] = input.data[off + i] * cos(m * theta) - input.data[off + i + 1] * sin(m * theta);
                res.data[off + i + 1] = input.data[off + i + 1] * cos(m * theta) + input.data[off + i] * sin(m * theta);
            }
        });
        return res;
    }

    std::vector<SingleHeadAttention<T>> heads;
    std::vector<Tensor<T>> k_cache, v_cache;
    std::vector<Tensor<T>> x;
    Linear<T> q_proj, k_proj, v_proj, out_proj;
    RMSNorm<T> k_norm, q_norm;
    int num_attention_head;
    int num_kv_head;
    int hidden_dim;
    int per_group_num;
    int f_token_num;
    double rope_base = 1e6;
};