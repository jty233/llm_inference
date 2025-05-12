#pragma once

#include "thread_pool.h"
#include "time_calc.h"
#include <cassert>
#include <cmath>
#include <functional>
#include <future>
#include <iomanip>
#include <ios>
#include <iostream>
#include <ostream>
#include <utility>
#include <vector>
#ifdef USE_SIMD
#include <immintrin.h>
#endif

template<typename T>
class Tensor{
public:
    static const int DIM_MAX = 4;

    Tensor(std::vector<T> data, std::vector<int> shape) :  data(std::move(data)), shape(std::move(shape)) {
        initShape();
    }

    Tensor() = default;

    Tensor(Tensor&& oth) :  data(std::move(oth.data)), shape(std::move(oth.shape)), jump(std::move(oth.jump)) {
    }

    Tensor(const Tensor& oth) : data(oth.data), shape(oth.shape), jump(oth.jump) {
    }

    Tensor<T>& operator=(Tensor<T>&& oth) {
        data = std::move(oth.data);
        shape = std::move(oth.shape);
        jump = std::move(oth.jump);
        return *this;
    }

    Tensor<T>& operator=(const Tensor<T>& oth) {
        data = oth.data;
        shape = oth.shape;
        jump = oth.jump;
        return *this;
    }

    void load(std::vector<T> data, std::vector<int> shape) {
        this->data = std::move(data);
        this->shape = std::move(shape);
        initShape();
    }

    void asShape(std::vector<int> shape) {
        this->shape = std::move(shape);
        int size = 1;
        for (int i : this->shape)
            size *= i;
        data.clear();
        data.resize(size, 0);
        initShape();
    }

    template<typename... Args>
    T& at(Args&&... args) {
        return at({args...});
    }

    template<typename... Args>
    T at(Args&&... args) const {
        return at({args...});
    }

    T& at(const std::vector<int>& idxs) {
        int idx = 0;
        int off = DIM_MAX - idxs.size();
        for (int i = off; i < DIM_MAX; i++) {
            assert(idxs[i - off] >= 0 && idxs[i - off] < shape[i]);
            idx += jump[i] * idxs[i - off];
        }
        return data[idx];
    }
    T at(const std::vector<int>& idxs) const {
        int idx = 0;
        int off = DIM_MAX - idxs.size();
        for (int i = off; i < DIM_MAX; i++) {
            assert(idxs[i - off] >= 0 && idxs[i - off] < shape[i]);
            idx += jump[i] * idxs[i - off];
        }
        return data[idx];
    }

    T& atOffset(int idx) {
        return data[idx];
    }

    int idxs2Offset(const std::vector<int>& idxs) const {
        int idx = 0;
        int off = DIM_MAX - idxs.size();
        for (int i = off; i < DIM_MAX; i++) {
            assert(idxs[i - off] >= 0 && idxs[i - off] < shape[i]);
            idx += jump[i] * idxs[i - off];
        }
        return idx;
    }

    static void forEachDim(const std::vector<int>& dims, const std::function<void(std::vector<int>)>& fun, int thread_level = -1, std::vector<int> cur_dim = {}) {
        if (cur_dim.size() == dims.size()) {
            fun(cur_dim);
            return;
        }
        cur_dim.push_back(0);
        for (int i = 0; i < dims[cur_dim.size() - 1]; i++) {
            cur_dim.back() = i;
            if (cur_dim.size() + thread_level == dims.size()) {
                future_list.emplace_back(thread_pool.assign(forEachDim, dims, fun, thread_level, cur_dim));
            }
            else {
                forEachDim(dims, fun, thread_level, cur_dim);
            }
        }

        if (cur_dim.size() == 1) {
            for (auto& f : future_list) {
                f.get();
            }
            future_list.clear();
        }
    }

    static Tensor<T> concat(const std::vector<Tensor<T>>& tensors) {
        std::vector<int> shape = tensors[0].shape;
        int ori_dim = shape.back();
        int num = tensors.size();
        shape.back() *= num;

        Tensor<T> res;
        res.asShape(shape);
        forEachDim(shape, [&] (std::vector<int> dim) {
            auto& val = res.at(dim);
            int idx = dim.back() / ori_dim;
            dim.back() %= ori_dim;
            val = tensors[idx].at(dim);
        });
        return res;
    }

    Tensor<T> matMul(const Tensor<T>& oth) const {
        assert(shape[DIM_MAX - 1] == oth.shape[DIM_MAX - 2]);
        assert(checkBroadCastValid(oth, DIM_MAX - 2));

        int k = shape[DIM_MAX - 2], u = shape[DIM_MAX - 1], v = oth.shape[DIM_MAX - 1];
        std::vector<int> new_shape;
        for (int i = 0; i < DIM_MAX - 2; i++) {
            new_shape.push_back(std::max(shape[i], oth.shape[i]));
        }
        new_shape.push_back(k);
        new_shape.push_back(v);
        Tensor<T> res;
        res.asShape(new_shape);
        new_shape.pop_back();
        forEachDim(new_shape, [&](std::vector<int> dim) {
            int addr_self = 0, addr_oth = 0, addr_res = 0;
            for (int i = 0; i < DIM_MAX - 1; i++) {
                addr_self += (dim[i] % shape[i]) * jump[i];
                addr_res += dim[i] * res.jump[i];
            }
            for (int i = 0; i < DIM_MAX - 2; i++) {
                addr_oth += (dim[i] % oth.shape[i]) * oth.jump[i];
            }
            for (int i = 0; i < u; i++) {
                auto val = data[addr_self + i];
                #ifdef USE_SIMD
                __m256 a_vec = _mm256_set1_ps(val);
                int j = 0;
                for (; j + 7 < v; j += 8) {
                    __m256 b_vec = _mm256_loadu_ps(oth.data.data() + addr_oth + j);
                    __m256 c_vec = _mm256_loadu_ps(res.data.data() + addr_res + j);
                    c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                    _mm256_storeu_ps(res.data.data() + addr_res + j, c_vec);
                }
                for (; j < v; ++j) {
                    res.data[addr_res + j]  += val * oth.data[addr_oth + j];
                }
                #else
                for (int j = 0; j < v; j++) {
                    res.data[addr_res + j] += val * oth.data[addr_oth + j];
                }
                #endif
                addr_oth += oth.jump[DIM_MAX - 2];
            }
        }, 0);
        
        return res;
    }

    Tensor<T> matMulTranspos(const Tensor<T>& oth_T) const {
        assert(shape[DIM_MAX - 1] == oth_T.shape[DIM_MAX - 1]);
        assert(checkBroadCastValid(oth_T, DIM_MAX - 2));

        int k = shape[DIM_MAX - 2], u = shape[DIM_MAX - 1], v = oth_T.shape[DIM_MAX - 2];
        std::vector<int> new_shape;
        for (int i = 0; i < DIM_MAX - 2; i++) {
            new_shape.push_back(std::max(shape[i], oth_T.shape[i]));
        }
        new_shape.push_back(k);
        new_shape.push_back(v);
        Tensor<T> res;
        res.asShape(new_shape);
        new_shape.pop_back();
        forEachDim(new_shape, [&](std::vector<int> dim) {
            int addr_self = 0, addr_oth = 0, addr_res = 0;
            for (int i = 0; i < DIM_MAX - 1; i++) {
                addr_self += (dim[i] % shape[i]) * jump[i];
                addr_res += dim[i] * res.jump[i];
            }
            for (int i = 0; i < DIM_MAX - 2; i++) {
                addr_oth += (dim[i] % oth_T.shape[i]) * oth_T.jump[i];
            }
            for (int i = 0; i < v; i++) {
                __m256 sum256 = _mm256_setzero_ps();
                int j = 0;
                for (; j + 7 < u; j += 8) {
                    __m256 a_vec = _mm256_loadu_ps(data.data() + addr_self + j);
                    __m256 b_vec = _mm256_loadu_ps(oth_T.data.data() + addr_oth + j);
                    sum256 = _mm256_fmadd_ps(a_vec, b_vec, sum256);
                }
                float sum = 0.0f;
                alignas(32) float buf[8];
                _mm256_store_ps(buf, sum256);
                for (int t = 0; t < 8; ++t) sum += buf[t];
                for (; j < u; ++j) sum += data[addr_self + j] * oth_T.data[addr_oth + j];
                res.data[addr_res + i] = sum;

                addr_oth += oth_T.jump[DIM_MAX - 2];
            }
        }, 0);
        
        return res;
    }

    Tensor<T> elementWiseMul(const Tensor<T>& oth) const {
        assert(checkBroadCastValid(oth, DIM_MAX));
        assert(shape.back() == oth.shape.back());
        std::vector<int> new_shape;
        for (int dim = 0; dim < DIM_MAX; dim++) {
            new_shape.push_back(std::max(shape[dim], oth.shape[dim]));
        }
        Tensor<T> res;
        res.asShape(new_shape);
        int last_dim = new_shape.back();
        new_shape.pop_back();
        forEachDim(new_shape, [&](std::vector<int> dim) {
            std::vector<int> dim_self, dim_oth;
            int addr_self = 0, addr_oth = 0, addr_res = 0;
            for (int i = 0; i < DIM_MAX - 1; i++) {
                addr_self += (dim[i] % shape[i]) * jump[i];
                addr_oth += (dim[i] % oth.shape[i]) * oth.jump[i];
                addr_res += dim[i] * res.jump[i];
            }
            for (int i = 0; i < last_dim; i++) {
                res.data[addr_res++] = data[addr_self++] * oth.data[addr_oth++];
            }
        }, -1);
        return res;
    }

    Tensor<T> mean() const {
        auto new_shape = shape;
        int last_dim = new_shape.back();
        new_shape.back() = 1;
        Tensor<T> mean;
        mean.asShape(new_shape);
        Tensor<T>::forEachDim(new_shape, [&](std::vector<int> dim) {
            T sum = 0;
            int off = idxs2Offset(dim);
            for (int i = 0; i < last_dim; i++) {
                sum += data[off + i];
            }
            mean.at(dim) = sum / last_dim;
        });
        return mean;
    }

    Tensor<T> sqrt(double eps = 1e-5) const {
        Tensor<T> res;
        res.asShape(shape);
        Tensor<T>::forEachDim(shape, [&](std::vector<int> dim) {
            res.at(dim) = std::sqrt(at(dim) + eps);
        });
        return res;
    }

    Tensor<T> operator+(const Tensor<T>& oth) {
        assert(checkBroadCastValid(oth, DIM_MAX));
        std::vector<int> new_shape;
        for (int dim = 0; dim < DIM_MAX; dim++) {
            new_shape.push_back(std::max(shape[dim], oth.shape[dim]));
        }
        Tensor<T> res;
        res.asShape(new_shape);
        int last_dim = new_shape.back();
        new_shape.pop_back();
        forEachDim(new_shape, [&](std::vector<int> dim) {
            std::vector<int> dim_self, dim_oth;
            int addr_self = 0, addr_oth = 0, addr_res = 0;
            for (int i = 0; i < DIM_MAX - 1; i++) {
                addr_self += (dim[i] % shape[i]) * jump[i];
                addr_oth += (dim[i] % oth.shape[i]) * oth.jump[i];
                addr_res += dim[i] * res.jump[i];
            }
            for (int i = 0; i < last_dim; i++) {
                res.data[addr_res++] = data[addr_self + (i % shape.back())] + oth.data[addr_oth + (i % oth.shape.back())];
            }
        }, -1);
        return res;
    }

    Tensor<T> operator-(const Tensor<T>& oth) const {
        assert(checkBroadCastValid(oth, DIM_MAX));
        std::vector<int> new_shape;
        for (int dim = 0; dim < DIM_MAX; dim++) {
            new_shape.push_back(std::max(shape[dim], oth.shape[dim]));
        }
        Tensor<T> res;
        res.asShape(new_shape);
        int last_dim = new_shape.back();
        new_shape.pop_back();
        forEachDim(new_shape, [&](std::vector<int> dim) {
            std::vector<int> dim_self, dim_oth;
            int addr_self = 0, addr_oth = 0, addr_res = 0;
            for (int i = 0; i < DIM_MAX - 1; i++) {
                addr_self += (dim[i] % shape[i]) * jump[i];
                addr_oth += (dim[i] % oth.shape[i]) * oth.jump[i];
                addr_res += dim[i] * res.jump[i];
            }
            for (int i = 0; i < last_dim; i++) {
                res.data[addr_res++] = data[addr_self + (i % shape.back())] - oth.data[addr_oth + (i % oth.shape.back())];
            }
        }, -1);
        return res;
    }

    Tensor<T> operator/ (const Tensor<T>& oth) const {
        assert(checkBroadCastValid(oth, DIM_MAX));
        std::vector<int> new_shape;
        for (int dim = 0; dim < DIM_MAX; dim++) {
            new_shape.push_back(std::max(shape[dim], oth.shape[dim]));
        }
        Tensor<T> res;
        res.asShape(new_shape);
        int last_dim = new_shape.back();
        new_shape.pop_back();
        forEachDim(new_shape, [&](std::vector<int> dim) {
            std::vector<int> dim_self, dim_oth;
            int addr_self = 0, addr_oth = 0, addr_res = 0;
            for (int i = 0; i < DIM_MAX - 1; i++) {
                addr_self += (dim[i] % shape[i]) * jump[i];
                addr_oth += (dim[i] % oth.shape[i]) * oth.jump[i];
                addr_res += dim[i] * res.jump[i];
            }
            for (int i = 0; i < last_dim; i++) {
                res.data[addr_res++] = data[addr_self + (i % shape.back())] / oth.data[addr_oth + (i % oth.shape.back())];
            }
        }, -1);
        return res;
    }

    void operator/=(float v) {
        forEachDim(shape, [&](std::vector<int> dim) {
            at(dim) /= v;
        });
    }

    Tensor<T> slice(const std::vector<std::pair<int, int>>& args) const {
        int off = DIM_MAX - args.size();
        std::vector<int> slice_shape(shape.begin(), shape.begin() + off);
        for (auto [beg, end] : args) {
            slice_shape.push_back(end - beg);
        }
        Tensor<T> res;
        res.asShape(slice_shape);
        forEachDim(slice_shape, [&](std::vector<int> dim) {
            std::vector<int> ori_dim(dim);
            for (int i = off; i < ori_dim.size(); i++) {
                ori_dim[i] += args[i - off].first;
            }
            res.at(dim) = at(ori_dim);
        }, 1);
        return res;
    }

    Tensor<T> transpose() const {
        TimeCalcGuard g("transpose");
        for (int i = 0; i < DIM_MAX - 2; i++) {
            assert(shape[i] == 1);
        }
        std::vector<int> new_shape = shape;
        std::swap(new_shape[DIM_MAX - 1], new_shape[DIM_MAX - 2]);
        Tensor<T> res;
        res.asShape(new_shape);
        forEachDim(new_shape, [&](std::vector<int> dim) {
            std::vector<int> ori_dim = dim;
            std::swap(ori_dim[DIM_MAX - 1], ori_dim[DIM_MAX - 2]);
            res.at(dim) = at(ori_dim);
        }, -1);
        return res;
    }

    friend std::ostream& operator<<(std::ostream& out, const Tensor<T>& ten) {
        out << std::fixed << std::setprecision(4);
        forEachDim(ten.shape, [&] (std::vector<int> dim) {
            if (dim.back() == 0) {
                out << '[';
            }
            if (dim.back() == ten.shape.back() - 1) {
                out << ten.at(dim) << ']' << std::endl;
                return;
            }
            out << ten.at(dim) << ',';
        });
        return out;
    }

    void initShape() {
        while (shape.size() < DIM_MAX) {
            shape.insert(shape.begin(), 1);
        }
        jump.resize(DIM_MAX);
        jump.back() = 1;
        for (int i = DIM_MAX - 2; i >= 0; i--) {
            jump[i] = jump[i + 1] * shape[i + 1];
        }
    }

    bool checkBroadCastValid(const Tensor<T>& oth, int checkDim) const {
        for (int idx = 0; idx < checkDim; idx++) {
            int max_shape = std::max(shape[idx], oth.shape[idx]);
            int min_shape = std::min(shape[idx], oth.shape[idx]);
            if (max_shape / min_shape * min_shape != max_shape) {
                return false;
            }
        }
        return true;
    }

    std::vector<T> data;
    std::vector<int> shape;
    std::vector<int> jump;
    static inline ThreadPool& thread_pool = ThreadPool::getInstance();
    static inline std::vector<std::future<void>> future_list;
};