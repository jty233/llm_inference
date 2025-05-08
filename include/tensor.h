#pragma once

#include "thread_pool.h"
#include "time_calc.h"
#include <cassert>
#include <functional>
#include <future>
#include <iomanip>
#include <ios>
#include <iostream>
#include <ostream>
#include <utility>
#include <vector>

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

    T& at(std::vector<int> idxs) {
        while (idxs.size() < DIM_MAX) {
            idxs.insert(idxs.begin(), 0);
        }
        int idx = 0;
        for (int i = 0; i < DIM_MAX; i++) {
            assert(idxs[i] >= 0 && idxs[i] < shape[i]);
            idx += jump[i] * idxs[i];
        }
        return data[idx];
    }
    T at(std::vector<int> idxs) const {
        while (idxs.size() < DIM_MAX) {
            idxs.insert(idxs.begin(), 0);
        }
        int idx = 0;
        for (int i = 0; i < DIM_MAX; i++) {
            assert(idxs[i] >= 0 && idxs[i] < shape[i]);
            idx += jump[i] * idxs[i];
        }
        return data[idx];
    }

    T& atOffset(int idx) {
        return data[idx];
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

    Tensor<T> operator*(const Tensor<T>& oth) const {
        TimeCalcGuard g("operator *");
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
        forEachDim(new_shape, [&](std::vector<int> dim) {
            T tmp = 0;
            int addr_self = 0, addr_oth = 0;
            for (int i = 0; i < DIM_MAX; i++) {
                addr_self += (dim[i] % shape[i]) * jump[i];
                addr_oth += (dim[i] % oth.shape[i]) * oth.jump[i];
            }
            addr_self -= (dim[DIM_MAX - 1] % shape[DIM_MAX - 1]) * jump[DIM_MAX - 1];
            addr_oth -= (dim[DIM_MAX - 2] % oth.shape[DIM_MAX - 2]) * oth.jump[DIM_MAX - 2];

            for (int i = 0; i < u; i++) {
                tmp += data[addr_self] * oth.data[addr_oth];
                addr_self += jump[DIM_MAX - 1];
                addr_oth += oth.jump[DIM_MAX - 2];
            }
            res.at(dim) = tmp;
        }, 1);
        
        return res;
    }

    Tensor<T> operator+(const Tensor<T>& oth) {
        TimeCalcGuard g("operator +");
        assert(checkBroadCastValid(oth, DIM_MAX));
        std::vector<int> new_shape;
        for (int dim = 0; dim < DIM_MAX; dim++) {
            new_shape.push_back(std::max(shape[dim], oth.shape[dim]));
        }
        Tensor<T> res;
        res.asShape(new_shape);
        forEachDim(new_shape, [&](std::vector<int> dim) {
            std::vector<int> dim_self, dim_oth;
            for (int i = 0; i < DIM_MAX; i++) {
                dim_self.push_back(dim[i] % shape[i]);
                dim_oth.push_back(dim[i] % oth.shape[i]);
            }
            res.at(dim) = at(dim_self) + oth.at(dim_oth);
        }, 1);
        return res;
    }

    Tensor<T> transpose() {
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
        }, 1);
        return res;
    }

    friend std::ostream& operator<<(std::ostream& out, const Tensor<T>& ten) {
        out << std::fixed << std::setprecision(4);
        forEachDim(ten.shape, [&] (std::vector<int> dim) {
            if (dim.back() == 0) {
                out << '[';
            }
            else if (dim.back() == ten.shape.back() - 1) {
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