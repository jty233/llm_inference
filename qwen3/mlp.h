#pragma once
#include "modules/linear.h"
#include "module.h"
#include "tensor.h"
template<typename T>
struct MLP {

    Tensor<T> forward(const Tensor<T>& input) {
        return down.forward(silu(gate.forward(input)).elementWiseMul(up.forward(input)));
    }

    Linear<T> gate, up, down;
};