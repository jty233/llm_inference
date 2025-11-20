#!/bin/bash

echo "尝试编译项目（不使用CUDA）"

# 编译不带CUDA的版本
g++ -std=c++20 -I. -I./include -I./3rdparty -I./3rdparty/re2 -DUSE_SIMD -mavx2 -mfma -ffast-math -O2 \
  src/main.cpp src/model_parse.cpp \
  -o llm_inference_cpu -L./3rdparty/build -lbpe.a

if [ $? -eq 0 ]; then
  echo "CPU版本编译成功！"
else
  echo "CPU版本编译失败。"
fi
