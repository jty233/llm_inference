#!/bin/bash

echo "Building CPU-only LLM Inference Engine..."

# Create build directory if it doesn't exist
mkdir -p 3rdparty/build
cd 3rdparty/build

# Build bpe library
if [ ! -f "libbpe.a" ]; then
    echo "Building BPE library..."
    g++ -O3 -std=c++20 -I.. -I../re2 -c ../bpe/bpe.cc
    ar rcs libbpe.a bpe.o
fi

cd ../../

# Build the main project
echo "Building main project..."
g++ -O3 -std=c++20 -mavx2 -mfma -ffast-math -Iinclude -I3rdparty -I3rdparty/re2 -I. src/main.cpp src/model_parse.cpp 3rdparty/build/libbpe.a 3rdparty/re2/obj/libre2.a -o llm_inference

echo "Build completed!"
echo "Run './llm_inference' to start the Qwen3 model inference."