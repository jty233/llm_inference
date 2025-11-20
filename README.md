# CPU-Only LLM Inference Engine

这是一个纯 CPU 实现的大型语言模型推理引擎，完全不依赖 CUDA 或其他 GPU 加速库，具有良好的通用性。

## 特性

- **纯 CPU 实现**：无需 GPU，可在任何支持 AVX2 指令集的 x86-64 CPU 上运行
- **高性能**：使用 SIMD 指令（AVX2/FMA）优化矩阵运算
- **多线程**：内置线程池支持并行计算
- **Qwen3 模型支持**：支持 Qwen3 模型的完整推理流程
- **Safetensors 格式**：支持加载 safetensors 格式的模型权重

## 构建依赖

- g++ (C++20 支持)
- AVX2 指令集支持的 CPU
- RE2 正则表达式库（已包含在 third-party 中）

## 构建方法

```bash
# 构建项目
./build.sh

# 运行推理
./llm_inference
```

## 项目结构

- `include/` - 头文件目录
- `src/` - 源文件目录
- `qwen3/` - Qwen3 模型实现
- `3rdparty/` - 第三方库 (RE2, BPE, JSON)
- `model/` - 模型文件目录

## 优化技术

1. **SIMD 优化**：使用 AVX2 指令加速矩阵乘法
2. **多线程计算**：通过线程池并行处理计算任务
3. **内存优化**：高效的张量存储和访问模式
4. **编译优化**：使用 -O3 -mavx2 -mfma -ffast-math 优化编译

## 说明

- 项目中虽然存在一些 Python 脚本包含 CUDA 相关代码，但核心 C++ 推理引擎完全基于 CPU 实现
- 模型推理性能依赖于 CPU 核心数量和时钟频率
- 对于大规模模型，推理速度会比 GPU 版本慢，但具有更好的通用性