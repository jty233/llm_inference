cmake_minimum_required(VERSION 3.10)
project(bpe CXX)

# ------------------------------
# 1. 强制所有库都只编译为静态库
# ------------------------------
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build all targets as static libraries" FORCE)

# ------------------------------
# 2. C++ 基本要求
# ------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------
# 3. ABI 版本号（可选，仅作标记）
# ------------------------------
set(SONAME 10)

# ==============================
# 4. 手动收集 RE2 底下的所有源文件（.cc/.c），
#    并排除掉 re2/fuzzing 目录下及 fuzz.cc 文件
# ==============================
file(GLOB_RECURSE RE2_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/re2/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/re2/*.c"
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 过滤掉凡是路径中包含 "re2/fuzzing/" 或 文件名以 fuzz 开头 的文件
list(FILTER RE2_SRCS EXCLUDE REGEX ".*/re2/fuzzing/.*")
list(FILTER RE2_SRCS EXCLUDE REGEX ".*/re2/util/fuzz.*\\.cc$")

# 生成一个名叫 re2_objects 的 OBJECT 库
add_library(re2_objects OBJECT ${RE2_SRCS})

# RE2 的头文件都在 re2/ 目录下，所以一定要把它加进 include path
target_include_directories(re2_objects
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/re2
)

# ==============================
# 5. 把 re2_objects 里的所有对象，再加上 bpe.cc，一起打包成 bpe 静态库
# ==============================
add_library(bpe STATIC
    $<TARGET_OBJECTS:re2_objects>   # RE2 编译后得到的所有 .o
    bpe/bpe.cc
    bpe/bpe.h
)

# 暴露给外部使用者的 include 路径：
#   - re2/        （RE2 的所有头文件都在这里）
#   - bpe/  （bpe.h 在这里）
target_include_directories(bpe
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/re2
        ${CMAKE_CURRENT_SOURCE_DIR}/bpe
)