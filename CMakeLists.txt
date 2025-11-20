cmake_minimum_required(VERSION 3.10.0)
project(llm_inference VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_STANDARD 17)
set(CUDA_STANDARD_REQUIRED ON)


include_directories(include 3rdparty . 3rdparty/re2)
add_compile_definitions(USE_SIMD)
add_compile_options(-mavx2 -mfma -ffast-math)

find_package(CUDA REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "*.cu" "src/*.cu")

add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

target_link_directories(${PROJECT_NAME} PUBLIC ./3rdparty/build)
target_link_libraries(${PROJECT_NAME} libbpe.a)

# 链接CUDA库
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
endif()