cmake_minimum_required(VERSION 3.10.0)
project(llm_inference VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include 3rdparty mnist gpt2 ${OpenCV_INCLUDE_DIRS})
add_compile_definitions(USE_SIMD)
add_compile_options(-mavx2 -mfma -ffast-math)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})