cmake_minimum_required(VERSION 3.31)

set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe")

project(fast_ml_engine LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/openmp>)
else()
    add_compile_options(-mavx2 -mfma -march=native -fopenmp)
endif()

add_library(math_lib
        include/tensor.hpp
        src/tensor.cpp
        src/cuda_kernels.cu
)
target_include_directories(math_lib PUBLIC include)

target_link_libraries(math_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(fast_ml_engine
        main.cpp
        include/cuda_kernels.hpp
)

if(NOT MSVC)
    target_link_options(fast_ml_engine PRIVATE -fopenmp)
endif()

target_link_libraries(fast_ml_engine PRIVATE math_lib OpenMP::OpenMP_CXX)