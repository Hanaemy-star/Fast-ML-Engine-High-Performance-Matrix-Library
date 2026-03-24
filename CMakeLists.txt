cmake_minimum_required(VERSION 3.31)
project(fast_ml_engine)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

if(MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-mavx2 -mfma -march=native -fopenmp)
endif()

add_library(math_lib
        include/tensor.hpp
        src/tensor.cpp
)
target_include_directories(math_lib PUBLIC include)

target_link_libraries(math_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(fast_ml_engine main.cpp)

if(NOT MSVC)
    target_link_options(fast_ml_engine PRIVATE -fopenmp)
endif()

target_link_libraries(fast_ml_engine PRIVATE math_lib OpenMP::OpenMP_CXX)