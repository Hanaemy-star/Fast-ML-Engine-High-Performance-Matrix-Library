cmake_minimum_required(VERSION 3.31)
project(fast_ml_engine)

set(CMAKE_CXX_STANDARD 20)


add_library(math_lib
        include/tensor.hpp
        src/tensor.cpp
)

target_include_directories(math_lib PUBLIC include)

add_executable(fast_ml_engine main.cpp)

target_link_libraries(fast_ml_engine PRIVATE math_lib)