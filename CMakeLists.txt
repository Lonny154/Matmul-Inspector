cmake_minimum_required(VERSION 3.16)

project(matmul_inspector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul-inspector
    src/main.cpp
    src/matrix.cpp
    src/inspector.cpp
)

target_include_directories(matmul-inspector PRIVATE include)