cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0104)
    # Let modern CMake select the compiler's default architecture. Users can
    # override it with CMAKE_CUDA_ARCHITECTURES without requiring a local GPU.
    cmake_policy(SET CMP0104 NEW)
endif()

project(matmul_inspector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MATMUL_INSPECTOR_ENABLE_CUDA "Build the CUDA backend when a compiler is available" ON)
if(MATMUL_INSPECTOR_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(STATUS "CUDA compiler/toolkit unavailable; building CPU-only")
    endif()
endif()

add_library(numeric src/numeric.cpp)
target_include_directories(numeric PUBLIC include)

add_library(matrix src/matrix.cpp)
target_include_directories(matrix PUBLIC include)

add_library(inspector src/inspector.cpp)
target_link_libraries(inspector PUBLIC matrix numeric)

if(MATMUL_INSPECTOR_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    add_library(cuda_matmul src/cuda_matmul.cu)
    set_target_properties(cuda_matmul PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(cuda_matmul PUBLIC matrix)
    target_compile_definitions(cuda_matmul PUBLIC MATMUL_INSPECTOR_HAS_CUDA=1)
endif()

add_executable(matmul-inspector src/main.cpp)
target_link_libraries(matmul-inspector PRIVATE inspector)
if(TARGET cuda_matmul)
    target_link_libraries(matmul-inspector PRIVATE cuda_matmul)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(numeric-tests tests/numeric_test.cpp)
    target_link_libraries(numeric-tests PRIVATE numeric)
    add_test(NAME numeric-tests COMMAND numeric-tests)

    add_executable(matrix-tests tests/matrix_test.cpp)
    target_link_libraries(matrix-tests PRIVATE inspector)
    add_test(NAME matrix-tests COMMAND matrix-tests)

    add_executable(comparison-tests tests/comparison_test.cpp)
    target_link_libraries(comparison-tests PRIVATE inspector)
    add_test(NAME comparison-tests COMMAND comparison-tests)

    if(TARGET cuda_matmul)
        add_executable(cuda-matmul-tests tests/cuda_matmul_test.cpp)
        target_link_libraries(cuda-matmul-tests PRIVATE cuda_matmul numeric)
        add_test(NAME cuda-matmul-tests COMMAND cuda-matmul-tests)
        set_tests_properties(cuda-matmul-tests PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
