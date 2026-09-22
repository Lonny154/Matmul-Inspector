cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0104)
    # Let modern CMake select the compiler's default architecture. Users can
    # override it with CMAKE_CUDA_ARCHITECTURES without requiring a local GPU.
    cmake_policy(SET CMP0104 NEW)
endif()

project(matmul_inspector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MATMUL_INSPECTOR_WARNINGS "Enable compiler warnings" ON)
option(MATMUL_INSPECTOR_GPU_TESTS "Register tests that execute CUDA kernels" ON)
if(MATMUL_INSPECTOR_WARNINGS)
    if(MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
    else()
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
    endif()
endif()

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

add_library(comparison src/comparison.cpp)
target_link_libraries(comparison PUBLIC matrix numeric)
add_library(benchmark src/benchmark.cpp)
target_include_directories(benchmark PUBLIC include)

add_library(inspector src/inspector.cpp)
target_link_libraries(inspector PUBLIC comparison)

set(MATMUL_INSPECTOR_CUDA_BUILT OFF)
if(MATMUL_INSPECTOR_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    set(MATMUL_INSPECTOR_CUDA_BUILT ON)
    add_library(cuda_matmul src/cuda_matmul.cu)
    set_target_properties(cuda_matmul PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(cuda_matmul PUBLIC matrix benchmark)
    target_compile_definitions(cuda_matmul PUBLIC MATMUL_INSPECTOR_HAS_CUDA=1)
    if(MATMUL_INSPECTOR_WARNINGS AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        if(MSVC)
            target_compile_options(cuda_matmul PRIVATE -Xcompiler=/W4)
        else()
            # NVCC emits nonstandard line directives in its generated host code.
            target_compile_options(cuda_matmul PRIVATE -Xcompiler=-Wall,-Wextra)
        endif()
    endif()
endif()

configure_file(cmake/build_options.hpp.in generated/build_options.hpp.in @ONLY)
find_package(Python3 COMPONENTS Interpreter QUIET)
configure_file(cmake/fp_unverified.hpp.in generated/fp_verification.hpp COPYONLY)
if(TARGET cuda_matmul AND Python3_Interpreter_FOUND)
    get_filename_component(cuda_bin_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    find_program(MATMUL_CUOBJDUMP cuobjdump HINTS "${cuda_bin_dir}")
    if(MATMUL_CUOBJDUMP)
        add_custom_target(verify-fp
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_fp.py"
                --binary $<TARGET_FILE:cuda_matmul> --cuobjdump "${MATMUL_CUOBJDUMP}"
                --header "${CMAKE_CURRENT_BINARY_DIR}/generated/fp_verification.hpp"
                --report "${CMAKE_CURRENT_BINARY_DIR}/fp_verification.json" --allow-unverified
            DEPENDS cuda_matmul "${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_fp.py"
            BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/fp_verification.json"
            VERBATIM)
    endif()
endif()
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>/build_options.hpp"
    INPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/build_options.hpp.in")
add_custom_target(build-provenance
    COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/generated/git_provenance.hpp"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/provenance.cmake"
    BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/generated/git_provenance.hpp"
    VERBATIM)
add_library(experiment src/experiment.cpp src/metadata.cpp)
target_link_libraries(experiment PUBLIC comparison benchmark)
target_include_directories(experiment PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/$<CONFIG>")
add_dependencies(experiment build-provenance)
if(TARGET verify-fp)
    add_dependencies(experiment verify-fp)
endif()

add_executable(matmul-inspector src/main.cpp src/cli.cpp)
target_link_libraries(matmul-inspector PRIVATE inspector experiment)
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

    add_executable(fp-tests tests/fp_test.cpp)
    target_link_libraries(fp-tests PRIVATE experiment)
    add_test(NAME fp-tests COMMAND fp-tests)
    set_tests_properties(fp-tests PROPERTIES LABELS "cpu")

    add_executable(experiment-tests tests/experiment_test.cpp)
    target_link_libraries(experiment-tests PRIVATE experiment)
    add_test(NAME experiment-tests COMMAND experiment-tests)
    set_tests_properties(numeric-tests matrix-tests comparison-tests experiment-tests PROPERTIES LABELS "cpu")

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME artifact-tests COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/artifact_test.py" $<TARGET_FILE:matmul-inspector>
            $<TARGET_FILE:experiment-tests> "${CMAKE_CURRENT_SOURCE_DIR}")
        set_tests_properties(artifact-tests PROPERTIES LABELS "cpu")
    endif()

    if(TARGET cuda_matmul AND MATMUL_INSPECTOR_GPU_TESTS)
        if(Python3_Interpreter_FOUND)
            add_test(NAME fp-replay-tests COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/fp_replay_test.py" $<TARGET_FILE:matmul-inspector>
                "${CMAKE_CURRENT_SOURCE_DIR}")
            set_tests_properties(fp-replay-tests PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")
        endif()
        add_executable(cuda-matmul-tests tests/cuda_matmul_test.cpp)
        target_link_libraries(cuda-matmul-tests PRIVATE cuda_matmul experiment numeric)
        add_test(NAME cuda-matmul-tests COMMAND cuda-matmul-tests)
        set_tests_properties(cuda-matmul-tests PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")
    endif()
endif()
