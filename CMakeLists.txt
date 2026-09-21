cmake_minimum_required(VERSION 3.16)

project(matmul_inspector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(numeric src/numeric.cpp)
target_include_directories(numeric PUBLIC include)

add_library(inspector
    src/matrix.cpp
    src/inspector.cpp
)
target_link_libraries(inspector PUBLIC numeric)

add_executable(matmul-inspector src/main.cpp)
target_link_libraries(matmul-inspector PRIVATE inspector)

include(CTest)
if(BUILD_TESTING)
    add_executable(numeric-tests tests/numeric_test.cpp)
    target_link_libraries(numeric-tests PRIVATE numeric)
    add_test(NAME numeric-tests COMMAND numeric-tests)

    add_executable(matrix-tests tests/matrix_test.cpp)
    target_link_libraries(matrix-tests PRIVATE inspector)
    add_test(NAME matrix-tests COMMAND matrix-tests)
endif()
