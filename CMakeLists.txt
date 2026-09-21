cmake_minimum_required(VERSION 3.16)

project(matmul_inspector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(numeric src/numeric.cpp)
target_include_directories(numeric PUBLIC include)

add_executable(matmul-inspector
    src/main.cpp
    src/matrix.cpp
    src/inspector.cpp
)

target_link_libraries(matmul-inspector PRIVATE numeric)

include(CTest)
if(BUILD_TESTING)
    add_executable(numeric-tests tests/numeric_test.cpp)
    target_link_libraries(numeric-tests PRIVATE numeric)
    add_test(NAME numeric-tests COMMAND numeric-tests)
endif()
