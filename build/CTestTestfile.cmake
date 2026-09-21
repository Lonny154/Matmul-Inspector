# CMake generated Testfile for 
# Source directory: /home/daniel/projects/matmul-inspector
# Build directory: /home/daniel/projects/matmul-inspector/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(numeric-tests "/home/daniel/projects/matmul-inspector/build/numeric-tests")
set_tests_properties(numeric-tests PROPERTIES  _BACKTRACE_TRIPLES "/home/daniel/projects/matmul-inspector/CMakeLists.txt;51;add_test;/home/daniel/projects/matmul-inspector/CMakeLists.txt;0;")
add_test(matrix-tests "/home/daniel/projects/matmul-inspector/build/matrix-tests")
set_tests_properties(matrix-tests PROPERTIES  _BACKTRACE_TRIPLES "/home/daniel/projects/matmul-inspector/CMakeLists.txt;55;add_test;/home/daniel/projects/matmul-inspector/CMakeLists.txt;0;")
add_test(comparison-tests "/home/daniel/projects/matmul-inspector/build/comparison-tests")
set_tests_properties(comparison-tests PROPERTIES  _BACKTRACE_TRIPLES "/home/daniel/projects/matmul-inspector/CMakeLists.txt;59;add_test;/home/daniel/projects/matmul-inspector/CMakeLists.txt;0;")
add_test(cuda-matmul-tests "/home/daniel/projects/matmul-inspector/build/cuda-matmul-tests")
set_tests_properties(cuda-matmul-tests PROPERTIES  SKIP_RETURN_CODE "77" _BACKTRACE_TRIPLES "/home/daniel/projects/matmul-inspector/CMakeLists.txt;64;add_test;/home/daniel/projects/matmul-inspector/CMakeLists.txt;0;")
