# matmul-inspector

Experimental C++ tool for inspecting matrix multiplication execution at a low level.

Current features:
- Row-major float matrix storage with optional padded rows
- Raw memory address inspection
- Trace individual matrix multiplication outputs back to contributing inputs
- Detect numerical and bitwise mismatches
- Display IEEE-754 bit patterns and hexadecimal representations
- Compute ULP distance between floating-point results
- Optional naive FP32 CUDA matmul and CPU/GPU result comparison

`Matrix(rows, cols)` creates tightly packed rows. Use
`Matrix(rows, cols, row_stride)` to add padding; the stride (leading dimension)
is measured in floats and must be at least `cols`. `row_stride()` reports the
physical distance between row starts. `data()` includes padding, while
`matrix(row, col)` accesses logical elements at `row * row_stride() + col`.

`matmul(a, b)` accepts either layout and returns packed rows. Use
`matmul(a, b, output_row_stride)` for a padded result. Inspector reports logical
elements with their actual storage addresses and skips padding.

Build and run the tests:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

CUDA is detected with CMake's `check_language(CUDA)`. If the compiler/toolkit is
unavailable, only the CPU targets and tests are built. To explicitly disable
CUDA, configure with `-DMATMUL_INSPECTOR_ENABLE_CUDA=OFF`. If `nvcc` is outside
your compiler search path, specify it on the first configuration:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
./build-cuda/matmul-inspector
```

The optional `cuda_matmul` target exports `MATMUL_INSPECTOR_HAS_CUDA=1` to its
consumers. With CMake 3.18 or newer, `CMAKE_CUDA_ARCHITECTURES` can select GPU
architectures; otherwise the compiler default is used. Building CUDA does not
require a GPU. The CUDA test returns CTest's configured skip code (77) when the
device/runtime is unavailable; failures after a successful runtime probe fail
the test. The demo also explains when GPU execution is skipped.

`cuda_matmul(a, b)` returns packed rows; `cuda_matmul(a, b, output_row_stride)`
returns padded rows. The backend uses the current/default CUDA device, allocates
with `cudaMalloc`, copies `rows * row_stride * sizeof(float)` bytes for each
input, and copies that full storage size back for the output. Output padding is
zero initialized. Each thread computes one logical output element with a
sequential FP32 dot product. Empty products return zeros without a kernel launch.
CUDA calls, launch errors, synchronization, and memory cleanup are checked;
execution failures throw exceptions. There is no CPU fallback for execution
errors and no tiling, shared memory, Tensor Cores, or cuBLAS.

The demo prints CPU and GPU results for the 2x3 times 3x2 example.
`Inspector::compare_results(expected, actual)` compares stored logical outputs
using the CPU result as the reference, reporting the first numerical mismatch
and first bitwise mismatch independently. Diagnostics include bit patterns, ULP
distance, absolute error, relative error, and the combined tolerance
`max(1e-6, 1e-5 * max(abs(actual), abs(expected)))`. The original convention of
reporting relative error as zero for a zero reference is retained; absolute
error and tolerance still determine numerical agreement.

No fast-math flags are added. Default CUDA compilation may fuse multiply-add,
so bitwise equality is not required for numerical agreement. See NVIDIA's
[floating-point guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
and CMake's [optional language detection](https://cmake.org/cmake/help/latest/module/CheckLanguage.html).

Long-term goal:
Explore tooling for tracing numerical divergence from high-level tensor operations down toward low-level execution and memory behavior.
