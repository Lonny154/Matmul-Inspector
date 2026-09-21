# matmul-inspector

Experimental C++ tool for inspecting matrix multiplication execution at a low level.

Current features:
- Row-major float matrix storage with optional padded rows
- Raw memory address inspection
- Trace individual matrix multiplication outputs back to contributing inputs
- Detect numerical and bitwise mismatches
- Display IEEE-754 bit patterns and hexadecimal representations
- Compute ULP distance between floating-point results

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
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Long-term goal:
Explore tooling for tracing numerical divergence from high-level tensor operations down toward low-level execution and memory behavior.
