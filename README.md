# matmul-inspector

Experimental C++ tool for inspecting matrix multiplication execution at a low level.

Current features:
- Contiguous float matrix storage
- Raw memory address inspection
- Trace individual matrix multiplication outputs back to contributing inputs
- Detect numerical and bitwise mismatches
- Display IEEE-754 bit patterns and hexadecimal representations
- Compute ULP distance between floating-point results

Long-term goal:
Explore tooling for tracing numerical divergence from high-level tensor operations down toward low-level execution and memory behavior.
