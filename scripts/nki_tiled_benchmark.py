
import time

import numpy as np
from nki_tiled_matmul import tiled_matmul_kernel

rng = np.random.default_rng(42)
a = rng.standard_normal((256, 256), dtype=np.float32)
b = rng.standard_normal((256, 256), dtype=np.float32)
a_t = np.ascontiguousarray(a.T)

# Compile, execute, and validate before timing.
result = tiled_matmul_kernel(a_t, b)
np.testing.assert_allclose(
    result, a @ b, rtol=1e-3, atol=1e-3
)

# Warm up to exclude initial compilation.
for _ in range(10):
    tiled_matmul_kernel(a_t, b)

samples_us = []

for _ in range(100):
    start = time.perf_counter_ns()
    tiled_matmul_kernel(a_t, b)
    end = time.perf_counter_ns()

    samples_us.append((end - start) / 1000)

print("Matrix size: 256x256")
print(f"Median call time: {np.median(samples_us):.2f} us")
print(f"p99 call time: {np.percentile(samples_us, 99):.2f} us")
print(f"Minimum call time: {np.min(samples_us):.2f} us")
