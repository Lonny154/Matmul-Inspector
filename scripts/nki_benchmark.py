
import time

import nki
import numpy as np

from nki_matmul import matmul_kernel

rng = np.random.default_rng(42)

a = rng.standard_normal((16, 16), dtype=np.float32)
b = rng.standard_normal((16, 16), dtype=np.float32)

# Compile and validate separately from timing.
c = matmul_kernel(a, b)
reference = a @ b

np.testing.assert_allclose(
    c, reference, rtol=1e-4, atol=1e-4
)
print("Correctness: PASS")

# Warm up the compiled kernel.
for _ in range(10):
    matmul_kernel(a, b)

# Measure repeated end-to-end calls.
samples_us = []

for _ in range(100):
    start = time.perf_counter_ns()
    matmul_kernel(a, b)
    end = time.perf_counter_ns()

    samples_us.append((end - start) / 1000)

samples_us = np.array(samples_us)

print("Matrix: 16x16")
print("Median call latency (us):", np.median(samples_us))
print("p99 call latency (us):", np.percentile(samples_us, 99))
print("Min call latency (us):", np.min(samples_us))
