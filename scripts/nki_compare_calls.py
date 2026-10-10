
import time
import numpy as np

from nki_matmul import matmul_kernel
from nki_tiled_matmul import tiled_matmul_kernel

rng = np.random.default_rng(42)

def measure(name, kernel, a, b):
    print(f"\n{name}", flush=True)

    # Initial compilation/execution
    result = kernel(a, b)
    print("Output shape:", result.shape, flush=True)

    # Warmup
    for _ in range(3):
        kernel(a, b)

    samples = []
    for i in range(10):
        start = time.perf_counter()
        kernel(a, b)
        samples.append(time.perf_counter() - start)

    print(f"Median: {np.median(samples):.4f} seconds")
    print(f"Min: {np.min(samples):.4f} seconds")

a16 = rng.standard_normal((16, 16), dtype=np.float32)
b16 = rng.standard_normal((16, 16), dtype=np.float32)

a256 = rng.standard_normal((256, 256), dtype=np.float32)
b256 = rng.standard_normal((256, 256), dtype=np.float32)

measure("16x16", matmul_kernel, a16, b16)

measure(
    "256x256 tiled",
    tiled_matmul_kernel,
    np.ascontiguousarray(a256.T),
    b256,
)
