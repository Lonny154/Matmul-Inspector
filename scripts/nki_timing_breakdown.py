
import time
import numpy as np

from nki_tiled_matmul import tiled_matmul_kernel

rng = np.random.default_rng(42)
a = rng.standard_normal((256, 256), dtype=np.float32)
b = rng.standard_normal((256, 256), dtype=np.float32)
a_t = np.ascontiguousarray(a.T)

for i in range(5):
    start = time.perf_counter()
    result = tiled_matmul_kernel(a_t, b)
    elapsed = time.perf_counter() - start

    print(f"Call {i + 1}: {elapsed:.4f} seconds", flush=True)

print("Correct:", np.allclose(result, a @ b, rtol=1e-3, atol=1e-3))
