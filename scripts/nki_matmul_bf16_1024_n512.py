import numpy as np
import ml_dtypes

from nki_tiled_matmul_bf16_512 import tiled_matmul_kernel


def main():
    size = 1024
    rng = np.random.default_rng(42)

    # Generate original FP32 inputs
    a_fp32 = rng.standard_normal(
        (size, size), dtype=np.float32
    )
    b_fp32 = rng.standard_normal(
        (size, size), dtype=np.float32
    )

    # Convert inputs to BF16
    a = a_fp32.astype(ml_dtypes.bfloat16)
    b = b_fp32.astype(ml_dtypes.bfloat16)

    # Kernel expects transposed A
    a_t = np.ascontiguousarray(a.T)

    # Execute on Trainium1
    c_nki = tiled_matmul_kernel(a_t, b)

    # Reference using BF16-rounded inputs
    c_bf16_ref = (
        a.astype(np.float32)
        @ b.astype(np.float32)
    )

    # Reference using original FP32 inputs
    c_fp32_ref = a_fp32 @ b_fp32

    output = np.asarray(c_nki, dtype=np.float32)

    kernel_error = np.abs(output - c_bf16_ref)
    precision_error = np.abs(output - c_fp32_ref)

    print("Matrix size:", size)
    print("Input dtype:", a.dtype)
    print("Output dtype:", output.dtype)

    print("Kernel max error:", kernel_error.max())
    print("Kernel mean error:", kernel_error.mean())

    print("FP32 max error:", precision_error.max())
    print("FP32 mean error:", precision_error.mean())

    np.testing.assert_allclose(
        output,
        c_bf16_ref,
        rtol=1e-3,
        atol=1e-3,
    )

    print("Correctness: PASS")


if __name__ == "__main__":
    main()