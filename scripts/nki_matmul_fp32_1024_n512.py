
import numpy as np

from nki_tiled_matmul_bf16_512 import tiled_matmul_kernel


def main():
    size = 1024
    rng = np.random.default_rng(42)

    a = rng.standard_normal(
        (size, size), dtype=np.float32
    )
    b = rng.standard_normal(
        (size, size), dtype=np.float32
    )

    # Kernel expects A transposed: [K, M]
    a_t = np.ascontiguousarray(a.T)

    # Execute on physical Trainium hardware
    c_nki = tiled_matmul_kernel(a_t, b)
    c_ref = a @ b

    error = np.abs(c_nki - c_ref)

    print(f"Matrix size: {size}x{size}")
    print("NKI shape:", c_nki.shape)
    print("Reference shape:", c_ref.shape)
    print("Max absolute error:", error.max())
    print("Mean absolute error:", error.mean())
    print(
        "Allclose:",
        np.allclose(
            c_nki,
            c_ref,
            rtol=1e-3,
            atol=1e-3,
        ),
    )

    np.testing.assert_allclose(
        c_nki, c_ref, rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    main()
