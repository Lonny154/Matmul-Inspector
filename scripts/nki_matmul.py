import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def matmul_kernel(a, b):
    M, K = a.shape
    K2, N = b.shape

    assert K == K2

    # Tensor Engine wants the contraction dimension K
    # in the partition dimension of SBUF.
    a_sb = nl.ndarray((K, M), dtype=a.dtype, buffer=nl.sbuf)
    b_sb = nl.ndarray((K, N), dtype=b.dtype, buffer=nl.sbuf)

    # Load A transposed so a_sb has shape [K, M].
    a_sb[...] = nl.load_transpose2d(a)
    b_sb[...] = nl.load(b)

    # Tensor Engine writes matmul output into PSUM.
    c_psum = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)

    nisa.nc_matmul(
        dst=c_psum,
        stationary=a_sb,
        moving=b_sb,
    )

    # PSUM cannot be stored directly to HBM.
    # Copy PSUM -> SBUF first.
    c_sb = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=c_sb, src=c_psum)

    c = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=c, src=c_sb)

    return c


def main():
    rng = np.random.default_rng(42)

    a = rng.standard_normal((16, 16), dtype=np.float32)
    b = rng.standard_normal((16, 16), dtype=np.float32)

    c_nki = matmul_kernel(a, b)

    c_ref = a @ b

    error = np.abs(c_nki - c_ref)

    print("NKI result shape:", c_nki.shape)
    print("Reference shape:", c_ref.shape)
    print("max abs error:", error.max())
    print("mean abs error:", error.mean())
    print(
        "allclose:",
        np.allclose(c_nki, c_ref, rtol=1e-4, atol=1e-4),
    )


if __name__ == "__main__":
    main()