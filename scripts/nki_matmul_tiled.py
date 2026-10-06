import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def matmul_kernel_tiled(a, b):
    M, K = a.shape
    K2, N = b.shape

    TILE_K = 128

    assert K == K2
    assert K % TILE_K == 0

    c_psum = nl.ndarray(
        (M, N),
        dtype=nl.float32,
        buffer=nl.psum,
    )

    for i in nl.static_range(K // TILE_K):

        a_tile = nl.ndarray(
            (TILE_K, M),
            dtype=a.dtype,
            buffer=nl.sbuf,
        )

        b_tile = nl.ndarray(
            (TILE_K, N),
            dtype=b.dtype,
            buffer=nl.sbuf,
        )

        # A starts as [M, K].
        # Take [M, TILE_K], then transpose -> [TILE_K, M].
        a_tile[...] = nl.load_transpose2d(
            a[:, i * TILE_K:(i + 1) * TILE_K]
        )

        # B already has layout [K, N].
        b_tile[...] = nl.load(
            b[i * TILE_K:(i + 1) * TILE_K, :]
        )

        nisa.nc_matmul(
            dst=c_psum,
            stationary=a_tile,
            moving=b_tile,
            accumulate=(i > 0),
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

    M = 64
    K = 256
    N = 64

    a = rng.standard_normal((M, K), dtype=np.float32)
    b = rng.standard_normal((K, N), dtype=np.float32)

    simulated_kernel = nki.simulate(matmul_kernel_tiled)
    c_nki = simulated_kernel(a, b)

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