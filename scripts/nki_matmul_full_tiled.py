import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def matmul_kernel_mn_tiled(a, b):
    M, K = a.shape
    K2, N = b.shape

    TILE_M = 128
    TILE_N = 512
    TILE_K = 128

    assert K == K2
    assert K % TILE_K == 0
    assert M % TILE_M == 0
    assert N % TILE_N == 0

    c = nl.ndarray(
        (M, N),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )

    for m in nl.static_range(M // TILE_M):
        for n in nl.static_range(N // TILE_N):

            c_psum = nl.ndarray(
                (TILE_M, TILE_N),
                dtype=nl.float32,
                buffer=nl.psum,
            )

            for k in nl.static_range(K // TILE_K):

                a_tile = nl.ndarray(
                    (TILE_K, TILE_M),
                    dtype=a.dtype,
                    buffer=nl.sbuf,
                )

                b_tile = nl.ndarray(
                    (TILE_K, TILE_N),
                    dtype=b.dtype,
                    buffer=nl.sbuf,
                )

                a_tile[...] = nl.load_transpose2d(
                    a[
                        m * TILE_M:(m + 1) * TILE_M,
                        k * TILE_K:(k + 1) * TILE_K
                    ]
                )

                b_tile[...] = nl.load(
                    b[
                        k * TILE_K:(k + 1) * TILE_K,
                        n * TILE_N:(n + 1) * TILE_N
                    ]
                )

                nisa.nc_matmul(
                    dst=c_psum,
                    stationary=a_tile,
                    moving=b_tile,
                    accumulate=(k > 0),
                )



            c_sb = nl.ndarray(
                (TILE_M, TILE_N),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )

            nisa.tensor_copy(
                dst=c_sb,
                src=c_psum,
            )

            nisa.dma_copy(
                dst=c[
                    m * TILE_M:(m + 1) * TILE_M,
                    n * TILE_N:(n + 1) * TILE_N
                ],
                src=c_sb,
            )

    return c



def main():
    rng = np.random.default_rng(42)

    M = 256
    K = 512
    N = 1024

    a = rng.standard_normal((M, K), dtype=np.float32)
    b = rng.standard_normal((K, N), dtype=np.float32)

    simulated_kernel = nki.simulate(matmul_kernel_mn_tiled)
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