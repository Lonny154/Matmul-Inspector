
import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def tiled_matmul_v2(a_t, b):
    K, M = a_t.shape
    K2, N = b.shape

    assert K == K2
    assert M % 128 == 0
    assert N % 128 == 0
    assert K % 128 == 0

    TM = 128
    TN = 128
    TK = 128

    result = nl.ndarray(
        (M, N),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )

    for mi in nl.affine_range(M // TM):

        # EXPERIMENT:
        # Preload A tiles once per M-block.
        # Reuse them across all N-blocks.
        a_tiles = []

        for ki in nl.static_range(K // TK):
            a_sb = nl.ndarray(
                (TK, TM),
                dtype=a_t.dtype,
                buffer=nl.sbuf,
            )

            nisa.dma_copy(
                dst=a_sb,
                src=a_t[
                    ki * TK:(ki + 1) * TK,
                    mi * TM:(mi + 1) * TM,
                ],
            )

            a_tiles.append(a_sb)

        for ni in nl.affine_range(N // TN):

            acc = nl.ndarray(
                (TM, TN),
                dtype=nl.float32,
                buffer=nl.psum,
            )

            for ki in nl.static_range(K // TK):

                # Only B is loaded in this loop.
                b_sb = nl.ndarray(
                    (TK, TN),
                    dtype=b.dtype,
                    buffer=nl.sbuf,
                )

                nisa.dma_copy(
                    dst=b_sb,
                    src=b[
                        ki * TK:(ki + 1) * TK,
                        ni * TN:(ni + 1) * TN,
                    ],
                )

                nisa.nc_matmul(
                    dst=acc,
                    stationary=a_tiles[ki],
                    moving=b_sb,
                    accumulate=(ki > 0),
                )

            c_sb = nl.ndarray(
                (TM, TN),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )

            nisa.tensor_copy(
                dst=c_sb,
                src=acc,
            )

            nisa.dma_copy(
                dst=result[
                    mi * TM:(mi + 1) * TM,
                    ni * TN:(ni + 1) * TN,
                ],
                src=c_sb,
            )

    return result


def main():
    size = 512
    rng = np.random.default_rng(42)

    a = rng.standard_normal(
        (size, size), dtype=np.float32
    )
    b = rng.standard_normal(
        (size, size), dtype=np.float32
    )

    a_t = np.ascontiguousarray(a.T)

    c_nki = tiled_matmul_v2(a_t, b)
    c_ref = a @ b

    error = np.abs(c_nki - c_ref)

    print("Kernel: A-tile reuse v2")
    print("Matrix:", c_nki.shape)
    print("Max absolute error:", error.max())
    print("Mean absolute error:", error.mean())

    np.testing.assert_allclose(
        c_nki, c_ref,
        rtol=1e-3,
        atol=1e-3,
    )

    print("Correctness: PASS")


if __name__ == "__main__":
    main()
