
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def k_blocked_matmul(a_t, b):
    K, M = a_t.shape
    K2, N = b.shape

    assert K == K2
    assert M % 128 == 0
    assert N % 512 == 0
    assert K % 256 == 0

    TM = 128
    TN = 512
    TK = 128

    result = nl.ndarray(
        (M, N),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )

    for mi in nl.affine_range(M // TM):
        for ni in nl.affine_range(N // TN):

            acc = nl.ndarray(
                (TM, TN),
                dtype=nl.float32,
                buffer=nl.psum,
            )

            # Group two physical K tiles per logical block.
            for kb in nl.affine_range(K // (2 * TK)):
                for inner in nl.static_range(2):
                    ki = kb * 2 + inner

                    a_sb = nl.ndarray(
                        (TK, TM),
                        dtype=a_t.dtype,
                        buffer=nl.sbuf,
                    )
                    b_sb = nl.ndarray(
                        (TK, TN),
                        dtype=b.dtype,
                        buffer=nl.sbuf,
                    )

                    nisa.dma_copy(
                        dst=a_sb,
                        src=a_t[
                            ki * TK:(ki + 1) * TK,
                            mi * TM:(mi + 1) * TM,
                        ],
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
                        stationary=a_sb,
                        moving=b_sb,
                        accumulate=(ki > 0),
                    )

            c_sb = nl.ndarray(
                (TM, TN),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )

            nisa.tensor_copy(dst=c_sb, src=acc)

            nisa.dma_copy(
                dst=result[
                    mi * TM:(mi + 1) * TM,
                    ni * TN:(ni + 1) * TN,
                ],
                src=c_sb,
            )

    return result
