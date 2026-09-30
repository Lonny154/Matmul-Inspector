"""Artifact operation identity and nominal models, matching operation.cpp.

Absent operation is accepted only for legacy matmul kernel names. Vector rows
use M=N=1 (scalar output), K=length, plus an explicit length column.
"""
MATMUL_KERNELS = {'cpu', 'naive', 'tiled', 'cuda-naive-fma', 'cuda-naive-no-fma', 'cuda-naive-reordered'}
KINDS = {'matmul', 'dot', 'reduction_sum'}


def identity(config, row=None):
    operation = config.get('operation')
    if operation is None:
        kernels = [config.get(k) for k in ('reference', 'candidate')]
        if row is not None:
            kernels += [row.get('kernel')]
        if any(k is not None and k not in MATMUL_KERNELS for k in kernels):
            raise ValueError('Operation missing for non-legacy kernels')
        operation = 'matmul'
    if operation not in KINDS:
        raise ValueError('Unsupported operation: ' + str(operation))
    if row is not None:
        if row.get('operation', operation) != operation:
            raise ValueError('Row operation disagrees with metadata')
        if operation != 'matmul':
            if (int(row['M']), int(row['N'])) != (1, 1) or int(row.get('length', 0)) != int(row['K']):
                raise ValueError('Vector row requires scalar output dimensions and explicit length=K')
            if row['kernel'] not in ('cpu', 'cpu-reverse', 'cuda-tree'):
                raise ValueError('Unsupported vector kernel')
        elif row.get('kernel') not in MATMUL_KERNELS:
            raise ValueError('Unsupported matmul kernel')
    return operation


def model(operation, m, n, k):
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (m,n,k)):
        raise ValueError('Dimensions must be positive integers')
    if operation == 'matmul':
        return 2*m*n*k, 4*(m*k+k*n+m*n)
    if operation not in ('dot', 'reduction_sum') or (m,n) != (1,1):
        raise ValueError('Unsupported operation or invalid vector shape')
    return (2*k, 4*(2*k+1)) if operation == 'dot' else (k-1, 4*(k+1))


def label(row):
    return ('×'.join(map(str, row['shape'])) if row.get('operation', 'matmul') == 'matmul'
            else 'length ' + str(row['shape'][2]))


VECTOR_FIXTURES = ('random', 'random_uniform', 'ascending_magnitude', 'descending_magnitude',
                   'alternating_sign', 'cancellation', 'large_dynamic_range', 'repeated_small_plus_large')

def generator(operation, fixture):
    if fixture in ('random', 'random_uniform'):
        return 'lcg32-v1'
    if operation == 'matmul':
        return fixture + '-v1' if fixture in ('cancellation', 'fma-sensitive') else None
    return 'vector-' + fixture + '-v1' if fixture in VECTOR_FIXTURES else None
