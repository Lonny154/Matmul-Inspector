import importlib.util

import pytest

from matmul_inspector.tensor_game import RankOneAction
from matmul_inspector.tensor_game_triton_lowering import create_lowering, lowering_manifest, reference_lowering


def test_reference_lowerings_share_contract_and_expected_metadata():
    conventional = reference_lowering("conventional")
    strassen = reference_lowering("strassen")
    assert (conventional.metadata.rank, conventional.metadata.preparation_additions, conventional.metadata.reconstruction_additions) == (8, 0, 4)
    assert (strassen.metadata.rank, strassen.metadata.preparation_additions, strassen.metadata.reconstruction_additions) == (7, 10, 8)
    assert (conventional.metadata.execution_preparation_updates, conventional.metadata.execution_reconstruction_updates) == (16, 8)
    assert (strassen.metadata.execution_preparation_updates, strassen.metadata.execution_reconstruction_updates) == (24, 12)
    assert conventional.metadata.flattening.startswith("row-major blocks")
    assert strassen.metadata.triton_dot_input_precision == "ieee"
    assert lowering_manifest(strassen) == lowering_manifest(reference_lowering("strassen"))
    tf32x3_manifest = lowering_manifest(strassen, "tf32x3")
    assert tf32x3_manifest["selected_dot_input_precision"] == "tf32x3"
    assert tf32x3_manifest["metadata"]["triton_dot_input_precision"] == "tf32x3"


def test_invalid_decomposition_is_rejected_before_runtime_import():
    with pytest.raises(ValueError, match="exactly reconstruct"):
        create_lowering("invalid", (RankOneAction((1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)),))
    assert importlib.util.find_spec("matmul_inspector.tensor_game_triton_runtime") is not None


def test_unknown_reference_is_clear():
    with pytest.raises(ValueError, match="unknown reference"):
        reference_lowering("winograd")


def test_unknown_precision_is_clear_without_gpu_execution():
    with pytest.raises(ValueError, match="unsupported dot precision"):
        lowering_manifest(reference_lowering("strassen"), "tf32")


def test_default_gpu_precision_matches_explicit_ieee():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(91)
    left = torch.randn((128, 128), device="cuda", dtype=torch.float32)
    right = torch.randn((128, 128), device="cuda", dtype=torch.float32)
    lowering = reference_lowering("conventional")
    assert torch.equal(lowering.execute(left, right), lowering.execute(left, right, "ieee"))


@pytest.mark.parametrize("algorithm", ("conventional", "strassen"))
@pytest.mark.parametrize("precision", ("ieee", "tf32x3"))
@pytest.mark.parametrize("size", (128, 256, 512, 1024))
def test_gpu_lowering_matches_pytorch(algorithm, precision, size):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(1200 + size)
    left = torch.randn((size, size), device="cuda", dtype=torch.float32)
    right = torch.randn((size, size), device="cuda", dtype=torch.float32)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        actual = reference_lowering(algorithm).execute(left, right, precision)
        expected = left @ right
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("algorithm", ("conventional", "strassen"))
@pytest.mark.parametrize("precision", ("ieee", "tf32x3"))
@pytest.mark.parametrize("case", ("zero", "identity", "negative"))
def test_gpu_lowering_special_cases(algorithm, precision, case):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    size = 64
    if case == "zero":
        left = torch.zeros((size, size), device="cuda")
        right = torch.zeros((size, size), device="cuda")
    elif case == "identity":
        left = torch.eye(size, device="cuda")
        right = torch.eye(size, device="cuda")
    else:
        torch.manual_seed(2401)
        left = -torch.rand((size, size), device="cuda")
        right = torch.rand((size, size), device="cuda") - 0.5
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        actual = reference_lowering(algorithm).execute(left, right, precision)
        expected = left @ right
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
