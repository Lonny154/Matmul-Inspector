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
    fused_manifest = lowering_manifest(strassen, strategy="fused_input")
    assert fused_manifest["execution_strategy"] == "fused_input"
    assert fused_manifest["strategy_metadata"]["preparation_kernel_launches"] == 0
    assert fused_manifest["strategy_metadata"]["full_size_input_combination_buffers"] == 0
    output_fused = lowering_manifest(strassen, strategy="fused_input", reconstruction="fused")
    assert output_fused["reconstruction_strategy"] == "fused"
    assert output_fused["strategy_metadata"]["reconstruction_kernel_launches"] == 1
    assert output_fused["strategy_metadata"]["intermediate_output_block_buffers"] == 0


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


def test_unknown_strategy_is_clear_without_gpu_execution():
    with pytest.raises(ValueError, match="unsupported execution strategy"):
        lowering_manifest(reference_lowering("strassen"), strategy="recursive")


def test_unknown_reconstruction_is_clear_without_gpu_execution():
    with pytest.raises(ValueError, match="unsupported reconstruction strategy"):
        lowering_manifest(reference_lowering("strassen"), reconstruction="atomic")


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
@pytest.mark.parametrize("strategy", ("unfused", "fused_input"))
@pytest.mark.parametrize("reconstruction", ("baseline", "fused"))
@pytest.mark.parametrize("size", (128, 256, 512, 1024))
def test_gpu_lowering_matches_pytorch(algorithm, precision, strategy, reconstruction, size):
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
        actual = reference_lowering(algorithm).execute(left, right, precision, strategy, reconstruction)
        expected = left @ right
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("algorithm", ("conventional", "strassen"))
@pytest.mark.parametrize("precision", ("ieee", "tf32x3"))
@pytest.mark.parametrize("strategy", ("unfused", "fused_input"))
@pytest.mark.parametrize("reconstruction", ("baseline", "fused"))
@pytest.mark.parametrize("case", ("zero", "identity", "negative"))
def test_gpu_lowering_special_cases(algorithm, precision, strategy, reconstruction, case):
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
        actual = reference_lowering(algorithm).execute(left, right, precision, strategy, reconstruction)
        expected = left @ right
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("algorithm", ("conventional", "strassen"))
def test_fused_and_unfused_outputs_match_within_tolerance(algorithm):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(8128)
    left = torch.randn((256, 256), device="cuda")
    right = torch.randn((256, 256), device="cuda")
    lowering = reference_lowering(algorithm)
    torch.testing.assert_close(
        lowering.execute(left, right, "ieee", "fused_input"),
        lowering.execute(left, right, "ieee", "unfused"),
        rtol=2e-4, atol=2e-4,
    )


@pytest.mark.parametrize("algorithm", ("conventional", "strassen"))
@pytest.mark.parametrize("size", (128, 256, 512, 1024))
def test_fused_reconstruction_matches_baseline_and_pytorch(algorithm, size):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(9100 + size)
    left = torch.randn((size, size), device="cuda")
    right = torch.randn((size, size), device="cuda")
    lowering = reference_lowering(algorithm)
    baseline = lowering.execute(left, right, "ieee", "fused_input", "baseline")
    fused = lowering.execute(left, right, "ieee", "fused_input", "fused")
    torch.testing.assert_close(fused, left @ right, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(fused, baseline, rtol=2e-4, atol=2e-4)
