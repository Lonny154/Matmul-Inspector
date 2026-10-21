"""GPU-gated checks for the selected HA-Schedule-3 implementations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from matmul_inspector.tensor_game import RankOneAction
from matmul_inspector.tensor_game_triton_lowering import create_lowering, reference_lowering


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/tensor-game/hardware-aware-rank89-v1/frozen-rank8/solutions.jsonl"


@pytest.fixture(scope="module")
def gpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from matmul_inspector import tensor_game_schedule_gpu_runtime as runtime
    return torch, runtime


@pytest.fixture(scope="module")
def lowerings():
    frozen = json.loads(FROZEN.read_text().splitlines()[0])
    actions = tuple(RankOneAction(**row) for row in frozen["actions"])
    return {
        "conventional": reference_lowering("conventional"),
        "strassen": reference_lowering("strassen"),
        "project-rank8": create_lowering("project-rank8", actions),
    }


@pytest.mark.parametrize("algorithm,mode", [
    ("conventional", "reference"),
    ("strassen", "reference"),
    ("project-rank8", "stored_form"),
    ("project-rank8", "recomputed_form"),
])
def test_selected_gpu_candidate_matches_pytorch(gpu, lowerings, algorithm, mode):
    torch, runtime = gpu
    size = 128
    workspace = runtime.allocate_workspace(lowerings[algorithm], size, mode)
    for seed in (11, 23, 37, 53, 71):
        torch.manual_seed(seed)
        left = torch.randn((size, size), device="cuda", dtype=torch.float16)
        right = torch.randn((size, size), device="cuda", dtype=torch.float16)
        output = runtime.execute_selected(lowerings[algorithm], left, right, workspace, mode)
        torch.testing.assert_close(output, left @ right, rtol=5e-3, atol=5e-2)


def test_rank8_pair_agrees_and_uses_distinct_compiled_variants(gpu, lowerings, tmp_path):
    torch, runtime = gpu
    torch.manual_seed(91)
    left = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    right = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    lowering = lowerings["project-rank8"]
    captures = {}
    outputs = {}
    for mode in ("stored_form", "recomputed_form"):
        captures[mode] = {}
        outputs[mode] = runtime.execute_selected(
            lowering, left, right, runtime.allocate_workspace(lowering, 128, mode),
            mode, capture=captures[mode],
        ).clone()
    torch.testing.assert_close(outputs["stored_form"], outputs["recomputed_form"], rtol=5e-3, atol=5e-2)
    assert "materialize_shared_u" in captures["stored_form"]
    assert "materialize_shared_u" not in captures["recomputed_form"]
    assert captures["stored_form"]["shared_form_gemm"].asm["ptx"] != captures["recomputed_form"]["shared_form_gemm"].asm["ptx"]
    assert captures["stored_form"]["shared_form_gemm"].asm["cubin"] != captures["recomputed_form"]["shared_form_gemm"].asm["cubin"]
    from scripts.run_hardware_aware_schedule_gpu_study import _compiler_capture
    for mode in ("stored_form", "recomputed_form"):
        label = f"rank8_{mode}"
        entry = _compiler_capture(captures[mode]["shared_form_gemm"], label, tmp_path / "compiler")
        assert entry["label"] == label
        assert entry["kernel_name"] == "_shared_form_gemm"
        assert entry["artifacts"]["ptx"]["path"] == f"compiler/{label}.ptx"
        assert (tmp_path / entry["artifacts"]["ptx"]["path"]).is_file()


def test_invalid_lowering_is_rejected(gpu, lowerings):
    torch, runtime = gpu
    left = torch.zeros((128, 128), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="unsupported GPU schedule mode"):
        runtime.allocate_workspace(lowerings["project-rank8"], 128, "invalid")
    workspace = runtime.allocate_workspace(lowerings["project-rank8"], 128, "recomputed_form")
    with pytest.raises(ValueError, match="unknown selected schedule mode"):
        runtime.execute_selected(lowerings["project-rank8"], left, left, workspace, "invalid")
    with pytest.raises(ValueError, match="lacks the shared"):
        runtime.execute_selected(lowerings["strassen"], left, left, workspace, "recomputed_form")


@pytest.mark.parametrize("algorithm,mode,expected_count", [
    ("conventional", "reference", 9),
    ("strassen", "reference", 8),
    ("project-rank8", "stored_form", 10),
    ("project-rank8", "recomputed_form", 9),
])
def test_actual_launch_count_matches_source_lowering(gpu, lowerings, algorithm, mode, expected_count):
    torch, runtime = gpu
    from torch.profiler import DeviceType, ProfilerActivity, profile

    lowering = lowerings[algorithm]
    workspace = runtime.allocate_workspace(lowering, 128, mode)
    left = torch.ones((128, 128), device="cuda", dtype=torch.float16)
    right = torch.ones_like(left)
    runtime.execute_selected(lowering, left, right, workspace, mode)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
        runtime.execute_selected(lowering, left, right, workspace, mode)
        torch.cuda.synchronize()
    names = [event.name for event in recorded.events() if event.device_type == DeviceType.CUDA]
    assert names == runtime.expected_kernel_names(len(lowering.actions), mode)
    assert len(names) == expected_count
