"""GPU-gated stage, graph, and launch diagnostics for HA-Schedule-4."""

import json
from pathlib import Path

import pytest

from matmul_inspector.tensor_game import RankOneAction
from matmul_inspector.tensor_game_triton_lowering import create_lowering
from matmul_inspector.tensor_game_schedule_mechanism_study import (
    DUMMY, RECOMPUTED, STORED, validate_topology,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/tensor-game/hardware-aware-rank89-v1/frozen-rank8/solutions.jsonl"


@pytest.fixture(scope="module")
def gpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from matmul_inspector import tensor_game_schedule_gpu_runtime as original
    from matmul_inspector import tensor_game_schedule_mechanism_runtime as staged
    actions = tuple(RankOneAction(**row) for row in json.loads(FROZEN.read_text().splitlines()[0])["actions"])
    return torch, original, staged, create_lowering("project-rank8", actions)


@pytest.mark.parametrize("seed", [11, 23, 37, 53, 71])
def test_standalone_consumers_and_stage_instrumentation_match_production(gpu, seed):
    torch, original, staged, lowering = gpu
    torch.manual_seed(seed)
    size = 128
    left = torch.randn((size, size), device="cuda", dtype=torch.float16)
    right = torch.randn_like(left)
    reference = left @ right
    workspaces = {mode: original.allocate_workspace(lowering, size, gpu_mode)
                  for mode, gpu_mode in ((STORED, "stored_form"), (RECOMPUTED, "recomputed_form"))}
    outputs = {}
    for mode, gpu_mode in ((STORED, "stored_form"), (RECOMPUTED, "recomputed_form")):
        workspace = workspaces[mode]
        outputs[mode] = original.execute_selected(lowering, left, right, workspace, gpu_mode).clone()
        products = [product.clone() for product in workspace["products"]]
        if mode == STORED:
            staged.run_materialization(lowering, left, workspace)
        for index in (0, 1):
            staged.run_consumer_product(lowering, left, right, workspace, mode, index)
            assert torch.equal(workspace["products"][index], products[index])
        staged.run_common_gemms(lowering, left, right, workspace)
        staged_output = staged.run_reconstruction(lowering, left, workspace).clone()
        assert torch.equal(staged_output, outputs[mode])
        torch.testing.assert_close(staged_output, reference, rtol=5e-3, atol=5e-2)
        assert torch.equal(staged.execute_staged(lowering, left, right, workspace, mode), outputs[mode])
    assert torch.equal(outputs[STORED], outputs[RECOMPUTED])


def test_graph_replay_and_dummy_diagnostic_preserve_output(gpu):
    torch, original, staged, lowering = gpu
    torch.manual_seed(107)
    size = 128
    left = torch.randn((size, size), device="cuda", dtype=torch.float16)
    right = torch.randn_like(left)
    outputs = {}
    for mode, gpu_mode in ((STORED, "stored_form"), (RECOMPUTED, "recomputed_form")):
        workspace = original.allocate_workspace(lowering, size, gpu_mode)
        original.execute_selected(lowering, left, right, workspace, gpu_mode)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                original.execute_selected(lowering, left, right, workspace, gpu_mode)
        except Exception as exc:
            pytest.skip(f"CUDA Graph capture unsupported: {exc}")
        graph.replay()
        outputs[mode] = workspace["output"].clone()
        torch.testing.assert_close(outputs[mode], left @ right, rtol=5e-3, atol=5e-2)
    assert torch.equal(outputs[STORED], outputs[RECOMPUTED])
    scratch = torch.ones((1,), device="cuda", dtype=torch.int32)
    before = scratch.clone()
    workspace = original.allocate_workspace(lowering, size, "recomputed_form")
    staged.run_dummy_launch(scratch)
    diagnostic = original.execute_selected(lowering, left, right, workspace, "recomputed_form").clone()
    assert torch.equal(scratch, before)
    assert torch.equal(diagnostic, outputs[RECOMPUTED])


def test_recorded_topology_and_common_cubins_match(gpu):
    torch, original, staged, lowering = gpu
    from torch.profiler import DeviceType, ProfilerActivity, profile

    left = torch.ones((128, 128), device="cuda", dtype=torch.float16)
    right = torch.ones_like(left)
    compiled = {}
    for mode, gpu_mode in ((STORED, "stored_form"), (RECOMPUTED, "recomputed_form")):
        workspace = original.allocate_workspace(lowering, 128, gpu_mode)
        capture = {}
        original.execute_selected(lowering, left, right, workspace, gpu_mode, capture=capture)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
            original.execute_selected(lowering, left, right, workspace, gpu_mode)
            torch.cuda.synchronize()
        names = [event.name for event in recorded.events() if event.device_type == DeviceType.CUDA]
        assert len(validate_topology(mode, names)) == (10 if mode == STORED else 9)
        compiled[mode] = capture
    for role in [*(f"common_gemm_{index}" for index in range(2, 8)), "reconstruction"]:
        assert compiled[STORED][role].asm["cubin"] == compiled[RECOMPUTED][role].asm["cubin"]
    assert compiled[STORED]["shared_form_gemm"].asm["cubin"] != compiled[RECOMPUTED]["shared_form_gemm"].asm["cubin"]
    scratch = torch.ones((1,), device="cuda", dtype=torch.int32)
    workspace = original.allocate_workspace(lowering, 128, "recomputed_form")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
        staged.run_dummy_launch(scratch)
        original.execute_selected(lowering, left, right, workspace, "recomputed_form")
        torch.cuda.synchronize()
    names = [event.name for event in recorded.events() if event.device_type == DeviceType.CUDA]
    assert len(validate_topology(DUMMY, names)) == 10


def test_invalid_staged_mode_is_rejected(gpu):
    torch, original, staged, lowering = gpu
    left = torch.ones((128, 128), device="cuda", dtype=torch.float16)
    right = torch.ones_like(left)
    workspace = original.allocate_workspace(lowering, 128, "recomputed_form")
    with pytest.raises(ValueError, match="unknown rank-8 mechanism mode"):
        staged.run_consumers(lowering, left, right, workspace, "bad")
    with pytest.raises(ValueError, match="first two products"):
        staged.run_consumer_product(lowering, left, right, workspace, RECOMPUTED, 2)
