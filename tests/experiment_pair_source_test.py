"""CPU-only trusted local source tests for the paired experiment path."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from matmul_inspector import cli, experiment, experiment_pair


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "examples/mi_workflow_3_local_pair.json").read_text())


def source_spec(path: Path) -> dict:
    spec = copy.deepcopy(EXAMPLE)
    spec["kernels"]["B"]["source"]["path"] = str(path)
    return spec


def fixture(tmp_path: Path, body: str = "def matmul_into(a, b, out, parameters):\n    return None\n") -> Path:
    path = tmp_path / "kernel.py"
    path.write_text(body)
    return path


def test_source_spec_validation_is_read_only_and_bounded(tmp_path, monkeypatch):
    marker = tmp_path / "imported"
    path = fixture(tmp_path, f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n"
                   "def matmul_into(a, b, out, parameters):\n    return None\n")
    spec = source_spec(path)
    validated = experiment_pair.validate_spec(spec)
    assert not marker.exists()
    assert validated["kernels"]["B"]["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert validated["benchmark"]["mode"] == "preallocated"
    for edit in [
        lambda s: s["kernels"]["B"]["source"].update(path=str(tmp_path / "missing.py")),
        lambda s: s["kernels"]["B"]["source"].pop("entry_point"),
        lambda s: s["kernels"]["B"]["source"].update(entry_point="other"),
        lambda s: s["kernels"]["B"]["source"].update(sha256="0" * 64),
        lambda s: s["kernels"]["B"]["parameters"].update(block_k=7),
        lambda s: s["benchmark"].update(mode="allocating"),
        lambda s: s["benchmark"].update(rounds=21),
    ]:
        changed = copy.deepcopy(spec)
        edit(changed)
        with pytest.raises(ValueError):
            experiment_pair.validate_spec(changed)
    assert not marker.exists()
    schema = json.loads((ROOT / "schemas/mi_workflow_3_pair_v2.schema.json").read_text())
    assert set(schema["required"]) == set(EXAMPLE)


def test_import_contract_requires_opt_in_and_hash(tmp_path):
    path = fixture(tmp_path)
    source = experiment_pair.validate_spec(source_spec(path))["kernels"]["B"]["source"]
    with pytest.raises(ValueError, match="allow-trusted-local-source"):
        experiment_pair.load_trusted_entry(source)
    assert callable(experiment_pair.load_trusted_entry(source, allowed=True))
    path.write_text("raise RuntimeError('new version')\n")
    with pytest.raises(experiment.StageFailure, match="hash changed"):
        experiment_pair.load_trusted_entry(source, allowed=True)


@pytest.mark.parametrize("body,stage", [
    ("raise RuntimeError('broken import')\n", "source_loading"),
    ("def other():\n    pass\n", "interface"),
    ("matmul_into = 7\n", "interface"),
    ("def matmul_into(a):\n    pass\n", "interface"),
])
def test_loader_errors_are_actionable(tmp_path, body, stage):
    path = fixture(tmp_path, body)
    source = experiment_pair.validate_spec(source_spec(path))["kernels"]["B"]["source"]
    with pytest.raises(experiment.StageFailure) as failure:
        experiment_pair.load_trusted_entry(source, allowed=True)
    assert failure.value.stage == stage


def test_output_and_return_contract_on_cpu():
    torch = pytest.importorskip("torch")
    reference = torch.ones((2, 2), dtype=torch.float16)
    tolerance = {"atol": 0.01, "rtol": 0.01}
    assert experiment_pair.check_output(torch, reference, reference.clone(), tolerance)["passed"]
    assert not experiment_pair.check_output(torch, reference, torch.full_like(reference, float("nan")),
                                            tolerance)["passed"]
    assert not experiment_pair.check_output(torch, reference, torch.zeros((3, 2), dtype=torch.float16),
                                            tolerance)["passed"]
    assert not experiment_pair.check_output(torch, reference, reference.float(), tolerance)["passed"]
    assert experiment_pair._compiled_capture(None) is None
    with pytest.raises(experiment.StageFailure, match="compiled launch result"):
        experiment_pair._compiled_capture(reference)


def fake_pair_runner(public, *, runtime_spec):
    snapshot = Path(runtime_spec["kernels"]["B"]["source"]["path"])
    assert snapshot.is_file()
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == public["kernels"]["B"]["source"]["sha256"]
    assert public["kernels"]["B"]["source"]["path"] == "sources/B.py"
    checks = {label: {"passed": True, "max_absolute_error": 0.0} for label in ("A", "B")}
    rows = []
    for index in range(1, 9):
        order = experiment_pair.pair_order(index)
        measured = {}
        for label, value in (("A", 0.20), ("B", 0.10)):
            measured[label] = {
                "cuda_event_elapsed_ms_per_operation": value,
                "cuda_event_elapsed_ms_per_block": value * 100,
                "host_submission_ms_per_block": value * 100 - 2,
                "host_wall_ms_per_block": value * 100 + 1,
                "cuda_event_minus_host_wall_ms_per_block": -1.0,
            }
        rows.append({"round": index, "order": order, "measurements": measured})
    return {"spec": public, "status": "success", "git": {"commit": "mock"},
            "environment": {"device": {"name": "mock"},
                            "versions": {"torch": "mock", "python": "mock"}},
            "execution_source_paths": {"B": str(snapshot)},
            "correctness": {"passed": True, "checks": checks},
            "benchmark": {"mode": "preallocated", "warmups_per_kernel": 30,
                          "iterations_per_measurement": 100,
                          "unranked_timing_controls": {label: {
                              "cuda_event_minus_host_wall_ms_per_block": -1.0} for label in ("A", "B")},
                          "rounds": rows}}


def unavailable_compiler(*_):
    return {"A": {"ptx": "UNAVAILABLE", "sass": "UNAVAILABLE"},
            "B": {"ptx": "UNAVAILABLE", "sass": "UNAVAILABLE"},
            "ptx_comparison": {"status": "UNAVAILABLE"}}


def test_opt_in_snapshot_report_and_offline_verify(tmp_path, monkeypatch, capsys):
    source = fixture(tmp_path)
    spec = source_spec(source)
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="allow-trusted-local-source"):
        experiment_pair.run_pair(spec, output, runner=fake_pair_runner)
    assert not output.exists()
    decision = experiment_pair.run_pair(spec, output, allow_trusted_local_source=True,
                                        runner=fake_pair_runner, compiler_capture=unavailable_compiler)
    assert decision["decision"] == "B_FASTER"
    assert (output / "sources/B.py").read_bytes() == source.read_bytes()
    assert "sources/B.py" in (output / "report.md").read_text()
    assert experiment_pair.verify_bundle(output)["status"] == "PASS"
    assert cli.main(["experiment", "pair", "--verify", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    # A source edit after the run cannot alter the retained decision.
    source.write_text("raise RuntimeError('later edit')\n")
    assert experiment_pair.verify_bundle(output)["status"] == "PASS"
    # Offline inspection must not import the retained source.
    def forbidden(*_args, **_kwargs):
        raise AssertionError("offline verifier imported source")
    monkeypatch.setattr(experiment_pair, "load_trusted_entry", forbidden)
    assert experiment_pair.verify_bundle(output)["status"] == "PASS"
    (output / "sources/B.py").write_text("changed\n")
    experiment_pair._manifest(output)
    with pytest.raises(ValueError):
        experiment_pair.verify_bundle(output)


def test_two_local_sources_snapshot_independently(tmp_path):
    a = fixture(tmp_path)
    b = tmp_path / "other.py"
    b.write_text("def matmul_into(a, b, out, parameters):\n    return 0\n")
    spec = source_spec(b)
    spec["kernels"]["A"] = copy.deepcopy(spec["kernels"]["B"])
    spec["kernels"]["A"]["source"]["path"] = str(a)
    checked = experiment_pair.validate_spec(spec)
    assert checked["kernels"]["A"]["source"]["sha256"] != checked["kernels"]["B"]["source"]["sha256"]

    def two_local_runner(public, *, runtime_spec):
        result = fake_pair_runner(public, runtime_spec=runtime_spec)
        a_snapshot = Path(runtime_spec["kernels"]["A"]["source"]["path"])
        assert a_snapshot.read_bytes() == a.read_bytes()
        result["execution_source_paths"]["A"] = str(a_snapshot)
        return result

    output = tmp_path / "two-local"
    experiment_pair.run_pair(spec, output, allow_trusted_local_source=True,
                             runner=two_local_runner, compiler_capture=unavailable_compiler)
    assert experiment_pair.verify_bundle(output)["status"] == "PASS"
    assert (output / "sources/A.py").read_bytes() != (output / "sources/B.py").read_bytes()


class SecretFailureBackend:
    def run(self, _spec, _log):
        raise experiment.StageFailure("source_loading", "secret-marker-do-not-save")


def test_worker_failure_redacts_local_exception_details(tmp_path, monkeypatch):
    spec = experiment_pair.validate_spec(source_spec(fixture(tmp_path)))
    monkeypatch.setattr(experiment_pair, "PairBackend", SecretFailureBackend)
    raw = experiment_pair.run_worker(spec, runtime_spec=spec)
    assert raw["failure"]["stage"] == "source_loading"
    assert "secret-marker-do-not-save" not in json.dumps(raw)
    assert raw["failure"]["traceback"] is None


def test_cli_requires_trust_flag_before_output_creation(tmp_path, capsys):
    spec = source_spec(fixture(tmp_path))
    source = tmp_path / "pair.json"
    source.write_text(json.dumps(spec))
    output = tmp_path / "blocked"
    assert cli.main(["experiment", "pair", "--spec", str(source), "--output", str(output)]) == 1
    assert "allow-trusted-local-source" in capsys.readouterr().err
    assert not output.exists()


def test_local_compiler_ptx_is_retained_and_replayed(tmp_path, monkeypatch):
    from matmul_inspector import experiment_compare
    pt_source = ROOT / "artifacts/mi-dogfood-1/baseline-v1/candidate.ptx"
    pt_text = pt_source.read_text()

    def registered_capture(_spec, destination):
        (destination / "candidate.ptx").write_text(pt_text)
        return {"status": "CAPTURED", "ptx": "CAPTURED", "sass": "UNAVAILABLE",
                "metadata": {"shared": 24576, "n_regs": 121}}

    def runner(public, *, runtime_spec):
        result = fake_pair_runner(public, runtime_spec=runtime_spec)
        result["local_compiler_capture"] = {"B": {
            "kernel_name": "fc1_kernel", "metadata": {"shared": 24576, "n_regs": 121},
            "ptx_text": pt_text, "sass_text": None}}
        return result

    monkeypatch.setattr(experiment_compare, "_compiler_capture", registered_capture)
    output = tmp_path / "compiler-pair"
    experiment_pair.run_pair(source_spec(fixture(tmp_path)), output,
                             allow_trusted_local_source=True, runner=runner)
    compiler = json.loads((output / "compiler.json").read_text())
    assert compiler["B"]["ptx"] == "CAPTURED"
    assert compiler["B"]["sass"] == "UNAVAILABLE"
    assert compiler["ptx_comparison"]["status"] == "CAPTURED"
    assert experiment_pair.verify_bundle(output)["status"] == "PASS"
