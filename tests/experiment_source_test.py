#!/usr/bin/env python3
"""CPU-only tests for preserved Triton source candidates."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matmul_inspector import experiment, experiment_batch, experiment_session


def source_config(path: Path) -> dict:
    config = copy.deepcopy(experiment.DEFAULT_CONFIG)
    config["benchmark"]["mode"] = "preallocated"
    config["candidate"] = {
        "implementation": "triton_source",
        "source": {"path": str(path), "entry_point": "matmul_into",
                   "interface_version": "matmul_into_v1"},
        "parameters": copy.deepcopy(experiment.DEFAULT_CONFIG["candidate"]["parameters"]),
    }
    return config


class LoadOnlyBackend:
    def run(self, config, log):
        experiment.CudaTritonBackend._load_source(config)
        raise experiment.StageFailure("execution", "loaded")


class HangingBackend:
    def run(self, config, log):
        time.sleep(10)
        return {}


class SourceExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="matmul-source-test-")
        self.root = Path(self.temporary.name)
        self.source = self.root / "candidate.py"
        self.source.write_text("def matmul_into(a, b, out, parameters):\n    return None\n",
                               encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_snapshot_is_byte_exact_and_records_content_identity(self):
        config = source_config(self.source)
        destination = self.root / "preserved" / "candidate.py"
        resolved = experiment.snapshot_candidate_source(config, destination)
        expected = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.assertEqual(destination.read_bytes(), self.source.read_bytes())
        self.assertEqual(resolved["candidate"]["source"]["sha256"], expected)
        self.assertEqual(resolved["candidate"]["source"]["path"], str(destination.resolve()))

    def test_missing_file_bad_interface_and_allocating_mode_reject(self):
        config = source_config(self.root / "missing.py")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            experiment.validate_config(config)
        config = source_config(self.source)
        config["candidate"]["source"]["entry_point"] = "other"
        with self.assertRaisesRegex(ValueError, "entry_point"):
            experiment.validate_config(config)
        config = source_config(self.source)
        config["benchmark"]["mode"] = "allocating"
        with self.assertRaisesRegex(ValueError, "preallocated"):
            experiment.validate_config(config)

    def test_source_import_and_interface_failures_are_worker_records(self):
        self.source.write_text("raise RuntimeError('broken import')\n", encoding="utf-8")
        result = experiment.run_experiment(
            source_config(self.source), output_root=self.root / "runs",
            backend_factory=LoadOnlyBackend)
        self.assertEqual(result["failure"]["stage"], "source_loading")
        self.assertEqual(result["source_identity"]["interface_version"], "matmul_into_v1")
        self.assertTrue((Path(result["run_directory"]) / "source" / "candidate.py").is_file())
        self.source.write_text("def other():\n    pass\n", encoding="utf-8")
        result = experiment.run_experiment(
            source_config(self.source), output_root=self.root / "runs",
            backend_factory=LoadOnlyBackend)
        self.assertEqual(result["failure"]["stage"], "interface")

    def test_timeout_is_cleaned_up_and_batch_continues(self):
        shared = source_config(self.source)
        candidate = shared.pop("candidate")
        batch_config = {"shared": shared, "candidates": [
            {"id": "hang", **candidate}, {"id": "next", **candidate}],
            "scheduling_seed": 0, "repetitions": 1}

        calls = 0
        def executor(config, *, output_root):
            nonlocal calls
            calls += 1
            config["timeout_seconds"] = 0.03
            factory = HangingBackend if calls == 1 else LoadOnlyBackend
            return experiment.run_experiment(config, output_root=output_root,
                                             backend_factory=factory)

        result = experiment_batch.run_batch(
            batch_config, output_root=self.root / "batches", experiment_executor=executor)
        self.assertEqual(calls, 2)
        self.assertEqual(len([entry for entry in result["entries"]
                              if entry["repetition"] is not None]), 2)


class SourceSessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="matmul-source-session-test-")
        self.root = Path(self.temporary.name)
        self.source = self.root / "candidate.py"
        self.source.write_text("def matmul_into(a, b, out, parameters):\n    return None\n",
                               encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def session_config(self):
        shared = source_config(self.source)
        shared.pop("candidate")
        defaults = experiment.DEFAULT_CONFIG["candidate"]["parameters"]
        return {"objective": "test source preservation", "shared": shared,
                "implementation": "triton_source", "allow_source_candidates": True,
                "allowed_parameters": {key: [value] for key, value in defaults.items()},
                "total_experiment_budget": 2, "scheduling_seed": 0}

    def candidate(self, identifier: str, path: Path):
        return {"id": identifier, "implementation": "triton_source",
                "source": {"path": str(path), "entry_point": "matmul_into",
                           "interface_version": "matmul_into_v1"},
                "parameters": {}, "hypothesis": "exercise the interface"}

    @staticmethod
    def proposal(identifier, candidate):
        return {"proposal_id": identifier, "candidates": [candidate],
                "prior_result_references": [], "repetitions": 1}

    def test_opt_in_required_and_duplicate_identity_uses_content(self):
        config = self.session_config()
        config["allow_source_candidates"] = False
        with self.assertRaisesRegex(ValueError, "allow_source_candidates"):
            experiment_session.validate_session_config(config)
        state = experiment_session.create_session(self.session_config(), output_root=self.root)
        first = experiment_session.validate_proposal(
            state, self.proposal("one", self.candidate("a", self.source)))
        state["candidate_fingerprints"][first["candidates"][0]["fingerprint"]] = [{}]
        copy_path = self.root / "renamed.py"
        copy_path.write_bytes(self.source.read_bytes())
        with self.assertRaisesRegex(ValueError, "duplicates"):
            experiment_session.validate_proposal(
                state, self.proposal("two", self.candidate("b", copy_path)))

    def test_accepted_snapshot_survives_original_edit_and_failure_consumes_budget(self):
        state = experiment_session.create_session(self.session_config(), output_root=self.root)
        original = self.source

        def failed_batch(config, *, output_root):
            preserved = Path(config["candidates"][0]["source"]["path"])
            self.assertNotEqual(preserved, original)
            self.assertIn("return None", preserved.read_text(encoding="utf-8"))
            original.write_text("raise RuntimeError('edited later')\n", encoding="utf-8")
            root = Path(output_root) / "batch"
            run = root / "experiments" / "run-1"
            run.mkdir(parents=True)
            result_path = run / "result.json"
            result_path.write_text("{}\n", encoding="utf-8")
            entry = {"repetition": 1, "candidate_id": "a", "status": "failed",
                     "failure_stage": "source_loading", "experiment_run_id": "run-1",
                     "result_path": str(result_path)}
            summary = {"status": "completed_with_failures", "experiments": [entry],
                       "complete_candidate_ranking": []}
            batch = {"status": "completed_with_failures", "batch_directory": str(root),
                     "entries": [entry], "summary": summary}
            (root / "batch_results.json").write_text(json.dumps(batch), encoding="utf-8")
            return batch

        record = experiment_session.submit_proposal(
            state["session_directory"],
            self.proposal("one", self.candidate("a", self.source)),
            batch_executor=failed_batch)
        self.assertEqual(record["consumed_units"], 1)
        self.assertEqual(experiment_session.load_session(
            state["session_directory"])["budget"]["remaining"], 1)


if __name__ == "__main__":
    unittest.main()
