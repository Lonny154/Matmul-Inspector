#!/usr/bin/env python3
"""CPU-only tests for persistent, budgeted experiment sessions."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matmul_inspector import experiment, experiment_session


def session_config(budget=24):
    shared = copy.deepcopy(experiment.DEFAULT_CONFIG)
    shared.pop("candidate")
    return {
        "objective": "Find robust FC1 GEMM configurations without overstating small differences.",
        "shared": shared,
        "implementation": "triton_fc1",
        "allowed_parameters": {
            "block_m": [64, 128], "block_n": [64, 128], "block_k": [32, 64],
            "num_warps": [4, 8], "num_stages": [2, 3], "group_size_m": [1, 4],
        },
        "total_experiment_budget": budget,
        "scheduling_seed": 9,
    }


def candidate(identifier="candidate-a", **parameters):
    return {"id": identifier, "implementation": "triton_fc1", "parameters": parameters,
            "hypothesis": "A controlled parameter change may alter resource use."}


def proposal(identifier="proposal-1", candidates=None, repetitions=1, references=None):
    return {"proposal_id": identifier, "candidates": candidates or [candidate()],
            "prior_result_references": references or [], "repetitions": repetitions}


class SyntheticBatch:
    def __init__(self, mode="success"):
        self.mode = mode
        self.calls = 0

    def __call__(self, config, *, output_root):
        self.calls += 1
        root = Path(output_root) / f"batch-{self.calls}"
        root.mkdir(parents=True)
        entries = []
        sequence = 0
        for repetition in range(1, config["repetitions"] + 1):
            for item in config["candidates"]:
                sequence += 1
                if self.mode == "interrupt" and sequence == 2:
                    entries.append({"repetition": repetition, "candidate_id": item["id"],
                                    "status": "failed", "failure_stage": "interruption",
                                    "experiment_run_id": None, "result_path": None})
                    batch = self._record(root, config, entries, "interrupted")
                    return batch
                run = root / "experiments" / f"run-{sequence}"
                run.mkdir(parents=True)
                result_path = run / "result.json"
                result_path.write_text("{}\n", encoding="utf-8")
                failed = self.mode == "failure" and sequence == 1
                entries.append({
                    "repetition": repetition, "candidate_id": item["id"],
                    "status": "failed" if failed else "success",
                    "correctness_passed": not failed,
                    "failure_stage": "correctness" if failed else None,
                    "experiment_run_id": f"run-{sequence}", "result_path": str(result_path),
                    "paired_speedup_baseline_over_candidate": None if failed else 1.0,
                })
                if self.mode == "crash" and sequence == 1:
                    self._record(root, config, entries, "running")
                    raise RuntimeError("synthetic batch crash")
        status = "completed_with_failures" if self.mode == "failure" else "success"
        return self._record(root, config, entries, status)

    @staticmethod
    def _record(root, config, entries, status):
        summary = {
            "status": status, "experiments": entries,
            "complete_candidate_ranking": [] if status != "success" else [
                {"rank": 1, "candidate_id": config["candidates"][0]["id"],
                 "median_run_speedup": 1.0}],
        }
        batch = {"status": status, "batch_directory": str(root), "entries": entries,
                 "summary": summary}
        (root / "batch_results.json").write_text(json.dumps(batch), encoding="utf-8")
        (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return batch


class ExperimentSessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="matmul-session-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def create(self, budget=24):
        return experiment_session.create_session(session_config(budget), output_root=self.root)

    def test_creation_reload_and_provenance(self):
        created = self.create()
        loaded = experiment_session.load_session(created["session_directory"])
        self.assertEqual(loaded["session_id"], created["session_id"])
        self.assertEqual(loaded["budget"], {"total": 24, "consumed": 0,
                                             "reserved": 0, "remaining": 24})
        self.assertIn("git", loaded["provenance"])
        self.assertTrue((Path(created["session_directory"]) /
                         "submitted_session_config.json").is_file())
        self.assertEqual(loaded["fixed"]["shared"]["benchmark"]["mode"], "allocating")

    def test_proposal_validation_rejects_shared_changes_and_out_of_range_values(self):
        state = self.create()
        path = state["session_directory"]
        changed = proposal()
        changed["shared"] = state["fixed"]["shared"]
        with self.assertRaises(experiment_session.SessionError):
            experiment_session.submit_proposal(path, changed, batch_executor=SyntheticBatch())
        bad = proposal("proposal-2", [candidate(block_m=32)])
        with self.assertRaises(experiment_session.SessionError):
            experiment_session.submit_proposal(path, bad, batch_executor=SyntheticBatch())
        loaded = experiment_session.load_session(path)
        self.assertEqual(loaded["budget"]["remaining"], 24)
        self.assertEqual(len(loaded["rejected_proposals"]), 2)

    def test_canonical_duplicates_need_an_explicit_repeat_reason(self):
        state = self.create()
        path = state["session_directory"]
        first = proposal(candidates=[candidate("original", block_m=64, block_n=128)])
        experiment_session.submit_proposal(path, first, batch_executor=SyntheticBatch())
        duplicate = proposal("proposal-2", [candidate("renamed", block_n=128, block_m=64)])
        with self.assertRaises(experiment_session.SessionError):
            experiment_session.submit_proposal(path, duplicate, batch_executor=SyntheticBatch())
        repeated_candidate = candidate("repeat", block_m=64, block_n=128)
        repeated_candidate["repeat_reason"] = "Verify a noisy prior result."
        repeated = proposal("proposal-3", [repeated_candidate])
        result = experiment_session.submit_proposal(
            path, repeated, batch_executor=SyntheticBatch())
        self.assertEqual(result["state"], "completed")

    def test_exact_budget_rejection_happens_before_execution(self):
        state = self.create(budget=3)
        executor = SyntheticBatch()
        costly = proposal(candidates=[candidate("a"), candidate("b", block_n=64)], repetitions=2)
        with self.assertRaises(experiment_session.SessionError):
            experiment_session.submit_proposal(state["session_directory"], costly,
                                                   batch_executor=executor)
        self.assertEqual(executor.calls, 0)
        self.assertEqual(experiment_session.load_session(
            state["session_directory"])["budget"]["remaining"], 3)

    def test_failed_experiments_consume_budget(self):
        state = self.create(budget=4)
        result = experiment_session.submit_proposal(
            state["session_directory"], proposal(), batch_executor=SyntheticBatch("failure"))
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["consumed_units"], 1)
        self.assertEqual(experiment_session.load_session(
            state["session_directory"])["budget"]["remaining"], 3)

    def test_interruption_releases_confirmed_unlaunched_units_only(self):
        state = self.create(budget=8)
        submitted = proposal(candidates=[candidate("a"), candidate("b", block_n=64)],
                             repetitions=2)
        result = experiment_session.submit_proposal(
            state["session_directory"], submitted, batch_executor=SyntheticBatch("interrupt"))
        self.assertEqual(result["state"], "needs_reconciliation")
        self.assertEqual(result["consumed_units"], 1)
        self.assertEqual(result["reserved_units"], 1)
        budget = experiment_session.load_session(state["session_directory"])["budget"]
        self.assertEqual(budget, {"total": 8, "consumed": 1, "reserved": 1, "remaining": 6})

    def test_crash_reconciliation_never_automatically_reruns(self):
        state = self.create(budget=6)
        executor = SyntheticBatch("crash")
        submitted = proposal(candidates=[candidate("a"), candidate("b", block_n=64)],
                             repetitions=2)
        result = experiment_session.submit_proposal(
            state["session_directory"], submitted, batch_executor=executor)
        self.assertEqual(executor.calls, 1)
        self.assertEqual(result["state"], "needs_reconciliation")
        self.assertEqual((result["consumed_units"], result["reserved_units"]), (1, 3))
        recovered = experiment_session.recover_session(
            state["session_directory"], result["proposal_id"], "release")
        self.assertEqual(recovered["reserved_units"], 0)
        self.assertEqual(executor.calls, 1)
        self.assertEqual(experiment_session.load_session(
            state["session_directory"])["budget"]["remaining"], 5)

    def test_accepted_prelaunch_crash_is_safely_released_by_inspection(self):
        state = self.create(budget=2)
        directory = Path(state["session_directory"])
        record = {
            "proposal_id": "accepted-crash", "state": "accepted", "cost": 1,
            "consumed_units": 0, "reserved_units": 1, "results": [],
            "candidates": [], "batch_directory": None, "observations": None,
        }
        state["proposals"].append(record)
        experiment_session._budget(state)
        (directory / "proposals" / "accepted-crash").mkdir(parents=True)
        experiment_session._atomic_json(directory / "session.json", state)
        recovered = experiment_session.recover_session(
            directory, "accepted-crash", "inspect")
        self.assertEqual((recovered["consumed_units"], recovered["reserved_units"]), (0, 0))
        self.assertEqual(experiment_session.load_session(directory)["budget"]["remaining"], 2)

    def test_concurrent_submission_is_rejected_by_lock(self):
        state = self.create()
        directory = Path(state["session_directory"])
        with experiment_session._session_lock(directory):
            with self.assertRaises(experiment_session.SessionLockError):
                experiment_session.submit_proposal(directory, proposal(),
                                                     batch_executor=SyntheticBatch())

    def test_context_export_is_atomic_and_separates_hypotheses(self):
        state = self.create()
        experiment_session.submit_proposal(
            state["session_directory"], proposal(), batch_executor=SyntheticBatch())
        destination = self.root / "context.json"
        context = experiment_session.export_context(state["session_directory"], destination)
        saved = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(saved, context)
        self.assertEqual(len(context["hypotheses"]), 1)
        self.assertEqual(len(context["observations"]["successful_results"]), 1)
        self.assertEqual(context["budget"]["consumed"], 1)
        self.assertFalse(list(self.root.glob(".context.json.*.tmp")))

    def test_new_fingerprints_include_immutable_benchmark_mode(self):
        allocating = self.create()
        preallocated_config = session_config()
        preallocated_config["shared"]["benchmark"]["mode"] = "preallocated"
        preallocated = experiment_session.create_session(preallocated_config, output_root=self.root)
        allocating_candidate = experiment_session.validate_proposal(
            allocating, proposal())["candidates"][0]
        preallocated_candidate = experiment_session.validate_proposal(
            preallocated, proposal())["candidates"][0]
        self.assertNotEqual(allocating_candidate["fingerprint"],
                            preallocated_candidate["fingerprint"])
        submitted = proposal("changed-mode")
        submitted["shared"] = copy.deepcopy(preallocated["fixed"]["shared"])
        with self.assertRaises(ValueError):
            experiment_session.validate_proposal(allocating, submitted)
        self.assertEqual(allocating["fixed"]["shared"]["benchmark"]["mode"], "allocating")

    def test_legacy_session_omission_means_allocating_without_rewrite(self):
        state = self.create()
        directory = Path(state["session_directory"])
        state["fixed"]["shared"]["benchmark"].pop("mode")
        state["candidate_fingerprints"] = {"historical-fingerprint": [{"proposal_id": "old"}]}
        experiment_session._atomic_json(directory / "session.json", state)
        loaded = experiment_session.load_session(directory)
        self.assertNotIn("mode", loaded["fixed"]["shared"]["benchmark"])
        context = experiment_session.export_context(directory)
        self.assertEqual(context["benchmark_mode"], "allocating")
        self.assertIn("historical-fingerprint", context["candidate_fingerprints"])
        persisted = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        self.assertNotIn("mode", persisted["fixed"]["shared"]["benchmark"])


if __name__ == "__main__":
    unittest.main()
