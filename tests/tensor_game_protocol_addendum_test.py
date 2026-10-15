import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from matmul_inspector.tensor_game_protocol_addendum import (
    content_digest,
    expected_trial_count,
    validate_protocol_addendum,
)


ABLATIONS = [
    "residual_informed_policy_plus_heuristic_value",
    "residual_informed_policy_plus_constant_max_moves_value",
    "residual_informed_policy_plus_checkpoint_value",
    "learned_policy_plus_heuristic_value",
    "learned_policy_plus_checkpoint_value",
]
MCTS = {
    "simulations": 200, "proposals": 12, "c_puct": 1.5, "node_limit": 200,
    "episode_wall_time_seconds": 5.0, "move_limit_policy": "construction_length_upper_bound",
}


class ProtocolAddendumTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"fixture checkpoint")
        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.plan_path = self.root / "evaluation_plan.json"
        self.plan = {
            "status": "frozen_before_held_out_evaluation", "held_out_results_present": False,
            "trial_seeds": [3, 11], "mcts": MCTS, "search_ablations": ABLATIONS,
            "checkpoint_comparisons": {"fixture": {
                "checkpoint_sha256": digest,
                "value_semantics": "tg3b_predicted_completion_length",
                "value_adapter": "completion_length_to_leaf_value_with_terminal_overrides",
            }},
        }
        self._write(self.plan_path, self.plan)
        self.addendum_path = self.root / "addendum.json"
        self.addendum = self._addendum(digest)
        self._write_addendum()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write(path, value):
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def _configuration(self, index, ablation):
        policy = "checkpoint_policy" if ablation.startswith("learned") else "residual_informed"
        value = (
            "checkpoint_value" if ablation.endswith("checkpoint_value") else
            "constant_max_moves" if "constant" in ablation else "nonzero_heuristic"
        )
        return {
            "configuration_id": f"config_{index}", "frozen_plan_ablation": ablation,
            "policy_source": "checkpoint:fixture" if policy == "checkpoint_policy" else policy,
            "value_source": "checkpoint:fixture" if value == "checkpoint_value" else value,
            "value_head_semantics": (
                "tg3b_predicted_completion_length" if value == "checkpoint_value" else "not_applicable"
            ),
            "action_proposal_method": policy, "leaf_evaluation_method": value,
            "search_parameters": MCTS, "role": "primary" if index else "control",
        }

    def _addendum(self, digest):
        return {
            "schema_version": 1, "addendum_id": "fixture", "approval_status": "pending_review",
            "original_plan": {"path": str(self.plan_path), "sha256": hashlib.sha256(self.plan_path.read_bytes()).hexdigest()},
            "held_out_evaluation_occurred": False,
            "checkpoint_path_resolutions": {"fixture": {
                "resolved_actual_path": str(self.checkpoint), "expected_sha256": digest,
                "observed_sha256": digest, "value_head_semantics": "tg3b_predicted_completion_length",
            }},
            "model_search_matrix": [self._configuration(i, value) for i, value in enumerate(ABLATIONS)],
            "bootstrap_analysis": {
                "confidence_level": 0.95, "resamples": 10000, "seed": 20261013,
                "resampling_unit": "originating_tensor_target_group",
                "keep_all_trial_seeds_together": True,
                "comparison_statistic": "paired_target_level_success_rate_difference",
                "confidence_interval_method": "percentile_cluster_bootstrap",
            },
            "wall_time_interpretation": {
                "scope": "entire_multi_decision_episode", "inner_search_uses_remaining_episode_time": True,
            },
            "trial_accounting": {"held_out_target_groups": 30, "expected_trial_count": 300},
        }

    def _write_addendum(self):
        self.addendum["content_digest"] = content_digest(self.addendum)
        self._write(self.addendum_path, self.addendum)

    def _validate(self, **kwargs):
        return validate_protocol_addendum(self.plan_path, self.addendum_path, load_models=False, **kwargs)

    def test_valid_addendum_and_trial_count(self):
        report = self._validate(output_path=self.root / "new-output")
        self.assertTrue(report["valid"])
        self.assertFalse(report["execution_ready"])
        self.assertEqual(expected_trial_count(self.plan, self.addendum), 300)
        self.assertEqual(report["unresolved_ambiguities"], [])

    def test_missing_checkpoint_resolution(self):
        self.checkpoint.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "missing checkpoint resolution"):
            self._validate()

    def test_checkpoint_digest_mismatch(self):
        self.checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint digest mismatch"):
            self._validate()

    def test_unsupported_value_semantics(self):
        self.addendum["checkpoint_path_resolutions"]["fixture"]["value_head_semantics"] = "unknown"
        self._write_addendum()
        with self.assertRaisesRegex(ValueError, "unsupported value semantics"):
            self._validate()

    def test_duplicate_configuration_ids(self):
        self.addendum["model_search_matrix"][1]["configuration_id"] = "config_0"
        self._write_addendum()
        with self.assertRaisesRegex(ValueError, "duplicate configuration IDs"):
            self._validate()

    def test_missing_configuration_mapping(self):
        del self.addendum["model_search_matrix"][0]["policy_source"]
        self._write_addendum()
        with self.assertRaisesRegex(ValueError, "missing configuration mappings"):
            self._validate()

    def test_grouped_bootstrap_and_wall_time_are_fixed(self):
        for section, key in (("bootstrap_analysis", "resampling_unit"), ("wall_time_interpretation", "scope")):
            changed = copy.deepcopy(self.addendum)
            changed[section][key] = "different"
            changed["content_digest"] = content_digest(changed)
            self._write(self.addendum_path, changed)
            with self.assertRaises(ValueError):
                self._validate()

    def test_addendum_integrity(self):
        self.addendum["trial_accounting"]["expected_trial_count"] = 301
        self._write(self.addendum_path, self.addendum)
        with self.assertRaisesRegex(ValueError, "content digest mismatch"):
            self._validate()

    def test_frozen_plan_digest_mismatch(self):
        self.addendum["original_plan"]["sha256"] = "0" * 64
        self._write_addendum()
        with self.assertRaisesRegex(ValueError, "frozen-plan digest mismatch"):
            self._validate()

    def test_execution_rejected_while_pending_review(self):
        with self.assertRaisesRegex(PermissionError, "verified approval artifact"):
            self._validate(require_approved=True)

    def test_separate_approval_artifact_is_verified(self):
        approval = {
            "approval_status": "approved", "explicit_user_approval": True,
            "frozen_plan_sha256": hashlib.sha256(self.plan_path.read_bytes()).hexdigest(),
            "addendum_content_digest": self.addendum["content_digest"],
            "approved_configuration_count": 5, "approved_trial_count": 300,
            "episode_wall_time_seconds": 5.0, "frozen_documents_unchanged": True,
        }
        approval["content_digest"] = content_digest(approval)
        path = self.root / "approval.json"
        self._write(path, approval)
        report = self._validate(approval_path=path)
        self.assertTrue(report["review_approved"])
        self.assertFalse(report["execution_ready"])  # Fixture validation intentionally disables model wiring.


if __name__ == "__main__":
    unittest.main()
