#!/usr/bin/env python3
"""CPU-only tests for kernel candidate sampling."""

from __future__ import annotations

import math
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matmul_inspector.sampling import (
    CandidateConstraint, SearchContext, SearchSpaceExhausted, TritonGemmConfig,
    TritonGemmSearchSpace, UniformSampler, WeightedSampler, make_sampling_records,
    read_jsonl, write_jsonl,
)


def space(*, constraints=()):
    return TritonGemmSearchSpace(
        {"block_m": [32, 64], "block_n": [64, 128], "block_k": [16],
         "num_warps": [4, 8], "num_stages": [2]},
        SearchContext(512, 768, 3072, "fp16", hardware_target="sm_89"),
        constraints=constraints, identifier="fc1-test",
    )


def equal_weights(search_space):
    return {name: {value: 1.0 for value in values}
            for name, values in search_space.values.items()}


class SamplingTests(unittest.TestCase):
    def test_uniform_fixed_seed_is_reproducible(self):
        self.assertEqual(
            UniformSampler(space(), seed=42).sample(6, unique=True),
            UniformSampler(space(), seed=42).sample(6, unique=True),
        )

    def test_unique_sampling_tracks_duplicates_across_calls_and_exhausts(self):
        sampler = UniformSampler(space(), seed=3)
        first = sampler.sample(5, unique=True)
        second = sampler.sample(3, unique=True)
        self.assertEqual(len(set(first + second)), 8)
        with self.assertRaisesRegex(SearchSpaceExhausted, "only 0 remain"):
            sampler.sample(unique=True)

    def test_nonunique_sampling_allows_replacement(self):
        one = TritonGemmSearchSpace(
            {"block_m": [32], "block_n": [64], "block_k": [16],
             "num_warps": [4], "num_stages": [2]},
            SearchContext(1, 1, 1, "fp16"),
        )
        self.assertEqual(len(UniformSampler(one, seed=1).sample(3)), 3)

    def test_search_space_and_candidate_validation(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            TritonGemmSearchSpace(
                {"block_m": [32, 32], "block_n": [64], "block_k": [16],
                 "num_warps": [4], "num_stages": [2]},
                SearchContext(1, 1, 1, "fp16"),
            )
        constrained = space(constraints=[CandidateConstraint(
            "large-tile-needs-eight-warps",
            lambda config, _: config.block_m * config.block_n <= 4096 or config.num_warps == 8,
        )])
        rejected = TritonGemmConfig(64, 128, 16, 4, 2)
        with self.assertRaisesRegex(ValueError, "large-tile-needs-eight-warps"):
            constrained.validate(rejected)
        self.assertNotIn(rejected, constrained.valid_candidates())

    def test_weighted_fixed_seed_is_reproducible_and_weights_have_effect(self):
        search_space = space()
        weights = equal_weights(search_space)
        weights["block_m"] = {32: 0.0, 64: 5.0}
        first = WeightedSampler(search_space, seed=9, weights=weights).sample(20)
        second = WeightedSampler(search_space, seed=9, weights=weights).sample(20)
        self.assertEqual(first, second)
        self.assertTrue(all(item.block_m == 64 for item in first))

    def test_weighted_unique_sampling_and_zero_weight_exhaustion(self):
        search_space = space()
        weights = equal_weights(search_space)
        weights["block_m"] = {32: 0.0, 64: 1.0}
        sampler = WeightedSampler(search_space, seed=2, weights=weights)
        self.assertEqual(len(sampler.sample(4, unique=True)), 4)
        with self.assertRaisesRegex(SearchSpaceExhausted, "strictly positive"):
            sampler.sample(unique=True)

    def test_weighted_unique_preflight_is_transactional(self):
        search_space = space()
        weights = equal_weights(search_space)
        weights["block_m"] = {32: 0.0, 64: 1.0}
        sampler = WeightedSampler(search_space, seed=2, weights=weights)
        rng_state = sampler._random.getstate()
        seen = set(sampler._seen)
        with self.assertRaisesRegex(SearchSpaceExhausted, "only 4 remain"):
            sampler.sample(5, unique=True)
        self.assertEqual(sampler._random.getstate(), rng_state)
        self.assertEqual(sampler._seen, seen)
        expected = WeightedSampler(search_space, seed=2, weights=weights).sample(4, unique=True)
        self.assertEqual(sampler.sample(4, unique=True), expected)

    def test_all_valid_combinations_have_zero_weight(self):
        constrained = space(constraints=[CandidateConstraint(
            "only-block-m-32", lambda config, _: config.block_m == 32,
        )])
        weights = equal_weights(constrained)
        weights["block_m"] = {32: 0.0, 64: 1.0}
        sampler = WeightedSampler(constrained, seed=4, weights=weights)
        rng_state = sampler._random.getstate()
        with self.assertRaisesRegex(SearchSpaceExhausted, "strictly positive"):
            sampler.sample(unique=True)
        self.assertEqual(sampler._random.getstate(), rng_state)
        self.assertEqual(sampler._seen, set())

    def test_invalid_weights_are_rejected(self):
        for invalid in (-1.0, math.inf, math.nan):
            weights = equal_weights(space())
            weights["block_m"][32] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                WeightedSampler(space(), seed=0, weights=weights)
        weights = equal_weights(space())
        weights["block_m"] = {32: 0.0, 64: 0.0}
        with self.assertRaisesRegex(ValueError, "positive"):
            WeightedSampler(space(), seed=0, weights=weights)

    def test_valid_candidate_enumeration_is_cached_for_all_samplers(self):
        evaluations = 0

        def count_validation(config, context):
            nonlocal evaluations
            evaluations += 1
            return True

        search_space = space(constraints=[CandidateConstraint("count", count_validation)])
        first = search_space.valid_candidates()
        second = search_space.valid_candidates()
        UniformSampler(search_space, seed=1).sample()
        WeightedSampler(search_space, seed=1, weights=equal_weights(search_space)).sample()
        self.assertIs(first, second)
        self.assertEqual(evaluations, 8)
        with self.assertRaises(TypeError):
            search_space.values["block_m"] = (128,)
        with self.assertRaises(AttributeError):
            search_space.constraints = ()

    def test_config_serialization_and_jsonl_round_trip(self):
        sampler = UniformSampler(space(), seed=17)
        candidates = sampler.sample(3, unique=True)
        records = make_sampling_records(candidates, sampler, timestamp="2026-10-13T12:00:00Z")
        self.assertEqual(TritonGemmConfig.from_dict(candidates[0].to_dict()), candidates[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            write_jsonl(path, records)
            self.assertEqual(read_jsonl(path), records)
        self.assertEqual([item.sampling_order for item in records], [0, 1, 2])
        self.assertEqual(len({item.candidate_id for item in records}), 3)

    def test_version_one_extensions_accept_additional_data(self):
        sampler = UniformSampler(space(), seed=17)
        record = make_sampling_records(sampler.sample(), sampler,
                                       timestamp="2026-10-13T12:00:00Z")[0]
        value = record.to_dict()
        value["extensions"] = {
            "compilation": {"status": "pending"},
            "future_namespace": {"arbitrary": [1, 2, 3]},
        }
        parsed = type(record).from_dict(value)
        self.assertEqual(parsed.extensions, value["extensions"])

    def test_version_one_schema_errors_are_explicit(self):
        sampler = UniformSampler(space(), seed=17)
        value = make_sampling_records(sampler.sample(), sampler,
                                      timestamp="2026-10-13T12:00:00Z")[0].to_dict()
        cases = []
        missing = dict(value)
        missing.pop("timestamp")
        cases.append((missing, "missing required fields: timestamp"))
        unknown = {**value, "new_top_level_field": True}
        cases.append((unknown, "unknown version-1"))
        unsupported = {**value, "schema_version": 2}
        cases.append((unsupported, "unsupported sampling record schema: 2"))
        malformed_context = {**value, "context": {"m": "not-an-integer"}}
        cases.append((malformed_context, "invalid version-1 sampling record data"))
        for malformed, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                type(make_sampling_records(sampler.sample(), sampler)[0]).from_dict(malformed)

    def test_jsonl_rejects_malformed_context_with_line_number(self):
        sampler = UniformSampler(space(), seed=17)
        value = make_sampling_records(sampler.sample(), sampler)[0].to_dict()
        value["context"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.jsonl"
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1.*context must be an object"):
                read_jsonl(path)


if __name__ == "__main__":
    unittest.main()
