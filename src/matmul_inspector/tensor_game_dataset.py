"""Exact synthetic demonstration datasets for future neural TensorGame work."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Sequence

from .tensor_game import (
    RankOneAction,
    Tensor,
    outer_product,
    subtract_action,
    sum_actions,
    tensor_shape,
    validate_action,
    zero_tensor,
)
from .tensor_game_search import canonicalize_action

SCHEMA_VERSION = 1
GENERATOR_VERSION = "tg3a.synthetic.v1"
SPLITS = ("train", "validation", "test")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def tensor_to_list(tensor: Tensor) -> list[list[list[int]]]:
    return [[list(row) for row in plane] for plane in tensor]


def tensor_from_list(value: Any) -> Tensor:
    if not isinstance(value, list):
        raise ValueError("tensor must be a nested list")
    tensor = tuple(tuple(tuple(row) for row in plane) for plane in value)
    tensor_shape(tensor)
    return tensor


def action_to_dict(action: RankOneAction) -> dict[str, list[int]]:
    return {"u": list(action.u), "v": list(action.v), "w": list(action.w)}


def action_from_dict(value: Any) -> RankOneAction:
    if not isinstance(value, dict) or set(value) != {"u", "v", "w"}:
        raise ValueError("action must contain exactly u, v, and w")
    return RankOneAction(value["u"], value["v"], value["w"])


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    shape: tuple[int, int, int] = (4, 4, 4)
    construction_lengths: tuple[int, ...] = (1, 2, 3, 4)
    decompositions_per_length: int = 3
    coefficients: tuple[int, ...] = (-1, 0, 1)
    zero_probability: float = 0.5
    seed: int = 0
    split_weights: tuple[int, int, int] = (8, 1, 1)
    family: str = "synthetic_rank_one_to_four"
    max_generation_attempts: int = 10000

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in self.shape):
            raise ValueError("shape must contain three positive integers")
        if not self.construction_lengths or any(x <= 0 for x in self.construction_lengths):
            raise ValueError("construction lengths must be positive")
        if self.decompositions_per_length <= 0:
            raise ValueError("decompositions_per_length must be positive")
        if set(self.coefficients) != {-1, 0, 1}:
            raise ValueError("TG-3A supports the coefficient set {-1, 0, 1}")
        if not 0 <= self.zero_probability < 1:
            raise ValueError("zero_probability must be in [0, 1)")
        if len(self.split_weights) != 3 or any(x < 0 for x in self.split_weights) or not sum(self.split_weights):
            raise ValueError("split_weights must contain three nonnegative values with positive sum")
        if not self.family:
            raise ValueError("family must not be empty")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SyntheticDatasetConfig":
        return cls(
            shape=tuple(value["shape"]),
            construction_lengths=tuple(value["construction_lengths"]),
            decompositions_per_length=value["decompositions_per_length"],
            coefficients=tuple(value["coefficients"]),
            zero_probability=value["zero_probability"],
            seed=value["seed"],
            split_weights=tuple(value["split_weights"]),
            family=value["family"],
            max_generation_attempts=value["max_generation_attempts"],
        )


@dataclass(frozen=True)
class SyntheticDataset:
    manifest: dict[str, Any]
    decompositions: tuple[dict[str, Any], ...]
    samples: tuple[dict[str, Any], ...]


def _sample_vector(length: int, config: SyntheticDatasetConfig, rng: random.Random) -> tuple[int, ...]:
    for _ in range(100):
        vector = tuple(
            0 if rng.random() < config.zero_probability else rng.choice((-1, 1))
            for _ in range(length)
        )
        if any(vector):
            return vector
    index = rng.randrange(length)
    return tuple(rng.choice((-1, 1)) if i == index else 0 for i in range(length))


def sample_action(config: SyntheticDatasetConfig, rng: random.Random) -> RankOneAction:
    action = RankOneAction(*(_sample_vector(length, config, rng) for length in config.shape))
    validate_action(action, config.shape, frozenset(config.coefficients))
    return canonicalize_action(action)


def _split_schedule(config: SyntheticDatasetConfig, total: int, rng: random.Random) -> list[str]:
    weights = config.split_weights
    counts = [total * weight // sum(weights) for weight in weights]
    if total >= 3:
        for index in (1, 2):
            if weights[index] and counts[index] == 0:
                counts[index] = 1
    counts[0] = total - counts[1] - counts[2]
    if counts[0] < 0:
        raise ValueError("dataset is too small for the requested nonempty splits")
    schedule = [split for split, count in zip(SPLITS, counts) for _ in range(count)]
    rng.shuffle(schedule)
    return schedule


def _remaining_without_one(actions: Sequence[RankOneAction], selected: RankOneAction) -> tuple[RankOneAction, ...]:
    result = list(actions)
    result.remove(selected)
    return tuple(result)


def _trajectory(
    decomposition_id: str,
    factors: tuple[RankOneAction, ...],
    target: Tensor,
    config: SyntheticDatasetConfig,
) -> tuple[dict[str, Any], ...]:
    residual = target
    samples = []
    for step in range(len(factors)):
        remaining = factors[step:]
        unique_next = tuple(dict.fromkeys(remaining))
        for action in unique_next:
            expected = _remaining_without_one(remaining, action)
            expected_residual = sum_actions(expected, config.shape) if expected else zero_tensor(config.shape)
            if subtract_action(residual, action) != expected_residual:
                raise AssertionError("stored next action does not produce its exact remaining residual")
        residual_digest = _digest(tensor_to_list(residual))
        sample_id = _digest([decomposition_id, step, residual_digest])
        samples.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "decomposition_id": decomposition_id,
                "step": step,
                "residual": tensor_to_list(residual),
                "residual_digest": residual_digest,
                "remaining_factors": [action_to_dict(action) for action in remaining],
                "next_actions": [action_to_dict(action) for action in unique_next],
                "known_completion_length": len(remaining),
                "shape": list(config.shape),
                "coefficients": list(config.coefficients),
                "generator_seed": config.seed,
                "generator_version": GENERATOR_VERSION,
            }
        )
        residual = subtract_action(residual, factors[step])
    if residual != zero_tensor(config.shape):
        raise AssertionError("generated trajectory did not terminate at exact zero")
    return tuple(samples)


def generate_dataset(config: SyntheticDatasetConfig) -> SyntheticDataset:
    """Generate a deterministic, exact, split-safe synthetic curriculum."""
    rng = random.Random(config.seed)
    requested = len(config.construction_lengths) * config.decompositions_per_length
    schedule = _split_schedule(config, requested, rng)
    length_schedule = [
        length
        for length in config.construction_lengths
        for _ in range(config.decompositions_per_length)
    ]
    rng.shuffle(length_schedule)
    config_payload = json.loads(json.dumps(asdict(config), sort_keys=True))
    dataset_id = f"{config.family}-{_digest(config_payload)[:16]}"
    target_splits: dict[str, str] = {}
    residual_splits: dict[str, str] = {}
    sequence_digests: set[str] = set()
    decompositions: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    stats = {
        "requested_decompositions": requested,
        "accepted_decompositions": 0,
        "generated_samples": 0,
        "attempts": 0,
        "rejected_zero_targets": 0,
        "rejected_duplicate_targets": 0,
        "rejected_cross_split_residuals": 0,
        "canonical_sequence_duplicates": 0,
    }
    for index, (length, split) in enumerate(zip(length_schedule, schedule)):
        accepted = False
        while not accepted:
            stats["attempts"] += 1
            if stats["attempts"] > config.max_generation_attempts:
                raise RuntimeError("synthetic generation exhausted its attempt limit")
            origin_seed = rng.randrange(2**63)
            local_rng = random.Random(origin_seed)
            factors = tuple(sample_action(config, local_rng) for _ in range(length))
            target = sum_actions(factors, config.shape)
            if target == zero_tensor(config.shape):
                stats["rejected_zero_targets"] += 1
                continue
            target_digest = _digest(tensor_to_list(target))
            if target_digest in target_splits:
                stats["rejected_duplicate_targets"] += 1
                continue
            sequence_digest = _digest(sorted(_digest(action_to_dict(action)) for action in factors))
            if sequence_digest in sequence_digests:
                stats["canonical_sequence_duplicates"] += 1
                continue
            decomposition_id = f"{dataset_id}-d{index:06d}-{target_digest[:12]}"
            candidate_samples = _trajectory(decomposition_id, factors, target, config)
            collision = any(
                sample["residual_digest"] in residual_splits
                and residual_splits[sample["residual_digest"]] != split
                for sample in candidate_samples
            )
            if collision:
                stats["rejected_cross_split_residuals"] += 1
                continue
            record = {
                "schema_version": SCHEMA_VERSION,
                "decomposition_id": decomposition_id,
                "dataset_id": dataset_id,
                "family": config.family,
                "split": split,
                "shape": list(config.shape),
                "coefficients": list(config.coefficients),
                "construction_length": length,
                "construction_length_semantics": "verified_upper_bound_not_proven_rank",
                "zero_probability": config.zero_probability,
                "origin_seed": origin_seed,
                "generator_seed": config.seed,
                "generator_version": GENERATOR_VERSION,
                "target": tensor_to_list(target),
                "target_digest": target_digest,
                "factors": [action_to_dict(action) for action in factors],
                "canonical_factor_multiset_digest": sequence_digest,
                "sample_ids": [sample["sample_id"] for sample in candidate_samples],
            }
            record["content_digest"] = _digest(record)
            for sample in candidate_samples:
                sample["split"] = split
                sample["dataset_id"] = dataset_id
                sample["content_digest"] = _digest(sample)
                residual_splits.setdefault(sample["residual_digest"], split)
            target_splits[target_digest] = split
            sequence_digests.add(sequence_digest)
            decompositions.append(record)
            samples.extend(candidate_samples)
            accepted = True
    stats["accepted_decompositions"] = len(decompositions)
    stats["generated_samples"] = len(samples)
    split_counts = {split: sum(record["split"] == split for record in decompositions) for split in SPLITS}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "dataset_id": dataset_id,
        "family": config.family,
        "config": config_payload,
        "decomposition_count": len(decompositions),
        "sample_count": len(samples),
        "split_decomposition_counts": split_counts,
        "statistics": stats,
    }
    dataset = SyntheticDataset(manifest, tuple(decompositions), tuple(samples))
    validate_dataset(dataset)
    return dataset


def _validate_action_record(value: Any, shape: tuple[int, int, int], coefficients: frozenset[int]) -> RankOneAction:
    action = action_from_dict(value)
    validate_action(action, shape, coefficients)
    if canonicalize_action(action) != action:
        raise ValueError("stored action is not in canonical sign form")
    return action


def validate_dataset(dataset: SyntheticDataset) -> None:
    manifest = dataset.manifest
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported synthetic dataset schema version")
    if manifest.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("unsupported synthetic generator version")
    config = SyntheticDatasetConfig.from_dict(manifest["config"])
    expected_dataset_id = f"{config.family}-{_digest(manifest['config'])[:16]}"
    if manifest.get("dataset_id") != expected_dataset_id or manifest.get("family") != config.family:
        raise ValueError("synthetic dataset identity does not match its configuration")
    if manifest["decomposition_count"] != len(dataset.decompositions) or manifest["sample_count"] != len(dataset.samples):
        raise ValueError("manifest record counts do not match dataset contents")
    by_decomposition: dict[str, list[dict[str, Any]]] = {}
    target_split: dict[str, str] = {}
    residual_split: dict[str, str] = {}
    for sample in dataset.samples:
        required = {
            "schema_version", "sample_id", "decomposition_id", "step", "residual",
            "residual_digest", "remaining_factors", "next_actions", "known_completion_length",
            "shape", "coefficients", "generator_seed", "generator_version", "split",
            "dataset_id", "content_digest",
        }
        if set(sample) != required or sample["schema_version"] != SCHEMA_VERSION:
            raise ValueError("corrupt synthetic sample schema")
        content = dict(sample)
        digest = content.pop("content_digest")
        if _digest(content) != digest:
            raise ValueError("synthetic sample content digest mismatch")
        residual = tensor_from_list(sample["residual"])
        if tuple(sample["shape"]) != config.shape or tensor_shape(residual) != config.shape:
            raise ValueError("synthetic sample shape mismatch")
        if _digest(sample["residual"]) != sample["residual_digest"]:
            raise ValueError("synthetic residual digest mismatch")
        if sample["dataset_id"] != manifest["dataset_id"] or sample["generator_version"] != GENERATOR_VERSION:
            raise ValueError("synthetic sample provenance mismatch")
        if sample["split"] not in SPLITS or sample["coefficients"] != list(config.coefficients):
            raise ValueError("synthetic sample split or coefficient metadata is invalid")
        if sample["sample_id"] != _digest(
            [sample["decomposition_id"], sample["step"], sample["residual_digest"]]
        ):
            raise ValueError("synthetic sample identifier mismatch")
        previous = residual_split.setdefault(sample["residual_digest"], sample["split"])
        if previous != sample["split"]:
            raise ValueError("identical residual appears across dataset splits")
        remaining = tuple(
            _validate_action_record(value, config.shape, frozenset(config.coefficients))
            for value in sample["remaining_factors"]
        )
        next_actions = tuple(
            _validate_action_record(value, config.shape, frozenset(config.coefficients))
            for value in sample["next_actions"]
        )
        if len(remaining) != sample["known_completion_length"] or set(next_actions) != set(remaining):
            raise ValueError("next-action targets do not match the remaining factor multiset")
        if sum_actions(remaining, config.shape) != residual:
            raise ValueError("remaining factors do not reconstruct the stored residual")
        for action in next_actions:
            rest = _remaining_without_one(remaining, action)
            expected = sum_actions(rest, config.shape) if rest else zero_tensor(config.shape)
            if subtract_action(residual, action) != expected:
                raise ValueError("next action does not produce a valid exact transition")
        by_decomposition.setdefault(sample["decomposition_id"], []).append(sample)
    for record in dataset.decompositions:
        required = {
            "schema_version", "decomposition_id", "dataset_id", "family", "split", "shape",
            "coefficients", "construction_length", "construction_length_semantics",
            "zero_probability", "origin_seed", "generator_seed", "generator_version", "target",
            "target_digest", "factors", "canonical_factor_multiset_digest", "sample_ids", "content_digest",
        }
        if set(record) != required or record["schema_version"] != SCHEMA_VERSION:
            raise ValueError("corrupt synthetic decomposition schema")
        content = dict(record)
        digest = content.pop("content_digest")
        if _digest(content) != digest:
            raise ValueError("synthetic decomposition content digest mismatch")
        target = tensor_from_list(record["target"])
        factors = tuple(
            _validate_action_record(value, config.shape, frozenset(config.coefficients))
            for value in record["factors"]
        )
        if len(factors) != record["construction_length"] or sum_actions(factors, config.shape) != target:
            raise ValueError("stored factors do not reconstruct the target exactly")
        expected_sequence_digest = _digest(sorted(_digest(action_to_dict(action)) for action in factors))
        if record["canonical_factor_multiset_digest"] != expected_sequence_digest:
            raise ValueError("canonical factor multiset digest mismatch")
        if _digest(record["target"]) != record["target_digest"]:
            raise ValueError("synthetic target digest mismatch")
        if (
            record["dataset_id"] != manifest["dataset_id"]
            or record["split"] not in SPLITS
            or record["generator_version"] != GENERATOR_VERSION
            or not record["decomposition_id"].startswith(f"{manifest['dataset_id']}-d")
            or not record["decomposition_id"].endswith(record["target_digest"][:12])
        ):
            raise ValueError("synthetic decomposition identity or provenance mismatch")
        if record["target_digest"] in target_split:
            raise ValueError("duplicate target tensor appears in the dataset")
        target_split[record["target_digest"]] = record["split"]
        rows = sorted(by_decomposition.get(record["decomposition_id"], ()), key=lambda row: row["step"])
        if [row["sample_id"] for row in rows] != record["sample_ids"]:
            raise ValueError("decomposition sample linkage is corrupt")
        if any(row["split"] != record["split"] for row in rows):
            raise ValueError("decomposition trajectory crosses dataset splits")


def save_dataset(dataset: SyntheticDataset, output: Path) -> None:
    """Atomically create a dataset directory and refuse silent overwrites."""
    if output.exists():
        raise FileExistsError(output)
    validate_dataset(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        decomposition_bytes = b"".join(_json_bytes(record) for record in dataset.decompositions)
        sample_bytes = b"".join(_json_bytes(record) for record in dataset.samples)
        manifest = dict(dataset.manifest)
        manifest["files"] = {
            "decompositions.jsonl": hashlib.sha256(decomposition_bytes).hexdigest(),
            "samples.jsonl": hashlib.sha256(sample_bytes).hexdigest(),
        }
        (temporary / "decompositions.jsonl").write_bytes(decomposition_bytes)
        (temporary / "samples.jsonl").write_bytes(sample_bytes)
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path.name} line {line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"invalid record in {path.name} line {line_number}")
        records.append(value)
    return tuple(records)


def load_dataset(path: Path) -> SyntheticDataset:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported synthetic dataset schema version")
    if set(manifest.get("files", {})) != {"decompositions.jsonl", "samples.jsonl"}:
        raise ValueError("synthetic dataset manifest has invalid file metadata")
    for filename, expected in manifest["files"].items():
        if hashlib.sha256((path / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"synthetic dataset file digest mismatch: {filename}")
    core_manifest = dict(manifest)
    core_manifest.pop("files")
    dataset = SyntheticDataset(
        core_manifest,
        _read_jsonl(path / "decompositions.jsonl"),
        _read_jsonl(path / "samples.jsonl"),
    )
    validate_dataset(dataset)
    return dataset


class TorchSyntheticDataset:
    """Optional DataLoader-compatible adapter with padded multi-action labels."""

    def __init__(self, dataset: SyntheticDataset, split: str = "train", dtype: str = "float32"):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        try:
            import torch
        except ImportError as error:
            raise ImportError("TorchSyntheticDataset requires optional PyTorch") from error
        if dtype not in {"float32", "float64", "int64"}:
            raise ValueError("dtype must be float32, float64, or int64")
        self.torch = torch
        self.dtype = getattr(torch, dtype)
        self.samples = tuple(sample for sample in dataset.samples if sample["split"] == split)
        self.shape = tuple(dataset.manifest["config"]["shape"])
        self.coefficients = tuple(dataset.manifest["config"]["coefficients"])
        self.max_targets = max(dataset.manifest["config"]["construction_lengths"])
        self.action_width = sum(self.shape)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        flat_residual = [value for plane in sample["residual"] for row in plane for value in row]
        exact_limit = {
            self.torch.float32: 2**24,
            self.torch.float64: 2**53,
            self.torch.int64: 2**63 - 1,
        }[self.dtype]
        if any(abs(value) > exact_limit for value in flat_residual):
            raise OverflowError(f"{self.dtype} conversion would not preserve every exact integer")
        actions = [value for action in sample["next_actions"] for value in (action["u"] + action["v"] + action["w"])]
        padded = actions + [0] * (self.max_targets * self.action_width - len(actions))
        mask = [True] * len(sample["next_actions"]) + [False] * (self.max_targets - len(sample["next_actions"]))
        return {
            "residual": self.torch.tensor(flat_residual, dtype=self.dtype).reshape(self.shape),
            "next_actions": self.torch.tensor(padded, dtype=self.torch.int64).reshape(self.max_targets, self.action_width),
            "action_mask": self.torch.tensor(mask, dtype=self.torch.bool),
            "known_completion_length": self.torch.tensor(sample["known_completion_length"], dtype=self.torch.int64),
            "allowed_coefficients": self.torch.tensor(self.coefficients, dtype=self.torch.int64),
        }
