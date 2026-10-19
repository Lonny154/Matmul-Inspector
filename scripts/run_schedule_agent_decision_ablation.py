#!/usr/bin/env python3
"""Compare matched Batch 2 model decisions with and without retained GPU evidence."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from matmul_inspector.tensor_game_schedule_agent import (
    LIMITS, OpenAIPlanner, live_hash_allowlists, validate_proposal,
)
from matmul_inspector.tensor_game_schedule_agent_ablation import (
    CONDITIONS, assess_proposal, canonical, common_context, compare_trials,
    condition_feedback, static_catalog,
)
from matmul_inspector.tensor_game_schedule_gpu_study import verify_artifact_hashes
from matmul_inspector.tensor_game_schedule_reselection import load_retained_corpus, select_batch1
from scripts.audit_schedule_agent_bundle import audit
from scripts.run_hardware_aware_schedule_lowering_study import REPO
from scripts.run_schedule_agent_prototype import contains_secret, read_rows, verified_batch2_feedback

EVIDENCE = REPO / "artifacts/schedule-agent/live-gpu-v4"
COMMITTED_V4_MANIFEST_SHA256 = "4467a6ffb3ae23b9cf35f647cf577b0f27f3e354fe93316803007d98c6b03045"
MAX_PAIRS = 3
MAX_API_CALLS = 6
MAX_REQUEST_BYTES = 24_000  # Narrower than Live-1's 32,000-byte call cap.


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_provenance() -> dict:
    import matmul_inspector.tensor_game_schedule_agent as agent_module
    import matmul_inspector.tensor_game_schedule_agent_ablation as ablation_module
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "git_branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
            "python": platform.python_version(),
            "runner_sha256": digest(Path(__file__)),
            "agent_module_sha256": digest(Path(agent_module.__file__)),
            "ablation_module_sha256": digest(Path(ablation_module.__file__))}


def load_inputs() -> dict:
    if digest(EVIDENCE / "hash_manifest.json") != COMMITTED_V4_MANIFEST_SHA256:
        raise ValueError("Live-1 source manifest differs from the committed v4 bundle")
    result = audit(EVIDENCE)
    if result["status"] != "PASS":
        raise ValueError("Live-1 evidence audit failed")
    candidates, _ = load_retained_corpus(REPO)
    batch1 = select_batch1(candidates, size=LIMITS.matrix_size)
    feedback = verified_batch2_feedback(EVIDENCE, candidates, batch1)
    catalog = static_catalog(candidates, batch1)
    control = condition_feedback(feedback, catalog, "control")
    treatment = condition_feedback(feedback, catalog, "treatment")
    if common_context(control) != common_context(treatment):
        raise ValueError("control and treatment contexts differ outside evidence")
    observations = read_rows(EVIDENCE / "batch1_observations.jsonl")
    return {"candidates": candidates, "batch1": batch1, "feedback": feedback,
            "catalog": catalog, "conditions": {"control": control, "treatment": treatment},
            "observations": observations,
            "source_manifest_sha256": digest(EVIDENCE / "hash_manifest.json"),
            "source_provenance": json.loads((EVIDENCE / "provenance.json").read_text()),
            "audit": result}


class Bundle:
    def __init__(self, path: Path):
        if path.exists():
            raise ValueError("output path already exists")
        path.mkdir(parents=True)
        self.path = path

    def write(self, name: str, value) -> None:
        content = canonical(value) + b"\n"
        manifest = {item.name: digest(item) for item in sorted(self.path.iterdir())
                    if item.is_file() and item.name not in ("hash_manifest.json", name)}
        manifest[name] = hashlib.sha256(content).hexdigest()
        manifest_bytes = canonical(manifest) + b"\n"
        used = sum(item.stat().st_size for item in self.path.iterdir()
                   if item.is_file() and item.name not in ("hash_manifest.json", name))
        if used + len(content) + len(manifest_bytes) > LIMITS.max_artifact_bytes:
            raise ValueError("ablation artifact budget exhausted")
        (self.path / name).write_bytes(content)
        (self.path / "hash_manifest.json").write_bytes(manifest_bytes)


def record_from_proposal(trial: int, condition: str, planner: OpenAIPlanner,
                         raw: dict, inputs: dict) -> dict:
    record = {"trial": trial, "condition": condition, "source": planner.source,
              "model": planner.model, "request_body": planner.last_request_body,
              "provider_metadata": planner.last_metadata, "proposal": raw}
    try:
        valid = validate_proposal(raw, batch=2, candidates=inputs["candidates"],
                                  batch1=inputs["batch1"])
    except Exception as error:
        # An invalid proposal remains raw. No fallback or repair is applied.
        rule = str(error).split(": ", 1)[0]
        secret = os.environ.get("OPENAI_API_KEY", "")
        record["validation"] = {"valid": False, "error_kind": type(error).__name__,
                                "rule": type(error).__name__ if secret and secret in rule else rule}
    else:
        record["validation"] = {"valid": True}
        record["assessment"] = assess_proposal(valid, inputs["candidates"],
            inputs["batch1"], inputs["observations"], inputs["feedback"])
    return record


def run(model: str, pairs: int, output: Path) -> dict:
    if not 1 <= pairs <= MAX_PAIRS:
        raise ValueError("paired trial count must be one through three")
    secret = os.environ.get("OPENAI_API_KEY", "")
    if not secret or secret in model or secret in str(output):
        raise ValueError("valid external API credential and safe model/output are required")
    inputs = load_inputs()
    bundle = Bundle(output)
    design = {"study": "Schedule-Agent-Live-2", "model": model,
              "code_provenance": code_provenance(),
              "pairs": pairs, "max_api_calls": MAX_API_CALLS,
              "max_total_reported_tokens": LIMITS.max_planner_total_tokens,
              "max_request_bytes": MAX_REQUEST_BYTES,
              "max_output_tokens_per_call": LIMITS.max_planner_output_tokens_per_call,
              "source_bundle": str(EVIDENCE.relative_to(REPO)),
              "source_manifest_sha256": inputs["source_manifest_sha256"],
              "source_provenance": inputs["source_provenance"],
              "source_audit": inputs["audit"],
              "common_context_sha256": hashlib.sha256(canonical(
                  common_context(inputs["conditions"]["control"]))).hexdigest(),
              "trial_orders": [list(CONDITIONS if index % 2 else reversed(CONDITIONS))
                               for index in range(1, pairs + 1)]}
    bundle.write("design.json", design)
    records: list[dict] = []
    calls = tokens = 0
    schema = None
    common_prompt = None
    try:
        for trial, order in enumerate(design["trial_orders"], 1):
            for condition in order:
                if calls >= MAX_API_CALLS or tokens + MAX_REQUEST_BYTES + (
                        LIMITS.max_planner_output_tokens_per_call) > LIMITS.max_planner_total_tokens:
                    raise ValueError("global model-call or token reservation budget exhausted")
                limits = replace(LIMITS, max_planner_calls=1,
                                 max_planner_request_bytes=MAX_REQUEST_BYTES)
                planner = OpenAIPlanner(model, limits, first_batch=2)
                try:
                    raw = planner.propose(2, inputs["conditions"][condition])
                except Exception as error:
                    if not contains_secret({"request": planner.last_request_body,
                                            "metadata": planner.last_metadata}, secret):
                        bundle.write(f"trial{trial}_{condition}.json", {
                            "trial": trial, "condition": condition, "source": planner.source,
                            "model": model, "request_body": planner.last_request_body,
                            "provider_metadata": planner.last_metadata,
                            "provider_error_kind": type(error).__name__,
                            "validation": {"valid": False}})
                    raise
                finally:
                    calls += planner.calls
                    tokens += planner.total_tokens
                if contains_secret({"proposal": raw, "metadata": planner.last_metadata,
                                    "request": planner.last_request_body}, secret):
                    raise ValueError("provider output contained a credential; raw output withheld")
                current_schema = planner.last_request_body["text"]["format"]["schema"]
                prompt = json.loads(planner.last_request_body["input"][0]["content"])
                current_common_prompt = {key: value for key, value in prompt.items()
                                         if key != "feedback"}
                if schema is None:
                    schema = current_schema
                elif schema != current_schema:
                    raise ValueError("matched conditions used different proposal schemas")
                if common_prompt is None:
                    common_prompt = current_common_prompt
                elif common_prompt != current_common_prompt:
                    raise ValueError("matched conditions used different tasks or limits")
                record = record_from_proposal(trial, condition, planner, raw, inputs)
                records.append(record)
                bundle.write(f"trial{trial}_{condition}.json", record)
                bundle.write("progress.json", {"api_calls": calls, "reported_tokens": tokens,
                                               "records": len(records)})
        summary = compare_trials(records)
        bundle.write("summary.json", summary)
        status = ("COMPLETED" if all(row["validation"]["valid"] for row in records)
                  else "COMPLETED_WITH_INVALID_PROPOSALS")
        return {"status": status, "api_calls": calls,
                "reported_tokens": tokens, "summary": summary, "output": str(output)}
    except Exception as error:
        bundle.write("failure.json", {"kind": type(error).__name__, "api_calls": calls,
                                      "reported_tokens": tokens, "records": len(records)})
        raise


def replay(path: Path) -> dict:
    manifest = verify_artifact_hashes(path)
    if set(manifest) != {item.name for item in path.iterdir() if item.is_file()} - {
            "hash_manifest.json"}:
        raise ValueError("ablation manifest omits or adds files")
    inputs = load_inputs()
    design = json.loads((path / "design.json").read_text())
    if design["source_manifest_sha256"] != inputs["source_manifest_sha256"] or (
            design["common_context_sha256"] != hashlib.sha256(canonical(
                common_context(inputs["conditions"]["control"]))).hexdigest()):
        raise ValueError("ablation source or common context changed")
    if (not 1 <= design["pairs"] <= MAX_PAIRS or
            design["trial_orders"] != [list(CONDITIONS if index % 2 else reversed(CONDITIONS))
                                       for index in range(1, design["pairs"] + 1)] or
            design["max_api_calls"] != MAX_API_CALLS or
            design["max_total_reported_tokens"] != LIMITS.max_planner_total_tokens):
        raise ValueError("saved paired design or budget changed")
    records = []
    schema = None
    common_prompt = None
    reported_tokens = 0
    for trial, order in enumerate(design["trial_orders"], 1):
        for condition in order:
            record = json.loads((path / f"trial{trial}_{condition}.json").read_text())
            if record["trial"] != trial or record["condition"] != condition:
                raise ValueError("trial identity changed")
            request = record["request_body"]
            usage = record["provider_metadata"]["usage"]
            if (record["source"] != OpenAIPlanner.source or
                    record["model"] != design["model"] or
                    request["model"] != design["model"] or request["store"] is not False or
                    record["provider_metadata"]["status"] != "completed" or
                    record["provider_metadata"]["api_calls"] != 1 or
                    any(type(usage.get(key)) is not int or usage[key] < 0 for key in (
                        "input_tokens", "output_tokens", "total_tokens")) or
                    usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"] or
                    usage["output_tokens"] > LIMITS.max_planner_output_tokens_per_call or
                    len(json.dumps(request, separators=(",", ":")).encode()) > MAX_REQUEST_BYTES):
                raise ValueError("saved provider response or request budget is invalid")
            reported_tokens += usage["total_tokens"]
            prompt = json.loads(request["input"][0]["content"])
            current_common_prompt = {key: value for key, value in prompt.items()
                                     if key != "feedback"}
            if common_prompt is None:
                common_prompt = current_common_prompt
            elif common_prompt != current_common_prompt:
                raise ValueError("saved tasks or limits differ between conditions")
            expected = {key: value for key, value in inputs["conditions"][condition].items()
                        if key != "eligible_batch2_hashes"}
            if prompt["batch"] != 2 or prompt["feedback"] != expected:
                raise ValueError("saved prompt differs from the intended condition")
            current_schema = request["text"]["format"]["schema"]
            scope, comparisons = live_hash_allowlists(2, inputs["conditions"][condition])
            properties = current_schema["properties"]
            if (current_schema["$defs"]["eligible_hash"]["enum"] != scope or
                    properties["comparison_preferences"]["items"]["enum"] != comparisons or
                    properties["requested_configurations"]["enum"] != [LIMITS.batch2] or
                    request["max_output_tokens"] != LIMITS.max_planner_output_tokens_per_call):
                raise ValueError("saved request changed schedule eligibility or budget")
            if schema is None:
                schema = current_schema
            elif schema != current_schema:
                raise ValueError("proposal schemas differ")
            if record["validation"]["valid"]:
                expected_assessment = assess_proposal(record["proposal"], inputs["candidates"],
                    inputs["batch1"], inputs["observations"], inputs["feedback"])
                if record["assessment"] != expected_assessment:
                    raise ValueError("saved decision assessment differs from replay")
            else:
                try:
                    validate_proposal(record["proposal"], batch=2,
                        candidates=inputs["candidates"], batch1=inputs["batch1"])
                except Exception as error:
                    if record["validation"]["rule"] != str(error).split(": ", 1)[0]:
                        raise ValueError("saved invalid-proposal rule differs from replay") from error
                else:
                    raise ValueError("saved invalid proposal now passes validation")
            records.append(record)
    if len(records) > MAX_API_CALLS or reported_tokens > LIMITS.max_planner_total_tokens:
        raise ValueError("saved model-only experiment exceeded its global budget")
    summary = compare_trials(records)
    if summary != json.loads((path / "summary.json").read_text()):
        raise ValueError("saved pair comparison differs from replay")
    return {"status": "PASS", "records": len(records),
            "selection_replay_exact": True, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--pairs", type=int, choices=range(1, MAX_PAIRS + 1), default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-bundle", type=Path)
    args = parser.parse_args()
    try:
        if args.replay_bundle:
            if args.model or args.output:
                parser.error("offline replay accepts only --replay-bundle")
            result = replay(args.replay_bundle)
        else:
            if not args.model or args.output is None:
                parser.error("model-only ablation requires --model and --output")
            result = run(args.model, args.pairs, args.output)
    except Exception as error:
        # Provider exceptions can contain arbitrary text. Keep CLI output typed.
        print(json.dumps({"status": "FAILED", "error_kind": type(error).__name__}),
              file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
