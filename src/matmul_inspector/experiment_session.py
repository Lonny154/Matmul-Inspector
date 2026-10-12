"""Persistent, budgeted search sessions built on the experiment batch runner."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import traceback
from typing import Any, Callable, Iterator, Mapping, Sequence
import uuid

from .experiment import (DEFAULT_CONFIG, SOURCE_IMPLEMENTATION, _git_metadata,
                         resolved_benchmark_mode, snapshot_candidate_source, validate_config)
from .experiment_batch import run_batch


SESSION_SCHEMA_VERSION = 1
SESSION_FILENAME = "session.json"
LOCK_FILENAME = ".session.lock"
PARAMETER_NAMES = frozenset(DEFAULT_CONFIG["candidate"]["parameters"])
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SessionError(RuntimeError):
    """A user-actionable session error."""


class SessionLockError(SessionError):
    """The session is already being changed by another process."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SessionError(f"cannot read JSON {path}: {error}") from error


def _session_directory(path: Path | str) -> Path:
    candidate = Path(path).resolve()
    return candidate.parent if candidate.name == SESSION_FILENAME else candidate


def load_session(path: Path | str) -> dict[str, Any]:
    directory = _session_directory(path)
    state = _read_json(directory / SESSION_FILENAME)
    if state.get("schema_version") != SESSION_SCHEMA_VERSION:
        raise SessionError(f"unsupported session schema: {state.get('schema_version')!r}")
    return state


@contextmanager
def _session_lock(directory: Path, *, blocking: bool = False) -> Iterator[None]:
    lock_path = directory / LOCK_FILENAME
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError as error:
            raise SessionLockError(f"session is locked by another process: {directory}") from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must match {IDENTIFIER_PATTERN.pattern}")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_shared(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("shared must be an object")
    required = {"workload", "seed", "correctness", "benchmark", "timeout_seconds", "baseline"}
    if set(raw) != required:
        missing, unknown = sorted(required - set(raw)), sorted(set(raw) - required)
        details = ([f"missing: {', '.join(missing)}"] if missing else [])
        details += ([f"unknown: {', '.join(unknown)}"] if unknown else [])
        raise ValueError("invalid shared fields (" + "; ".join(details) + ")")
    placeholder = _json_copy(DEFAULT_CONFIG["candidate"])
    placeholder["parameters"]["block_k"] = 1
    resolved = validate_config({**raw, "candidate": placeholder})
    return {key: resolved[key] for key in required}


def _validate_allowed(shared: Mapping[str, Any], implementation: Any,
                      raw: Any) -> dict[str, list[int]]:
    if implementation not in {"triton_fc1", SOURCE_IMPLEMENTATION}:
        raise ValueError("implementation must be 'triton_fc1' or 'triton_source'")
    if not isinstance(raw, Mapping) or set(raw) != PARAMETER_NAMES:
        raise ValueError("allowed_parameters must contain exactly: " +
                         ", ".join(sorted(PARAMETER_NAMES)))
    allowed: dict[str, list[int]] = {}
    defaults = DEFAULT_CONFIG["candidate"]["parameters"]
    for name in sorted(PARAMETER_NAMES):
        values = raw[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"allowed_parameters.{name} must be a non-empty list")
        if len(values) != len(set(values)):
            raise ValueError(f"allowed_parameters.{name} contains duplicates")
        allowed[name] = []
        for value in values:
            _positive_int(value, f"allowed_parameters.{name} value")
            parameters = {**defaults, name: value}
            # Parameter constraints are shared with the registered kernel. Source
            # paths are proposal-specific and are validated when submitted.
            validate_config({**shared, "candidate": {
                "implementation": "triton_fc1", "parameters": parameters,
            }})
            allowed[name].append(value)
    return allowed


def validate_session_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("session configuration must be a JSON object")
    required = {"objective", "shared", "implementation", "allowed_parameters",
                "total_experiment_budget", "scheduling_seed"}
    optional = {"allow_source_candidates"}
    if not required <= set(raw) or set(raw) - required - optional:
        missing, unknown = sorted(required - set(raw)), sorted(set(raw) - required)
        unknown = sorted(set(raw) - required - optional)
        details = ([f"missing: {', '.join(missing)}"] if missing else [])
        details += ([f"unknown: {', '.join(unknown)}"] if unknown else [])
        raise ValueError("invalid session fields (" + "; ".join(details) + ")")
    objective = raw["objective"]
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("objective must be a non-empty string")
    shared = _validate_shared(raw["shared"])
    implementation = raw["implementation"]
    allow_source = raw.get("allow_source_candidates", False)
    if not isinstance(allow_source, bool):
        raise ValueError("allow_source_candidates must be a boolean")
    if implementation == SOURCE_IMPLEMENTATION and not allow_source:
        raise ValueError("triton_source sessions require allow_source_candidates=true")
    if implementation == SOURCE_IMPLEMENTATION and shared["benchmark"]["mode"] != "preallocated":
        raise ValueError("triton_source sessions require benchmark.mode='preallocated'")
    allowed = _validate_allowed(shared, implementation, raw["allowed_parameters"])
    total = _positive_int(raw["total_experiment_budget"], "total_experiment_budget")
    seed = raw["scheduling_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("scheduling_seed must be a non-negative integer")
    return {"objective": objective.strip(), "shared": shared, "implementation": implementation,
            "allowed_parameters": allowed, "total_experiment_budget": total,
            "scheduling_seed": seed, "allow_source_candidates": allow_source}


def _environment() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": platform.python_version()}
    for package in ("torch", "triton"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"platform": platform.platform(), "versions": versions}


def _budget(state: dict[str, Any]) -> None:
    total = state["budget"]["total"]
    consumed = sum(item.get("consumed_units", 0) for item in state["proposals"])
    reserved = sum(item.get("reserved_units", 0) for item in state["proposals"])
    state["budget"].update({"consumed": consumed, "reserved": reserved,
                            "remaining": total - consumed - reserved})


def create_session(raw_config: Mapping[str, Any], *,
                   output_root: Path | str = "artifacts/experiment-sessions") -> dict[str, Any]:
    config = validate_session_config(raw_config)
    created = datetime.now(timezone.utc)
    session_id = f"{created.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    directory = Path(output_root).resolve() / session_id
    directory.mkdir(parents=True, exist_ok=False)
    state = {
        "schema_version": SESSION_SCHEMA_VERSION, "session_id": session_id,
        "created_at_utc": created.isoformat(), "updated_at_utc": created.isoformat(),
        "session_directory": str(directory), "objective": config["objective"],
        "fixed": {"shared": config["shared"], "implementation": config["implementation"],
                  "allowed_parameters": config["allowed_parameters"],
                  "scheduling_seed": config["scheduling_seed"],
                  "allow_source_candidates": config["allow_source_candidates"]},
        "budget": {"total": config["total_experiment_budget"], "consumed": 0,
                   "reserved": 0, "remaining": config["total_experiment_budget"]},
        "proposals": [], "rejected_proposals": [], "candidate_fingerprints": {},
        "provenance": {"git": _git_metadata(Path(__file__).resolve().parents[2]),
                       "environment": _environment()},
    }
    _atomic_json(directory / "submitted_session_config.json", raw_config)
    _atomic_json(directory / SESSION_FILENAME, state)
    return state


def _resolve_candidate(state: Mapping[str, Any], raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"candidates[{index}] must be an object")
    allowed_fields = {"id", "implementation", "parameters", "hypothesis", "repeat_reason"}
    if state["fixed"]["implementation"] == SOURCE_IMPLEMENTATION:
        allowed_fields.add("source")
    unknown = sorted(set(raw) - allowed_fields)
    required = {"id", "implementation", "parameters", "hypothesis"}
    missing = sorted(required - set(raw))
    if unknown or missing:
        raise ValueError(f"candidates[{index}] fields invalid; missing={missing}, unknown={unknown}")
    identifier = _identifier(raw["id"], f"candidates[{index}].id")
    if raw["implementation"] != state["fixed"]["implementation"]:
        raise ValueError(f"candidate {identifier}: unsupported implementation")
    if not isinstance(raw["parameters"], Mapping):
        raise ValueError(f"candidate {identifier}: parameters must be an object")
    unknown_parameters = sorted(set(raw["parameters"]) - PARAMETER_NAMES)
    if unknown_parameters:
        raise ValueError(f"candidate {identifier}: unknown parameters: {', '.join(unknown_parameters)}")
    parameters = {**DEFAULT_CONFIG["candidate"]["parameters"], **raw["parameters"]}
    candidate_config = {
        "implementation": raw["implementation"], "parameters": parameters,
        **({"source": raw.get("source")} if raw["implementation"] == SOURCE_IMPLEMENTATION else {}),
    }
    resolved_config = validate_config({**state["fixed"]["shared"], "candidate": candidate_config})
    for name, value in parameters.items():
        if value not in state["fixed"]["allowed_parameters"][name]:
            raise ValueError(f"candidate {identifier}: {name}={value} is outside allowed values")
    hypothesis = raw["hypothesis"]
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError(f"candidate {identifier}: hypothesis must be non-empty")
    repeat_reason = raw.get("repeat_reason")
    if repeat_reason is not None and (not isinstance(repeat_reason, str) or not repeat_reason.strip()):
        raise ValueError(f"candidate {identifier}: repeat_reason must be non-empty when provided")
    source = resolved_config["candidate"].get("source")
    if source:
        payload = {
            "implementation": raw["implementation"], "parameters": parameters,
            "source_sha256": source["sha256"], "entry_point": source["entry_point"],
            "interface_version": source["interface_version"],
            "shared": state["fixed"]["shared"],
        }
    else:
        # Keep this exact legacy payload so old registered-candidate identities
        # and duplicate detection remain stable.
        payload = {"implementation": raw["implementation"], "parameters": parameters,
                   "shared": state["fixed"]["shared"]}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"id": identifier, "implementation": raw["implementation"],
            "parameters": parameters, "hypothesis": hypothesis.strip(),
            "repeat_reason": repeat_reason.strip() if repeat_reason else None,
            **({"source": source} if source else {}),
            "fingerprint": fingerprint}


def _known_references(state: Mapping[str, Any]) -> set[str]:
    known = set()
    for proposal in state["proposals"]:
        if proposal.get("batch_directory"):
            known.add(proposal["proposal_id"])
        for result in proposal.get("results", []):
            if result.get("result_path"):
                known.add(result["result_path"])
    return known


def validate_proposal(state: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("proposal must be a JSON object")
    required = {"proposal_id", "candidates", "prior_result_references", "repetitions"}
    if set(raw) != required:
        raise ValueError("proposal fields must contain exactly: " + ", ".join(sorted(required)))
    proposal_id = _identifier(raw["proposal_id"], "proposal_id")
    used_ids = {item["proposal_id"] for item in state["proposals"]}
    used_ids.update(item.get("proposal_id") for item in state["rejected_proposals"])
    if proposal_id in used_ids:
        raise ValueError(f"proposal_id already exists: {proposal_id}")
    candidates_raw = raw["candidates"]
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError("candidates must be a non-empty list")
    candidates = [_resolve_candidate(state, candidate, index)
                  for index, candidate in enumerate(candidates_raw)]
    identifiers = [candidate["id"] for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate IDs must be unique within a proposal")
    seen = set(state["candidate_fingerprints"])
    for candidate in candidates:
        duplicate = candidate["fingerprint"] in seen
        if duplicate and not candidate["repeat_reason"]:
            raise ValueError(
                f"candidate {candidate['id']} duplicates a previously submitted configuration; "
                "provide repeat_reason to repeat it"
            )
        seen.add(candidate["fingerprint"])
    references = raw["prior_result_references"]
    if not isinstance(references, list) or any(not isinstance(item, str) for item in references):
        raise ValueError("prior_result_references must be a list of strings")
    invalid_references = sorted(set(references) - _known_references(state))
    if invalid_references:
        raise ValueError("unknown prior result references: " + ", ".join(invalid_references))
    repetitions = _positive_int(raw["repetitions"], "repetitions")
    return {"proposal_id": proposal_id, "candidates": candidates,
            "prior_result_references": references, "repetitions": repetitions,
            "cost": repetitions * len(candidates)}


def _record_rejection(directory: Path, state: dict[str, Any], raw: Any,
                      error: BaseException) -> None:
    record = {"proposal_id": raw.get("proposal_id") if isinstance(raw, Mapping) else None,
              "rejected_at_utc": _now(), "submitted": _json_copy(raw),
              "validation_errors": [str(error)]}
    state["rejected_proposals"].append(record)
    state["updated_at_utc"] = _now()
    _atomic_json(directory / SESSION_FILENAME, state)
    rejected = directory / "rejected"
    rejected.mkdir(exist_ok=True)
    _atomic_json(rejected / f"{uuid.uuid4().hex}.json", record)


def _latest_batch(batch_root: Path) -> dict[str, Any] | None:
    candidates = sorted(batch_root.glob("*/batch_results.json"),
                        key=lambda path: path.stat().st_mtime_ns)
    return _read_json(candidates[-1]) if candidates else None


def _evidence(batch: Mapping[str, Any] | None) -> tuple[int, int, list[dict[str, Any]]]:
    if not batch:
        return 0, 0, []
    entries = [entry for entry in batch.get("entries", []) if entry.get("repetition") is not None]
    confirmed = sum(bool(entry.get("experiment_run_id") and entry.get("result_path")) for entry in entries)
    uncertain = len(entries) - confirmed
    return confirmed, uncertain, entries


BatchExecutor = Callable[..., dict[str, Any]]


def submit_proposal(session: Path | str, raw_proposal: Mapping[str, Any], *,
                    batch_executor: BatchExecutor = run_batch) -> dict[str, Any]:
    directory = _session_directory(session)
    with _session_lock(directory):
        state = load_session(directory)
        try:
            proposal = validate_proposal(state, raw_proposal)
        except BaseException as error:
            _record_rejection(directory, state, raw_proposal, error)
            raise SessionError(str(error)) from error
        if proposal["cost"] > state["budget"]["remaining"]:
            error = ValueError(
                f"proposal costs {proposal['cost']} units but only {state['budget']['remaining']} remain")
            _record_rejection(directory, state, raw_proposal, error)
            raise SessionError(str(error)) from error

        if state["fixed"]["implementation"] == SOURCE_IMPLEMENTATION:
            source_directory = directory / "sources" / proposal["proposal_id"]
            try:
                for candidate in proposal["candidates"]:
                    full_config = {**state["fixed"]["shared"], "candidate": {
                        "implementation": candidate["implementation"],
                        "parameters": candidate["parameters"],
                        "source": candidate["source"],
                    }}
                    preserved = snapshot_candidate_source(
                        full_config, source_directory / f"{candidate['id']}.py")
                    candidate["source"] = preserved["candidate"]["source"]
            except BaseException as error:
                _record_rejection(directory, state, raw_proposal, error)
                raise SessionError(str(error)) from error

        record = {**proposal, "state": "accepted", "accepted_at_utc": _now(),
                  "reserved_units": proposal["cost"], "consumed_units": 0,
                  "batch_directory": None, "batch_summary_path": None, "results": [],
                  "observations": None, "recovery": None}
        state["proposals"].append(record)
        for candidate in proposal["candidates"]:
            state["candidate_fingerprints"].setdefault(candidate["fingerprint"], []).append({
                "proposal_id": proposal["proposal_id"], "candidate_id": candidate["id"],
                "repeat_reason": candidate["repeat_reason"],
            })
        _budget(state)
        state["updated_at_utc"] = _now()
        proposal_directory = directory / "proposals" / proposal["proposal_id"]
        proposal_directory.mkdir(parents=True, exist_ok=False)
        _atomic_json(proposal_directory / "accepted_proposal.json", record)
        _atomic_json(directory / SESSION_FILENAME, state)

        record["state"] = "running"
        record["started_at_utc"] = _now()
        _atomic_json(directory / SESSION_FILENAME, state)
        batch_root = proposal_directory / "batches"
        batch_config = {
            "shared": state["fixed"]["shared"],
            "candidates": [{"id": candidate["id"],
                            "implementation": candidate["implementation"],
                            "parameters": candidate["parameters"],
                            **({"source": candidate["source"]}
                               if candidate.get("source") else {})}
                           for candidate in proposal["candidates"]],
            "scheduling_seed": state["fixed"]["scheduling_seed"] + len(state["proposals"]) - 1,
            "repetitions": proposal["repetitions"],
        }
        batch: dict[str, Any] | None = None
        execution_error: BaseException | None = None
        execution_traceback: str | None = None
        try:
            batch = batch_executor(batch_config, output_root=batch_root)
        except BaseException as error:
            execution_error = error
            execution_traceback = traceback.format_exc()
            batch = _latest_batch(batch_root)

        confirmed, uncertain, entries = _evidence(batch)
        record["consumed_units"] = confirmed
        # If the batch call itself escaped unexpectedly, no absent artifact is
        # proof that the corresponding invocation never launched.
        record["reserved_units"] = proposal["cost"] - confirmed if execution_error else uncertain
        record["results"] = entries
        if batch:
            record["batch_directory"] = batch.get("batch_directory")
            if record["batch_directory"]:
                record["batch_summary_path"] = str(Path(record["batch_directory"]) / "summary.json")
            record["observations"] = batch.get("summary")
        if execution_error is not None:
            record["execution_error"] = {"type": type(execution_error).__name__,
                                         "message": str(execution_error),
                                         "traceback": execution_traceback}
            record["state"] = "needs_reconciliation"
        elif uncertain:
            record["state"] = "needs_reconciliation"
        elif batch and batch.get("status") == "interrupted":
            record["state"] = "interrupted"
        elif batch and batch.get("status") == "success":
            record["state"] = "completed"
        else:
            record["state"] = "failed"
        record["finished_at_utc"] = _now()
        _budget(state)
        state["updated_at_utc"] = _now()
        _atomic_json(proposal_directory / "proposal_state.json", record)
        _atomic_json(directory / SESSION_FILENAME, state)
        return _json_copy(record)


def recover_session(session: Path | str, proposal_id: str, resolution: str) -> dict[str, Any]:
    directory = _session_directory(session)
    with _session_lock(directory):
        state = load_session(directory)
        matches = [proposal for proposal in state["proposals"]
                   if proposal["proposal_id"] == proposal_id]
        if not matches:
            raise SessionError(f"unknown proposal: {proposal_id}")
        proposal = matches[0]
        original_state = proposal["state"]
        if original_state not in ("accepted", "running", "needs_reconciliation"):
            raise SessionError(f"proposal {proposal_id} does not need reconciliation")
        batch = _latest_batch(directory / "proposals" / proposal_id / "batches")
        confirmed, entry_uncertain, entries = _evidence(batch)
        additional = max(0, confirmed - proposal["consumed_units"])
        moved = min(additional, proposal["reserved_units"])
        proposal["consumed_units"] += moved
        proposal["reserved_units"] -= moved
        proposal["results"] = entries or proposal["results"]
        final_batch = bool(batch and batch.get("status") in {
            "success", "completed_with_failures", "interrupted", "configuration_failed",
        })
        if final_batch:
            # A final batch record proves omitted scheduled entries never launched.
            proposal["reserved_units"] = min(proposal["reserved_units"], entry_uncertain)
        elif original_state == "accepted" and batch is None:
            # The accepted state is persisted before the transition to running.
            # With no batch directory, launch had not begun.
            proposal["reserved_units"] = 0
        reserved = proposal["reserved_units"]
        if resolution == "inspect":
            if proposal["reserved_units"]:
                proposal["state"] = "needs_reconciliation"
                _budget(state)
                state["updated_at_utc"] = _now()
                _atomic_json(directory / "proposals" / proposal_id / "proposal_state.json", proposal)
                _atomic_json(directory / SESSION_FILENAME, state)
                raise SessionError("launch ambiguity remains; use --resolution consume or release")
        elif resolution == "consume":
            proposal["consumed_units"] += proposal["reserved_units"]
            proposal["reserved_units"] = 0
        elif resolution == "release":
            proposal["reserved_units"] = 0
        else:
            raise SessionError("resolution must be inspect, consume, or release")
        if resolution == "inspect" and batch and batch.get("status") == "success":
            proposal["state"] = "completed"
        elif resolution == "inspect" and batch and batch.get("status") == "interrupted":
            proposal["state"] = "interrupted"
        else:
            proposal["state"] = "failed"
        proposal["recovery"] = {"resolved_at_utc": _now(), "resolution": resolution,
                                "units": reserved}
        _budget(state)
        state["updated_at_utc"] = _now()
        _atomic_json(directory / "proposals" / proposal_id / "proposal_state.json", proposal)
        _atomic_json(directory / SESSION_FILENAME, state)
        return _json_copy(proposal)


def session_status(session: Path | str) -> dict[str, Any]:
    state = load_session(session)
    return {"schema_version": state["schema_version"], "session_id": state["session_id"],
            "objective": state["objective"],
            "benchmark_mode": resolved_benchmark_mode(state["fixed"]["shared"]),
            "budget": state["budget"],
            "proposal_states": [{"proposal_id": proposal["proposal_id"],
                                 "state": proposal["state"], "cost": proposal["cost"],
                                 "consumed_units": proposal["consumed_units"],
                                 "reserved_units": proposal["reserved_units"]}
                                for proposal in state["proposals"]],
            "rejected_proposal_count": len(state["rejected_proposals"])}


def export_context(session: Path | str, output: Path | str | None = None) -> dict[str, Any]:
    state = load_session(session)
    successes, failures, hypotheses = [], [], []
    for proposal in state["proposals"]:
        for candidate in proposal["candidates"]:
            hypotheses.append({"proposal_id": proposal["proposal_id"],
                               "candidate_id": candidate["id"],
                               "hypothesis": candidate["hypothesis"],
                               "repeat_reason": candidate["repeat_reason"],
                               "fingerprint": candidate["fingerprint"]})
        for result in proposal.get("results", []):
            observation = {"proposal_id": proposal["proposal_id"], **result}
            (successes if result.get("status") == "success" else failures).append(observation)
    context = {
        "schema_version": SESSION_SCHEMA_VERSION, "session_id": state["session_id"],
        "objective": state["objective"],
        "benchmark_mode": resolved_benchmark_mode(state["fixed"]["shared"]),
        "fixed": state["fixed"], "budget": state["budget"],
        "proposal_history": state["proposals"], "rejected_proposals": state["rejected_proposals"],
        "candidate_fingerprints": state["candidate_fingerprints"],
        "hypotheses": hypotheses,
        "observations": {"successful_results": successes, "failures": failures,
                         "descriptive_rankings": [
                             {"proposal_id": proposal["proposal_id"],
                              "ranking": (proposal.get("observations") or {}).get(
                                  "complete_candidate_ranking", [])}
                             for proposal in state["proposals"]]},
        "limitations": ["Rankings are descriptive and do not establish statistical significance.",
                        "Rankings from different benchmark modes must not be combined.",
                        "Legacy sessions without benchmark.mode are interpreted as allocating "
                        "without rewriting their records or fingerprints.",
                        "Historical experiments are not imported; this session starts fresh."],
    }
    if output is not None:
        _atomic_json(Path(output), context)
    return context


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Manage persistent, budgeted experiment sessions")
    subparsers = result.add_subparsers(dest="action", required=True)
    create = subparsers.add_parser("create", help="create a session")
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--output-root", type=Path, default=Path("artifacts/experiment-sessions"))
    for name in ("status", "context"):
        command = subparsers.add_parser(name, help=f"show session {name}")
        command.add_argument("--session", type=Path, required=True)
        if name == "context":
            command.add_argument("--output", type=Path)
    submit = subparsers.add_parser("submit", help="validate and execute one proposal")
    submit.add_argument("--session", type=Path, required=True)
    submit.add_argument("--proposal", type=Path, required=True)
    recover = subparsers.add_parser("recover", help="resolve an ambiguous launch")
    recover.add_argument("--session", type=Path, required=True)
    recover.add_argument("--proposal-id", required=True)
    recover.add_argument("--resolution", choices=("inspect", "consume", "release"), required=True)
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    args = parser().parse_args(arguments)
    try:
        if args.action == "create":
            state = create_session(_read_json(args.config), output_root=args.output_root)
            print(f"session directory: {state['session_directory']}")
        elif args.action == "status":
            print(json.dumps(session_status(args.session), indent=2, sort_keys=True))
        elif args.action == "context":
            context = export_context(args.session, args.output)
            print(json.dumps(context, indent=2, sort_keys=True))
        elif args.action == "submit":
            proposal = submit_proposal(args.session, _read_json(args.proposal))
            print(json.dumps(proposal, indent=2, sort_keys=True))
            return 0 if proposal["state"] == "completed" else 1
        else:
            recovered = recover_session(args.session, args.proposal_id, args.resolution)
            print(json.dumps(recovered, indent=2, sort_keys=True))
        return 0
    except (SessionError, ValueError) as error:
        print(f"session error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
