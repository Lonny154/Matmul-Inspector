#!/usr/bin/env python3
"""Prepare, train, and evaluate the controlled TG-4I experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from matmul_inspector.tensor_game_candidate_coverage import decode
from matmul_inspector.tensor_game_coverage_training import (
    CoverageDataset,
    VERSION,
    checkpoint_sha256,
    collate_coverage,
    coverage_loss,
    labels_digest,
    validate_separation,
    verified_labels,
)
from matmul_inspector.tensor_game_experience import action_from_payload, action_identity
from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_offline_learning import (
    OfflineTrainingConfig,
    _move_tensors,
    configure_trainable,
    sampled_support_log_probabilities,
)
from matmul_inspector.tensor_game_out_of_sample import (
    EvaluationPolicy,
    generate_fresh_targets,
    tensor_tuple,
)
from matmul_inspector.tensor_game_search import HeuristicValueEvaluator, SampledMCTS, SearchConfig

WEIGHTS = (0.1, 0.5, 1.0)
SEEDS = (3, 19, 35)


def load_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows, fields=()):
    names = list(rows[0]) if rows else list(fields)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    training_manifest = load_json(args.experience / "manifest.json")
    target_records = training_manifest["metadata"]["target_records"]
    fresh = generate_fresh_targets(
        known_paths=args.known,
        seed=20261016,
        per_depth=4,
    )
    heldout = {
        **{key: value for key, value in fresh.items() if key != "targets"},
        "reserved_for": "TG-4I final evaluation; sealed before labels or training",
        "targets": [
            row for row in fresh["targets"] if row["construction_length"] in (1, 2)
        ],
    }
    write_json(args.output / "heldout_targets.json", heldout)
    labels = verified_labels(target_records, "train") + verified_labels(
        target_records, "validation"
    )
    tg4g = load_json(args.tg4g)
    separation = validate_separation(labels, (heldout, tg4g))
    with (args.output / "training_labels.jsonl").open("w") as stream:
        for row in labels:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    label_validation = {
        **separation,
        "labels_digest": labels_digest(labels),
        "file_sha256": sha(args.output / "training_labels.jsonl"),
        "all_positives_legal": True,
        "all_continuations_verified": True,
        "all_encoding_round_trips": True,
        "tg4g_excluded": True,
        "new_heldout_excluded": True,
    }
    write_json(args.output / "label_validation.json", label_validation)
    config = OfflineTrainingConfig(
        mode="policy_only",
        epochs=30,
        batch_size=128,
        learning_rate=0.001,
        policy_coefficient=1.0,
        value_coefficient=1.0,
        gradient_clip=1.0,
        patience=30,
        seed=42,
        device="cpu",
        strict_torso_freeze=True,
    )
    prereg = {
        "version": VERSION,
        "status": "frozen_before_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "experience": str(args.experience),
        "experience_manifest_sha256": sha(args.experience / "manifest.json"),
        "experience_decisions_sha256": sha(args.experience / "decisions.json"),
        "labels_sha256": label_validation["file_sha256"],
        "heldout_manifest_sha256": sha(args.output / "heldout_targets.json"),
        "tg4g_exclusion_sha256": sha(args.tg4g),
        "initial_checkpoint": str(args.checkpoint),
        "initial_checkpoint_sha256": sha(args.checkpoint / "best_model_state.pt"),
        "model_architecture": "existing TG-3B factorized ternary policy/value MLP",
        "configurations": ["frozen", "mcts_only", "mcts_plus_coverage_0.1", "mcts_plus_coverage_0.5", "mcts_plus_coverage_1.0"],
        "training_config": asdict(config),
        "optimizer": "AdamW with framework-default weight decay",
        "matched_budget": "30 fixed epochs for every trained configuration; no early stopping",
        "random_seeds": {"training": 42, "search": list(SEEDS)},
        "coverage_loss": "negative log globally normalized canonical positive-set probability mass",
        "coverage_weights": list(WEIGHTS),
        "candidate_cutoffs": [1, 12, 32, 256],
        "checkpoint_selection": "within run minimize validation MCTS CE plus lambda times validation coverage loss; across coverage weights maximize validation positive top-12 micro recall, then minimize validation MCTS CE, then choose smaller lambda",
        "search": {"candidate_budget": 12, "simulation_budget_by_depth": {"1": 16, "2": 64}, "node_limit": 2000, "timeout_seconds": 10.0, "c_puct": 1.5, "value": "nonzero_heuristic"},
        "primary_metrics": ["validation_verified_positive_top12_recall", "heldout_exact_discovery"],
        "secondary_metrics": ["MCTS_validation_cross_entropy", "reference_recall_at_1_12_32_256", "search_cost", "training_cost"],
        "maximum_compute_budget": "4 runs x 30 epochs on CPU; 72 selected final searches with 10-second per-search hard cap",
        "stopping": "stop on nonfinite loss, invalid gradient, integrity failure, or exhausted fixed epoch budget",
        "heldout_policy": "new manifest sealed now; unavailable for checkpoint or lambda selection; opened once after selection",
    }
    write_json(args.output / "preregistration.json", prereg)
    digest = sha(args.output / "preregistration.json")
    (args.output / "preregistration.sha256").write_text(f"{digest}  preregistration.json\n")
    print(digest)


def load_labels(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def evaluate_losses(model, dataset, config, batch_size, device):
    import torch

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_coverage)
    mcts_total = coverage_total = 0.0
    examples = coverage_examples = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = _move_tensors(batch, device)
            logits, _ = model(batch["residual"])
            log_policy = sampled_support_log_probabilities(logits, batch["actions"], batch["action_mask"], config.shape)
            losses = -(batch["search_policy"] * torch.where(batch["action_mask"], log_policy, 0.0)).sum(1)
            mcts_total += float(losses.sum())
            examples += len(losses)
            if len(batch["coverage_indices"]):
                selected = logits[batch["coverage_indices"]]
                loss = coverage_loss(selected, batch["positive_actions"], batch["positive_mask"], config.shape)
                coverage_total += float(loss) * len(selected)
                coverage_examples += len(selected)
    return {"mcts_cross_entropy": mcts_total / examples, "coverage_loss": coverage_total / coverage_examples, "examples": examples, "coverage_examples": coverage_examples}


def train_one(name, weight, args, labels):
    import torch

    config = OfflineTrainingConfig(mode="policy_only", epochs=30, batch_size=128, learning_rate=0.001, policy_coefficient=1.0, value_coefficient=1.0, gradient_clip=1.0, patience=30, seed=42, device="cpu", strict_torso_freeze=True)
    model, model_config, source_manifest = load_checkpoint(args.checkpoint, "cpu")
    train = CoverageDataset(args.experience, "train", labels)
    validation = CoverageDataset(args.experience, "validation", labels)
    random.seed(config.seed); torch.manual_seed(config.seed); torch.use_deterministic_algorithms(True)
    trainable = configure_trainable(model, config.mode, config.strict_torso_freeze)
    generator = torch.Generator().manual_seed(config.seed)
    loader = torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=generator, collate_fn=collate_coverage)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    directory = args.output / "checkpoints" / name
    directory.mkdir(parents=True)
    checkpoint = directory / "best_model_state.pt"
    best = math.inf; best_epoch = None; metrics = []; started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train(); totals = Counter()
        for batch in loader:
            batch = _move_tensors(batch, "cpu"); optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch["residual"])
            log_policy = sampled_support_log_probabilities(logits, batch["actions"], batch["action_mask"], model_config.shape)
            mcts = -(batch["search_policy"] * torch.where(batch["action_mask"], log_policy, 0.0)).sum(1).mean()
            coverage = logits.sum() * 0
            if weight and len(batch["coverage_indices"]):
                coverage = coverage_loss(logits[batch["coverage_indices"]], batch["positive_actions"], batch["positive_mask"], model_config.shape)
            loss = mcts + weight * coverage
            if not torch.isfinite(loss): raise RuntimeError("nonfinite training loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), config.gradient_clip)
            if not torch.isfinite(gradient): raise RuntimeError("nonfinite gradient")
            optimizer.step(); count = len(logits)
            totals.update(examples=count, total_loss=float(loss.detach())*count, mcts_loss=float(mcts.detach())*count, coverage_loss=float(coverage.detach())*count, gradient_norm=float(gradient)*count)
        validation_metrics = evaluate_losses(model, validation, model_config, config.batch_size, "cpu")
        objective = validation_metrics["mcts_cross_entropy"] + weight * validation_metrics["coverage_loss"]
        metrics.append({"configuration":name,"coverage_weight":weight,"epoch":epoch,"train_total_loss":totals["total_loss"]/totals["examples"],"train_mcts_loss":totals["mcts_loss"]/totals["examples"],"train_coverage_loss":totals["coverage_loss"]/totals["examples"],"gradient_norm":totals["gradient_norm"]/totals["examples"],"validation_mcts_cross_entropy":validation_metrics["mcts_cross_entropy"],"validation_coverage_loss":validation_metrics["coverage_loss"],"validation_objective":objective})
        if objective < best:
            best=objective; best_epoch=epoch; torch.save(model.state_dict(),checkpoint)
    manifest={"version":VERSION,"configuration":name,"coverage_weight":weight,"source_checkpoint_sha256":sha(args.checkpoint/source_manifest["checkpoint"]),"training_config":asdict(config),"optimizer":"AdamW","trainable_parameters":trainable,"best_epoch":best_epoch,"best_validation_objective":best,"checkpoint":"best_model_state.pt","checkpoint_sha256":checkpoint_sha256(checkpoint),"training_seconds":time.perf_counter()-started,"fixed_epochs_completed":30,"environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu"}}
    write_json(directory/"experiment.json",manifest)
    return manifest,metrics


def load_trained(path, source):
    import torch
    model, config, _ = load_checkpoint(source,"cpu")
    model.load_state_dict(torch.load(path/"best_model_state.pt",map_location="cpu",weights_only=True)); model.eval()
    return model,config


def decoded_recall(models, labels, split):
    rows=[]
    for label in [x for x in labels if x["split"]==split]:
        positives={x["action_id"] for x in label["positive_actions"]}
        reference=action_identity(action_from_payload(next(x["action"] for x in label["positive_actions"] if "validated_construction_continuation" in x["provenance"])))
        residual=tensor_tuple(label["residual"])
        for name,(model,config) in models.items():
            for cutoff in (1,12,32,256):
                started=time.perf_counter(); decoded=decode(model,config,residual,cutoff); elapsed=time.perf_counter()-started
                ids=[action_identity(x[0]) for x in decoded]; recalled=positives&set(ids)
                rows.append({"scope":split,"state_id":label["state_id"],"target_id":label["target_id"],"constructed_depth":label["constructed_depth"],"configuration":name,"cutoff":cutoff,"positive_count":len(positives),"positive_recalled":len(recalled),"any_positive_recalled":bool(recalled),"reference_recalled":reference in ids,"reference_rank":ids.index(reference)+1 if reference in ids else None,"distinct_candidates":len(set(ids)),"generation_seconds":elapsed})
    return rows


def train(args):
    prereg=load_json(args.output/"preregistration.json")
    if sha(args.output/"preregistration.json")!=(args.output/"preregistration.sha256").read_text().split()[0]: raise ValueError("preregistration hash mismatch")
    labels=load_labels(args.output/"training_labels.jsonl")
    if sha(args.output/"training_labels.jsonl")!=prereg["labels_sha256"]: raise ValueError("label hash mismatch")
    manifests={}; metrics=[]
    for name,weight in (("mcts_only",0.0),)+tuple((f"coverage_{w}",w) for w in WEIGHTS):
        manifests[name],rows=train_one(name,weight,args,labels); metrics.extend(rows)
    write_csv(args.output/"training_metrics.csv",metrics)
    frozen=load_checkpoint(args.checkpoint,"cpu")[:2]
    models={"frozen":frozen,**{name:load_trained(args.output/"checkpoints"/name,args.checkpoint) for name in manifests}}
    recall=decoded_recall(models,labels,"validation"); write_csv(args.output/"validation_candidate_recall.csv",recall)
    choices=[]
    for weight in WEIGHTS:
        name=f"coverage_{weight}"; top=[r for r in recall if r["configuration"]==name and r["cutoff"]==12]
        manifest=manifests[name]
        model,config=models[name]; val=CoverageDataset(args.experience,"validation",labels)
        losses=evaluate_losses(model,val,config,128,"cpu")
        choices.append({"name":name,"weight":weight,"positive_recalled":sum(r["positive_recalled"] for r in top),"positive_total":sum(r["positive_count"] for r in top),"validation_mcts_cross_entropy":losses["mcts_cross_entropy"]})
    selected=sorted(choices,key=lambda x:(-x["positive_recalled"],x["validation_mcts_cross_entropy"],x["weight"]))[0]
    checkpoint_manifest={"source":{"name":"frozen","checkpoint_sha256":prereg["initial_checkpoint_sha256"]},"trained":manifests,"coverage_selection_candidates":choices,"selected_coverage":selected,"heldout_accessed":False}
    write_json(args.output/"checkpoint_manifest.json",checkpoint_manifest)
    print(selected["name"])


def search_rows(targets,models,scope):
    rows=[]
    for target in targets:
        depth=target["construction_length"]; simulations={1:16,2:64}[depth]
        tensor=tensor_tuple(target["target"])
        for seed in SEEDS:
            for name,(model,config) in models.items():
                policy=EvaluationPolicy(model,config,model,config,"frozen_top12")
                result=SampledMCTS(tensor,policy,SearchConfig(simulations=simulations,max_actions=12,max_moves=depth,seed=seed,c_puct=1.5,node_limit=2000,max_wall_time_seconds=10.0),HeuristicValueEvaluator()).run()
                rows.append({"scope":scope,"target_id":target["target_id"],"target_digest":target["target_digest"],"constructed_depth":depth,"configuration":name,"seed":seed,"success":result.exact_reconstruction,"simulations":result.simulations_run,"simulations_to_first_success":result.simulations_to_first_success,"expanded_nodes":result.expanded_nodes,"candidate_evaluations":result.accepted_proposals,"model_forward_passes":policy.forward_passes,"proposal_seconds":result.proposal_generation_seconds,"search_seconds":result.search_wall_seconds,"termination_reason":result.termination_reason})
    return rows


def reference_recall_rows(targets,models,scope):
    labels=[]
    for target in targets:
        labels.append({"state_id":target["target_digest"],"target_id":target["target_id"],"constructed_depth":target["construction_length"],"split":scope,"residual":target["target"],"positive_actions":[{"action":target["canonical_factors"][0],"action_id":action_identity(action_from_payload(target["canonical_factors"][0])),"provenance":["validated_construction_continuation"]}]})
    return decoded_recall(models,labels,scope)


def evaluate(args):
    prereg=load_json(args.output/"preregistration.json"); manifest=load_json(args.output/"checkpoint_manifest.json")
    if manifest["heldout_accessed"]: raise ValueError("heldout already accessed")
    selected=manifest["selected_coverage"]["name"]
    models={"frozen":load_checkpoint(args.checkpoint,"cpu")[:2],"mcts_only":load_trained(args.output/"checkpoints"/"mcts_only",args.checkpoint),selected:load_trained(args.output/"checkpoints"/selected,args.checkpoint)}
    heldout=load_json(args.output/"heldout_targets.json")
    if sha(args.output/"heldout_targets.json")!=prereg["heldout_manifest_sha256"]: raise ValueError("heldout hash mismatch")
    recall=reference_recall_rows(heldout["targets"],models,"heldout")
    write_csv(args.output/"candidate_recall.csv",recall)
    results=search_rows(heldout["targets"],models,"heldout"); write_csv(args.output/"discovery_results.csv",results)
    index={(r["target_id"],r["seed"],r["configuration"]):r for r in results}; pairs=[]
    for baseline,candidate in (("frozen","mcts_only"),("frozen",selected),("mcts_only",selected)):
        for target in heldout["targets"]:
            for seed in SEEDS:
                a=index[target["target_id"],seed,baseline]["success"]; b=index[target["target_id"],seed,candidate]["success"]
                pairs.append({"target_id":target["target_id"],"constructed_depth":target["construction_length"],"seed":seed,"baseline":baseline,"candidate":candidate,"baseline_success":a,"candidate_success":b,"transition":"shared_success" if a and b else "regression" if a else "improvement" if b else "shared_failure"})
    write_csv(args.output/"paired_outcomes.csv",pairs)
    training=list(csv.DictReader((args.output/"training_metrics.csv").open())); val=list(csv.DictReader((args.output/"validation_candidate_recall.csv").open()))
    summary={"version":VERSION,"preregistration_sha256":sha(args.output/"preregistration.json"),"heldout_manifest_sha256":sha(args.output/"heldout_targets.json"),"selected_coverage":manifest["selected_coverage"],"results":{},"validation_recall":{},"paired":dict(Counter(f"{r['baseline']}_vs_{r['candidate']}:{r['transition']}" for r in pairs)),"training_seconds":{name:data["training_seconds"] for name,data in manifest["trained"].items()},"heldout_opened_once":True,"uncertainty":"eight unique targets are too few for useful target-cluster intervals; exact target and repeated-seed counts reported"}
    for name in models:
        rr=[r for r in results if r["configuration"]==name]; top12=[r for r in recall if r["configuration"]==name and r["cutoff"]==12]
        summary["results"][name]={"trial_successes":sum(r["success"] for r in rr),"trials":len(rr),"targets_solved":len({r["target_id"] for r in rr if r["success"]}),"unique_targets":8,"reference_top12":sum(r["reference_recalled"] for r in top12),"reference_states":8,"mean_search_seconds":sum(float(r["search_seconds"]) for r in rr)/len(rr),"mean_expansions":sum(int(r["expanded_nodes"]) for r in rr)/len(rr)}
        vr=[r for r in val if r["configuration"]==name and r["cutoff"]=="12"]
        summary["validation_recall"][name]={"positive_recalled":sum(int(r["positive_recalled"]) for r in vr),"positive_total":sum(int(r["positive_count"]) for r in vr),"states_with_any":sum(r["any_positive_recalled"]=="True" for r in vr),"states":len(vr)}
    manifest["heldout_accessed"]=True; manifest["heldout_evaluated_configurations"]=list(models); write_json(args.output/"checkpoint_manifest.json",manifest)
    write_json(args.output/"summary.json",summary); print(json.dumps(summary["results"],indent=2))


def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    for command in ("prepare","train","evaluate"):
        p=sub.add_parser(command); p.add_argument("--output",type=Path,required=True); p.add_argument("--experience",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True)
        if command=="prepare": p.add_argument("--known",type=Path,action="append",required=True); p.add_argument("--tg4g",type=Path,required=True)
    args=parser.parse_args(); {"prepare":prepare,"train":train,"evaluate":evaluate}[args.command](args)


if __name__=="__main__": main()
