"""Deterministic structurally targeted corpus utilities for TG-4V."""
from __future__ import annotations
import random
from collections import Counter
from typing import Any

from .tensor_game import RankOneAction,sum_actions,validate_action
from .tensor_game_experience import tensor_digest
from .tensor_game_out_of_sample import axis_orbit_digests,known_target_orbits

SHAPE=(4,4,4)
SCHEDULE={
 1:("sparse","sparse","dense","dense","tie_heavy","tie_heavy"),
 2:("repeated_term","sign_equivalent","intermediate","intermediate","dense","tie_heavy"),
 3:("repeated_term","sign_equivalent","dense","tie_heavy","intermediate","sparse"),
 4:("repeated_term","sign_equivalent","dense","tie_heavy","intermediate","sparse"),
}

def payload(action):return {"u":list(action.u),"v":list(action.v),"w":list(action.w)}
def vector(rng,kind):
    if kind=="dense":return tuple(rng.choice((-1,1)) for _ in range(4))
    if kind=="sparse":
        index=rng.randrange(4);return tuple(rng.choice((-1,1)) if i==index else 0 for i in range(4))
    indices=rng.sample(range(4),2);return tuple(rng.choice((-1,1)) if i in indices else 0 for i in range(4))
def action(rng,kind):return RankOneAction(vector(rng,kind),vector(rng,kind),vector(rng,kind))
def sign_form(value,index):
    if index%3==0:return value
    if index%3==1:return RankOneAction(tuple(-x for x in value.u),tuple(-x for x in value.v),value.w)
    return RankOneAction(tuple(-x for x in value.u),value.v,tuple(-x for x in value.w))

def factors_for(rng,depth,category):
    if category=="repeated_term":
        base=action(rng,"intermediate");return tuple(base for _ in range(depth))
    if category=="sign_equivalent":
        base=action(rng,"intermediate");return tuple(sign_form(base,i) for i in range(depth))
    if category=="tie_heavy":
        offset=rng.randrange(4);return tuple(RankOneAction(tuple(1 if i==(offset+j)%4 else 0 for i in range(4)),tuple(1 if i==(offset+2*j)%4 else 0 for i in range(4)),tuple(1 if i==(offset+3*j)%4 else 0 for i in range(4))) for j in range(depth))
    return tuple(action(rng,category) for _ in range(depth))

def characteristics(target,category,factors):
    values=[x for plane in target for row in plane for x in row];nonzero=sum(x!=0 for x in values)
    return {"primary_category":category,"categories":[category,"near_equivalent_decomposition"] if category in {"repeated_term","sign_equivalent","tie_heavy"} else [category],"nonzero_count":nonzero,"density":nonzero/64,"max_abs":max(map(abs,values)),"repeated_reference_actions":len(factors)-len(set(factors)),"sign_equivalent_reference":category=="sign_equivalent","tie_heavy_by_construction":category=="tie_heavy"}

def generate_readiness_targets(known_paths,seed=20261018):
    rng=random.Random(seed);known=known_target_orbits(known_paths);selected=[];orbits=set();rejections=Counter()
    for depth,categories in SCHEDULE.items():
        for index,category in enumerate(categories):
            for _ in range(10000):
                factors=factors_for(rng,depth,category)
                for item in factors:validate_action(item,SHAPE,frozenset((-1,0,1)))
                target=sum_actions(factors,SHAPE);orbit=axis_orbit_digests(target)
                if orbit&known:rejections["historical_exact_or_axis"]+=1;continue
                if orbit&orbits:rejections["within_manifest_exact_or_axis"]+=1;continue
                digest=tensor_digest(target);selected.append({"target_id":f"tg4v-d{depth}-{index:02d}-{digest[:12]}","target_digest":digest,"target":[[[x for x in row] for row in plane] for plane in target],"construction_length":depth,"construction_length_semantics":"verified upper bound; not proven rank","canonical_factors":[payload(x) for x in factors],"split":"tg4v_independent_readiness","workload":characteristics(target,category,factors)});orbits.update(orbit);break
            else:raise RuntimeError(f"could not generate {category} depth {depth}")
    counts=Counter(x["workload"]["primary_category"] for x in selected)
    return {"schema_version":"tg4v.targets.v1","generator":"structurally_targeted_legal_rank_one_v1","generator_seed":seed,"target_count":len(selected),"depth_counts":dict(Counter(x["construction_length"] for x in selected)),"category_counts":dict(counts),"category_minimums":{"sparse":4,"intermediate":4,"dense":5,"repeated_term":3,"sign_equivalent":3,"tie_heavy":5},"known_source_files":[str(x) for x in known_paths],"known_orbit_digest_count":len(known),"equivalence_exclusion":"exact tensor and six axis permutations; not general basis-change equivalence","rejections":dict(rejections),"unsupported_or_unverified_categories":["distinct alternative decomposition existence is not proven"],"targets":selected}

def decide_readiness(criteria:dict[str,Any],evidence:dict[str,Any])->dict[str,Any]:
    """Apply preregistered gates without silently promoting the default."""
    hard={
        "reference_and_candidate_correctness":evidence["correctness_failures"]==0,
        "no_exhaustive_only_target":evidence["exhaustive_only_targets"]==0,
        "no_excess_timeouts":evidence["bounded_timeouts"]<=evidence["exhaustive_timeouts"],
        "default_and_cli_compatibility":evidence["integration_failures"]==0,
    }
    performance={
        "mean_speedup":evidence["mean_speedup"]>=criteria["minimum_mean_speedup"],
        "paired_faster_fraction":evidence["paired_faster_fraction"]>=criteria["minimum_paired_faster_fraction"],
        "memory_peak":evidence["bounded_mean_python_peak_bytes"]<=criteria["maximum_mean_python_peak_bytes"],
        "memory_retained":evidence["bounded_mean_python_retained_bytes"]<=criteria["maximum_mean_python_retained_bytes"],
        "zero_regression_bound":evidence["regression_upper_bound"]<=criteria["maximum_zero_regression_upper_bound"],
    }
    if not all(hard.values()):decision="NO-GO"
    elif all(performance.values()):decision="GO"
    else:decision="CONDITIONAL"
    return {"decision":decision,"default_changed":False,"hard_gates":hard,"performance_and_uncertainty_gates":performance,"failed_gates":[k for k,v in {**hard,**performance}.items() if not v],"meaning":{"GO":"evidence supports a later, separately reviewed default change","CONDITIONAL":"no correctness regression observed, but evidence misses at least one preregistered promotion gate","NO-GO":"a correctness, discovery, timeout, or compatibility hard gate failed"}[decision]}
