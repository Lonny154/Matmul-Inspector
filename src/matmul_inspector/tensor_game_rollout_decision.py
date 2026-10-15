"""Independent corpus and release-gate helpers for TG-4W."""
from __future__ import annotations
import math,random
from collections import Counter
from typing import Any

from .tensor_game import RankOneAction,sum_actions,validate_action
from .tensor_game_default_readiness import SHAPE,action,factors_for,payload,characteristics
from .tensor_game_experience import tensor_digest
from .tensor_game_out_of_sample import axis_orbit_digests,known_target_orbits

CATEGORIES=("sparse","intermediate","dense","repeated_term","sign_equivalent","tie_heavy")

def generate_rollout_targets(known_paths,seed=20261019,count=150):
    """Generate a balanced corpus without consulting either provider."""
    if count%len(CATEGORIES):raise ValueError("count must be divisible by six")
    rng=random.Random(seed);known=known_target_orbits(known_paths);orbits=set();targets=[];rejections=Counter()
    jobs=[]
    for category in CATEGORIES:
        depths=(2,3,4) if category in {"repeated_term","sign_equivalent"} else (1,2,3,4)
        jobs.extend((category,depths[i%len(depths)]) for i in range(count//len(CATEGORIES)))
    rng.shuffle(jobs)
    for index,(category,depth) in enumerate(jobs):
        for _ in range(20000):
            if category=="tie_heavy":
                coordinates=tuple((rng.randrange(4),rng.randrange(4),rng.randrange(4)) for _ in range(depth))
                factors=tuple(RankOneAction(tuple(int(i==u) for i in range(4)),tuple(int(i==v) for i in range(4)),tuple(int(i==w) for i in range(4))) for u,v,w in coordinates)
            else:factors=factors_for(rng,depth,category)
            for item in factors:validate_action(item,SHAPE,frozenset((-1,0,1)))
            target=sum_actions(factors,SHAPE);orbit=axis_orbit_digests(target)
            if orbit&known:rejections["historical_exact_or_axis"]+=1;continue
            if orbit&orbits:rejections["within_manifest_exact_or_axis"]+=1;continue
            digest=tensor_digest(target);targets.append({"target_id":f"tg4w-d{depth}-{index:03d}-{digest[:12]}","target_digest":digest,"target":[[[v for v in row] for row in plane] for plane in target],"construction_length":depth,"construction_length_semantics":"verified upper bound; not proven rank","canonical_factors":[payload(x) for x in factors],"split":"tg4w_independent_rollout","workload":characteristics(target,category,factors)});orbits.update(orbit);break
        else:raise RuntimeError(f"could not generate target {index}")
    return {"schema_version":"tg4w.targets.v1","generator":"balanced_structurally_targeted_legal_rank_one_v1","generator_seed":seed,"target_count":count,"depth_counts":dict(Counter(x["construction_length"] for x in targets)),"category_counts":dict(Counter(x["workload"]["primary_category"] for x in targets)),"category_minimums":{x:count//len(CATEGORIES) for x in CATEGORIES},"known_source_files":[str(x) for x in known_paths],"known_orbit_digest_count":len(known),"equivalence_exclusion":"exact tensor and six axis permutations; not arbitrary basis-change equivalence","rejections":dict(rejections),"unsupported_or_unverified_categories":["existence of distinct alternative decompositions is not proven"],"targets":targets}

def upper_zero_event_bound(n:int,alpha=.05)->float:return 1-math.pow(alpha,1/n)

def decide_release(criteria:dict[str,Any],evidence:dict[str,Any])->dict[str,Any]:
    hard={"zero_exhaustive_only_targets":evidence["exhaustive_only_targets"]==0,"candidate_correctness":evidence["correctness_failures"]==0,"timeouts_not_worse":evidence["bounded_timeouts"]<=evidence["exhaustive_timeouts"],"compatibility":evidence["compatibility_failures"]==0,"memory_stability":evidence["memory_stability_failures"]==0}
    release={"reliability_margin":evidence["regression_upper_bound"]<=criteria["maximum_regression_upper_bound"],"mean_speedup":evidence["mean_speedup"]>=criteria["minimum_mean_speedup"],"paired_consistency":evidence["paired_faster_fraction"]>=criteria["minimum_paired_faster_fraction"],"cross_machine_reproducibility":evidence["genuine_machine_count"]>=criteria["minimum_genuine_machine_count"]}
    decision="NO-GO" if not all(hard.values()) else "GO" if all(release.values()) else "CONDITIONAL"
    return {"decision":decision,"default_changed":False,"hard_gates":hard,"release_gates":release,"failed_gates":[k for k,v in {**hard,**release}.items() if not v]}
