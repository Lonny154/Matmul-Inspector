import json
import torch
from torch.profiler import DeviceType, ProfilerActivity, profile

from matmul_inspector.tensor_game_schedule_gpu_study import _schedule
from matmul_inspector.tensor_game_schedule_hbm_lowering import classify_hbm_schedule
from matmul_inspector.tensor_game_schedule_hbm_runtime import allocate_hbm, execute_hbm
from matmul_inspector.tensor_game_schedule_mixed_lowering import classify_mixed_schedule
from matmul_inspector.tensor_game_schedule_mixed_runtime import allocate_mixed, execute_mixed
from scripts.run_hardware_aware_schedule_lowering_study import FROZEN, make_dag, project_actions, source_rows

rows = source_rows()
actions = project_actions(FROZEN)
cases = []
for size in (128, 256, 512, 1024):
    for algorithm, prefix in (("conventional", "bbe200b8"),
                              ("project-rank8", "00d4dd6a"),
                              ("project-rank8", "4cfed2b2"),
                              ("strassen", "63a9d1d3")):
        source = next(row for row in rows if row["algorithm"] == algorithm
                      and row["matrix_size"] == 128 and row["schedule_hash"].startswith(prefix))
        row = next(row for row in rows if row["algorithm"] == algorithm
                   and row["matrix_size"] == size and row["schedule"]["groups"] == source["schedule"]["groups"]
                   and row["schedule"]["recompute_allowed"] == source["schedule"]["recompute_allowed"])
        cases.append(("HBM", size, algorithm, row))
    for algorithm in ("conventional", "strassen", "project-rank8"):
        row = next(row for row in rows if row["algorithm"] == algorithm
                   and row["matrix_size"] == size and row["policy"] == "FUSE_OUTPUT_RECONSTRUCTION")
        cases.append(("MIXED", size, algorithm, row))

for kind, size, algorithm, row in cases:
    dag = make_dag(algorithm, size, actions)
    schedule = _schedule(row["schedule"])
    plan = (classify_hbm_schedule if kind == "HBM" else classify_mixed_schedule)(
        dag, schedule, lowering_name=row["lowering_name"])
    workspace = (allocate_hbm if kind == "HBM" else allocate_mixed)(dag, schedule, plan)
    execute = execute_hbm if kind == "HBM" else execute_mixed
    torch.manual_seed(37)
    left = torch.randn((size, size), device="cuda", dtype=torch.float16)
    right = torch.randn_like(left)
    actual = execute(dag, schedule, plan, left, right, workspace)
    reference = left @ right
    torch.cuda.synchronize()
    numerical = (bool(torch.isfinite(actual).all()) if algorithm == "strassen" and size >= 512
                 else bool(torch.allclose(actual, reference, rtol=0.005, atol=0.05)))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
        execute(dag, schedule, plan, left, right, workspace)
        torch.cuda.synchronize()
    names = [event.name for event in recorded.events() if event.device_type == DeviceType.CUDA]
    expected = plan.schedule_plan.expected_launch_count if kind == "HBM" else plan.expected_launch_count
    sequence = plan.schedule_plan.launch_sequence if kind == "HBM" else plan.launch_sequence
    prefix_match = all(actual_name.startswith(launch["kernel"])
                       for actual_name, launch in zip(names, sequence))
    print(json.dumps({"kind": kind, "size": size, "algorithm": algorithm,
                      "schedule_hash": row["schedule_hash"], "abstract_groups": len(schedule.groups),
                      "expected_launches": expected, "observed_launches": len(names),
                      "prefix_match": prefix_match, "numerical_gate": numerical,
                      "last_planned_kernel": sequence[-1]["kernel"],
                      "last_observed_kernel": names[-1] if names else None}, sort_keys=True))
