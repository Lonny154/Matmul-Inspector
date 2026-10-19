import json
import torch


def pytest_runtest_makereport(item, call):
    if call.when != "call" or call.excinfo is None:
        return
    for entry in call.excinfo.traceback:
        local = entry.frame.f_locals
        if "names" not in local or "plan" not in local:
            continue
        plan = local["plan"]
        sequence = plan.schedule_plan.launch_sequence if hasattr(plan, "schedule_plan") else plan.launch_sequence
        names = local["names"]
        workspace = local.get("workspace")
        output = workspace.get("output") if workspace else None
        size = local.get("size", 0)
        algorithm = local.get("algorithm")
        if output is not None:
            torch.cuda.synchronize()
            if algorithm == "strassen" and size >= 512:
                profiled_output_pass = bool(torch.isfinite(output).all())
            else:
                profiled_output_pass = bool(torch.allclose(
                    output, local["left"] @ local["right"], rtol=0.005, atol=0.05))
        else:
            profiled_output_pass = None
        print("PROFILER_PROBE " + json.dumps({
            "nodeid": item.nodeid,
            "abstract_groups": len(local["schedule"].groups) if "schedule" in local else None,
            "expected_count": len(sequence),
            "observed_count": len(names),
            "planned_kernels": [row["kernel"] for row in sequence],
            "observed_kernels": names,
            "profiled_output_pass": profiled_output_pass,
        }, sort_keys=True))
        break
