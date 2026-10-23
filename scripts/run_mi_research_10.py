"""Compile and check the frozen four-CTA Perminov-58 candidate once."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
import time

try:
    from scripts import freeze_mi_research_10 as frozen
    from scripts import mi_research_10_preregistration as preregistration
    from scripts import run_mi_research_2c_graph as evidence
    from scripts import run_mi_research_3b_correctness as diagnostics
    from scripts import run_mi_research_4_feasibility as cases
    from scripts import run_mi_research_6_gpu_probe as loader
    from scripts import run_mi_research_7 as compiler_prior
except ModuleNotFoundError:
    import freeze_mi_research_10 as frozen
    import mi_research_10_preregistration as preregistration
    import run_mi_research_2c_graph as evidence
    import run_mi_research_3b_correctness as diagnostics
    import run_mi_research_4_feasibility as cases
    import run_mi_research_6_gpu_probe as loader
    import run_mi_research_7 as compiler_prior

OUT = frozen.OUT / "gpu-validation-v1"
PLAN = frozen.PLAN
TOLERANCE = {"atol": 0.01, "rtol": 0.01}


def static_counts(ttir: str, ptx: str) -> dict:
    return {"ttir_dot": ttir.count("tt.dot"),
            "ttir_addf": ttir.count("arith.addf"),
            "ttir_subf": ttir.count("arith.subf"),
            "ptx_mma_sync": ptx.count("mma.sync"),
            "ptx_add_f32": len(re.findall(r"\badd\.f32\b", ptx)),
            "ptx_sub_f32": len(re.findall(r"\bsub\.f32\b", ptx)),
            "ptx_ld_local": len(re.findall(r"\bld\.local\b", ptx)),
            "ptx_st_local": len(re.findall(r"\bst\.local\b", ptx))}


def verify() -> dict:
    frozen.main()
    manifest = json.loads((OUT / "hash_manifest.json").read_text())
    actual = {str(p.relative_to(OUT)) for p in OUT.rglob("*")
              if p.is_file() and p.name != "hash_manifest.json"}
    if actual != set(manifest) or any(evidence.digest(OUT / name) != sha
                                     for name, sha in manifest.items()):
        raise ValueError("GPU evidence manifest mismatch")
    raw = json.loads((OUT / "raw.json").read_text())
    if raw["frozen_plan_sha256"] != evidence.digest(PLAN) or raw["candidate_source_sha256"] != evidence.digest(frozen.candidate.TARGET):
        raise ValueError("frozen identity mismatch")
    compiler = raw["compiler"]
    if compiler["ptx_sha256"] != evidence.digest(OUT / "candidate.ptx") or any(
        sha != evidence.digest(OUT / f"candidate.{kind}") for kind, sha in compiler["ir_sha256"].items()
    ) or raw["static_counts"] != static_counts((OUT / "candidate.ttir").read_text(),
                                                (OUT / "candidate.ptx").read_text()):
        raise ValueError("compiler evidence mismatch")
    if [r["case"] for r in raw["correctness"]] != list(cases.CASE_NAMES):
        raise ValueError("correctness case coverage changed")
    passed = all(check["passed"] and check["failing_elements"] == 0 and
                 check["candidate_nonfinite_count"] == 0
                 for row in raw["correctness"] for check in row["checks"])
    if not passed or raw["topology"]["kernel_events"] != 1 or raw["topology"]["grid"] != [2, 2, 1]:
        raise ValueError("correctness or topology gate failed")
    return {"status": "PASS", "correctness_cases": len(raw["correctness"]),
            "source_sha256": raw["candidate_source_sha256"], "counts": raw["static_counts"]}


def run() -> dict:
    frozen.main()
    plan = json.loads(PLAN.read_text())
    preregistration.validate(plan)
    if OUT.exists():
        raise FileExistsError("GPU evidence already exists")
    if plan["execution_set"] != ["perminov58_four_cta"] or plan["compilation_status_at_freeze"] != "not_started":
        raise ValueError("execution set changed")
    import torch
    import triton
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    OUT.mkdir(parents=True)
    for source, name in ((PLAN, "frozen_plan.json"), (frozen.candidate.TARGET, "candidate_source.py"),
                         (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(source, OUT / name)
    original = {"precision": torch.get_float32_matmul_precision(),
                "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "reduced": bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)}
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        module = loader.load_snapshot(OUT / "candidate_source.py")
        a, b = cases._inputs(torch, 96, "seed-0")
        out = torch.empty((96, 96), device="cuda", dtype=torch.float16)
        torch.cuda.synchronize(); start = time.perf_counter_ns()
        compiled = module.call(a, b, out)
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter_ns() - start) / 1e6
        compiler = compiler_prior._capture_compiler(compiled, OUT, "candidate")
        correctness = []
        for name in cases.CASE_NAMES:
            a, b = cases._inputs(torch, 96, name)
            reference = torch.matmul(a.float(), b.float()).half()
            checks = []
            for _ in range(2):
                out.fill_(float("nan"))
                module.call(a, b, out)
                torch.cuda.synchronize()
                checks.append(diagnostics.summarize_error(torch, out, reference, a=a, b=b))
            correctness.append({"case": name, "checks": checks,
                                "output_poisoned_before_each_execution": True})
        if any(not x["passed"] or x["failing_elements"] or x["candidate_nonfinite_count"]
               for row in correctness for x in row["checks"]):
            raise ValueError("numerical correctness failed")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            module.call(a, b, out)
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(OUT / "trace.json"))
        events = json.loads((OUT / "trace.json").read_text())["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        if len(kernels) != 1:
            raise ValueError("expected one physical kernel")
        topology = {"kernel_events": len(kernels), "grid": kernels[0].get("args", {}).get("grid"),
                    "block": kernels[0].get("args", {}).get("block"), "kernel_name": kernels[0]["name"]}
        if topology["grid"] != [2, 2, 1] or topology["block"] != [128, 1, 1]:
            raise ValueError("CTA topology changed")
        properties = torch.cuda.get_device_properties(0)
        raw = {"schema_version": 1, "study": "MI-Research-10",
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "frozen_plan_sha256": evidence.digest(PLAN),
               "candidate_source_sha256": evidence.digest(frozen.candidate.TARGET),
               "diagnostic_script_sha256": evidence.digest(Path(__file__)),
               "first_compile_and_call_ms": first_call_ms,
               "environment": {"gpu": properties.name, "compute_capability": f"{properties.major}.{properties.minor}",
                               "sm_count": properties.multi_processor_count,
                               "torch": torch.__version__, "triton": triton.__version__,
                               "python": sys.version.split()[0], "cuda_runtime": torch.version.cuda},
               "precision": {"fp32_matmul_precision": torch.get_float32_matmul_precision(),
                             "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                             "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction},
               "tolerance": TOLERANCE, "compiler": compiler,
               "static_counts": static_counts((OUT / "candidate.ttir").read_text(), (OUT / "candidate.ptx").read_text()),
               "correctness": correctness, "topology": topology,
               "timing": "NOT_MEASURED"}
        evidence.write_json(OUT / "raw.json", raw)
        evidence.manifest(OUT)
        return verify()
    except Exception as error:
        evidence.write_json(OUT / "failure.json", {"type": type(error).__name__, "message": str(error)})
        evidence.manifest(OUT)
        raise
    finally:
        torch.set_float32_matmul_precision(original["precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = original["reduced"]


if __name__ == "__main__":
    print(json.dumps(verify() if "--verify" in sys.argv else run(), sort_keys=True))
