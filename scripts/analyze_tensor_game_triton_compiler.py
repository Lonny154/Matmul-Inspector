#!/usr/bin/env python3
"""Summarize saved Tensor Game Triton compiler artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--nvdisasm", default="nvdisasm")
    parser.add_argument("--sass-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")

    rows = []
    for cubin in sorted(args.artifact_root.glob("*/*.cubin")):
        directory = cubin.parent
        metadata_path = directory / "_fp32_gemm_kernel.json"
        ptx_path = directory / "_fp32_gemm_kernel.ptx"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ptx = ptx_path.read_text(encoding="utf-8")
        resource_text = subprocess.run(
            [args.cuobjdump, "--dump-resource-usage", str(cubin)],
            check=True, capture_output=True, text=True,
        ).stdout
        sass = subprocess.run(
            [args.nvdisasm, "-c", str(cubin)],
            check=True, capture_output=True, text=True,
        ).stdout
        sass_bytes = sass.encode()
        sass_path = None
        if args.sass_dir is not None:
            sass_path = args.sass_dir / f"{directory.name}.sass"
            sass_path.parent.mkdir(parents=True, exist_ok=True)
            sass_path.write_text(sass, encoding="utf-8")
        match = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", resource_text)
        if match is None:
            raise RuntimeError(f"could not parse resource usage for {cubin}")
        rows.append({
            "specialization": directory.name,
            "compiler_hash": metadata["hash"],
            "target": metadata["target"],
            "triton_version": metadata["triton_version"],
            "num_warps": metadata["num_warps"],
            "num_stages": metadata["num_stages"],
            "dynamic_shared_bytes": metadata["shared"],
            "registers_per_thread": int(match.group(1)),
            "stack_bytes": int(match.group(2)),
            "static_shared_bytes": int(match.group(3)),
            "local_bytes": int(match.group(4)),
            "ptx_version": re.search(r"\.version\s+([^\s]+)", ptx).group(1),
            "ptx_target": re.search(r"\.target\s+([^\s]+)", ptx).group(1),
            "ptx_fma_rn_f32_static_occurrences": ptx.count("fma.rn.f32"),
            "ptx_mma_static_occurrences": len(re.findall(r"\b(?:mma|wgmma)\.", ptx)),
            "sass_path": str(sass_path) if sass_path is not None else None,
            "sass_bytes": len(sass_bytes),
            "sass_sha256": hashlib.sha256(sass_bytes).hexdigest(),
            "sass_ffma_static_occurrences": len(re.findall(r"\bFFMA\b", sass)),
            "sass_hmma_static_occurrences": len(re.findall(r"\b(?:HMMA|IMMA|BMMA)\b", sass)),
            "resource_tool_output": resource_text.strip(),
        })
    payload = {
        "schema": "tensor_game_triton_compiler_summary.v1",
        "notes": [
            "PTX counts are static text occurrences, not dynamic instruction counts.",
            "cuobjdump reports final-binary registers/local/stack/static shared memory; Triton metadata separately reports dynamic shared memory.",
            "PTX is virtual ISA and is not final SASS.",
            "SASS counts are static disassembly occurrences, not dynamic instruction counts.",
        ],
        "specializations": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
