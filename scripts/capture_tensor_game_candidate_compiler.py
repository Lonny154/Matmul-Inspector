#!/usr/bin/env python3
"""Compile one dataset candidate and retain bounded Triton/compiler evidence."""

from __future__ import annotations
import argparse, hashlib, json, re, shutil, subprocess
from pathlib import Path

def count(pattern,text): return len(re.findall(pattern,text))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--candidate-id",required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--size",type=int,default=1024); p.add_argument("--cache",type=Path,required=True); a=p.parse_args()
    if a.output.exists(): p.error("--output must not exist")
    import torch,triton
    from matmul_inspector.tensor_game import RankOneAction
    from matmul_inspector.tensor_game_candidate_dataset import validate_decomposition
    from matmul_inspector.tensor_game_triton_lowering import create_lowering
    records=json.loads((a.dataset/"representatives.json").read_text()); record=next((r for r in records if r["candidate_id"]==a.candidate_id),None)
    if record is None: p.error("candidate is not in representatives.json")
    actions=tuple(RankOneAction(x["u"],x["v"],x["w"]) for x in record["actions"]); validate_decomposition(actions); lowering=create_lowering(a.candidate_id,actions)
    left=torch.randn((a.size,a.size),device="cuda");right=torch.randn((a.size,a.size),device="cuda"); lowering.execute(left,right,"ieee","fused_input","fused");torch.cuda.synchronize()
    a.output.mkdir(parents=True); rows=[]; preserved=set()
    for cubin in sorted(a.cache.rglob("*.cubin")):
        directory=cubin.parent; stem=cubin.stem; metadata=json.loads((directory/f"{stem}.json").read_text()); ptx=(directory/f"{stem}.ptx").read_text(); sass=subprocess.run(["/usr/local/cuda/bin/nvdisasm","-c",str(cubin)],check=True,capture_output=True,text=True).stdout; resources=subprocess.run(["/usr/local/cuda/bin/cuobjdump","--dump-resource-usage",str(cubin)],check=True,capture_output=True,text=True).stdout
        match=re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",resources)
        kind="reconstruction" if "reconstruction" in stem else "fused_input"
        preserve=kind not in preserved
        paths={}
        if preserve:
            preserved.add(kind); destination=a.output/kind; destination.mkdir()
            for suffix in (".ttir",".ttgir",".ptx"):
                source=directory/f"{stem}{suffix}"; target=destination/f"kernel{suffix}"; shutil.copy2(source,target); paths[suffix[1:]]=str(target.relative_to(a.output))
            sass_path=destination/"kernel.sass"; sass_path.write_text(sass); paths["sass"]=str(sass_path.relative_to(a.output))
        rows.append({"cache_key":directory.name,"kernel":stem,"kind":kind,"compiler_hash":metadata.get("hash"),"target":metadata.get("target"),"triton_version":metadata.get("triton_version",triton.__version__),"num_warps":metadata.get("num_warps"),"num_stages":metadata.get("num_stages"),"dynamic_shared_bytes":metadata.get("shared"),"registers_per_thread":int(match.group(1)) if match else None,"stack_bytes":int(match.group(2)) if match else None,"static_shared_bytes":int(match.group(3)) if match else None,"local_bytes":int(match.group(4)) if match else None,"cubin_sha256":hashlib.sha256(cubin.read_bytes()).hexdigest(),"ptx_sha256":hashlib.sha256(ptx.encode()).hexdigest(),"sass_sha256":hashlib.sha256(sass.encode()).hexdigest(),"ptx_static":{"fma_f32":ptx.count("fma.rn.f32"),"global_loads":count(r"\bld\.global",ptx),"global_stores":count(r"\bst\.global",ptx)},"sass_static":{"ffma":count(r"\bFFMA\b",sass),"global_loads":count(r"\bLDG(?:\.|\b)",sass),"global_stores":count(r"\bSTG(?:\.|\b)",sass)},"preserved_text_artifacts":paths,"resource_tool_output":resources.strip()})
    payload={"schema":"tensor_game_candidate_compiler.v1","candidate_id":a.candidate_id,"family_id":record["family_id"],"rank":record["rank"],"size":a.size,"precision":"ieee","strategy":"fused_input","reconstruction":"fused","torch":torch.__version__,"triton":triton.__version__,"kernels":rows,"notes":["CUBIN hashes refer to uncommitted cache binaries.","Static instruction counts are text occurrences, not dynamic execution counts.","One input and one reconstruction text artifact are preserved per candidate."]}
    (a.output/"summary.json").write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps({"candidate":a.candidate_id,"kernels":len(rows),"preserved":list(preserved)}));return 0
if __name__=="__main__":raise SystemExit(main())
