import csv,json
from pathlib import Path
import pytest
from examples.tensor_game_cross_machine_verification import RESULT_FIELDS,compare,load_inputs,schedule,validate_resume

ROOT=Path("artifacts/tensor-game/tg4w-rollout-decision-v1")
pytestmark=pytest.mark.skipif(
    not ROOT.is_dir(),
    reason="optional TG-4X evidence test requires the ignored frozen artifact bundle",
)
def test_frozen_inputs_and_schedule_are_exact():
    manifest,spec,hashes=load_inputs(ROOT)
    assert hashes["target_manifest.json"].startswith("c55792") and hashes["evaluation_spec.json"].startswith("1f5126")
    assert len(schedule(manifest,spec))==300 and len(schedule(manifest,spec,4))==8

def test_resume_requires_an_exact_schedule_prefix():
    manifest,spec,_=load_inputs(ROOT);planned=schedule(manifest,spec,4);rows=[]
    for target,seed,provider,order in planned[:2]:
        row={name:"" for name in RESULT_FIELDS};row.update({"target_id":target["target_id"],"seed":seed,"provider":provider});rows.append(row)
    validate_resume(rows,planned)
    rows.reverse()
    with pytest.raises(ValueError,match="exact prefix"):validate_resume(rows,planned)

def test_hash_mismatch_is_rejected(tmp_path):
    for name in ("target_manifest.json","evaluation_spec.json"):(tmp_path/name).write_bytes((ROOT/name).read_bytes())
    (tmp_path/"target_manifest.json").write_text("{}")
    with pytest.raises(ValueError,match="hash mismatch"):load_inputs(tmp_path)
