#!/usr/bin/env python3
"""CPU-only tests for the PTX inspection and comparison utility."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ptx_inspector as ptx


BASELINE = r'''// retained compiler annotation
.version 8.0
.target sm_80
.address_size 64
.file 1 "baseline.cu"
.extern .shared .align 16 .b8 dynamic_smem[];

.visible .entry kernel_a(
    .param .u64 kernel_a_param_0
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .f32 %f1, %f2;
    .shared .align 4 .b8 tile[64];
    .loc 1 10 2
$L0:
    ld.global.v2.f32
        {%f1, %f2},
        [%r1];
    @!%p1 bra.uni $L1;
    mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
        {%f1, %f2},
        {%r1, %r2},
        {%r2, %r3},
        {%f1, %f2};
    mystery.op.foo {%r1, %r2};
$L1:
    st.shared.v2.f32 [tile], {%f1, %f2};
    bar.sync 0;
    ret;
}

.func helper(.param .b32 helper_param)
{
    .reg .b32 %h<2>;
    add.u32 %h1, %h0, 1;
    ret;
}
'''


SECOND_KERNEL = r'''
.visible .entry kernel_b(.param .u32 value)
{
    ret;
}
'''


class PtxInspectorTests(unittest.TestCase):
    def test_metadata_symbols_and_resources(self):
        module = ptx.parse_module(BASELINE)
        self.assertEqual((module.version, module.target, module.address_size), ("8.0", "sm_80", 64))
        self.assertEqual([item.name for item in module.symbols], ["kernel_a", "helper"])
        kernel = module.symbols[0]
        summary = ptx.symbol_summary(kernel)
        self.assertEqual(summary["ptx_declared_registers_by_type"], {"b32": 4, "f32": 2, "pred": 2})
        self.assertEqual(summary["ptx_declared_static_shared_bytes"], 64)
        self.assertEqual(module.module_shared_memory[0].name, "dynamic_smem")
        self.assertTrue(module.module_shared_memory[0].dynamic)
        self.assertIn("physical register", ptx.module_summary(module, [kernel])["resource_note"])

    def test_multiline_vector_and_brace_operands(self):
        kernel = ptx.parse_module(BASELINE).symbols[0]
        load = kernel.instructions[0]
        self.assertEqual(load.mnemonic, "ld.global.v2.f32")
        self.assertEqual(load.operands, "{%f1, %f2}, [%r1]")
        mma = next(item for item in kernel.instructions if item.base == "mma")
        self.assertEqual(mma.category, "tensor")
        self.assertIn("{%r1, %r2}", mma.operands)
        self.assertEqual(kernel.labels, ["$L0", "$L1"])
        self.assertEqual(len([item for item in kernel.instructions if item.mnemonic == "ret"]), 1)

    def test_predicates_memory_sync_and_unknown(self):
        kernel = ptx.parse_module(BASELINE).symbols[0]
        branch = next(item for item in kernel.instructions if item.base == "bra")
        self.assertEqual(branch.predicate, "!%p1")
        self.assertEqual(branch.category, "control_flow")
        categories = {item.mnemonic: item.category for item in kernel.instructions}
        self.assertEqual(categories["ld.global.v2.f32"], "memory_global")
        self.assertEqual(categories["st.shared.v2.f32"], "memory_shared")
        self.assertEqual(categories["bar.sync"], "synchronization")
        self.assertEqual(categories["mystery.op.foo"], "other")

    def test_unqualified_load_store_use_generic_addressing_category(self):
        load = ptx.parse_instruction("ld.f32 %f1, [%rd1];")
        store = ptx.parse_instruction("st.f32 [%rd2], %f2;")
        self.assertEqual(load.category, "memory_generic")
        self.assertEqual(store.category, "memory_generic")

    def test_tensor_and_fence_families(self):
        for mnemonic in ("mma.sync.aligned.m16n8k8", "wmma.mma.sync", "wgmma.mma_async"):
            self.assertEqual(ptx.classify_instruction(mnemonic), "tensor")
        for mnemonic in ("bar.sync", "mbarrier.init.shared.b64", "fence.acq_rel.gpu"):
            self.assertEqual(ptx.classify_instruction(mnemonic), "synchronization")

    def test_normalization_is_conservative(self):
        candidate = BASELINE.replace('.file 1 "baseline.cu"', '.file 7 "elsewhere.cu"').replace(".loc 1 10 2", ".loc 7 99 4")
        self.assertNotEqual(BASELINE, candidate)
        self.assertEqual(ptx.normalize_ptx(BASELINE), ptx.normalize_ptx(candidate))
        self.assertIn(".extern .shared", ptx.normalize_ptx(BASELINE))
        self.assertIn(".visible .entry kernel_a", ptx.normalize_ptx(BASELINE))
        changed = candidate.replace("bar.sync 0", "membar.gl")
        self.assertNotEqual(ptx.normalize_ptx(BASELINE), ptx.normalize_ptx(changed))
        self.assertIn("// retained compiler annotation", ptx.normalize_ptx(BASELINE))

    def test_kernel_selection_ambiguity_and_explicit_selection(self):
        module = ptx.parse_module(BASELINE + SECOND_KERNEL)
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            ptx.select_comparison(module, module, None, None, None)
        left, right = ptx.select_comparison(module, module, "kernel_a", None, None)
        self.assertEqual((left.name, right.name), ("kernel_a", "kernel_a"))
        with self.assertRaisesRegex(ValueError, "supplied together"):
            ptx.select_comparison(module, module, None, "kernel_a", None)

    def test_version_target_and_instruction_changes(self):
        left = ptx.parse_module(BASELINE)
        right = ptx.parse_module(BASELINE.replace(".version 8.0", ".version 8.1").replace("sm_80", "sm_90").replace("bar.sync 0", "membar.gl"))
        left_kernel, right_kernel = left.symbols[0], right.symbols[0]
        comparison = ptx.compare_symbols(left_kernel, right_kernel)
        sync = {row["name"]: row for row in comparison["instruction_mnemonics"]}
        self.assertEqual(sync["bar.sync"]["delta"], -1)
        self.assertEqual(sync["membar.gl"]["delta"], 1)
        self.assertNotEqual(left.version, right.version)
        self.assertNotEqual(left.target, right.target)

    def test_malformed_input_diagnostics(self):
        with self.assertRaisesRegex(ValueError, "declare"):
            ptx.parse_module(".version 8.0\n")
        with self.assertRaisesRegex(ValueError, "unbalanced"):
            ptx.parse_module(BASELINE.rsplit("}", 1)[0])
        malformed_operand = BASELINE.replace("{%f1, %f2},", "{%f1, %f2],", 1)
        with self.assertRaisesRegex(ValueError, "unmatched|unbalanced"):
            ptx.parse_module(malformed_operand)

    def test_inspect_and_compare_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.ptx"
            candidate = root / "candidate.ptx"
            baseline.write_text(BASELINE)
            candidate.write_text(BASELINE.replace('.file 1 "baseline.cu"', '.file 2 "candidate.cu"'))
            inspect_output = root / "inspection"
            inspection = ptx.run(["inspect", "--input", str(baseline), "--kernel", "kernel_a",
                                  "--output", str(inspect_output)])
            self.assertEqual(inspection["schema_version"], 1)
            self.assertEqual(inspection["input"]["size_bytes"], len(baseline.read_bytes()))
            self.assertEqual(inspection["input"]["sha256"], ptx.sha256(baseline.read_bytes()))
            self.assertTrue((inspect_output / "analysis.json").is_file())
            inspection_report = (inspect_output / "report.md").read_text()
            self.assertIn("PTX-declared registers", inspection_report)
            self.assertIn("not final physical register allocation", inspection_report)
            with self.assertRaises(FileExistsError):
                ptx.run(["inspect", "--input", str(baseline), "--output", str(inspect_output)])

            compare_output = root / "comparison"
            comparison = ptx.run(["compare", "--baseline", str(baseline), "--candidate", str(candidate),
                                  "--kernel", "kernel_a", "--output", str(compare_output)])
            self.assertFalse(comparison["comparison"]["raw"]["identical"])
            self.assertTrue(comparison["comparison"]["normalized"]["identical"])
            self.assertEqual(comparison["comparison"]["normalized"]["scope"], "full_module")
            self.assertEqual(comparison["comparison"]["kernel"]["scope"], "selected_entry_only")
            self.assertTrue((compare_output / "comparison.json").is_file())
            self.assertTrue((compare_output / "ptx" / "normalized.diff").is_file())
            report = (compare_output / "report.md").read_text()
            self.assertIn("## Full-module text comparison", report)
            self.assertIn("compare the complete PTX modules", report)
            self.assertIn("## Selected-entry structural comparison", report)
            self.assertIn("### Full mnemonics", report)
            self.assertIn("not final physical register allocation", report)
            captured = json.loads((compare_output / "comparison.json").read_text())
            self.assertEqual(captured["tool"], "ptx-inspector")


if __name__ == "__main__":
    unittest.main()
