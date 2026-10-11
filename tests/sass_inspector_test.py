#!/usr/bin/env python3
"""CPU-only tests for the nvdisasm SASS inspection utility."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sass_inspector as sass


CAPTURE = r'''.target sm_89
.section .debug_frame { }
/*0000*/ .byte 0x00
.section .text.mi_naive_fma,"ax",@progbits
.sectioninfo @"SHI_REGISTERS=28"
.type mi_naive_fma,@function
.size mi_naive_fma,(.L_end-mi_naive_fma)
mi_naive_fma:
.text.mi_naive_fma:
/*0000*/                   MOV R1, c[0x0][0x28] ;
/*0010*/              @P0  FFMA R8, R4, R5, R8 ;
.L_loop:
/*0020*/                   CALL.REL.NOINC `($__internal_3_$__cuda_sm20_div_u64) ;
/*0030*/                   EXIT ;
.type $__internal_3_$__cuda_sm20_div_u64,@function
.size $__internal_3_$__cuda_sm20_div_u64,(.L_helper_end-$__internal_3_$__cuda_sm20_div_u64)
$__internal_3_$__cuda_sm20_div_u64:
/*0040*/                   IADD3 R4, R4, 0x1, RZ ;
/*0050*/                   RET.REL.NODEC R20 0x0 ;
.section .nv.info.mi_naive_fma,"a",@progbits
/*9990*/                   LDG.E R0, [R2] ;
'''


NO_FMA = CAPTURE.replace(".text.mi_naive_fma", ".text.mi_naive_no_fma").replace(
    "mi_naive_fma", "mi_naive_no_fma"
).replace("FFMA R8, R4, R5, R8", "FMUL R9, R4, R5")


class SassInspectorTests(unittest.TestCase):
    def test_sections_functions_resources_and_metadata_exclusion(self):
        module = sass.parse_module(CAPTURE)
        self.assertEqual(module.target, "sm_89")
        self.assertEqual([item.name for item in module.functions], [
            "mi_naive_fma", "$__internal_3_$__cuda_sm20_div_u64"
        ])
        primary, helper = module.functions
        self.assertEqual((primary.role, helper.role), ("primary", "helper"))
        self.assertEqual(primary.allocated_registers_per_thread, 28)
        self.assertIsNone(helper.allocated_registers_per_thread)
        self.assertNotIn("LDG", [item.base for item in primary.instructions])
        self.assertEqual(module.ambiguities, [])

    def test_addresses_predicates_modifiers_calls_and_unknowns(self):
        primary = sass.parse_module(CAPTURE).functions[0]
        ffma = next(item for item in primary.instructions if item.base == "FFMA")
        self.assertEqual((ffma.address, ffma.address_value), ("0010", 0x10))
        self.assertEqual(ffma.predicate, "@P0")
        call = primary.instructions[-2]
        self.assertEqual(call.mnemonic, "CALL.REL.NOINC")
        self.assertEqual(call.modifiers, ["REL", "NOINC"])
        self.assertEqual(call.call_target, "$__internal_3_$__cuda_sm20_div_u64")
        unknown = sass.parse_instruction("/*00a0*/ MYSTERY.X R1, {R2, R3} ;")
        self.assertEqual(unknown.category, "other")

    def test_labels_remain_in_control_flow_order_while_addresses_are_omitted(self):
        primary = sass.parse_module(CAPTURE).functions[0]
        normalized = sass.normalize_function(primary)
        self.assertLess(normalized.index("MOV"), normalized.index(".L_loop:"))
        self.assertLess(normalized.index(".L_loop:"), normalized.index("CALL"))
        self.assertNotIn("/*0010*/", normalized)
        self.assertEqual(primary.instructions[1].address, "0010")

    def test_classification_families(self):
        expected = {
            "FFMA": "floating_point_arithmetic", "FMUL": "floating_point_arithmetic",
            "FADD": "floating_point_arithmetic", "FSETP.GT": "comparison_predicate",
            "LDG.E": "memory_global", "LDS": "memory_shared", "LDL": "memory_local",
            "LDC": "memory_constant", "CALL.REL.NOINC": "call", "BAR.SYNC": "synchronization",
            "HMMA.1688.F32": "tensor",
        }
        for mnemonic, category in expected.items():
            self.assertEqual(sass.classify_instruction(mnemonic), category)

    def test_embedded_helper_is_excluded_from_primary_counts(self):
        module = sass.parse_module(CAPTURE)
        primary, helper = module.functions
        summary = sass.function_summary(primary)
        self.assertEqual(summary["base_operations"]["FFMA"], 1)
        self.assertNotIn("IADD3", summary["base_operations"])
        self.assertEqual(len(helper.instructions), 2)

    def test_ambiguous_boundaries_are_not_attributed(self):
        source = CAPTURE.replace("mi_naive_fma:\n", "wrong_symbol:\n", 1)
        module = sass.parse_module(source)
        self.assertFalse(any(item.name == "mi_naive_fma" for item in module.functions))
        reasons = {item["reason"] for item in module.ambiguities}
        self.assertIn("missing_or_ambiguous_exact_symbol_label", reasons)
        self.assertIn("orphan_instruction_region", reasons)
        self.assertIn("section_primary_function_not_resolved", reasons)

    def test_explicit_pair_selection_and_ambiguous_default(self):
        left = sass.parse_module(CAPTURE)
        right = sass.parse_module(NO_FMA)
        selected = sass.select_comparison(left, right, None, "mi_naive_fma", "mi_naive_no_fma")
        self.assertEqual((selected[0].name, selected[1].name), ("mi_naive_fma", "mi_naive_no_fma"))
        with self.assertRaisesRegex(ValueError, "explicit selection"):
            sass.select_comparison(left, right, None, None, None)

    def test_raw_and_normalized_scopes_and_output_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root / "a.sass", root / "b.sass"
            baseline.write_text(CAPTURE)
            candidate.write_text(NO_FMA)
            inspection_dir = root / "inspection"
            result = sass.run(["inspect", "--input", str(baseline), "--kernel", "mi_naive_fma",
                               "--output", str(inspection_dir)])
            captured = json.loads((inspection_dir / "analysis.json").read_text())
            instruction = captured["analysis"]["selected_functions"][0]["instructions"][0]
            self.assertEqual(instruction["address"], "0000")
            self.assertIn("physical 32-bit register allocation", (inspection_dir / "report.md").read_text())
            with self.assertRaises(FileExistsError):
                sass.run(["inspect", "--input", str(baseline), "--output", str(inspection_dir)])

            comparison_dir = root / "comparison"
            comparison = sass.run([
                "compare", "--baseline", str(baseline), "--candidate", str(candidate),
                "--baseline-kernel", "mi_naive_fma", "--candidate-kernel", "mi_naive_no_fma",
                "--output", str(comparison_dir)
            ])
            self.assertEqual(comparison["comparison"]["raw"]["scope"], "full_file")
            self.assertEqual(comparison["comparison"]["function"]["scope"], "selected_function_only")
            report = (comparison_dir / "report.md").read_text()
            self.assertIn("Normalized similarity is neither binary nor semantic equivalence", report)
            self.assertTrue((comparison_dir / "sass" / "normalized.diff").is_file())
            self.assertEqual(result["tool"], "sass-inspector")


if __name__ == "__main__":
    unittest.main()
