"""Generated output-block ownership of the frozen Perminov 55-gate circuit.

Run scripts/generate_mi_research_7b_kernel.py to check the source slices.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused(a_ptr, b_ptr, c_ptr):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    col = tl.program_id(1) * 16 + tl.arange(0, 16)
    inner = tl.arange(0, 32)
    output_block = tl.program_id(2)
    if output_block == 0:
        row = tl.program_id(0) * 16 + tl.arange(0, 16)
        col = tl.program_id(1) * 16 + tl.arange(0, 16)
        inner = tl.arange(0, 32)
        A0 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A1 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 1 * 32 + inner[None, :]).to(tl.float32)
        A2 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        A5 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        A8 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        B0 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B3 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B5 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        B6 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B7 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        B8 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        Ug00 = (-A5) + A2
        Ug01 = (-Ug00) + A0
        Ug02 = Ug01 + A1
        Ug03 = Ug02 - A5
        Ug04 = Ug03 - A8
        Ug05 = Ug04 - A1
        Vg01 = B8 + B5
        M05 = tl.dot(Ug05, Vg01, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg04 = B6 + B0
        M17 = tl.dot(Ug00, Vg04, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg08 = Vg04 + B7
        Vg05 = (-B3) + B0
        Vg06 = Vg05 - Vg01
        M09 = tl.dot(Ug01, Vg06, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg09 = Vg08 - Vg06
        M00 = tl.dot(Ug02, Vg09, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg11 = Vg09 - B3
        M18 = tl.dot(Ug04, Vg11, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M07 = tl.dot(A8, B6, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M10 = tl.dot(A8, B7, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg00 = (-M18) - M10
        Wg01 = Wg00 + M05
        Wg02 = (-M07) + Wg01
        M12 = tl.dot(A5, B7, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg04 = M00 - M12
        Wg05 = Wg04 + M09
        Wg06 = M17 + Wg05
        Wg11 = Wg02 + Wg06
        tl.store(c_ptr + (0 * 32 + row[:, None]) * 96 + 0 * 32 + col[None, :], Wg11)
    if output_block == 1:
        row = tl.program_id(0) * 16 + tl.arange(0, 16)
        col = tl.program_id(1) * 16 + tl.arange(0, 16)
        inner = tl.arange(0, 32)
        A0 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A1 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 1 * 32 + inner[None, :]).to(tl.float32)
        A2 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        A5 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        B0 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B1 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        B3 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B4 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        B5 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        B6 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B7 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        B8 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        Ug00 = (-A5) + A2
        Ug01 = (-Ug00) + A0
        Ug02 = Ug01 + A1
        Vg01 = B8 + B5
        Vg04 = B6 + B0
        M17 = tl.dot(Ug00, Vg04, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg08 = Vg04 + B7
        Vg05 = (-B3) + B0
        Vg06 = Vg05 - Vg01
        M09 = tl.dot(Ug01, Vg06, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg09 = Vg08 - Vg06
        Vg10 = Vg08 + B1
        M00 = tl.dot(Ug02, Vg09, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M04 = tl.dot(A0, Vg10, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg13 = Vg09 + B4
        M20 = tl.dot(A1, Vg13, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg16 = M04 + M20
        M12 = tl.dot(A5, B7, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg04 = M00 - M12
        Wg05 = Wg04 + M09
        Wg06 = M17 + Wg05
        Wg17 = Wg16 - Wg06
        tl.store(c_ptr + (0 * 32 + row[:, None]) * 96 + 1 * 32 + col[None, :], Wg17)
    if output_block == 2:
        row = tl.program_id(0) * 16 + tl.arange(0, 16)
        col = tl.program_id(1) * 16 + tl.arange(0, 16)
        inner = tl.arange(0, 32)
        A0 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A1 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 1 * 32 + inner[None, :]).to(tl.float32)
        A2 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        A5 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        A8 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        B0 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B2 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        B3 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B5 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        B6 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B7 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        B8 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        Ug00 = (-A5) + A2
        Ug01 = (-Ug00) + A0
        Ug02 = Ug01 + A1
        Ug03 = Ug02 - A5
        Ug04 = Ug03 - A8
        Ug05 = Ug04 - A1
        Vg00 = B8 + B2
        M02 = tl.dot(A0, Vg00, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg01 = B8 + B5
        M05 = tl.dot(Ug05, Vg01, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg04 = B6 + B0
        Vg08 = Vg04 + B7
        Vg05 = (-B3) + B0
        Vg06 = Vg05 - Vg01
        Vg09 = Vg08 - Vg06
        Vg11 = Vg09 - B3
        M18 = tl.dot(Ug04, Vg11, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg12 = Vg11 - B5
        M11 = tl.dot(Ug03, Vg12, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M07 = tl.dot(A8, B6, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M10 = tl.dot(A8, B7, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg00 = (-M18) - M10
        Wg01 = Wg00 + M05
        Wg02 = (-M07) + Wg01
        Wg03 = M11 + Wg02
        Wg22 = M02 - Wg03
        tl.store(c_ptr + (0 * 32 + row[:, None]) * 96 + 2 * 32 + col[None, :], Wg22)
    if output_block == 3:
        row = tl.program_id(0) * 16 + tl.arange(0, 16)
        col = tl.program_id(1) * 16 + tl.arange(0, 16)
        inner = tl.arange(0, 32)
        A0 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A1 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 1 * 32 + inner[None, :]).to(tl.float32)
        A2 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        A3 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A4 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 1 * 32 + inner[None, :]).to(tl.float32)
        A5 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        A6 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A8 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        B0 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B2 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        B3 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B5 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        B6 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B7 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        B8 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        Ug00 = (-A5) + A2
        Ug01 = (-Ug00) + A0
        Ug02 = Ug01 + A1
        Ug03 = Ug02 - A5
        Ug04 = Ug03 - A8
        Ug05 = Ug04 - A1
        Ug07 = Ug02 - A4
        Ug08 = Ug07 - A3
        Ug09 = Ug08 - A6
        Ug11 = (-A4) + A1
        Ug12 = (-Ug11) + Ug09
        Vg01 = B8 + B5
        M05 = tl.dot(Ug05, Vg01, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg02 = B2 + B0
        M13 = tl.dot(Ug08, Vg02, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg03 = Vg02 - B3
        M01 = tl.dot(Ug09, Vg03, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg04 = B6 + B0
        Vg08 = Vg04 + B7
        Vg05 = (-B3) + B0
        Vg06 = Vg05 - Vg01
        M19 = tl.dot(Ug12, Vg05, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M09 = tl.dot(Ug01, Vg06, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg09 = Vg08 - Vg06
        M00 = tl.dot(Ug02, Vg09, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg11 = Vg09 - B3
        M18 = tl.dot(Ug04, Vg11, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg12 = Vg11 - B5
        M11 = tl.dot(Ug03, Vg12, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg09 = (-M13) - M11
        M07 = tl.dot(A8, B6, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M10 = tl.dot(A8, B7, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg00 = (-M18) - M10
        Wg01 = Wg00 + M05
        Wg02 = (-M07) + Wg01
        Wg03 = M11 + Wg02
        M12 = tl.dot(A5, B7, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg04 = M00 - M12
        Wg05 = Wg04 + M09
        Wg12 = (-M19) + Wg05
        M16 = tl.dot(A6, B2, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg07 = M16 + Wg03
        Wg08 = M01 + Wg07
        Wg10 = Wg09 + Wg08
        Wg13 = Wg12 + Wg10
        tl.store(c_ptr + (1 * 32 + row[:, None]) * 96 + 0 * 32 + col[None, :], Wg13)
    if output_block == 4:
        row = tl.program_id(0) * 16 + tl.arange(0, 16)
        col = tl.program_id(1) * 16 + tl.arange(0, 16)
        inner = tl.arange(0, 32)
        A3 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A4 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 1 * 32 + inner[None, :]).to(tl.float32)
        A5 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        B1 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        B4 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        B7 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        M12 = tl.dot(A5, B7, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M14 = tl.dot(A3, B1, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M21 = tl.dot(A4, B4, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg18 = M14 + M21
        Wg19 = Wg18 + M12
        tl.store(c_ptr + (1 * 32 + row[:, None]) * 96 + 1 * 32 + col[None, :], Wg19)
    if output_block == 5:
        row = tl.program_id(0) * 16 + tl.arange(0, 16)
        col = tl.program_id(1) * 16 + tl.arange(0, 16)
        inner = tl.arange(0, 32)
        A0 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A1 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 1 * 32 + inner[None, :]).to(tl.float32)
        A2 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        A3 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A4 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 1 * 32 + inner[None, :]).to(tl.float32)
        A5 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        A6 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A8 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        B0 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B2 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        B3 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B5 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        B6 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B7 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        B8 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        Ug00 = (-A5) + A2
        Ug01 = (-Ug00) + A0
        Ug02 = Ug01 + A1
        Ug03 = Ug02 - A5
        Ug04 = Ug03 - A8
        Ug05 = Ug04 - A1
        Ug07 = Ug02 - A4
        Ug08 = Ug07 - A3
        Ug09 = Ug08 - A6
        Ug11 = (-A4) + A1
        Ug12 = (-Ug11) + Ug09
        Vg01 = B8 + B5
        M05 = tl.dot(Ug05, Vg01, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg02 = B2 + B0
        Vg03 = Vg02 - B3
        M01 = tl.dot(Ug09, Vg03, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg07 = Vg03 - B5
        M22 = tl.dot(Ug07, Vg07, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg04 = B6 + B0
        Vg08 = Vg04 + B7
        Vg05 = (-B3) + B0
        Vg06 = Vg05 - Vg01
        M19 = tl.dot(Ug12, Vg05, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M09 = tl.dot(Ug01, Vg06, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg09 = Vg08 - Vg06
        Wg23 = M22 + M19
        Wg24 = Wg23 - M09
        Vg11 = Vg09 - B3
        M18 = tl.dot(Ug04, Vg11, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg12 = Vg11 - B5
        M11 = tl.dot(Ug03, Vg12, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M07 = tl.dot(A8, B6, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M10 = tl.dot(A8, B7, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg00 = (-M18) - M10
        Wg01 = Wg00 + M05
        Wg02 = (-M07) + Wg01
        Wg03 = M11 + Wg02
        M16 = tl.dot(A6, B2, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg07 = M16 + Wg03
        Wg08 = M01 + Wg07
        Wg25 = Wg24 - Wg08
        tl.store(c_ptr + (1 * 32 + row[:, None]) * 96 + 2 * 32 + col[None, :], Wg25)
    if output_block == 6:
        row = tl.program_id(0) * 16 + tl.arange(0, 16)
        col = tl.program_id(1) * 16 + tl.arange(0, 16)
        inner = tl.arange(0, 32)
        A0 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A1 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 1 * 32 + inner[None, :]).to(tl.float32)
        A2 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        A3 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A4 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 1 * 32 + inner[None, :]).to(tl.float32)
        A5 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        A6 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A7 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 1 * 32 + inner[None, :]).to(tl.float32)
        A8 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        B0 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B2 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        B3 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B5 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        B6 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B7 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        B8 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        Ug00 = (-A5) + A2
        Ug01 = (-Ug00) + A0
        Ug02 = Ug01 + A1
        Ug03 = Ug02 - A5
        Ug04 = Ug03 - A8
        Ug05 = Ug04 - A1
        Ug07 = Ug02 - A4
        Ug08 = Ug07 - A3
        Ug09 = Ug08 - A6
        Ug10 = Ug09 - A7
        M08 = tl.dot(Ug10, B3, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg01 = B8 + B5
        M05 = tl.dot(Ug05, Vg01, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg02 = B2 + B0
        M13 = tl.dot(Ug08, Vg02, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg03 = Vg02 - B3
        M01 = tl.dot(Ug09, Vg03, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg04 = B6 + B0
        Vg08 = Vg04 + B7
        Vg05 = (-B3) + B0
        Vg06 = Vg05 - Vg01
        Vg09 = Vg08 - Vg06
        Vg11 = Vg09 - B3
        M18 = tl.dot(Ug04, Vg11, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg12 = Vg11 - B5
        M11 = tl.dot(Ug03, Vg12, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg09 = (-M13) - M11
        M07 = tl.dot(A8, B6, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M10 = tl.dot(A8, B7, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg00 = (-M18) - M10
        Wg01 = Wg00 + M05
        Wg02 = (-M07) + Wg01
        Wg14 = (-M08) + Wg01
        Wg03 = M11 + Wg02
        M16 = tl.dot(A6, B2, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg07 = M16 + Wg03
        Wg08 = M01 + Wg07
        Wg10 = Wg09 + Wg08
        Wg15 = Wg14 - Wg10
        tl.store(c_ptr + (2 * 32 + row[:, None]) * 96 + 0 * 32 + col[None, :], Wg15)
    if output_block == 7:
        row = tl.program_id(0) * 16 + tl.arange(0, 16)
        col = tl.program_id(1) * 16 + tl.arange(0, 16)
        inner = tl.arange(0, 32)
        A6 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A7 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 1 * 32 + inner[None, :]).to(tl.float32)
        A8 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        B1 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        B4 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        B7 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        M03 = tl.dot(A7, B4, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M10 = tl.dot(A8, B7, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M15 = tl.dot(A6, B1, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg20 = M03 + M15
        Wg21 = Wg20 + M10
        tl.store(c_ptr + (2 * 32 + row[:, None]) * 96 + 1 * 32 + col[None, :], Wg21)
    if output_block == 8:
        row = tl.program_id(0) * 16 + tl.arange(0, 16)
        col = tl.program_id(1) * 16 + tl.arange(0, 16)
        inner = tl.arange(0, 32)
        A0 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A1 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 1 * 32 + inner[None, :]).to(tl.float32)
        A2 = tl.load(a_ptr + (0 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        A5 = tl.load(a_ptr + (1 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        A6 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 0 * 32 + inner[None, :]).to(tl.float32)
        A7 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 1 * 32 + inner[None, :]).to(tl.float32)
        A8 = tl.load(a_ptr + (2 * 32 + row[:, None]) * 96 + 2 * 32 + inner[None, :]).to(tl.float32)
        B0 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B2 = tl.load(b_ptr + (0 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        B3 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B5 = tl.load(b_ptr + (1 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        B6 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 0 * 32 + col[None, :]).to(tl.float32)
        B7 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 1 * 32 + col[None, :]).to(tl.float32)
        B8 = tl.load(b_ptr + (2 * 32 + inner[:, None]) * 96 + 2 * 32 + col[None, :]).to(tl.float32)
        Ug00 = (-A5) + A2
        Ug01 = (-Ug00) + A0
        Ug02 = Ug01 + A1
        Ug03 = Ug02 - A5
        Ug04 = Ug03 - A8
        Ug05 = Ug04 - A1
        Ug06 = Ug04 + A7
        M06 = tl.dot(Ug06, B5, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg01 = B8 + B5
        M05 = tl.dot(Ug05, Vg01, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg26 = M06 - M05
        Vg04 = B6 + B0
        Vg08 = Vg04 + B7
        Vg05 = (-B3) + B0
        Vg06 = Vg05 - Vg01
        Vg09 = Vg08 - Vg06
        Vg11 = Vg09 - B3
        M18 = tl.dot(Ug04, Vg11, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Vg12 = Vg11 - B5
        M11 = tl.dot(Ug03, Vg12, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M07 = tl.dot(A8, B6, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        M10 = tl.dot(A8, B7, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg00 = (-M18) - M10
        Wg01 = Wg00 + M05
        Wg02 = (-M07) + Wg01
        Wg03 = M11 + Wg02
        M16 = tl.dot(A6, B2, tl.zeros((16, 16), tl.float32), input_precision="tf32x3")
        Wg07 = M16 + Wg03
        Wg27 = Wg26 + Wg07
        tl.store(c_ptr + (2 * 32 + row[:, None]) * 96 + 2 * 32 + col[None, :], Wg27)


def call(left, right, output):
    if (left.shape != (96, 96) or right.shape != (96, 96) or
            output.shape != (96, 96) or left.dtype != torch.float16 or
            right.dtype != torch.float16 or output.dtype != torch.float16 or
            not left.is_cuda or not right.is_cuda or not output.is_cuda or
            not left.is_contiguous() or not right.is_contiguous() or
            not output.is_contiguous()):
        raise ValueError("expected contiguous CUDA FP16 96x96 inputs/output")
    return _fused[(2, 2, 9)](left, right, output, num_warps=4, num_stages=2)
