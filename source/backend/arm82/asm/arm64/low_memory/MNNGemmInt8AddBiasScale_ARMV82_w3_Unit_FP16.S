//
//  MNNGemmInt8AddBiasScale_ARMV82_w3_Unit_FP16.S
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//
//  ARMv8.2 sdot path, fp16 output, 3-bit quantized weights.
//  Bit-plane (2 + 1) split.  Per cell (UNIT=8 OC * SRC_UNIT=4 IC = 32 weights):
//
//    main plane (8 bytes):  same 4-OC-per-byte packing as w2 sdot
//      bytes 0..3: bits[6:7]=oc0, [4:5]=oc1, [2:3]=oc2, [0:1]=oc3 at IC=b
//      bytes 4..7: bits[6:7]=oc4, [4:5]=oc5, [2:3]=oc6, [0:1]=oc7 at IC=b-4
//
//    aux plane (4 bytes):   2 OCs * 4 IC per byte (low / high nibble)
//      byte 0: bits[3:0]=oc0_aux at IC[0..3], bits[7:4]=oc1_aux at IC[0..3]
//      byte 1: bits[3:0]=oc2_aux at IC[0..3], bits[7:4]=oc3_aux at IC[0..3]
//      byte 2: oc4 / oc5 (same scheme)
//      byte 3: oc6 / oc7
//
//  Total: 12 bytes/cell (= 32 weights * 3 bits).
//
//  Unpack produces unsigned [0, 7] in oc03/oc47.  The C++ post-process applies
//  originOffset = -4 to re-center to signed [-4, 3].  Two preloaded const vectors:
//    idx_main + shifts_main (same as w2 sdot)  in v6/v7
//    idx_aux  + shifts_aux                       in v28/v29
//  See UNPACK_W3_SDOT below.
//

#if defined(__aarch64__)
#include "MNNAsmGlobal.h"

.text
.align 5

// Unpack a single w3 cell.  Inputs:
//   v_w_main  — 8 packed main bytes (only lower 8 bytes of the reg are valid)
//   v_w_aux   — 4 packed aux bytes  (only lower 4 bytes valid)
//   v_idx_m / v_shifts_m  — main extract constants (same as w2 sdot)
//   v_idx_a / v_shifts_a  — aux extract constants
// Outputs:
//   v_oc03  — unsigned int8 weights for OC 0-3 * 4 IC  (16 lanes, range [0, 7])
//   v_oc47  — unsigned int8 weights for OC 4-7 * 4 IC
// Both inputs are clobbered (v_w_main used as ext scratch + movi mask3,
// v_w_aux used as ext scratch).
// Unsigned-domain output; C++ applies originOffset = -4 to re-center to [-4, 3].
// 13 ALU ops/cell (was 16; saved 1 movi + 2 bic, replaced 2 sub with 2 orr).
.macro UNPACK_W3_SDOT v_w_main, v_w_aux, v_oc03, v_oc47, \
                      v_idx_m, v_shifts_m, v_idx_a, v_shifts_a, v_t1, v_t2
    // ---- Main extract for OC 0-3 (bytes 0..3 of v_w_main) ----
    tbl  \v_t1\().16b,    {\v_w_main\().16b}, \v_idx_m\().16b   // bytes 0..3 replicated 4x
    ext  \v_w_main\().16b, \v_w_main\().16b, \v_w_main\().16b, #4 // rotate so 4..7 -> 0..3
    tbl  \v_t2\().16b,    {\v_w_main\().16b}, \v_idx_m\().16b   // OC 4-7 main
    ushl \v_t1\().16b,    \v_t1\().16b, \v_shifts_m\().16b
    ushl \v_t2\().16b,    \v_t2\().16b, \v_shifts_m\().16b
    movi \v_w_main\().16b, #3                                  // mask3
    and  \v_t1\().16b,    \v_t1\().16b, \v_w_main\().16b      // v_t1 = main_03 (range [0, 3])
    and  \v_t2\().16b,    \v_t2\().16b, \v_w_main\().16b      // v_t2 = main_47

    // ---- Aux extract for OC 0-3 (bytes 0..1 of v_w_aux) ----
    tbl  \v_oc03\().16b,  {\v_w_aux\().16b}, \v_idx_a\().16b   // [aux[0]x8, aux[1]x8]
    ext  \v_w_aux\().16b, \v_w_aux\().16b, \v_w_aux\().16b, #2 // rotate aux 2..3 -> 0..1
    tbl  \v_oc47\().16b,  {\v_w_aux\().16b}, \v_idx_a\().16b   // OC 4-7 aux
    ushl \v_oc03\().16b,  \v_oc03\().16b, \v_shifts_a\().16b   // aux bit -> bit 2 (= 4 or 0)
    ushl \v_oc47\().16b,  \v_oc47\().16b, \v_shifts_a\().16b
    // ushl leaves garbage in bits 0,1,3-7 (other aux bits of the same byte
    // shifted into wrong lanes).  Mask to bit 2 only before orr with main.
    movi \v_w_aux\().16b, #4
    and  \v_oc03\().16b,  \v_oc03\().16b, \v_w_aux\().16b
    and  \v_oc47\().16b,  \v_oc47\().16b, \v_w_aux\().16b

    // ---- Combine: unsigned 0..7 = main_2bit | aux_at_bit_2 ----
    orr  \v_oc03\().16b,  \v_t1\().16b, \v_oc03\().16b
    orr  \v_oc47\().16b,  \v_t2\().16b, \v_oc47\().16b
.endm

.macro ADD_BIAS_FLOAT d0, d1, d2, d3, z0
    fadd \d0\().4s, \d0\().4s, \z0\().4s
    fadd \d1\().4s, \d1\().4s, \z0\().4s
    fadd \d2\().4s, \d2\().4s, \z0\().4s
    fadd \d3\().4s, \d3\().4s, \z0\().4s
.endm

.macro SET_BIAS d0, d1, d2, d3
    movi \d0\().16b, #0
    movi \d1\().16b, #0
    movi \d2\().16b, #0
    movi \d3\().16b, #0
.endm
.macro Int32ToFloat z0, z1, z2, z3
    scvtf \z0\().4s, \z0\().4s
    scvtf \z1\().4s, \z1\().4s
    scvtf \z2\().4s, \z2\().4s
    scvtf \z3\().4s, \z3\().4s
.endm
.macro MUL_SCALE s, d0, d1, d2, d3
    fmul \d0\().4s, \d0\().4s, \s\().4s
    fmul \d1\().4s, \d1\().4s, \s\().4s
    fmul \d2\().4s, \d2\().4s, \s\().4s
    fmul \d3\().4s, \d3\().4s, \s\().4s
.endm
.macro MLA_WEIGHTZERO d0, s0, s1, idx // idx for xKernelSum
    fmla \d0\().4s, \s1\().4s, \s0\().s[\idx]
.endm
.macro MUL_EXTRA_SCALE s, d0, d1, d2, d3
    fmul \d0\().4s, \d0\().4s, \s\().s[0]
    fmul \d1\().4s, \d1\().4s, \s\().s[1]
    fmul \d2\().4s, \d2\().4s, \s\().s[2]
    fmul \d3\().4s, \d3\().4s, \s\().s[3]
.endm
.macro ReLU_FP16 s0, s1, s2, s3, z0, z1 // z0:min z1:max
    fmin \s0\().8h, \s0\().8h, \z1\().8h
    fmin \s1\().8h, \s1\().8h, \z1\().8h
    fmin \s2\().8h, \s2\().8h, \z1\().8h
    fmin \s3\().8h, \s3\().8h, \z1\().8h
    fmax \s0\().8h, \s0\().8h, \z0\().8h
    fmax \s1\().8h, \s1\().8h, \z0\().8h
    fmax \s2\().8h, \s2\().8h, \z0\().8h
    fmax \s3\().8h, \s3\().8h, \z0\().8h
.endm

.macro Float32ToHalf s0, s1, s2, s3, d0, d1
    fcvtn \d0\().4h,  \s0\().4s
    fcvtn2 \d0\().8h, \s1\().4s
    fcvtn \d1\().4h,  \s2\().4s
    fcvtn2 \d1\().8h, \s3\().4s
.endm
.macro ADD_FLOAT d0, d1, d2, d3, s0, s1, s2, s3
    fadd \d0\().4s, \d0\().4s, \s0\().4s
    fadd \d1\().4s, \d1\().4s, \s1\().4s
    fadd \d2\().4s, \d2\().4s, \s2\().4s
    fadd \d3\().4s, \d3\().4s, \s3\().4s
.endm

.macro REVERT_INPUT_DEQUANT_BIAS rg0, rg1, rg2, rg3
mul \rg1, \rg2, \rg3
sub \rg0, \rg0, \rg1
.endm

.macro REVERT_WEIGHT_KERNEL_SUM rg0, rg1, rg2, rg3
mul \rg1, \rg2, \rg3 // blocknum * ocUp4 * sizeof(float)
sub \rg0, \rg0, \rg1, LSL #5 // revert weight kernel sum
.endm

asm_function MNNGemmInt8AddBiasScale_ARMV82_w3_Unit_FP16
/* 
struct QuanPostTreatParameters {
    const float* scale;
    const float* biasFloat;
    int32_t maxValue;
    int32_t minValue;
    int32_t useInt8 = 1; // Save result as int8_t dataType; otherwise float32.
    float roundValuePos = 0.5f;
    float roundValueNeg = -0.5f;
    float* srcKernelSum;
    float* weightKernelSum;
    float* fp32minmax;
    ssize_t blockNum = 1;
    const int32_t* bias = nullptr;
    const float* inputScale = nullptr;
    const float* inputBias = nullptr;
    float* accumBuffer = nullptr;
};
*/

//void MNNGemmInt8AddBiasScale_ARMV82_w4_Unit_FP16(int8_t* dst, const int8_t* src, 
//    const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
// const QuanPostTreatParameters* parameters, size_t realDstCount);

//Auto: x0:dst, x1:src, x2:weight, x3:src_depth_quad, x4:dst_step
//x5:dst_depth_quad, x6: parameters, x7: realDstCount

//Load from x6: x9: bias, x8: xKernelSum, x23: fp32minmax
ldr x9, [x6, #8]

stp d14, d15, [sp, #(-16 * 9)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]
stp x23, x24, [sp, #(16 * 6)]
stp x25, x26, [sp, #(16 * 7)]
stp x27, x28, [sp, #(16 * 8)]

ldr x8, [x6, #40]  // srcKernelSum
ldr x28, [x6, #48] // weightKernelSum
ldr x24, [x6, #80]  // inputScale
ldr x27, [x6, #88]  // inputBias
ldr x10, [x6, #96]  // accumBuffer

ldr x23, [x6, #56]  // fp32minmax
ldr x26, [x6, #64]  // blockNum
lsl x22, x7, #2 // eDest * SRC_UNIT

// x25 = inputScale/inputBias post-index stride per block in TILE_1.
// TILE_1 only loads one fp32 value, so the stride is:
//   - no inputBias: mInputBlockNum == 1, re-read the same input scale for every block
//   - inputBias set: mInputBlockNum == blockNum, advance by realDstCount * sizeof(float)
mov x25, #0
cbz x27, _w3_load_consts
mov x25, x22

_w3_load_consts:
    // Main extract:
    //   v6 = idx_main    = [0,1,2,3 x4]
    //   v7 = shifts_main = [-6×4, -4×4, -2×4, 0×4]
    // Aux extract:
    //   v28 = idx_aux    = [0×8, 1×8]
    //   v29 = shifts_aux = [2,1,0,-1,-2,-3,-4,-5] x2  (bring bit i to bit 2)
    adr x16, .L_w3_sdot_unpack_consts
    ld1 {v6.16b, v7.16b},  [x16], #32
    ld1 {v28.16b, v29.16b}, [x16]

// MVP: TILE_12 / TILE_8 / TILE_4 fall through to single-batch TILE_1.
// Larger tile widths can be added later for prefill speedup.
TILE_12:
TILE_8:
TILE_4:
    cbz x7, End
    b TILE_1_Init

TILE_1_Init:
    cbz x23, TILE_1
    ld1r {v26.4s}, [x23], #4 // f32 min
    ld1r {v27.4s}, [x23] // f32 max
    sub x23, x23, #4
TILE_1:
    // v2 = unpack idx, v7 = unpack shifts (preloaded at function entry, kept across all paths)
    mov x6, x0
    mov x12, x2
    mov x14, x5
    mov x20, x9
    mov x15, x8 // input kernel sum
    mov x21, x24 // input dequant scale
L8LoopDz_TILE_1:
    mov x11, x1
    mov x19, #0
TILE1_BLOCKNUM:
    mov x13, x3

    movi v8.16b, #0           // accumulator OC 0-3
    movi v9.16b, #0           // accumulator OC 4-7

    cmp x22, #4
    bne L8LoopSz_TILE_1_lu1   // realDstCount != 1: lu4 src layout assumes batch=1
    cmp x13, #4
    blt L8LoopSz_TILE_1_lu1   // < 4 cells: skip lu4 unrolling

    movi v30.16b, #0          // shadow accum OC 0-3 for ILP
    movi v31.16b, #0

    L8LoopSz_TILE_1_lu4:
        // 4 cells of w3 weight: each cell = 16B (8 main + 4 aux + 4 zero-pad).
        ld1 {v10.16b, v11.16b, v12.16b, v13.16b}, [x12], #64
        ld1 {v0.4s}, [x11], #16              // 16B src = 4 IC chunks * 1 batch
        sub x13, x13, #4

        // Each cell's aux+pad is in bytes 8..15 of its loaded reg.  ext rotates the
        // upper half down so v_w_aux bytes 0..3 hold the 4 aux bytes (4..7 = pad).
        ext v22.16b, v10.16b, v10.16b, #8
        UNPACK_W3_SDOT v10, v22, v14, v18, v6, v7, v28, v29, v26, v27   // cell 0

        ext v23.16b, v11.16b, v11.16b, #8
        UNPACK_W3_SDOT v11, v23, v15, v19, v6, v7, v28, v29, v26, v27   // cell 1

        ext v24.16b, v12.16b, v12.16b, #8
        UNPACK_W3_SDOT v12, v24, v16, v20, v6, v7, v28, v29, v26, v27   // cell 2

        ext v25.16b, v13.16b, v13.16b, #8
        UNPACK_W3_SDOT v13, v25, v17, v21, v6, v7, v28, v29, v26, v27   // cell 3

        cmp x13, #4
        // alternate accum sets to keep two sdot dispatch ports busy
        .inst 0x4f80e1c8 // sdot v8.4s,  v14.16b, v0.4b[0]
        .inst 0x4f80e249 // sdot v9.4s,  v18.16b, v0.4b[0]
        .inst 0x4fa0e1fe // sdot v30.4s, v15.16b, v0.4b[1]
        .inst 0x4fa0e27f // sdot v31.4s, v19.16b, v0.4b[1]
        .inst 0x4f80ea08 // sdot v8.4s,  v16.16b, v0.4b[2]
        .inst 0x4f80ea89 // sdot v9.4s,  v20.16b, v0.4b[2]
        .inst 0x4fa0ea3e // sdot v30.4s, v17.16b, v0.4b[3]
        .inst 0x4fa0eabf // sdot v31.4s, v21.16b, v0.4b[3]
        bge L8LoopSz_TILE_1_lu4

    add v8.4s, v8.4s, v30.4s
    add v9.4s, v9.4s, v31.4s
    cbz x13, L8LoopSzEnd_TILE_1   // exhausted -> skip lu1

    L8LoopSz_TILE_1_lu1:
        ld1 {v4.8b},      [x12], #8                  // 8 bytes main = 32 * 2 bits
        ld1 {v5.s}[0],    [x12], #4                  // 4 bytes aux  = 32 * 1 bit
        add x12, x12, #4                              // skip 4 zero-pad bytes (cellStride=16)
        ld1 {v0.s}[0],    [x11], x22                 // src: 4 IC for 1 batch
        subs x13, x13, #1
        UNPACK_W3_SDOT v4, v5, v3, v12, v6, v7, v28, v29, v10, v11

        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4f80e189 // sdot v9.4s, v12.16b, v0.4b[0]
        bne L8LoopSz_TILE_1_lu1

    L8LoopSzEnd_TILE_1:

    L8Tile1Quan:
    // v2 = preloaded unpack idx (must not be clobbered).
    // v3 / v5 = xKernelSum / inputDequantBias scratch.
    // sdot w3's lu4 inner loop heavily uses v10..v27 (weight + aux + unpack), so we
    // keep the [x10] AccumBuffer running accum here (unlike i8mm w4) and only apply
    // P1.2 scale_eff fusion to shorten the v8/v9 critical path by one fmul.
    ld1 {v0.4s, v1.4s}, [x12], #32   // scale
    ld1 {v4.s}[0], [x24], x25         // input scale (loaded earlier)
    ld1 {v3.s}[0], [x8], x22          // x kernel sum
    ld1 {v24.4s, v25.4s}, [x12], #32 // weight quan zeropoint
    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s
    // P1.2: fuse scale * input_scale off the v8/v9 path.
    fmul v0.4s, v0.4s, v4.s[0]
    fmul v1.4s, v1.4s, v4.s[0]
    fmul v8.4s, v8.4s, v0.4s          // single fmul on v8 critical path
    fmul v9.4s, v9.4s, v1.4s

    TILE1_L8_MLA_TERM:
    MLA_WEIGHTZERO v8, v3, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9, v3, v25, 0 // tile:0, oc:4-7

    cbz x27, TILE1_ADD_DSTV
    ld1 {v5.s}[0], [x27], x25
    ld1 {v24.4s, v25.4s}, [x28], #32
    MLA_WEIGHTZERO v8, v5, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9, v5, v25, 0 // tile:0, oc:4-7

    TILE1_ADD_DSTV:
    cbz x19, TILE1_L8_ACCUM_BUFFER
    ld1 {v10.4s, v11.4s}, [x10]
    fadd v8.4s, v8.4s, v10.4s
    fadd v9.4s, v9.4s, v11.4s

    TILE1_L8_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE1_POST
    st1 {v8.4s, v9.4s}, [x10]
    b TILE1_BLOCKNUM

    TILE1_POST:
    sub x14, x14, #1
    cbz x9, TILE1_CVT_FP16
    ld1 {v10.4s, v11.4s}, [x20], #32
    fadd v8.4s, v8.4s, v10.4s
    fadd v9.4s, v9.4s, v11.4s

    TILE1_CVT_FP16:
    fcvtn v0.4h, v8.4s
    fcvtn2 v0.8h, v9.4s
    cbz x23, TILE1_STORE
    ld1r {v24.8h}, [x23], #2 // f16 min
    ld1r {v25.8h}, [x23] // f16 max
    sub x23, x23, #2
    fmax v0.8h, v24.8h, v0.8h
    fmin v0.8h, v25.8h, v0.8h
    TILE1_STORE:
    st1 {v0.8h}, [x6], x4

    L8Tile1LoopCheck:
    cbz x14, Tile1End
    mov x8, x15
    mov x24, x21 // revert input dequant scale
    cbz x27, L8LoopDz_TILE_1
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x22
    b L8LoopDz_TILE_1
Tile1End:
    subs x7, x7, #1
    beq End

    add x0, x0, #16
    add x24, x21, #4
    add x1, x1, #4
    add x8, x15, #4
    cbz x27, TILE_1
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x22
    REVERT_WEIGHT_KERNEL_SUM x28, x14, x26, x5
    add x27, x27, #4
    b TILE_1

End:
ldp x27, x28, [sp, #(16 * 8)]
ldp x25, x26, [sp, #(16 * 7)]
ldp x23, x24, [sp, #(16 * 6)]
ldp x19, x20, [sp, #(16 * 5)]
ldp x21, x22, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 9)
ret

// ===== W2 sdot unpack constants =====
// Lives in .text (read-only) right after the function so adr-relative load works
// on both Mach-O (macOS) and ELF (Linux/Android) without a separate section.
.align 4
.L_w3_sdot_unpack_consts:
    // === Main extract (same as w2 sdot) ===
    // 16 bytes: tbl idx_main — replicate input bytes 0..3 four times.
    .byte 0, 1, 2, 3, 0, 1, 2, 3
    .byte 0, 1, 2, 3, 0, 1, 2, 3
    // 16 bytes: shifts_main.  -6 / -4 / -2 / 0 per OC group of 4 lanes.
    .byte -6, -6, -6, -6, -4, -4, -4, -4
    .byte -2, -2, -2, -2,  0,  0,  0,  0

    // === Aux extract (bit-plane high bit) ===
    // 16 bytes: tbl idx_aux — replicate aux byte 0 to lanes 0..7, byte 1 to lanes 8..15.
    .byte 0, 0, 0, 0, 0, 0, 0, 0
    .byte 1, 1, 1, 1, 1, 1, 1, 1
    // 16 bytes: shifts_aux.  Byte i (in 0..7) wants its bit i at output bit 2,
    // which is shift = 2 - i.  The byte covers 2 OCs of 4 IC each.
    .byte 2, 1, 0, -1, -2, -3, -4, -5
    .byte 2, 1, 0, -1, -2, -3, -4, -5

#endif // __aarch64__
