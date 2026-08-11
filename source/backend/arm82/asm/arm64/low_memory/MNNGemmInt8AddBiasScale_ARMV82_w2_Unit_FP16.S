//
//  MNNGemmInt8AddBiasScale_ARMV82_w2_Unit_FP16.S
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//
//  ARMv8.2 sdot path, fp16 output, 2-bit quantized weights.
//
//  Per-cell layout (UNIT=8 OC * SRC_UNIT=4 IC = 32 weights = 8 bytes packed):
//    byte i (i in 0..3): OC0..OC3 weights at IC=i
//                        bits[6:7]=OC0, [4:5]=OC1, [2:3]=OC2, [0:1]=OC3
//    byte i (i in 4..7): OC4..OC7 weights at IC=(i-4)
//                        bits[6:7]=OC4, [4:5]=OC5, [2:3]=OC6, [0:1]=OC7
//
//  Per-cell unpack uses a single 16B constant `idx` and a 16B `shifts` table
//  (loaded once at function entry into v_idx, v_shifts), plus a fresh
//  `movi #3` scratch for the per-OC-group AND. See UNPACK_W2_SDOT below.
//

#if defined(__aarch64__)
#include "MNNAsmGlobal.h"

.text
.align 5

// Unpack a single w2 cell: 8 packed bytes in v_w -> v_lo (OC 0-3, 4 IC) + v_hi (OC 4-7, 4 IC).
// idx and shifts are preloaded constants.  v_w is destroyed (used as ext scratch and mask).
// 8 ALU ops + 1 implicit movi for mask3.  Inputs/outputs are byte-vector regs.
.macro UNPACK_W2_SDOT v_w, v_lo, v_hi, v_idx, v_shifts
    tbl  \v_lo\().16b, {\v_w\().16b}, \v_idx\().16b      // bytes 0..3 replicated 4x
    ext  \v_w\().16b, \v_w\().16b, \v_w\().16b, #4         // rotate so bytes 4..7 are at 0..3
    tbl  \v_hi\().16b, {\v_w\().16b}, \v_idx\().16b
    ushl \v_lo\().16b, \v_lo\().16b, \v_shifts\().16b      // -6/-4/-2/0 per OC group of 4 lanes
    ushl \v_hi\().16b, \v_hi\().16b, \v_shifts\().16b
    movi \v_w\().16b, #3                                  // reuse v_w as mask3 (it is dead now)
    and  \v_lo\().16b, \v_lo\().16b, \v_w\().16b
    and  \v_hi\().16b, \v_hi\().16b, \v_w\().16b
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

asm_function MNNGemmInt8AddBiasScale_ARMV82_w2_Unit_FP16
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

// x25 = inputScale post-index stride per block in TILE_1 (4-byte ld1 .s[0]).
//   - no inputBias (x27 == 0): mInputBlockNum == 1, re-read same scale -> stride 0
//   - inputBias set (x27 != 0): mInputBlockNum == blockNum, advance batch*4 -> stride = x22
// (The old `mov x25, #-32 / sub x25, x22, #32` was leftover from a TILE_12 32-byte
//  load that no longer exists; the current TILE_1 path loads only 4 bytes.)
mov x25, #0
cbz x27, _w2_load_consts
mov x25, x22

_w2_load_consts:
    // Load idx [0,1,2,3 x4] into v6, shifts [-6×4,-4×4,-2×4,0×4] into v7.
    // Kept across all TILE paths; the unpack macro reuses v_w as ext scratch
    // and as a fresh `movi #3` mask register.  v6/v7 must be consecutive for
    // the 2-register ld1 form.
    adr x16, .L_w2_sdot_unpack_consts
    ld1 {v6.16b, v7.16b}, [x16]

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

    // lu4 hard-codes `ld1 {v0.4s}, [x11], #16` = 4 IC chunks of *batch 0* only.
    // Src layout is [IC_chunk][batch][SRC_UNIT bytes], so this is correct only
    // when batch=1 (x22=4).  For batch>=2, fall through to the lu1 path which
    // uses `ld1 {v0.s}[0], [x11], x22` and walks IC chunks correctly per batch.
    cmp x22, #4
    bne L8LoopSz_TILE_1_lu1
    cmp x13, #4
    blt L8LoopSz_TILE_1_lu1   // < 4 cells: skip lu4 unrolling

    movi v30.16b, #0          // shadow accum OC 0-3 for ILP
    movi v31.16b, #0          // shadow accum OC 4-7

    L8LoopSz_TILE_1_lu4:
        // 4 cells of w2 weight (4 * 8B = 32B) + 16B src (= 4 IC chunks * 1 batch * 4 bytes)
        ld1 {v10.8b, v11.8b, v12.8b, v13.8b}, [x12], #32
        ld1 {v0.4s}, [x11], #16
        sub x13, x13, #4

        UNPACK_W2_SDOT v10, v14, v18, v6, v7   // cell 0: v14 = OC 0-3, v18 = OC 4-7
        UNPACK_W2_SDOT v11, v15, v19, v6, v7   // cell 1
        UNPACK_W2_SDOT v12, v16, v20, v6, v7   // cell 2
        UNPACK_W2_SDOT v13, v17, v21, v6, v7   // cell 3

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
        ld1 {v4.8b}, [x12], #8                      // 8 bytes = 1 cell of w2 (32 weights)
        ld1 {v0.s}[0], [x11], x22                    // src: 4 IC for 1 batch
        subs x13, x13, #1
        UNPACK_W2_SDOT v4, v3, v12, v6, v7           // v3 = OC 0-3 × 4 IC, v12 = OC 4-7 × 4 IC

        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4f80e189 // sdot v9.4s, v12.16b, v0.4b[0]
        bne L8LoopSz_TILE_1_lu1

    L8LoopSzEnd_TILE_1:

    L8Tile1Quan:
    // v2 = preloaded unpack idx, MUST NOT be clobbered.
    // v3 / v5 = xKernelSum / inputDequantBias scratch.
    // v10 / v11 = per-OC-chunk running fp32 accum.  The [x10] AccumBuffer is shared
    // across OC chunks within one kernel call; the FIRST block of each OC chunk must
    // zero-init v10/v11 explicitly (else we'd inherit the prior OC chunk's tail).
    ld1 {v0.4s, v1.4s}, [x12], #32   // scale
    ld1 {v4.s}[0], [x24], x25         // input scale (loaded earlier for scale_eff fusion)
    ld1 {v3.s}[0], [x8], x22          // x kernel sum
    ld1 {v24.4s, v25.4s}, [x12], #32 // weight quan zeropoint
    cbz x19, _sdot_w2_first_block_zero
    ld1 {v10.4s, v11.4s}, [x10]       // running fp32 accum from prior block
    b _sdot_w2_init_done
_sdot_w2_first_block_zero:
    movi v10.4s, #0
    movi v11.4s, #0
_sdot_w2_init_done:
    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s

    // P1.2 fuse scale * input_scale off the v10/v11 critical path.
    fmul v0.4s, v0.4s, v4.s[0]
    fmul v1.4s, v1.4s, v4.s[0]

    // Per-block dequant chain into v10/v11 directly (replaces fmul v8/v9 + fadd v10/v11).
    fmla v10.4s, v8.4s, v0.4s         // v10 += accum * scale_eff
    fmla v11.4s, v9.4s, v1.4s
    TILE1_L8_MLA_TERM:
    MLA_WEIGHTZERO v10, v3, v24, 0
    MLA_WEIGHTZERO v11, v3, v25, 0

    cbz x27, TILE1_ADD_DSTV
    ld1 {v5.s}[0], [x27], x25
    ld1 {v24.4s, v25.4s}, [x28], #32
    MLA_WEIGHTZERO v10, v5, v24, 0
    MLA_WEIGHTZERO v11, v5, v25, 0

    TILE1_ADD_DSTV:
    // v10/v11 already hold the updated running accum; no fadd needed.

    TILE1_L8_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE1_POST
    st1 {v10.4s, v11.4s}, [x10]
    b TILE1_BLOCKNUM

    TILE1_POST:
    sub x14, x14, #1
    cbz x9, TILE1_CVT_FP16
    ld1 {v24.4s, v25.4s}, [x20], #32  // bias (use v24/v25 since v10/v11 hold result)
    fadd v10.4s, v10.4s, v24.4s
    fadd v11.4s, v11.4s, v25.4s

    TILE1_CVT_FP16:
    fcvtn v0.4h, v10.4s
    fcvtn2 v0.8h, v11.4s
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
.L_w2_sdot_unpack_consts:
    // First 16 bytes: tbl idx — replicate bytes 0..3 four times.
    .byte 0, 1, 2, 3, 0, 1, 2, 3
    .byte 0, 1, 2, 3, 0, 1, 2, 3
    // Next 16 bytes: per-lane signed shifts.  After tbl, lanes 0..3 hold byte 0..3
    // (replicated); shift by -6 brings bits[6:7] (OC0) to LSB.  Lanes 4..7 shift
    // by -4 (OC1), 8..11 by -2 (OC2), 12..15 by 0 (OC3).
    .byte -6, -6, -6, -6, -4, -4, -4, -4
    .byte -2, -2, -2, -2,  0,  0,  0,  0

#endif // __aarch64__
