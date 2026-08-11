//
//  MNNGemmInt8AddBiasScale_ARMV86_w3_Unit_FP16.S
//  MNN
//
//  Created by MNN on 2026/05/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//
//  3-bit weight variant using bit-plane (2 + 1) split storage.
//  Per inner-loop cell: 16 bytes "main plane" (2-bit) + 8 bytes "aux plane" (1-bit) = 24 B.
//  Main plane same as w2 layout (bits[6:7]=oc0,1, [4:5]=oc2,3, [2:3]=oc4,5, [0:1]=oc6,7).
//  Aux plane: byte i (i=0..7) holds 8 weights for IC=i, where bit (7-j) = high bit of OC j.
//
//  Per inner iter: weight pointer x2 advances by 24 (16 main + 8 aux).
//  Unpack: produce 4 NEON regs of 16 int8 each holding final 3-bit values [0, 7]:
//    aux portion = (extracted 1-bit) << 2, then ORR with main 2-bit unpack.
//  Order: do aux unpack first (using v3-v7 as scratch BEFORE src is loaded),
//         then 2-bit unpack with merge, then load src and run smmla.
//

#if defined(__aarch64__)
#include "MNNAsmGlobal.h"

.text
.align 5

.macro SET_0_5 d0, d1, d2, d3, d4
    movi \d0\().16b, #0
    movi \d1\().16b, #0
    movi \d2\().16b, #0
    movi \d3\().16b, #0
    movi \d4\().16b, #0
.endm

.macro SET_0_4 d0, d1, d2, d3
    movi \d0\().16b, #0
    movi \d1\().16b, #0
    movi \d2\().16b, #0
    movi \d3\().16b, #0
.endm

.macro SET_0_2 d0, d1
    movi \d0\().16b, #0
    movi \d1\().16b, #0
.endm

.macro ADD_BIAS_FLOAT d0, d1, d2, d3, z0
    fadd \d0\().4s, \d0\().4s, \z0\().4s
    fadd \d1\().4s, \d1\().4s, \z0\().4s
    fadd \d2\().4s, \d2\().4s, \z0\().4s
    fadd \d3\().4s, \d3\().4s, \z0\().4s
.endm

.macro ADD_FLOAT d0, d1, d2, d3, s0, s1, s2, s3
    fadd \d0\().4s, \d0\().4s, \s0\().4s
    fadd \d1\().4s, \d1\().4s, \s1\().4s
    fadd \d2\().4s, \d2\().4s, \s2\().4s
    fadd \d3\().4s, \d3\().4s, \s3\().4s
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

.macro ReLU_FP16_2 s0, s1, z0, z1 // z0:min z1:max
    fmin \s0\().8h, \s0\().8h, \z1\().8h
    fmin \s1\().8h, \s1\().8h, \z1\().8h
    fmax \s0\().8h, \s0\().8h, \z0\().8h
    fmax \s1\().8h, \s1\().8h, \z0\().8h
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
.macro MUL_INPUT_SCALE s, d0, d1, d2, d3
    fmul \d0\().4s, \d0\().4s, \s\().s[0]
    fmul \d1\().4s, \d1\().4s, \s\().s[1]
    fmul \d2\().4s, \d2\().4s, \s\().s[2]
    fmul \d3\().4s, \d3\().4s, \s\().s[3]
.endm
.macro MLA_WEIGHTZERO d0, s0, s1, idx // idx for xKernelSum
    fmla \d0\().4s, \s1\().4s, \s0\().s[\idx]
.endm
.macro Float32ToHalf s0, s1, s2, s3, d0, d1
    fcvtn \d0\().4h,  \s0\().4s
    fcvtn2 \d0\().8h, \s1\().4s
    fcvtn \d1\().4h,  \s2\().4s
    fcvtn2 \d1\().8h, \s3\().4s
.endm

// ============================================================================
// W3 unpack — IC-major aux version
//
// Aux is duplicated to both halves before entry.  `shifts` starts at {-5,-4}
// for OC0/1 and `idx` is a byte vector of +2, so each following OC pair is
// extracted by adding 2 to the signed shift counts.  The high bit lands at bit2
// and `mask1` is therefore the value 4.
// ============================================================================
.macro W3_UNPACK_SERIAL main, aux_dup, mask1, idx, shifts, mask3, t, \
                       oc01, oc23, oc45, oc67
    // Aux high-bit unpack.  Each pair is independent after its shift vector is ready.
    ushl \oc01\().16b, \aux_dup\().16b, \shifts\().16b
    add  \t\().16b,    \shifts\().16b,  \idx\().16b
    ushl \oc23\().16b, \aux_dup\().16b, \t\().16b
    add  \t\().16b,    \t\().16b,       \idx\().16b
    ushl \oc45\().16b, \aux_dup\().16b, \t\().16b
    add  \t\().16b,    \t\().16b,       \idx\().16b
    ushl \oc67\().16b, \aux_dup\().16b, \t\().16b
    and  \oc01\().16b, \oc01\().16b,    \mask1\().16b
    and  \oc23\().16b, \oc23\().16b,    \mask1\().16b
    and  \oc45\().16b, \oc45\().16b,    \mask1\().16b
    and  \oc67\().16b, \oc67\().16b,    \mask1\().16b

    // Main 2-bit unpack and combine.
    ushr \t\().16b,    \main\().16b, #6
    add  \oc01\().16b, \oc01\().16b, \t\().16b
    ushr \t\().16b,    \main\().16b, #4
    and  \t\().16b,    \t\().16b,    \mask3\().16b
    add  \oc23\().16b, \oc23\().16b, \t\().16b
    ushr \t\().16b,    \main\().16b, #2
    and  \t\().16b,    \t\().16b,    \mask3\().16b
    add  \oc45\().16b, \oc45\().16b, \t\().16b
    and  \t\().16b,    \main\().16b, \mask3\().16b
    add  \oc67\().16b, \oc67\().16b, \t\().16b
.endm

// Keep the wider-register TILE_1 call sites on the same implementation.  The
// old parallel path helped only the OC-major aux layout; IC-major aux already
// has short dependency chains and fewer instructions.
.macro W3_UNPACK_PARALLEL main, aux_dup, aux2, aux4, aux6, mask1, idx, shifts, mask3, t, \
                          oc01, oc23, oc45, oc67
    W3_UNPACK_SERIAL \main, \aux_dup, \mask1, \idx, \shifts, \mask3, \t, \oc01, \oc23, \oc45, \oc67
.endm

.macro W3_UNPACK_PAIR main0, aux0, main1, aux1, mask1, idx, shifts, mask3, t0, t1, \
                      a01, a23, a45, a67, b01, b23, b45, b67
    ushl \a01\().16b, \aux0\().16b, \shifts\().16b
    ushl \b01\().16b, \aux1\().16b, \shifts\().16b
    add  \t0\().16b,  \shifts\().16b, \idx\().16b
    ushl \a23\().16b, \aux0\().16b, \t0\().16b
    ushl \b23\().16b, \aux1\().16b, \t0\().16b
    add  \t0\().16b,  \t0\().16b, \idx\().16b
    ushl \a45\().16b, \aux0\().16b, \t0\().16b
    ushl \b45\().16b, \aux1\().16b, \t0\().16b
    add  \t0\().16b,  \t0\().16b, \idx\().16b
    ushl \a67\().16b, \aux0\().16b, \t0\().16b
    ushl \b67\().16b, \aux1\().16b, \t0\().16b
    and  \a01\().16b, \a01\().16b, \mask1\().16b
    and  \b01\().16b, \b01\().16b, \mask1\().16b
    and  \a23\().16b, \a23\().16b, \mask1\().16b
    and  \b23\().16b, \b23\().16b, \mask1\().16b
    and  \a45\().16b, \a45\().16b, \mask1\().16b
    and  \b45\().16b, \b45\().16b, \mask1\().16b
    and  \a67\().16b, \a67\().16b, \mask1\().16b
    and  \b67\().16b, \b67\().16b, \mask1\().16b

    ushr \t0\().16b,  \main0\().16b, #6
    ushr \t1\().16b,  \main1\().16b, #6
    add  \a01\().16b, \a01\().16b, \t0\().16b
    add  \b01\().16b, \b01\().16b, \t1\().16b
    ushr \t0\().16b,  \main0\().16b, #4
    ushr \t1\().16b,  \main1\().16b, #4
    and  \t0\().16b,  \t0\().16b, \mask3\().16b
    and  \t1\().16b,  \t1\().16b, \mask3\().16b
    add  \a23\().16b, \a23\().16b, \t0\().16b
    add  \b23\().16b, \b23\().16b, \t1\().16b
    ushr \t0\().16b,  \main0\().16b, #2
    ushr \t1\().16b,  \main1\().16b, #2
    and  \t0\().16b,  \t0\().16b, \mask3\().16b
    and  \t1\().16b,  \t1\().16b, \mask3\().16b
    add  \a45\().16b, \a45\().16b, \t0\().16b
    add  \b45\().16b, \b45\().16b, \t1\().16b
    and  \t0\().16b,  \main0\().16b, \mask3\().16b
    and  \t1\().16b,  \main1\().16b, \mask3\().16b
    add  \a67\().16b, \a67\().16b, \t0\().16b
    add  \b67\().16b, \b67\().16b, \t1\().16b
.endm

.macro W3_TILE1_LU2_STEP
    // 2 cells of w3 weight stored as [main0(16) | aux0(8) | main1(16) | aux1(8)]
    ld1 {v5.16b}, [x12], #16              // main plane cell 0
    ld1 {v7.8b},  [x12], #8               // aux plane cell 0
    ld1 {v6.16b}, [x12], #16              // main plane cell 1
    ld1 {v8.8b},  [x12], #8               // aux plane cell 1
    mov v7.d[1], v7.d[0]                  // replicate aux cell 0 to 16 bytes
    mov v8.d[1], v8.d[0]                  // replicate aux cell 1
    ld1 {v2.8b}, [x11], x22               // src tile 0
    ld1 {v3.8b}, [x11], x22               // src tile 1

    // Cell 0/1: unpack into two independent register sets.
    W3_UNPACK_PAIR v5, v7, v6, v8, v15, v13, v14, v30, v21, v28, v9, v10, v11, v12, v0, v1, v4, v20

    // Cell 0: smmla into set 0
    .inst 0x4e89a450 // smmla v16.4s, v2.16b, v9.16b
    .inst 0x4e8aa451 // smmla v17.4s, v2.16b, v10.16b
    .inst 0x4e8ba452 // smmla v18.4s, v2.16b, v11.16b
    .inst 0x4e8ca453 // smmla v19.4s, v2.16b, v12.16b

    // Cell 1: smmla into set 1
    .inst 0x4e80a478 // smmla v24.4s, v3.16b, v0.16b
    .inst 0x4e81a479 // smmla v25.4s, v3.16b, v1.16b
    .inst 0x4e84a47a // smmla v26.4s, v3.16b, v4.16b
    .inst 0x4e94a47b // smmla v27.4s, v3.16b, v20.16b
.endm

.macro REVERT_INPUT_DEQUANT_BIAS rg0, rg1, rg2, rg3
mul \rg1, \rg2, \rg3
sub \rg0, \rg0, \rg1
.endm

.macro REVERT_WEIGHT_KERNEL_SUM rg0, rg1, rg2, rg3
mul \rg1, \rg2, \rg3 // blocknum * ocUp8 * sizeof(float)
sub \rg0, \rg0, \rg1, LSL #5 // revert weight kernel sum
.endm
asm_function MNNGemmInt8AddBiasScale_ARMV86_w3_Unit_FP16
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
    float* weightQuanBias;
    float* fp32minmax;
    ssize_t blockNum = 1;
    const int32_t* bias;
    const float* extraScale = nullptr;
};
*/
//void MNNGemmInt8AddBiasScale_ARMV86_w3_Unit_FP16(int8_t* dst, const int8_t* src,
//    const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
// const QuanPostTreatParameters* parameters, size_t realDstCount);

//Auto: x0:dst, x1:src, x2:weight, x3:src_depth_quad, x4:dst_step
//x5:dst_depth_quad, x6: parameters, x7: realDstCount

//Load from x6: x9: biasFloat, x8: srcKernelSum, x28: weightQuanBias, x14: fp32minmax
/* For FP16
UNIT = 8;
SRC_UNIT = 8;
DST_XUNIT = 10;
 */
ldr x9, [x6, #8]

stp d14, d15, [sp, #(-16 * 10)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]
stp x23, x24, [sp, #(16 * 6)]
stp x25, x26, [sp, #(16 * 7)]
stp x27, x28, [sp, #(16 * 8)]
ldr x8, [x6, #40] // srcKernelSum
ldr x28, [x6, #48] // weightKernelSum
ldr x14, [x6, #56]  // fp32minmax
ldr x26, [x6, #64]  // blockNum
lsl x22, x7, #3 // eDest * GEMM_INT8_SRC_UNIT
ldr x23, [x6, #80]  // input scale
ldr x27, [x6, #88]  // input bias
ldr x10, [x6, #96]  // accumbuffer

mov x15, x8      // input kernel sum
mov x21, x23     // input dequant scale
mov x24, #-32       // inputBlockNum=1
// Load address of W3 unpack constants once. x16 stays valid throughout the function.
adr x16, .L_w3_unpack_consts
cbz x27, TILE_10
mov x24, #8         // inputBlockNum>1

TILE_10:
    cmp x7, #10
    blt TILE_8

mov x20, x9      // bias
mov x6, x27      // input bias
sub x4, x4, #128 // For Tile10
mov x12, #-256
LoopDz_TILE_10:
    mov x11, x1 // src
    mov x19, #0

TILE10_BLOCKNUM:
    mov x13, x3 // src_depth_quad
    movi v2.16b, #3

    SET_0_5 v12, v16, v20, v24, v28 // oc:0,1,0,1
    SET_0_5 v13, v17, v21, v25, v29 // oc:2,3,2,3
    SET_0_5 v14, v18, v22, v26, v30 // oc:4,5,4,5
    SET_0_5 v15, v19, v23, v27, v31 // oc:6,7,6,7

LoopSz_TILE_10:
    ld1 {v0.16b}, [x2], #16                            // main plane: 16B
    ld1 {v1.8b},  [x2], #8                             // aux plane:  8B
    mov v1.d[1], v1.d[0]                                // replicate aux to 16 bytes
    movi v3.16b, #4                                     // aux high-bit mask (will be clobbered by src load)
    ld1 {v4.16b, v5.16b}, [x16]                         // shift step, initial shifts
    W3_UNPACK_SERIAL v0, v1, v3, v4, v5, v2, v6, v8, v9, v10, v11
    ld1 {v3.16b, v4.16b, v5.16b, v6.16b}, [x11], #64    // src: E0-E9
    ld1 {v7.16b}, [x11], #16

    subs x13, x13, #1
    .inst 0x4e88a46c // smmla v12.4s, v3.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a46d // smmla v13.4s, v3.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    .inst 0x4e8aa46e // smmla v14.4s, v3.16b, v10.16b // tile0-oc4, tile0-oc5, tile1-oc4, tile1-oc5
    .inst 0x4e8ba46f // smmla v15.4s, v3.16b, v11.16b // tile0-oc6, tile0-oc7, tile1-oc6, tile1-oc7
    
    .inst 0x4e88a490 // smmla v16.4s, v4.16b, v8.16b // tile2-oc0, tile2-oc1, tile3-oc0, tile3-oc1
    .inst 0x4e89a491 // smmla v17.4s, v4.16b, v9.16b // tile2-oc2, tile2-oc3, tile3-oc2, tile3-oc3
    .inst 0x4e8aa492 // smmla v18.4s, v4.16b, v10.16b // tile2-oc4, tile2-oc5, tile3-oc4, tile3-oc5
    .inst 0x4e8ba493 // smmla v19.4s, v4.16b, v11.16b // tile2-oc6, tile2-oc7, tile3-oc6, tile3-oc7
    
    .inst 0x4e88a4b4 // smmla v20.4s, v5.16b, v8.16b // tile4-oc0, tile4-oc1, tile5-oc0, tile5-oc1
    .inst 0x4e89a4b5 // smmla v21.4s, v5.16b, v9.16b // tile4-oc2, tile4-oc3, tile5-oc2, tile5-oc3
    .inst 0x4e8aa4b6 // smmla v22.4s, v5.16b, v10.16b // tile4-oc4, tile4-oc5, tile5-oc4, tile5-oc5
    .inst 0x4e8ba4b7 // smmla v23.4s, v5.16b, v11.16b // tile4-oc6, tile4-oc7, tile5-oc6, tile5-oc7

    .inst 0x4e88a4d8 // smmla v24.4s, v6.16b, v8.16b // tile6-oc0, tile6-oc1, tile7-oc0, tile7-oc1
    .inst 0x4e89a4d9 // smmla v25.4s, v6.16b, v9.16b // tile6-oc2, tile6-oc3, tile7-oc2, tile7-oc3
    .inst 0x4e8aa4da // smmla v26.4s, v6.16b, v10.16b // tile6-oc4, tile6-oc5, tile7-oc4, tile7-oc5
    .inst 0x4e8ba4db // smmla v27.4s, v6.16b, v11.16b // tile6-oc6, tile6-oc7, tile7-oc6, tile7-oc7

    .inst 0x4e88a4fc // smmla v28.4s, v7.16b, v8.16b // tile8-oc0, tile8-oc1, tile9-oc0, tile9-oc1
    .inst 0x4e89a4fd // smmla v29.4s, v7.16b, v9.16b // tile8-oc2, tile8-oc3, tile9-oc2, tile9-oc3
    .inst 0x4e8aa4fe // smmla v30.4s, v7.16b, v10.16b // tile8-oc4, tile8-oc5, tile9-oc4, tile9-oc5
    .inst 0x4e8ba4ff // smmla v31.4s, v7.16b, v11.16b // tile8-oc6, tile8-oc7, tile9-oc6, tile9-oc7
    bne LoopSz_TILE_10
LoopSzEnd_TILE_10:
    // transpose
    uzp1 v0.2d, v12.2d, v13.2d // E0: oc:0-3
    uzp2 v1.2d, v12.2d, v13.2d // E1: oc:0-3
    uzp1 v2.2d, v14.2d, v15.2d // E0: oc:4-7
    uzp2 v3.2d, v14.2d, v15.2d // E1: oc:4-7

    uzp1 v4.2d, v16.2d, v17.2d
    uzp2 v5.2d, v16.2d, v17.2d
    uzp1 v6.2d, v18.2d, v19.2d
    uzp2 v7.2d, v18.2d, v19.2d

    uzp1 v8.2d, v20.2d, v21.2d
    uzp2 v9.2d, v20.2d, v21.2d
    uzp1 v10.2d, v22.2d, v23.2d
    uzp2 v11.2d, v22.2d, v23.2d

    uzp1 v12.2d, v24.2d, v25.2d
    uzp2 v13.2d, v24.2d, v25.2d
    uzp1 v14.2d, v26.2d, v27.2d
    uzp2 v15.2d, v26.2d, v27.2d

    uzp1 v16.2d, v28.2d, v29.2d
    uzp2 v17.2d, v28.2d, v29.2d
    uzp1 v18.2d, v30.2d, v31.2d
    uzp2 v19.2d, v30.2d, v31.2d
    Int32ToFloat v0, v1, v2, v3
    Int32ToFloat v4, v5, v6, v7
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    Int32ToFloat v16, v17, v18, v19

Tile10Quan:
    ld1 {v20.4s, v21.4s}, [x2], #32  // scale
    ld1 {v22.4s, v23.4s}, [x8], #32 // x kernel sum
    ld1 {v24.d}[0], [x8], #8
    ld1 {v25.4s, v26.4s}, [x2], #32 // weight quan zeropoint
    MUL_SCALE v20, v0, v1, v4, v5
    MUL_SCALE v21, v2, v3, v6, v7
    MUL_SCALE v20, v8, v9, v12, v13
    MUL_SCALE v21, v10, v11, v14, v15
    fmul v16.4s, v16.4s, v20.4s
    fmul v17.4s, v17.4s, v20.4s
    fmul v18.4s, v18.4s, v21.4s
    fmul v19.4s, v19.4s, v21.4s

    ld1 {v27.4s, v28.4s}, [x23], #32
    ld1 {v29.d}[0], [x23], x24
    MUL_INPUT_SCALE v27, v0, v1, v4, v5
    MUL_INPUT_SCALE v28, v8, v9, v12, v13
    MUL_INPUT_SCALE v27, v2, v3, v6, v7
    MUL_INPUT_SCALE v28, v10, v11, v14, v15
    fmul v16.4s, v16.4s, v29.s[0]
    fmul v17.4s, v17.4s, v29.s[1]
    fmul v18.4s, v18.4s, v29.s[0]
    fmul v19.4s, v19.4s, v29.s[1]

    TILE10_MLA:
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v26, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v3, v22, v26, 1 // tile:1, oc:4-7

    MLA_WEIGHTZERO v4, v22, v25, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v5, v22, v25, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v6, v22, v26, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v7, v22, v26, 3 // tile:3, oc:4-7

    MLA_WEIGHTZERO v8,  v23, v25, 0 // tile:4, oc:0-3
    MLA_WEIGHTZERO v9,  v23, v25, 1 // tile:5, oc:0-3
    MLA_WEIGHTZERO v10, v23, v26, 0 // tile:4, oc:4-7
    MLA_WEIGHTZERO v11, v23, v26, 1 // tile:5, oc:4-7

    MLA_WEIGHTZERO v12, v23, v25, 2 // tile:6, oc:0-3
    MLA_WEIGHTZERO v13, v23, v25, 3 // tile:7, oc:0-3
    MLA_WEIGHTZERO v14, v23, v26, 2 // tile:6, oc:4-7
    MLA_WEIGHTZERO v15, v23, v26, 3 // tile:7, oc:4-7

    MLA_WEIGHTZERO v16, v24, v25, 0 // tile:8, oc:0-3
    MLA_WEIGHTZERO v17, v24, v25, 1 // tile:9, oc:0-3
    MLA_WEIGHTZERO v18, v24, v26, 0 // tile:8, oc:4-7
    MLA_WEIGHTZERO v19, v24, v26, 1 // tile:9, oc:4-7
    
    cbz x6, TILE10_ADD_DSTV
    ld1 {v20.4s, v21.4s}, [x27], #32 // input dequant bias
    ld1 {v22.2s}, [x27], #8
    ld1 {v23.4s, v24.4s}, [x28], #32 // weight kernel sum
    MLA_WEIGHTZERO v0, v20, v23, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v20, v23, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v20, v24, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v3, v20, v24, 1 // tile:1, oc:4-7

    MLA_WEIGHTZERO v4, v20, v23, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v5, v20, v23, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v6, v20, v24, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v7, v20, v24, 3 // tile:3, oc:4-7

    MLA_WEIGHTZERO v8,  v21, v23, 0 // tile:4, oc:0-3
    MLA_WEIGHTZERO v9,  v21, v23, 1 // tile:5, oc:0-3
    MLA_WEIGHTZERO v10, v21, v24, 0 // tile:4, oc:4-7
    MLA_WEIGHTZERO v11, v21, v24, 1 // tile:5, oc:4-7

    MLA_WEIGHTZERO v12, v21, v23, 2 // tile:6, oc:0-3
    MLA_WEIGHTZERO v13, v21, v23, 3 // tile:7, oc:0-3
    MLA_WEIGHTZERO v14, v21, v24, 2 // tile:6, oc:4-7
    MLA_WEIGHTZERO v15, v21, v24, 3 // tile:7, oc:4-7

    MLA_WEIGHTZERO v16, v22, v23, 0 // tile:8, oc:0-3
    MLA_WEIGHTZERO v17, v22, v23, 1 // tile:9, oc:0-3
    MLA_WEIGHTZERO v18, v22, v24, 0 // tile:8, oc:4-7
    MLA_WEIGHTZERO v19, v22, v24, 1 // tile:9, oc:4-7


    TILE10_ADD_DSTV:
    cbz x19, TILE10_TEMP_BUFFER
    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], #64
    ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10], #64
    ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x10], #64
    ADD_FLOAT v0, v1, v2, v3, v20, v21, v22, v23
    ADD_FLOAT v4, v5, v6, v7, v24, v25, v26, v27
    ADD_FLOAT v8, v9, v10, v11, v28, v29, v30, v31

    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], #64
    ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10], x12
    ADD_FLOAT v12, v13, v14, v15, v20, v21, v22, v23
    ADD_FLOAT v16, v17, v18, v19, v24, v25, v26, v27

    TILE10_TEMP_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE10_POST
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10], #64
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10], #64
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10], x12
    b TILE10_BLOCKNUM

    TILE10_POST:
    cbz x9, TILE10_CVT_FP16
    ld1 {v20.4s, v21.4s}, [x9], #32  // bias
    ADD_BIAS_FLOAT v0, v1, v4, v5, v20
    ADD_BIAS_FLOAT v2, v3, v6, v7, v21
    ADD_BIAS_FLOAT v8, v9, v12, v13, v20
    ADD_BIAS_FLOAT v10, v11, v14, v15, v21
    fadd v16.4s, v16.4s, v20.4s
    fadd v17.4s, v17.4s, v20.4s
    fadd v18.4s, v18.4s, v21.4s
    fadd v19.4s, v19.4s, v21.4s

    TILE10_CVT_FP16:
    // float32->float16
    Float32ToHalf v0, v2, v1, v3, v20, v21
    Float32ToHalf v4, v6, v5, v7, v22, v23
    Float32ToHalf v8, v10, v9, v11, v24, v25
    Float32ToHalf v12, v14, v13, v15, v26, v27
    Float32ToHalf v16, v18, v17, v19, v30, v31

    cbz x14, TILE10_STORE
    ld1r {v29.8h}, [x14], #2 // f32 min
    ld1r {v28.8h}, [x14] // f32 max
    sub x14, x14, #2

    ReLU_FP16 v20, v21, v22, v23, v29, v28
    ReLU_FP16 v24, v25, v26, v27, v29, v28
    ReLU_FP16_2 v30, v31, v29, v28

    TILE10_STORE:
    
    st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64
    st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x0], #64
    st1 {v30.8h, v31.8h}, [x0], x4

Tile10LoopCheck:
    subs x5, x5, #1  // dz--
    mov x8, x15 // revert input kernel sum
    mov x27, x6 // revert input dequant bias
    mov x23, x21 // revert input dequant scale
    bne LoopDz_TILE_10
    b End

TILE_8:
    mov x25, #0
    cbz x27, RELU_VECTOR
    lsr x25, x22, #1         // input block quant: realDstCount * sizeof(float)
    RELU_VECTOR:
    cbz x14, TILE_Remain
    ld1r {v29.8h}, [x14], #2 // f32 min
    ld1r {v31.8h}, [x14] // f32 max
    sub x14, x14, #2
    TILE_Remain:
    movi v30.16b, #3
    cmp x7, #8
    blt TILE_4
    sub x4, x4, #64 // just for Tile8, revert it when Tile8end
    mov x24, x5 // dst_depth_quad
    mov x6, x0 // dst
    mov x12, x2 // weight
    mov x20, x9 // bias

LoopDz_TILE_8:
    mov x11, x1 // src
    mov x19, #0

TILE8_BLOCKNUM:
    mov x13, x3 // src_depth_quad

    SET_0_4 v12, v16, v20, v24 // oc:0,1,0,1
    SET_0_4 v13, v17, v21, v25 // oc:2,3,2,3
    SET_0_4 v14, v18, v22, v26 // oc:4,5,4,5
    SET_0_4 v15, v19, v23, v27 // oc:6,7,6,7
LoopSz_TILE_8:
    ld1 {v0.16b}, [x12], #16                            // main plane: 16B
    ld1 {v1.8b},  [x12], #8                             // aux plane:  8B
    mov v1.d[1], v1.d[0]                                // replicate aux to 16 bytes
    movi v7.16b, #4                                     // aux high-bit mask (TILE_8 doesn't use v7 for src)
    ld1 {v3.16b, v4.16b}, [x16]                         // shift step, initial shifts
    W3_UNPACK_SERIAL v0, v1, v7, v3, v4, v30, v2, v8, v9, v10, v11
    ld1 {v3.16b, v4.16b, v5.16b, v6.16b}, [x11], x22    // src: E0-E7
    subs x13, x13, #1

    .inst 0x4e88a46c // smmla v12.4s, v3.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a46d // smmla v13.4s, v3.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    .inst 0x4e8aa46e // smmla v14.4s, v3.16b, v10.16b // tile0-oc4, tile0-oc5, tile1-oc4, tile1-oc5
    .inst 0x4e8ba46f // smmla v15.4s, v3.16b, v11.16b // tile0-oc6, tile0-oc7, tile1-oc6, tile1-oc7
    
    .inst 0x4e88a490 // smmla v16.4s, v4.16b, v8.16b // tile2-oc0, tile2-oc1, tile3-oc0, tile3-oc1
    .inst 0x4e89a491 // smmla v17.4s, v4.16b, v9.16b // tile2-oc2, tile2-oc3, tile3-oc2, tile3-oc3
    .inst 0x4e8aa492 // smmla v18.4s, v4.16b, v10.16b // tile2-oc4, tile2-oc5, tile3-oc4, tile3-oc5
    .inst 0x4e8ba493 // smmla v19.4s, v4.16b, v11.16b // tile2-oc6, tile2-oc7, tile3-oc6, tile3-oc7
    
    .inst 0x4e88a4b4 // smmla v20.4s, v5.16b, v8.16b // tile4-oc0, tile4-oc1, tile5-oc0, tile5-oc1
    .inst 0x4e89a4b5 // smmla v21.4s, v5.16b, v9.16b // tile4-oc2, tile4-oc3, tile5-oc2, tile5-oc3
    .inst 0x4e8aa4b6 // smmla v22.4s, v5.16b, v10.16b // tile4-oc4, tile4-oc5, tile5-oc4, tile5-oc5
    .inst 0x4e8ba4b7 // smmla v23.4s, v5.16b, v11.16b // tile4-oc6, tile4-oc7, tile5-oc6, tile5-oc7

    .inst 0x4e88a4d8 // smmla v24.4s, v6.16b, v8.16b // tile6-oc0, tile6-oc1, tile7-oc0, tile7-oc1
    .inst 0x4e89a4d9 // smmla v25.4s, v6.16b, v9.16b // tile6-oc2, tile6-oc3, tile7-oc2, tile7-oc3
    .inst 0x4e8aa4da // smmla v26.4s, v6.16b, v10.16b // tile6-oc4, tile6-oc5, tile7-oc4, tile7-oc5
    .inst 0x4e8ba4db // smmla v27.4s, v6.16b, v11.16b // tile6-oc6, tile6-oc7, tile7-oc6, tile7-oc7
    bne LoopSz_TILE_8

LoopSzEnd_TILE_8:
    uzp1 v0.2d, v12.2d, v13.2d // E0: oc:0-3
    uzp2 v1.2d, v12.2d, v13.2d // E1: oc:0-3
    uzp1 v2.2d, v14.2d, v15.2d // E0: oc:4-7
    uzp2 v3.2d, v14.2d, v15.2d // E1: oc:4-7

    uzp1 v4.2d, v16.2d, v17.2d
    uzp2 v5.2d, v16.2d, v17.2d
    uzp1 v6.2d, v18.2d, v19.2d
    uzp2 v7.2d, v18.2d, v19.2d

    uzp1 v8.2d, v20.2d, v21.2d
    uzp2 v9.2d, v20.2d, v21.2d
    uzp1 v10.2d, v22.2d, v23.2d
    uzp2 v11.2d, v22.2d, v23.2d

    uzp1 v12.2d, v24.2d, v25.2d
    uzp2 v13.2d, v24.2d, v25.2d
    uzp1 v14.2d, v26.2d, v27.2d
    uzp2 v15.2d, v26.2d, v27.2d
    Int32ToFloat v0, v1, v2, v3
    Int32ToFloat v4, v5, v6, v7
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15

Tile8Quan:
    ld1 {v20.4s, v21.4s}, [x12], #32  // scale
    ld1 {v22.4s, v23.4s}, [x8] // x kernel sum
    ld1 {v25.4s, v26.4s}, [x12], #32 // weight quan zeropoint
    MUL_SCALE v20, v0, v1, v4, v5
    MUL_SCALE v21, v2, v3, v6, v7
    MUL_SCALE v20, v8, v9, v12, v13
    MUL_SCALE v21, v10, v11, v14, v15
    add x8, x8, x22, LSR #1

    ld1 {v27.4s, v28.4s}, [x23], x25 // input dequant scale
    MUL_INPUT_SCALE v27, v0, v1, v4, v5
    MUL_INPUT_SCALE v28, v8, v9, v12, v13
    MUL_INPUT_SCALE v27, v2, v3, v6, v7
    MUL_INPUT_SCALE v28, v10, v11, v14, v15

    TILE8_MLA:
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v26, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v3, v22, v26, 1 // tile:1, oc:4-7

    MLA_WEIGHTZERO v4, v22, v25, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v5, v22, v25, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v6, v22, v26, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v7, v22, v26, 3 // tile:3, oc:4-7

    MLA_WEIGHTZERO v8,  v23, v25, 0 // tile:4, oc:0-3
    MLA_WEIGHTZERO v9,  v23, v25, 1 // tile:5, oc:0-3
    MLA_WEIGHTZERO v10, v23, v26, 0 // tile:4, oc:4-7
    MLA_WEIGHTZERO v11, v23, v26, 1 // tile:5, oc:4-7

    MLA_WEIGHTZERO v12, v23, v25, 2 // tile:6, oc:0-3
    MLA_WEIGHTZERO v13, v23, v25, 3 // tile:7, oc:0-3
    MLA_WEIGHTZERO v14, v23, v26, 2 // tile:6, oc:4-7
    MLA_WEIGHTZERO v15, v23, v26, 3 // tile:7, oc:4-7

    cbz x27, TILE8_ADD_DSTV
    ld1 {v22.4s, v23.4s}, [x27], x25 // input dequant bias
    ld1 {v25.4s, v26.4s}, [x28], #32 // weight kernel sum
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v26, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v3, v22, v26, 1 // tile:1, oc:4-7

    MLA_WEIGHTZERO v4, v22, v25, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v5, v22, v25, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v6, v22, v26, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v7, v22, v26, 3 // tile:3, oc:4-7

    MLA_WEIGHTZERO v8,  v23, v25, 0 // tile:4, oc:0-3
    MLA_WEIGHTZERO v9,  v23, v25, 1 // tile:5, oc:0-3
    MLA_WEIGHTZERO v10, v23, v26, 0 // tile:4, oc:4-7
    MLA_WEIGHTZERO v11, v23, v26, 1 // tile:5, oc:4-7

    MLA_WEIGHTZERO v12, v23, v25, 2 // tile:6, oc:0-3
    MLA_WEIGHTZERO v13, v23, v25, 3 // tile:7, oc:0-3
    MLA_WEIGHTZERO v14, v23, v26, 2 // tile:6, oc:4-7
    MLA_WEIGHTZERO v15, v23, v26, 3 // tile:7, oc:4-7
    
    TILE8_ADD_DSTV:
    cbz x19, TILE8_TEMP_BUFFER
    ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10], #64
    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], #64
    ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10], #64
    ADD_FLOAT v0, v1, v2, v3, v16, v17, v18, v19
    ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10]
    ADD_FLOAT v4, v5, v6, v7, v20, v21, v22, v23
    ADD_FLOAT v8, v9, v10, v11, v24, v25, v26, v27
    ADD_FLOAT v12, v13, v14, v15, v16, v17, v18, v19
    sub x10, x10, #192

    TILE8_TEMP_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE8_POST
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10], #64
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10]
    sub x10, x10, #192
    b TILE8_BLOCKNUM

    TILE8_POST:
    sub x24, x24, #1
    cbz x9, TILE8_CVT_FP16
    ld1 {v16.4s, v17.4s}, [x20], #32 // bias
    ADD_BIAS_FLOAT v0, v1, v4, v5, v16
    ADD_BIAS_FLOAT v2, v3, v6, v7, v17
    ADD_BIAS_FLOAT v8, v9, v12, v13, v16
    ADD_BIAS_FLOAT v10, v11, v14, v15, v17

    TILE8_CVT_FP16:
    // float32->float16
    Float32ToHalf v0, v2, v1, v3, v20, v21
    Float32ToHalf v4, v6, v5, v7, v22, v23
    Float32ToHalf v8, v10, v9, v11, v24, v25
    Float32ToHalf v12, v14, v13, v15, v26, v27
    cbz x14, TILE8_STORE
    ReLU_FP16 v20, v21, v22, v23, v29, v31
    ReLU_FP16 v24, v25, v26, v27, v29, v31

    TILE8_STORE:
    st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x6], #64
    st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x6], x4

Tile8LoopCheck:
    cbz x24, Tile8End
    mov x8, x15       // revert input kernel sum
    mov x23, x21      // revert input dequant scale
    cbz x27, LoopDz_TILE_8
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    b LoopDz_TILE_8

Tile8End:
    add x0, x0, #128
    sub x7, x7, #8
    add x1, x1, #64
    add x23, x21, #32 // input dequant scale
    add x8, x15, #32  // input kernel sum
    add x4, x4, #64 // Revert x4 for following tile.
    cbz x27, TILE_4
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x24, x26, x5
    add x27, x27, #32 // update input dequant bias: + 8 * sizeof(float)

TILE_4:
    cmp x7, #4
    blt TILE_2
    mov x24, x5 // dst_depth_quad
    mov x6, x0 // dst
    mov x12, x2 // weight
    mov x20, x9 // bias
    mov x15, x8
    mov x21, x23

LoopDz_TILE_4:
    mov x11, x1 // src
    mov x19, #0
TILE4_BLOCKNUM:
    mov x13, x3 // src_depth_quad
    SET_0_4 v12, v13, v14, v15
    SET_0_4 v16, v17, v18, v19

LoopSz_TILE_4:
    ld1 {v2.16b}, [x12], #16          // main plane: 16B
    ld1 {v0.8b},  [x12], #8           // aux plane:  8B
    mov v0.d[1], v0.d[0]               // replicate aux to 16 bytes
    movi v1.16b, #4                    // aux high-bit mask
    ld1 {v6.16b, v7.16b}, [x16]        // shift step, initial shifts
    W3_UNPACK_SERIAL v2, v0, v1, v6, v7, v30, v3, v8, v9, v10, v11
    ld1 {v4.16b, v5.16b}, [x11], x22   // src
    subs x13, x13, #1
    .inst 0x4e88a48c // smmla v12.4s, v4.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a48d // smmla v13.4s, v4.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    .inst 0x4e8aa48e // smmla v14.4s, v4.16b, v10.16b // tile0-oc4, tile0-oc5, tile1-oc4, tile1-oc5
    .inst 0x4e8ba48f // smmla v15.4s, v4.16b, v11.16b // tile0-oc6, tile0-oc7, tile1-oc6, tile1-oc7
    
    .inst 0x4e88a4b0 // smmla v16.4s, v5.16b, v8.16b // tile2-oc0, tile2-oc1, tile3-oc0, tile3-oc1
    .inst 0x4e89a4b1 // smmla v17.4s, v5.16b, v9.16b // tile2-oc2, tile2-oc3, tile3-oc2, tile3-oc3
    .inst 0x4e8aa4b2 // smmla v18.4s, v5.16b, v10.16b // tile2-oc4, tile2-oc5, tile3-oc4, tile3-oc5
    .inst 0x4e8ba4b3 // smmla v19.4s, v5.16b, v11.16b // tile2-oc6, tile2-oc7, tile3-oc6, tile3-oc7
    bne LoopSz_TILE_4
LoopSzEnd_TILE_4:
    uzp1 v0.2d, v12.2d, v13.2d // E0: oc:0-3
    uzp2 v1.2d, v12.2d, v13.2d // E1: oc:0-3
    uzp1 v2.2d, v14.2d, v15.2d // E0: oc:4-7
    uzp2 v3.2d, v14.2d, v15.2d // E1: oc:4-7

    uzp1 v4.2d, v16.2d, v17.2d
    uzp2 v5.2d, v16.2d, v17.2d
    uzp1 v6.2d, v18.2d, v19.2d
    uzp2 v7.2d, v18.2d, v19.2d
    Int32ToFloat v0, v1, v2, v3
    Int32ToFloat v4, v5, v6, v7

Tile4Quan:
    ld1 {v20.4s, v21.4s}, [x12], #32  // scale
    ld1 {v22.4s}, [x8] // x kernel sum
    ld1 {v25.4s, v26.4s}, [x12], #32 // weight quan zeropoint
    MUL_SCALE v20, v0, v1, v4, v5
    MUL_SCALE v21, v2, v3, v6, v7
    add x8, x8, x22, LSR #1

    ld1 {v27.4s}, [x23], x25
    MUL_INPUT_SCALE v27, v0, v1, v4, v5
    MUL_INPUT_SCALE v27, v2, v3, v6, v7

    TILE4_MLA:
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v26, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v3, v22, v26, 1 // tile:1, oc:4-7

    MLA_WEIGHTZERO v4, v22, v25, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v5, v22, v25, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v6, v22, v26, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v7, v22, v26, 3 // tile:3, oc:4-7

    cbz x27, TILE4_ADD_DSTV
    ld1 {v22.4s}, [x27], x25 // input dequant bias
    ld1 {v25.4s, v26.4s}, [x28], #32 // weight kernel sum
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v26, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v3, v22, v26, 1 // tile:1, oc:4-7

    MLA_WEIGHTZERO v4, v22, v25, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v5, v22, v25, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v6, v22, v26, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v7, v22, v26, 3 // tile:3, oc:4-7

    TILE4_ADD_DSTV:
    cbz x19, TILE4_TEMP_BUFFER
    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], #64
    ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10]
    ADD_FLOAT v0, v1, v2, v3, v20, v21, v22, v23
    ADD_FLOAT v4, v5, v6, v7, v24, v25, v26, v27
    sub x10, x10, #64

    TILE4_TEMP_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE4_POST
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10]
    sub x10, x10, #64
    b TILE4_BLOCKNUM

    TILE4_POST:
    sub x24, x24, #1
    cbz x9, TILE4_CVT_FP16
    ld1 {v16.4s, v17.4s}, [x20], #32 // bias
    ADD_BIAS_FLOAT v0, v1, v4, v5, v16
    ADD_BIAS_FLOAT v2, v3, v6, v7, v17

    TILE4_CVT_FP16:
     // float32->float16
    Float32ToHalf v0, v2, v1, v3, v20, v21
    Float32ToHalf v4, v6, v5, v7, v22, v23
    cbz x14, TILE4_STORE
    ReLU_FP16 v20, v21, v22, v23, v29, v31

    TILE4_STORE:
    st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x6], x4

Tile4LoopCheck:
    cbz x24, Tile4End
    mov x8, x15
    mov x23, x21
    cbz x27, LoopDz_TILE_4
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    b LoopDz_TILE_4

Tile4End:
    add x0, x0, #64
    sub x7, x7, #4
    add x1, x1, #32
    add x8, x15, #16
    add x23, x21, #16
    cbz x27, TILE_2
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x24, x26, x5
    add x27, x27, #16

TILE_2:
    cmp x7, #2
    blt TILE_1
    mov x24, x5 // dst_depth_quad
    mov x6, x0 // dst
    mov x12, x2 // weight
    mov x20, x9 // bias
    mov x15, x8
    mov x21, x23

LoopDz_TILE_2:
    mov x11, x1 // src
    mov x19, #0
TILE2_BLOCKNUM:
    mov x13, x3 // src_depth_quad
    SET_0_4 v12, v13, v14, v15
LoopSz_TILE_2:
    ld1 {v2.16b}, [x12], #16           // main plane: 16B
    ld1 {v0.8b},  [x12], #8            // aux plane:  8B
    mov v0.d[1], v0.d[0]                // replicate aux to 16 bytes
    movi v1.16b, #4                     // aux high-bit mask
    ld1 {v5.16b, v6.16b}, [x16]         // shift step, initial shifts
    W3_UNPACK_SERIAL v2, v0, v1, v5, v6, v30, v3, v8, v9, v10, v11
    ld1 {v4.16b}, [x11], x22            // src

    .inst 0x4e88a48c // smmla v12.4s, v4.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a48d // smmla v13.4s, v4.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    .inst 0x4e8aa48e // smmla v14.4s, v4.16b, v10.16b // tile0-oc4, tile0-oc5, tile1-oc4, tile1-oc5
    .inst 0x4e8ba48f // smmla v15.4s, v4.16b, v11.16b // tile0-oc6, tile0-oc7, tile1-oc6, tile1-oc7
    subs x13, x13, #1
    bne LoopSz_TILE_2

    uzp1 v0.2d, v12.2d, v13.2d // E0: oc:0-3
    uzp2 v1.2d, v12.2d, v13.2d // E1: oc:0-3
    uzp1 v2.2d, v14.2d, v15.2d // E0: oc:4-7
    uzp2 v3.2d, v14.2d, v15.2d // E1: oc:4-7
    Int32ToFloat v0, v1, v2, v3

Tile2Quan:
    ld1 {v20.4s, v21.4s}, [x12], #32  // scale
    ld1 {v22.d}[0], [x8] // x kernel sum
    ld1 {v25.4s, v26.4s}, [x12], #32 // weight quan zeropoint
    fmul v0.4s, v0.4s, v20.4s
    fmul v1.4s, v1.4s, v20.4s
    fmul v2.4s, v2.4s, v21.4s
    fmul v3.4s, v3.4s, v21.4s
    add x8, x8, x22, LSR #1

    ld1 {v27.d}[0], [x23], x25
    fmul v0.4s, v0.4s, v27.s[0]
    fmul v1.4s, v1.4s, v27.s[1]
    fmul v2.4s, v2.4s, v27.s[0]
    fmul v3.4s, v3.4s, v27.s[1]

    TILE2_MLA:
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v26, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v3, v22, v26, 1 // tile:1, oc:4-7

    cbz x27, TILE2_ADD_DSTV
    ld1 {v22.2s}, [x27], x25 // input dequant bias
    ld1 {v25.4s, v26.4s}, [x28], #32 // weight kernel sum
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v26, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v3, v22, v26, 1 // tile:1, oc:4-7

    TILE2_ADD_DSTV:
    cbz x19, TILE2_TEMP_BUFFER
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10]
    ADD_FLOAT v0, v1, v2, v3, v4, v5, v6, v7

    TILE2_TEMP_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE2_POST
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10]
    b TILE2_BLOCKNUM

    TILE2_POST:
    sub x24, x24, #1
    cbz x9, TILE2_CVT_FP16
    ld1 {v16.4s, v17.4s}, [x20], #32 // bias
    fadd v0.4s, v0.4s, v16.4s
    fadd v1.4s, v1.4s, v16.4s
    fadd v2.4s, v2.4s, v17.4s
    fadd v3.4s, v3.4s, v17.4s

    // float32->float16
    TILE2_CVT_FP16:
    Float32ToHalf v0, v2, v1, v3, v20, v21
    cbz x14, TILE2_STORE
    fmax v20.8h, v20.8h, v29.8h
    fmax v21.8h, v21.8h, v29.8h
    fmin v20.8h, v20.8h, v31.8h
    fmin v21.8h, v21.8h, v31.8h

    TILE2_STORE:
    st1 {v20.8h, v21.8h}, [x6], x4

Tile2LoopCheck:
    cbz x24, Tile2End
    mov x8, x15
    mov x23, x21
    cbz x27, LoopDz_TILE_2
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    b LoopDz_TILE_2
Tile2End:
    add x0, x0, #32
    sub x7, x7, #2
    add x1, x1, #16
    add x8, x15, #8
    add x23, x21, #8
    cbz x27, TILE_1
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x24, x26, x5
    add x27, x27, #8

TILE_1:
    cmp x7, #1
    blt End
    mov x24, x5 // dst_depth_quad
    mov x6, x0 // dst
    mov x12, x2 // weight
    mov x20, x9 // bias
    mov x15, x8
    mov x21, x23

LoopDz_TILE_1:
    mov x11, x1 // src
    mov x19, #0
    // running fp32 accumulator across blocks within this OC chunk
    movi v22.4s, #0 // oc:0-3 running accum
    movi v23.4s, #0 // oc:4-7 running accum
    // W3 unpack constants stay live across all blockNum iterations in TILE_1.
    movi v15.16b, #4                     // aux high-bit mask
    ld1 {v13.16b, v14.16b}, [x16]        // v13 = shift step, v14 = initial shifts
TILE1_BLOCKNUM:
    mov x13, x3 // src_depth_quad

    movi v16.4s, #0 // oc:0,1,0,1 (set 0)
    movi v17.4s, #0 // oc:2,3,2,3
    movi v18.4s, #0 // oc:4,5,4,5
    movi v19.4s, #0 // oc:6,7,6,7

    cmp x13, #2
    blt LoopSz1_TILE_1_lu1

    movi v24.4s, #0                      // shadow accumulators for lu2
    movi v25.4s, #0
    movi v26.4s, #0
    movi v27.4s, #0

    cmp x13, #8
    beq LoopSz1_TILE_1_b64

LoopSz1_TILE_1_lu2:
    W3_TILE1_LU2_STEP

    sub x13, x13, #2
    cmp x13, #2
    bge LoopSz1_TILE_1_lu2

    b LoopSz1_TILE_1_lu2_merge

LoopSz1_TILE_1_b64:
    W3_TILE1_LU2_STEP
    W3_TILE1_LU2_STEP
    W3_TILE1_LU2_STEP
    W3_TILE1_LU2_STEP
    mov x13, #0

LoopSz1_TILE_1_lu2_merge:
    add v16.4s, v16.4s, v24.4s
    add v17.4s, v17.4s, v25.4s
    add v18.4s, v18.4s, v26.4s
    add v19.4s, v19.4s, v27.4s

    cbz x13, LoopSzEnd_TILE_1

LoopSz1_TILE_1_lu1:
    // v13/v14/v15 hold W3 unpack constants; load main/aux into other regs.
    ld1 {v5.16b}, [x12], #16             // main plane: 16B
    ld1 {v6.8b},  [x12], #8              // aux plane:  8B
    mov v6.d[1], v6.d[0]                  // replicate aux to 16 bytes
    // TILE_1 lu1 keeps the same call shape as the old parallel macro; IC-major aux ignores aux scratch regs.
    // t=v7, outputs=v0,v1,v3,v4 (matching smmla operands)
    W3_UNPACK_PARALLEL v5, v6, v8, v9, v10, v15, v13, v14, v30, v7, v0, v1, v3, v4
    ld1 {v2.8b}, [x11], x22              // src
    subs x13, x13, #1

    .inst 0x4e80a450 // smmla v16.4s, v2.16b, v0.16b
    .inst 0x4e81a451 // smmla v17.4s, v2.16b, v1.16b
    .inst 0x4e83a452 // smmla v18.4s, v2.16b, v3.16b
    .inst 0x4e84a453 // smmla v19.4s, v2.16b, v4.16b

    bne LoopSz1_TILE_1_lu1

LoopSzEnd_TILE_1:
    uzp1 v25.2d, v16.2d, v17.2d
    uzp1 v26.2d, v18.2d, v19.2d
    scvtf v25.4s, v25.4s
    scvtf v26.4s, v26.4s

Tile1Quan:
    ld1 {v0.4s, v1.4s}, [x12], #32  // scale
    ld1 {v4.s}[0], [x23], x25       // input scale
    ld1 {v6.s}[0], [x8]             // x kernel sum
    ld1 {v8.4s, v9.4s}, [x12], #32  // weight quan zeropoint
    add x8, x8, x22, LSR #1

    fmul v0.4s, v0.4s, v4.s[0]      // scale_eff
    fmul v1.4s, v1.4s, v4.s[0]
    fmla v22.4s, v25.4s, v0.4s
    fmla v23.4s, v26.4s, v1.4s
    TILE1_MLA:
    MLA_WEIGHTZERO v22, v6, v8, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v23, v6, v9, 0 // tile:0, oc:4-7

    cbz x27, TILE1_ADD_DSTV
    ld1 {v6.s}[0], [x27], x25 // input dequant bias
    ld1 {v8.4s, v9.4s}, [x28], #32 // weight kernel sum
    MLA_WEIGHTZERO v22, v6, v8, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v23, v6, v9, 0 // tile:0, oc:4-7

    TILE1_ADD_DSTV:
    // v22/v23 already hold the running per-OC-chunk accumulator.

    TILE1_TEMP_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    bne TILE1_BLOCKNUM

    TILE1_POST:
    sub x24, x24, #1
    cbz x9, TILE1_CVT_FP16
    ld1 {v16.4s, v17.4s}, [x20], #32 // bias
    fadd v22.4s, v22.4s, v16.4s
    fadd v23.4s, v23.4s, v17.4s

    TILE1_CVT_FP16:
    fcvtn v0.4h, v22.4s
    fcvtn2 v0.8h, v23.4s
    cbz x14, TILE1_STORE
    fmax v0.8h, v0.8h, v29.8h
    fmin v0.8h, v0.8h, v31.8h
    TILE1_STORE:
    st1 {v0.8h}, [x6], x4

Tile1LoopEnd:
    cbz x24, End
    mov x8, x15
    mov x23, x21
    cbz x27, LoopDz_TILE_1
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    b LoopDz_TILE_1

End:
ldp x27, x28, [sp, #(16 * 8)]
ldp x25, x26, [sp, #(16 * 7)]
ldp x23, x24, [sp, #(16 * 6)]
ldp x19, x20, [sp, #(16 * 5)]
ldp x21, x22, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 10)
ret

// ===== W3 unpack constant table =====
// Lives in .text (read-only) right after the function so adr-relative load works on
// both Mach-O (macOS) and ELF (Linux/Android) without needing a separate section.
.align 4
.L_w3_unpack_consts:
    // First 16 bytes: signed shift-count increment from one OC pair to the next.
    .byte 2,2,2,2,2,2,2,2
    .byte 2,2,2,2,2,2,2,2
    // Next 16 bytes: initial signed shift counts for OC0/1.  Aux byte bit7 is
    // OC0 and bit6 is OC1; both are shifted into bit2, then masked with 4.
    .byte -5,-5,-5,-5,-5,-5,-5,-5
    .byte -4,-4,-4,-4,-4,-4,-4,-4

#endif // __aarch64__
