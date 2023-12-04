//
//  MNNGemmHybridInt4_smmla.S
//  MNN
//
//  Created by MNN on 2023/10/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.macro Int32ToFloat z0, z1, z2, z3
    scvtf \z0\().4s, \z0\().4s
    scvtf \z1\().4s, \z1\().4s
    scvtf \z2\().4s, \z2\().4s
    scvtf \z3\().4s, \z3\().4s
.endm

.macro MulScale d0, d1, d2, d3, s
    fmul \d0\().4s, \d0\().4s, \s\().4s
    fmul \d1\().4s, \d1\().4s, \s\().4s
    fmul \d2\().4s, \d2\().4s, \s\().4s
    fmul \d3\().4s, \d3\().4s, \s\().4s
.endm

.macro Float32ToHalf s0, s1, s2, s3, d0, d1
    fcvtn \d0\().4h,  \s0\().4s
    fcvtn2 \d0\().8h, \s1\().4s
    fcvtn \d1\().4h,  \s2\().4s
    fcvtn2 \d1\().8h, \s3\().4s
.endm

.macro Dequant c0, a0, z0, b0, s0, idx
    fmul \c0\().8h, \c0\().8h, \a0\().8h
    fmla \c0\().8h, \z0\().8h, \s0\().h[\idx]
    fadd \c0\().8h, \c0\().8h, \b0\().8h
.endm

asm_function MNNGemmHybridInt4FP16_smmla

//struct QuanPostTreatParameters {
//    const float* scale;
//    const int32_t* bias;
//    int32_t maxValue;
//    int32_t minValue;
//    int32_t useInt8;
//};

//void MNNGemmHybridInt4_smmla(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realSize, float** param); 


// Auto: x0: C*, x1: A*, x2:B*, x3: src_depth_quad, x4: dst_step, x5: dst_depth_quad, x6: realSize, x7: param
// load from param: x7: alpha*, x8: zero*, x9: bias*, x10: sums*, x11: scales*
stp d14, d15, [sp, #(-16 * 9)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]
stp x23, x24, [sp, #(16 * 6)]
stp x25, x26, [sp, #(16 * 7)]
stp x27, x28, [sp, #(16 * 8)]

ldr x8, [x7, #0]
ldr x9, [x7, #8]
ldr x10, [x7, #16]
ldr x11, [x7, #24]
ldr x12, [x7, #32]

Start:
lsl x13, x3, #5 // x13 = src_depth_quad * UNIT * UNIT_SRC / 2(int4) = src_depth_quad * 32  = src_depth_quad << 5
b TILE_4

TILE_12:
    cmp x6, #12
    blt TILE_10
    sub x14, x4, #128  // dst_step
    lsr x15, x14, #1   // src_step = dst_step / 2
    mov x16, x5 // dst_depth_quad
    mov x17, x0 // dst
    mov x18, x2 // weight
    // dequant info
    mov x19, x8 // alpha
    mov x20, x9 // zero
    mov x21, x10 // bias
LoopDz_TILE_12:
    // dequant info for batch
    mov x22, x11 // sums
    mov x23, x12 // scales
    mov x24, x1  // src
    mov x25, x18 // weight
    mov x26, x3  // src_depth_quad
    // init
    dup v8.4s, wzr
    dup v9.4s, wzr
    dup v10.4s, wzr
    dup v11.4s, wzr
    dup v12.4s, wzr
    dup v13.4s, wzr
    dup v14.4s, wzr
    dup v15.4s, wzr
    dup v16.4s, wzr
    dup v17.4s, wzr
    dup v18.4s, wzr
    dup v19.4s, wzr
    dup v20.4s, wzr
    dup v21.4s, wzr
    dup v22.4s, wzr
    dup v23.4s, wzr
    dup v24.4s, wzr
    dup v25.4s, wzr
    dup v26.4s, wzr
    dup v27.4s, wzr
    dup v28.4s, wzr
    dup v29.4s, wzr
    dup v30.4s, wzr
    dup v31.4s, wzr
    // mask
    mov w27, #0x0f
    dup v6.16b, w27
    // offset
    mov w27, #8
    dup v7.16b, w27
LoopSz_TILE_12:
    // src    : 6 x [2 x 8] : (v4-5) * 3
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 6 x 4 x [4] : v8-v31
    ld1 {v0.16b, v1.16b}, [x25], #32    // weight
    // int4 to int8: v0, v1, v2, v3
    ushr v2.16b, v0.16b, #4
    and v3.16b, v0.16b, v6.16b
    ushr v4.16b, v1.16b, #4
    and v5.16b, v1.16b, v6.16b
    sub v2.16b, v2.16b, v7.16b
    sub v3.16b, v3.16b, v7.16b
    sub v4.16b, v4.16b, v7.16b
    sub v5.16b, v5.16b, v7.16b
    zip1 v0.16b, v2.16b, v3.16b
    zip2 v1.16b, v2.16b, v3.16b
    zip1 v2.16b, v4.16b, v5.16b
    zip2 v3.16b, v4.16b, v5.16b

    ld1 {v4.16b, v5.16b}, [x24], #32   // src
    .inst 0x4e80a488 // smmla v8.4s, v4.16b, v0.16b
    .inst 0x4e81a489 // smmla v9.4s, v4.16b, v1.16b
    .inst 0x4e82a48a // smmla v10.4s, v4.16b, v2.16b
    .inst 0x4e83a48b // smmla v11.4s, v4.16b, v3.16b
    .inst 0x4e80a4ac // smmla v12.4s, v5.16b, v0.16b
    .inst 0x4e81a4ad // smmla v13.4s, v5.16b, v1.16b
    .inst 0x4e82a4ae // smmla v14.4s, v5.16b, v2.16b
    .inst 0x4e83a4af // smmla v15.4s, v5.16b, v3.16b
    ld1 {v4.16b, v5.16b}, [x24], #32   // src
    .inst 0x4e80a490 // smmla v16.4s, v4.16b, v0.16b
    .inst 0x4e81a491 // smmla v17.4s, v4.16b, v1.16b
    .inst 0x4e82a492 // smmla v18.4s, v4.16b, v2.16b
    .inst 0x4e83a493 // smmla v19.4s, v4.16b, v3.16b
    .inst 0x4e80a4b4 // smmla v20.4s, v5.16b, v0.16b
    .inst 0x4e81a4b5 // smmla v21.4s, v5.16b, v1.16b
    .inst 0x4e82a4b6 // smmla v22.4s, v5.16b, v2.16b
    .inst 0x4e83a4b7 // smmla v23.4s, v5.16b, v3.16b
    ld1 {v4.16b, v5.16b}, [x24], x15   // src
    .inst 0x4e80a498 // smmla v24.4s, v4.16b, v0.16b
    .inst 0x4e81a499 // smmla v25.4s, v4.16b, v1.16b
    .inst 0x4e82a49a // smmla v26.4s, v4.16b, v2.16b
    .inst 0x4e83a49b // smmla v27.4s, v4.16b, v3.16b
    .inst 0x4e80a4bc // smmla v28.4s, v5.16b, v0.16b
    .inst 0x4e81a4bd // smmla v29.4s, v5.16b, v1.16b
    .inst 0x4e82a4be // smmla v30.4s, v5.16b, v2.16b
    .inst 0x4e83a4bf // smmla v31.4s, v5.16b, v3.16b
    subs x26, x26, #1
    bne LoopSz_TILE_12

LoopSzEnd_TILE_12:
    add x18, x18, x13
    sub x16, x16, #1
    Int32ToFloat  v8,  v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    Int32ToFloat v16, v17, v18, v19
    Int32ToFloat v20, v21, v22, v23
    Int32ToFloat v24, v25, v26, v27
    Int32ToFloat v28, v29, v30, v31
    // using float scale dequant for precison
    ld1 {v4.8h}, [x23], #16    // scales
    ld1 {v5.d}[0], [x23], #8   // scales
    fcvtl v6.4s, v4.4h
    fcvtl2 v7.4s, v4.8h
    // [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11]
    dup v0.4s, v6.s[0]
    mov v0.s[2], v6.s[1]
    mov v0.s[3], v6.s[1]
    dup v1.4s, v6.s[2]
    mov v1.s[2], v6.s[3]
    mov v1.s[3], v6.s[3]
    dup v2.4s, v7.s[0]
    mov v2.s[2], v7.s[1]
    mov v2.s[3], v7.s[1]
    dup v3.4s, v7.s[2]
    mov v3.s[2], v7.s[3]
    mov v3.s[3], v7.s[3]
    fcvtl v7.4s, v5.4h
    dup v4.4s, v7.s[0]
    mov v4.s[2], v7.s[1]
    mov v4.s[3], v7.s[1]
    dup v5.4s, v7.s[2]
    mov v5.s[2], v7.s[3]
    mov v5.s[3], v7.s[3]
    MulScale  v8,  v9, v10, v11, v0
    MulScale v12, v13, v14, v15, v1
    MulScale v16, v17, v18, v19, v2
    MulScale v20, v21, v22, v23, v3
    MulScale v24, v25, v26, v27, v4
    MulScale v28, v29, v30, v31, v5
    Float32ToHalf  v8,  v9, v10, v11,  v6,  v7
    Float32ToHalf v12, v13, v14, v15,  v8,  v9
    Float32ToHalf v16, v17, v18, v19, v10, v11
    Float32ToHalf v20, v21, v22, v23, v12, v13
    Float32ToHalf v24, v25, v26, v27, v14, v15
    Float32ToHalf v28, v29, v30, v31, v16, v17
    uzp1  v5.4s,  v6.4s,  v7.4s
    uzp2  v6.4s,  v6.4s,  v7.4s
    uzp1  v7.4s,  v8.4s,  v9.4s
    uzp2  v8.4s,  v8.4s,  v9.4s
    uzp1  v9.4s, v10.4s, v11.4s
    uzp2 v10.4s, v10.4s, v11.4s
    uzp1 v11.4s, v12.4s, v13.4s
    uzp2 v12.4s, v12.4s, v13.4s
    uzp1 v13.4s, v14.4s, v15.4s
    uzp2 v14.4s, v14.4s, v15.4s
    uzp1 v15.4s, v16.4s, v17.4s
    uzp2 v16.4s, v16.4s, v17.4s
Tile12Dequant:
    ld1 {v0.8h}, [x19], #16  // alpha
    ld1 {v1.8h}, [x20], #16  // zero
    ld1 {v2.8h}, [x21], #16  // bias
    ld1 {v3.8h}, [x22], #16  // sums
    ld1 {v4.d}[0], [x22], #8 // sums
    // alpha * sum + (zero * sumx) + bias
    Dequant  v5, v0, v1, v2, v3, 0
    Dequant  v6, v0, v1, v2, v3, 1
    Dequant  v7, v0, v1, v2, v3, 2
    Dequant  v8, v0, v1, v2, v3, 3
    Dequant  v9, v0, v1, v2, v3, 4
    Dequant v10, v0, v1, v2, v3, 5
    Dequant v11, v0, v1, v2, v3, 6
    Dequant v12, v0, v1, v2, v3, 7
    Dequant v13, v0, v1, v2, v4, 0
    Dequant v14, v0, v1, v2, v4, 1
    Dequant v15, v0, v1, v2, v4, 2
    Dequant v16, v0, v1, v2, v4, 3
    st1 { v5.8h,  v6.8h,  v7.8h,  v8.8h}, [x17], #64
    st1 { v9.8h, v10.8h, v11.8h, v12.8h}, [x17], #64
    st1 {v13.8h, v14.8h, v15.8h, v16.8h}, [x17], x14
    cmp x16, #1
    bge LoopDz_TILE_12
Tile12End:
    sub x6, x6, #12     // bach -= 12
    add x0, x0, #192    // dst += 12 * 8 * sizeof(float16_t)
    add x1, x1, #96     // src += 12 * 8 * sizeof(int8_t)
    add x11, x11, #24   // sum += 12 * sizeof(float16_t)
    add x12, x12, #24   // scale += 12 * sizeof(float16_t)s
    b TILE_12

TILE_10:
    cmp x6, #10
    blt TILE_8
    sub x14, x4, #128  // dst_step
    lsr x15, x14, #1   // src_step = dst_step / 2
    mov x16, x5 // dst_depth_quad
    mov x17, x0 // dst
    mov x18, x2 // weight
    // dequant info
    mov x19, x8 // alpha
    mov x20, x9 // zero
    mov x21, x10 // bias
LoopDz_TILE_10:
    // dequant info for batch
    mov x22, x11 // sums
    mov x23, x12 // scales
    mov x24, x1  // src
    mov x25, x18 // weight
    mov x26, x3  // src_depth_quad
    // init
    dup v12.4s, wzr
    dup v13.4s, wzr
    dup v14.4s, wzr
    dup v15.4s, wzr
    dup v16.4s, wzr
    dup v17.4s, wzr
    dup v18.4s, wzr
    dup v19.4s, wzr
    dup v20.4s, wzr
    dup v21.4s, wzr
    dup v22.4s, wzr
    dup v23.4s, wzr
    dup v24.4s, wzr
    dup v25.4s, wzr
    dup v26.4s, wzr
    dup v27.4s, wzr
    dup v28.4s, wzr
    dup v29.4s, wzr
    dup v30.4s, wzr
    dup v31.4s, wzr
    // mask
    mov w27, #0x0f
    dup v10.16b, w27
    // offset
    mov w27, #8
    dup v11.16b, w27
LoopSz_TILE_10:
    // src    : 5 x [2 x 8] : v4-8
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 5 x 4 x [4] : v12-v31
    ld1 {v0.16b, v1.16b}, [x25], #32    // weight
    // int4 to int8: v0, v1, v2, v3
    ushr v4.16b, v0.16b, #4
    and v5.16b, v0.16b, v10.16b
    sub v4.16b, v4.16b, v11.16b
    sub v5.16b, v5.16b, v11.16b
    ushr v6.16b, v1.16b, #4
    and v7.16b, v1.16b, v10.16b
    sub v6.16b, v6.16b, v11.16b
    sub v7.16b, v7.16b, v11.16b
    zip1 v0.16b, v4.16b, v5.16b
    zip2 v1.16b, v4.16b, v5.16b
    zip1 v2.16b, v6.16b, v7.16b
    zip2 v3.16b, v6.16b, v7.16b
    ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x24], #64   // src
    ld1 {v8.16b}, [x24], x15   // src

    .inst 0x4e80a48c // smmla v12.4s, v4.16b, v0.16b
    .inst 0x4e81a48d // smmla v13.4s, v4.16b, v1.16b
    .inst 0x4e82a48e // smmla v14.4s, v4.16b, v2.16b
    .inst 0x4e83a48f // smmla v15.4s, v4.16b, v3.16b
    .inst 0x4e80a4b0 // smmla v16.4s, v5.16b, v0.16b
    .inst 0x4e81a4b1 // smmla v17.4s, v5.16b, v1.16b
    .inst 0x4e82a4b2 // smmla v18.4s, v5.16b, v2.16b
    .inst 0x4e83a4b3 // smmla v19.4s, v5.16b, v3.16b
    .inst 0x4e80a4d4 // smmla v20.4s, v6.16b, v0.16b
    .inst 0x4e81a4d5 // smmla v21.4s, v6.16b, v1.16b
    .inst 0x4e82a4d6 // smmla v22.4s, v6.16b, v2.16b
    .inst 0x4e83a4d7 // smmla v23.4s, v6.16b, v3.16b
    .inst 0x4e80a4f8 // smmla v24.4s, v7.16b, v0.16b
    .inst 0x4e81a4f9 // smmla v25.4s, v7.16b, v1.16b
    .inst 0x4e82a4fa // smmla v26.4s, v7.16b, v2.16b
    .inst 0x4e83a4fb // smmla v27.4s, v7.16b, v3.16b
    .inst 0x4e80a51c // smmla v28.4s, v8.16b, v0.16b
    .inst 0x4e81a51d // smmla v29.4s, v8.16b, v1.16b
    .inst 0x4e82a51e // smmla v30.4s, v8.16b, v2.16b
    .inst 0x4e83a51f // smmla v31.4s, v8.16b, v3.16b
    subs x26, x26, #1
    bne LoopSz_TILE_10

LoopSzEnd_TILE_10:
    add x18, x18, x13
    sub x16, x16, #1
    Int32ToFloat v12, v13, v14, v15
    Int32ToFloat v16, v17, v18, v19
    Int32ToFloat v20, v21, v22, v23
    Int32ToFloat v24, v25, v26, v27
    Int32ToFloat v28, v29, v30, v31
    // using float scale dequant for precison
    ld1 {v4.8h}, [x23], #16    // scales
    ld1 {v5.s}[0], [x23], #4   // scales
    fcvtl v6.4s, v4.4h
    fcvtl2 v7.4s, v4.8h
    fcvtl v8.4s, v5.4h
    // [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
    dup v0.4s, v6.s[0]
    mov v0.s[2], v6.s[1]
    mov v0.s[3], v6.s[1]
    dup v1.4s, v6.s[2]
    mov v1.s[2], v6.s[3]
    mov v1.s[3], v6.s[3]
    dup v2.4s, v7.s[0]
    mov v2.s[2], v7.s[1]
    mov v2.s[3], v7.s[1]
    dup v3.4s, v7.s[2]
    mov v3.s[2], v7.s[3]
    mov v3.s[3], v7.s[3]
    dup v4.4s, v8.s[0]
    mov v4.s[2], v8.s[1]
    mov v4.s[3], v8.s[1]
    MulScale v12, v13, v14, v15, v0
    MulScale v16, v17, v18, v19, v1
    MulScale v20, v21, v22, v23, v2
    MulScale v24, v25, v26, v27, v3
    MulScale v28, v29, v30, v31, v4
    Float32ToHalf v12, v13, v14, v15, v10, v11
    Float32ToHalf v16, v17, v18, v19, v12, v13
    Float32ToHalf v20, v21, v22, v23, v14, v15
    Float32ToHalf v24, v25, v26, v27, v16, v17
    Float32ToHalf v28, v29, v30, v31, v18, v19
    uzp1  v9.4s, v10.4s, v11.4s
    uzp2 v10.4s, v10.4s, v11.4s
    uzp1 v11.4s, v12.4s, v13.4s
    uzp2 v12.4s, v12.4s, v13.4s
    uzp1 v13.4s, v14.4s, v15.4s
    uzp2 v14.4s, v14.4s, v15.4s
    uzp1 v15.4s, v16.4s, v17.4s
    uzp2 v16.4s, v16.4s, v17.4s
    uzp1 v17.4s, v18.4s, v19.4s
    uzp2 v18.4s, v18.4s, v19.4s
Tile10Dequant:
    ld1 {v0.8h}, [x19], #16  // alpha
    ld1 {v1.8h}, [x20], #16  // zero
    ld1 {v2.8h}, [x21], #16  // bias
    ld1 {v3.8h}, [x22], #16  // sums
    ld1 {v4.s}[0], [x22], #4 // sums
    // alpha * sum + (zero * sumx) + bias
    Dequant  v9, v0, v1, v2, v3, 0
    Dequant v10, v0, v1, v2, v3, 1
    Dequant v11, v0, v1, v2, v3, 2
    Dequant v12, v0, v1, v2, v3, 3
    Dequant v13, v0, v1, v2, v3, 4
    Dequant v14, v0, v1, v2, v3, 5
    Dequant v15, v0, v1, v2, v3, 6
    Dequant v16, v0, v1, v2, v3, 7
    Dequant v17, v0, v1, v2, v4, 0
    Dequant v18, v0, v1, v2, v4, 1
    st1 { v9.8h, v10.8h, v11.8h, v12.8h}, [x17], #64
    st1 {v13.8h, v14.8h, v15.8h, v16.8h}, [x17], #64
    st1 {v17.8h, v18.8h}, [x17], x14
    cmp x16, #1
    bge LoopDz_TILE_10
Tile10End:
    sub x6, x6, #10     // bach -= 10
    add x0, x0, #160    // dst += 10 * 8 * sizeof(float16_t)
    add x1, x1, #80     // src += 10 * 8 * sizeof(int8_t)
    add x11, x11, #20   // sum += 10 * sizeof(float16_t)
    add x12, x12, #20   // scale += 10 * sizeof(float16_t)
    b TILE_10

TILE_8:
    cmp x6, #8
    blt TILE_1
    sub x14, x4, #64  // dst_step
    lsr x15, x4, #1   // src_step = dst_step / 2
    mov x16, x5 // dst_depth_quad
    mov x17, x0 // dst
    mov x18, x2 // weight
    // dequant info
    mov x19, x8 // alpha
    mov x20, x9 // zero
    mov x21, x10 // bias
LoopDz_TILE_8:
    // dequant info for batch
    mov x22, x11 // sums
    mov x23, x12 // scales
    mov x24, x1  // src
    mov x25, x18 // weight
    mov x26, x3  // src_depth_quad
    // init
    dup v16.4s, wzr
    dup v17.4s, wzr
    dup v18.4s, wzr
    dup v19.4s, wzr
    dup v20.4s, wzr
    dup v21.4s, wzr
    dup v22.4s, wzr
    dup v23.4s, wzr
    dup v24.4s, wzr
    dup v25.4s, wzr
    dup v26.4s, wzr
    dup v27.4s, wzr
    dup v28.4s, wzr
    dup v29.4s, wzr
    dup v30.4s, wzr
    dup v31.4s, wzr
    // mask
    mov w27, #0x0f
    dup v10.16b, w27
    // offset
    mov w27, #8
    dup v11.16b, w27
LoopSz_TILE_8:
    // src    : 4 x [2 x 8] : v4-7
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 4 x 4 x [4] : v16-v31
    ld1 {v0.16b, v1.16b}, [x25], #32    // weight
    // int4 to int8: v0, v1, v2, v3
    ushr v4.16b, v0.16b, #4
    and v5.16b, v0.16b, v10.16b
    sub v4.16b, v4.16b, v11.16b
    sub v5.16b, v5.16b, v11.16b
    ushr v6.16b, v1.16b, #4
    and v7.16b, v1.16b, v10.16b
    sub v6.16b, v6.16b, v11.16b
    sub v7.16b, v7.16b, v11.16b
    zip1 v0.16b, v4.16b, v5.16b
    zip2 v1.16b, v4.16b, v5.16b
    zip1 v2.16b, v6.16b, v7.16b
    zip2 v3.16b, v6.16b, v7.16b
    ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x24], x15   // src
    .inst 0x4e80a490 // smmla v16.4s, v4.16b, v0.16b
    .inst 0x4e81a491 // smmla v17.4s, v4.16b, v1.16b
    .inst 0x4e82a492 // smmla v18.4s, v4.16b, v2.16b
    .inst 0x4e83a493 // smmla v19.4s, v4.16b, v3.16b
    .inst 0x4e80a4b4 // smmla v20.4s, v5.16b, v0.16b
    .inst 0x4e81a4b5 // smmla v21.4s, v5.16b, v1.16b
    .inst 0x4e82a4b6 // smmla v22.4s, v5.16b, v2.16b
    .inst 0x4e83a4b7 // smmla v23.4s, v5.16b, v3.16b
    .inst 0x4e80a4d8 // smmla v24.4s, v6.16b, v0.16b
    .inst 0x4e81a4d9 // smmla v25.4s, v6.16b, v1.16b
    .inst 0x4e82a4da // smmla v26.4s, v6.16b, v2.16b
    .inst 0x4e83a4db // smmla v27.4s, v6.16b, v3.16b
    .inst 0x4e80a4fc // smmla v28.4s, v7.16b, v0.16b
    .inst 0x4e81a4fd // smmla v29.4s, v7.16b, v1.16b
    .inst 0x4e82a4fe // smmla v30.4s, v7.16b, v2.16b
    .inst 0x4e83a4ff // smmla v31.4s, v7.16b, v3.16b
    subs x26, x26, #1
    bne LoopSz_TILE_8

LoopSzEnd_TILE_8:
    add x18, x18, x13
    sub x16, x16, #1
    Int32ToFloat v16, v17, v18, v19
    Int32ToFloat v20, v21, v22, v23
    Int32ToFloat v24, v25, v26, v27
    Int32ToFloat v28, v29, v30, v31
    // using float scale dequant for precison
    ld1 {v4.8h}, [x23]  // scales
    fcvtl v5.4s, v4.4h
    fcvtl2 v6.4s, v4.8h
    dup v0.4s, v5.s[0]
    mov v0.s[2], v5.s[1]
    mov v0.s[3], v5.s[1]
    dup v1.4s, v5.s[2]
    mov v1.s[2], v5.s[3]
    mov v1.s[3], v5.s[3]
    dup v2.4s, v6.s[0]
    mov v2.s[2], v6.s[1]
    mov v2.s[3], v6.s[1]
    dup v3.4s, v6.s[2]
    mov v3.s[2], v6.s[3]
    mov v3.s[3], v6.s[3]
    MulScale v16, v17, v18, v19, v0
    MulScale v20, v21, v22, v23, v1
    MulScale v24, v25, v26, v27, v2
    MulScale v28, v29, v30, v31, v3
    Float32ToHalf v16, v17, v18, v19, v12, v13
    Float32ToHalf v20, v21, v22, v23, v14, v15
    Float32ToHalf v24, v25, v26, v27, v16, v17
    Float32ToHalf v28, v29, v30, v31, v18, v19
    uzp1 v11.4s, v12.4s, v13.4s
    uzp2 v12.4s, v12.4s, v13.4s
    uzp1 v13.4s, v14.4s, v15.4s
    uzp2 v14.4s, v14.4s, v15.4s
    uzp1 v15.4s, v16.4s, v17.4s
    uzp2 v16.4s, v16.4s, v17.4s
    uzp1 v17.4s, v18.4s, v19.4s
    uzp2 v18.4s, v18.4s, v19.4s
Tile8Dequant:
    ld1 {v0.8h}, [x19], #16  // alpha
    ld1 {v1.8h}, [x20], #16  // zero
    ld1 {v2.8h}, [x21], #16  // bias
    ld1 {v3.8h}, [x22]  // sums
    // alpha * sum + (zero * sumx) + bias
    Dequant v11, v0, v1, v2, v3, 0
    Dequant v12, v0, v1, v2, v3, 1
    Dequant v13, v0, v1, v2, v3, 2
    Dequant v14, v0, v1, v2, v3, 3
    Dequant v15, v0, v1, v2, v3, 4
    Dequant v16, v0, v1, v2, v3, 5
    Dequant v17, v0, v1, v2, v3, 6
    Dequant v18, v0, v1, v2, v3, 7
    st1 {v11.8h, v12.8h, v13.8h, v14.8h}, [x17], #64
    st1 {v15.8h, v16.8h, v17.8h, v18.8h}, [x17], x14
    cmp x16, #1
    bge LoopDz_TILE_8
Tile8End:
    sub x6, x6, #8      // bach -= 8
    add x0, x0, #128    // dst += 8 * 8 * sizeof(float16_t)
    add x1, x1, #64     // src += 8 * 8 * sizeof(int8_t)
    add x11, x11, #16   // sum += 8 * sizeof(float16_t)
    add x12, x12, #16   // scale += 8 * sizeof(float16_t)
    b TILE_8

TILE_4:
    cmp x6, #4
    blt TILE_2
    mov x14, x4       // dst_step
    lsr x15, x4, #1   // src_step = dst_step / 2
    mov x16, x5 // dst_depth_quad
    mov x17, x0 // dst
    mov x18, x2 // weight
    // dequant info
    mov x19, x8 // alpha
    mov x20, x9 // zero
    mov x21, x10 // bias
LoopDz_TILE_4:
    // dequant info for batch
    mov x22, x11 // sums
    mov x23, x12 // scales
    mov x24, x1  // src
    mov x25, x18 // weight
    mov x26, x3  // src_depth_quad
    // init
    dup v16.4s, wzr
    dup v17.4s, wzr
    dup v18.4s, wzr
    dup v19.4s, wzr
    dup v20.4s, wzr
    dup v21.4s, wzr
    dup v22.4s, wzr
    dup v23.4s, wzr
    dup v24.4s, wzr
    dup v25.4s, wzr
    dup v26.4s, wzr
    dup v27.4s, wzr
    dup v28.4s, wzr
    dup v29.4s, wzr
    dup v30.4s, wzr
    dup v31.4s, wzr
    // mask
    mov w27, #0x0f
    dup v10.16b, w27
    // offset
    mov w27, #8
    dup v11.16b, w27
LoopSz_TILE_4:
    // src    : 2 x [2 x 8] : v4-5
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 2 x 4 x [4] : v16-23
    ld1 {v0.16b, v1.16b}, [x25], #32    // weight
    // int4 to int8: v0, v1, v2, v3
    ushr v4.16b, v0.16b, #4
    and v5.16b, v0.16b, v10.16b
    sub v4.16b, v4.16b, v11.16b
    sub v5.16b, v5.16b, v11.16b
    ushr v6.16b, v1.16b, #4
    and v7.16b, v1.16b, v10.16b
    sub v6.16b, v6.16b, v11.16b
    sub v7.16b, v7.16b, v11.16b
    zip1 v0.16b, v4.16b, v5.16b
    zip2 v1.16b, v4.16b, v5.16b
    zip1 v2.16b, v6.16b, v7.16b
    zip2 v3.16b, v6.16b, v7.16b
    ld1 {v4.16b, v5.16b}, [x24], x15   // src
    .inst 0x4e80a490 // smmla v16.4s, v4.16b, v0.16b
    .inst 0x4e81a491 // smmla v17.4s, v4.16b, v1.16b
    .inst 0x4e82a492 // smmla v18.4s, v4.16b, v2.16b
    .inst 0x4e83a493 // smmla v19.4s, v4.16b, v3.16b
    .inst 0x4e80a4b4 // smmla v20.4s, v5.16b, v0.16b
    .inst 0x4e81a4b5 // smmla v21.4s, v5.16b, v1.16b
    .inst 0x4e82a4b6 // smmla v22.4s, v5.16b, v2.16b
    .inst 0x4e83a4b7 // smmla v23.4s, v5.16b, v3.16b
    subs x26, x26, #1
    bne LoopSz_TILE_4

LoopSzEnd_TILE_4:
    add x18, x18, x13
    sub x16, x16, #1
    Int32ToFloat v16, v17, v18, v19
    Int32ToFloat v20, v21, v22, v23
    // using float scale dequant for precison
    ld1 {v4.d}[0], [x23]  // scales
    fcvtl v5.4s, v4.4h
    dup v0.4s, v5.s[0]
    mov v0.s[2], v5.s[1]
    mov v0.s[3], v5.s[1]
    dup v1.4s, v5.s[2]
    mov v1.s[2], v5.s[3]
    mov v1.s[3], v5.s[3]
    MulScale v16, v17, v18, v19, v0
    MulScale v20, v21, v22, v23, v1
    Float32ToHalf v16, v17, v18, v19, v12, v13
    Float32ToHalf v20, v21, v22, v23, v14, v15
    uzp1 v11.4s, v12.4s, v13.4s
    uzp2 v12.4s, v12.4s, v13.4s
    uzp1 v13.4s, v14.4s, v15.4s
    uzp2 v14.4s, v14.4s, v15.4s
Tile4Dequant:
    ld1 {v0.8h}, [x19], #16  // alpha
    ld1 {v1.8h}, [x20], #16  // zero
    ld1 {v2.8h}, [x21], #16  // bias
    ld1 {v3.d}[0], [x22]  // sums
    // alpha * sum + (zero * sumx) + bias
    Dequant v11, v0, v1, v2, v3, 0
    Dequant v12, v0, v1, v2, v3, 1
    Dequant v13, v0, v1, v2, v3, 2
    Dequant v14, v0, v1, v2, v3, 3
    st1 {v11.8h, v12.8h, v13.8h, v14.8h}, [x17], x14
    cmp x16, #1
    bge LoopDz_TILE_4
Tile4End:
    sub x6, x6, #4      // bach -= 4
    add x0, x0, #64     // dst += 4 * 8 * sizeof(float16_t)
    add x1, x1, #32     // src += 4 * 8 * sizeof(int8_t)
    add x11, x11, #8    // sum += 4 * sizeof(float16_t)
    add x12, x12, #8    // scale += 4 * sizeof(float16_t)
    b TILE_4

TILE_2:
    cmp x6, #2
    blt TILE_1
    mov x14, x4       // dst_step
    lsr x15, x4, #1   // src_step = dst_step / 2
    mov x16, x5 // dst_depth_quad
    mov x17, x0 // dst
    mov x18, x2 // weight
    // dequant info
    mov x19, x8 // alpha
    mov x20, x9 // zero
    mov x21, x10 // bias
LoopDz_TILE_2:
    mov x22, x11 // sums
    mov x23, x12 // scales
    mov x24, x1  // src
    mov x25, x18 // weight
    mov x26, x3  // src_depth_quad
    // init
    dup v16.4s, wzr
    dup v17.4s, wzr
    dup v18.4s, wzr
    dup v19.4s, wzr
    // mask
    mov w27, #0x0f
    dup v14.16b, w27
    // offset
    mov w27, #8
    dup v15.16b, w27
LoopSz_TILE_2:
    // src    : 1 x [2 x 8] : v4
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 1 x 4 x [4] : v16-19
    ld1 {v0.16b, v1.16b}, [x25], #32    // weight
    // int4 to int8: v0, v1, v2, v3
    ushr v8.16b, v0.16b, #4
    and v9.16b, v0.16b, v14.16b
    sub v8.16b, v8.16b, v15.16b
    sub v9.16b, v9.16b, v15.16b
    ushr v10.16b, v1.16b, #4
    and v11.16b, v1.16b, v14.16b
    sub v10.16b, v10.16b, v15.16b
    sub v11.16b, v11.16b, v15.16b
    zip1 v0.16b, v8.16b, v9.16b
    zip2 v1.16b, v8.16b, v9.16b
    zip1 v2.16b, v10.16b, v11.16b
    zip2 v3.16b, v10.16b, v11.16b
    ld1 {v4.16b}, [x24], x15   // src
    .inst 0x4e80a490 // smmla v16.4s, v4.16b, v0.16b
    .inst 0x4e81a491 // smmla v17.4s, v4.16b, v1.16b
    .inst 0x4e82a492 // smmla v18.4s, v4.16b, v2.16b
    .inst 0x4e83a493 // smmla v19.4s, v4.16b, v3.16b
    subs x26, x26, #1
    bne LoopSz_TILE_2

LoopSzEnd_TILE_2:
    add x18, x18, x13
    sub x16, x16, #1
    uzp1 v13.2d, v16.2d, v17.2d
    uzp1 v14.2d, v18.2d, v19.2d
    uzp2 v15.2d, v16.2d, v17.2d
    uzp2 v16.2d, v18.2d, v19.2d
    Int32ToFloat v13, v14, v15, v16
    // using float scale dequant for precison
    ld1 {v4.s}[0], [x23]  // scales
    fcvtl v5.4s, v4.4h
    fmul v13.4s, v13.4s, v5.s[0]
    fmul v14.4s, v14.4s, v5.s[0]
    fmul v15.4s, v15.4s, v5.s[1]
    fmul v16.4s, v16.4s, v5.s[1]
    fcvtn v12.4h,  v13.4s
    fcvtn2 v12.8h, v14.4s
    fcvtn v13.4h,  v15.4s
    fcvtn2 v13.8h, v16.4s
Tile2Dequant:
    ld1 {v0.8h}, [x19], #16  // alpha
    ld1 {v1.8h}, [x20], #16  // zero
    ld1 {v2.8h}, [x21], #16  // bias
    ld1 {v3.s}[0], [x22]  // sums
    // alpha * sum + (zero * sumx) + bias
    Dequant v12, v0, v1, v2, v3, 0
    Dequant v13, v0, v1, v2, v3, 1
    st1 {v12.8h, v13.8h}, [x17], x14
    cmp x16, #1
    bge LoopDz_TILE_2
Tile2End:
    sub x6, x6, #2      // batch -= 2
    add x0, x0, #32     // dst += 2 * 8 * sizeof(float16_t)
    add x1, x1, #16     // dst += 2 * 8 * sizeof(int8_t)
    add x11, x11, #4    // sum += 2 * sizeof(float16_t)
    add x12, x12, #4    // scale += 2 * sizeof(float16_t)
    b TILE_2


TILE_1:
    cmp x6, #1
    blt End
    mov x14, x4       // dst_step
    lsr x15, x4, #1   // src_step = dst_step / 2
    mov x16, x5 // dst_depth_quad
    mov x17, x0 // dst
    mov x18, x2 // weight
    // dequant info
    mov x19, x8 // alpha
    mov x20, x9 // zero
    mov x21, x10 // bias
LoopDz_TILE_1:
    mov x22, x11 // sums
    mov x23, x12 // scales
    mov x24, x1  // src
    mov x25, x18 // weight
    mov x26, x3  // src_depth_quad
    // init
    dup v16.4s, wzr
    dup v17.4s, wzr
    dup v18.4s, wzr
    dup v19.4s, wzr
    // mask
    mov w27, #0x0f
    dup v14.16b, w27
    // offset
    mov w27, #8
    dup v15.16b, w27

LoopSz_TILE_1:
    // src    : 1 x [1 x 8] : v4
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 1 x 4 x [2] : v16-v19
    prfm pldl1keep, [x25, #64]   // 预取下一次权重数据
    prfm pldl1keep, [x24, x15]   // 预取下一次源数据
    ld1 {v0.16b, v1.16b}, [x25], #32    // weight
    // int4 to int8: v0, v1, v2, v3
    ushr v8.16b, v0.16b, #4
    and v9.16b, v0.16b, v14.16b
    ushr v10.16b, v1.16b, #4
    and v11.16b, v1.16b, v14.16b
    sub v8.16b, v8.16b, v15.16b
    sub v9.16b, v9.16b, v15.16b
    sub v10.16b, v10.16b, v15.16b
    sub v11.16b, v11.16b, v15.16b
    zip1 v0.16b, v8.16b, v9.16b
    zip2 v1.16b, v8.16b, v9.16b
    zip1 v2.16b, v10.16b, v11.16b
    zip2 v3.16b, v10.16b, v11.16b
    ld1 {v4.8b}, [x24], x15   // src
    .inst 0x4e84a410 // smmla v16.4s, v0.16b, v4.16b
    .inst 0x4e84a431 // smmla v17.4s, v1.16b, v4.16b
    .inst 0x4e84a452 // smmla v18.4s, v2.16b, v4.16b
    .inst 0x4e84a473 // smmla v19.4s, v3.16b, v4.16b
    subs x26, x26, #1
    bne LoopSz_TILE_1

LoopSzEnd_TILE_1:
    add x18, x18, x13
    sub x16, x16, #1
    uzp1 v15.4s, v16.4s, v17.4s
    uzp1 v16.4s, v18.4s, v19.4s
    scvtf v15.4s, v15.4s
    scvtf v16.4s, v16.4s
    // using float scale dequant for precison
    ld1 {v4.h}[0], [x23]  // scales
    fcvtl v5.4s, v4.4h
    fmul v15.4s, v15.4s, v5.s[0]
    fmul v16.4s, v16.4s, v5.s[0]
    fcvtn v17.4h,  v15.4s
    fcvtn2 v17.8h, v16.4s
Tile1Dequant:
    ld1 {v0.8h}, [x19], #16  // alpha
    ld1 {v1.8h}, [x20], #16  // zero
    ld1 {v2.8h}, [x21], #16  // bias
    ld1 {v3.h}[0], [x22]  // sums
    // alpha * sum + (zero * sumx) + bias
    fmla v2.8h, v0.8h, v17.8h
    fmla v2.8h, v1.8h, v3.h[0]
    st1 {v2.8h}, [x17], x14
    cmp x16, #1
    bge LoopDz_TILE_1
Tile1End:
    sub x6, x6, #1      // batch -= 1
    add x0, x0, #16     // dst += 1 * 8 * sizeof(float16_t)
    add x1, x1, #8      // dst += 1 * 8 * sizeof(int8_t)
    add x11, x11, #2   // sum += 1 * sizeof(float16_t)
    add x12, x12, #2   // scale += 1 * sizeof(float16_t)
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

#endif