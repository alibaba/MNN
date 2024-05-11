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

.macro Float32ToHalf s0, s1, s2, s3, d0, d1
    fcvtn \d0\().4h,  \s0\().4s
    fcvtn2 \d0\().8h, \s1\().4s
    fcvtn \d1\().4h,  \s2\().4s
    fcvtn2 \d1\().8h, \s3\().4s
.endm

.macro MulScale d0, d1, d2, d3, s, idx0, idx1, alpha0, alpha1
    fmul \d0\().4s, \d0\().4s, \s\().s[\idx0]
    fmul \d1\().4s, \d1\().4s, \s\().s[\idx0]
    fmul \d2\().4s, \d2\().4s, \s\().s[\idx1]
    fmul \d3\().4s, \d3\().4s, \s\().s[\idx1]
    fmul \d0\().4s, \d0\().4s, \alpha0\().4s
    fmul \d1\().4s, \d1\().4s, \alpha1\().4s
    fmul \d2\().4s, \d2\().4s, \alpha0\().4s
    fmul \d3\().4s, \d3\().4s, \alpha1\().4s
.endm

.macro MulScale_New d0, d1, d2, d3, s, a1, a2, a3, a4
    fmul \d0\().4s, \d0\().4s, \s\().4s
    fmul \d1\().4s, \d1\().4s, \s\().4s
    fmul \d2\().4s, \d2\().4s, \s\().4s
    fmul \d3\().4s, \d3\().4s, \s\().4s
    fmul \d0\().4s, \d0\().4s, \a1\().4s
    fmul \d1\().4s, \d1\().4s, \a2\().4s
    fmul \d2\().4s, \d2\().4s, \a3\().4s
    fmul \d3\().4s, \d3\().4s, \a4\().4s
.endm

.macro Dequant c0, z0, b0, s0, idx
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
// mask
movi v10.16b, #15
// offset
movi v11.16b, #8
TILE_8:
    cmp x6, #8
    blt TILE_4
    //mov x14, x4       // dst_step
    lsr x15, x4, #1   // src_step = dst_step / 2
    sub x14, x4, #64
    mov x27, x5 // dst_depth_quad
    mov x28, x0 // dst
    mov x7, x2 // weight
    // dequant info
    mov x19, x8 // alpha
    mov x20, x9 // zero
    mov x21, x10 // bias
LoopDz_TILE_8:
    // dequant info for batch
    mov x22, x11 // sums
    mov x23, x12 // scales
    mov x24, x1  // src
    mov x25, x7 // weight
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

    ld1 {v14.8h}, [x23]  // scales
    ld1 {v15.8h}, [x19], #16  // alpha

LoopSz_TILE_8:
    // src    : 2 x [2 x 8] : v4-5
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 2 x 4 x [4] : v16-31
    ld1 {v8.16b, v9.16b}, [x25], #32    // weight
    ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x24], x15   // src
    // int4 to int8: v0, v1, v2, v3
    ushr v0.16b, v8.16b, #4
    and v1.16b, v8.16b, v10.16b
    ushr v2.16b, v9.16b, #4
    and v3.16b, v9.16b, v10.16b

    .inst 0x4e80a490 // smmla v16.4s, v4.16b, v0.16b // batch=0,1, oc=0,1
    .inst 0x4e81a491 // smmla v17.4s, v4.16b, v1.16b // batch=0,1, oc=2,3
    .inst 0x4e82a492 // smmla v18.4s, v4.16b, v2.16b // batch=0,1, oc=4,5
    .inst 0x4e83a493 // smmla v19.4s, v4.16b, v3.16b // batch=0,1, oc=6,7
    .inst 0x4e80a4b4 // smmla v20.4s, v5.16b, v0.16b // batch=2,3, oc=0,1
    .inst 0x4e81a4b5 // smmla v21.4s, v5.16b, v1.16b // batch=2,3, oc=2,3
    .inst 0x4e82a4b6 // smmla v22.4s, v5.16b, v2.16b // batch=2,3, oc=4,5
    .inst 0x4e83a4b7 // smmla v23.4s, v5.16b, v3.16b // batch=2,3, oc=6,7

    .inst 0x4e80a4d8 // smmla v24.4s, v6.16b, v0.16b // batch=4,5, oc=0,1
    .inst 0x4e81a4d9 // smmla v25.4s, v6.16b, v1.16b // batch=4,5, oc=2,3
    .inst 0x4e82a4da // smmla v26.4s, v6.16b, v2.16b // batch=4,5, oc=4,5
    .inst 0x4e83a4db // smmla v27.4s, v6.16b, v3.16b // batch=4,5, oc=6,7
    .inst 0x4e80a4fc // smmla v28.4s, v7.16b, v0.16b // batch=6,7, oc=0,1
    .inst 0x4e81a4fd // smmla v29.4s, v7.16b, v1.16b // batch=6,7, oc=2,3
    .inst 0x4e82a4fe // smmla v30.4s, v7.16b, v2.16b // batch=6,7, oc=4,5
    .inst 0x4e83a4ff // smmla v31.4s, v7.16b, v3.16b // batch=6,7, oc=6,7
    subs x26, x26, #1
    bne LoopSz_TILE_8

LoopSzEnd_TILE_8:
    add x7, x7, x13
    fcvtl v8.4s, v15.4h // oc:0-3
    fcvtl2 v9.4s, v15.8h // oc:4-7
    fcvtl v12.4s, v14.4h // scales: batch 0,1,2,3
    fcvtl2 v13.4s, v14.8h // scales: batch 4,5,6,7
    sub x27, x27, #1
    Int32ToFloat v16, v17, v18, v19
    Int32ToFloat v20, v21, v22, v23
    Int32ToFloat v24, v25, v26, v27
    Int32ToFloat v28, v29, v30, v31

    zip1 v0.4s, v12.4s, v12.4s // scales: batch 0,0,1,1
    zip2 v1.4s, v12.4s, v12.4s // scales: batch 2,2,3,3
    zip1 v2.4s, v13.4s, v13.4s // scales: batch 4,4,5,5
    zip2 v3.4s, v13.4s, v13.4s // scales: batch 6,6,7,7
    trn1 v4.2d, v8.2d, v8.2d // alpha: oc 0,1,0,1
    trn2 v5.2d, v8.2d, v8.2d // alpha: oc 2,3,2,3
    trn1 v6.2d, v9.2d, v9.2d // alpha: oc 4,5,4,5
    trn2 v7.2d, v9.2d, v9.2d // alpha: oc 6,7,6,7

    MulScale_New v16, v17, v18, v19, v0, v4, v5, v6, v7
    MulScale_New v20, v21, v22, v23, v1, v4, v5, v6, v7
    MulScale_New v24, v25, v26, v27, v2, v4, v5, v6, v7
    MulScale_New v28, v29, v30, v31, v3, v4, v5, v6, v7
    Float32ToHalf v16, v17, v18, v19, v0, v1 // (batch,oc) v12:(0,0)(0,1)(1,0)(1,1)(0,2)(0,3)(1,3)(1,2)
    Float32ToHalf v20, v21, v22, v23, v12, v13 // batch=2,3 v14:(2,0)(2,1)(3,0)(3,1)(2,2)(2,3)(3,3)(3,2)
    Float32ToHalf v24, v25, v26, v27, v14, v15 // batch=4,5
    Float32ToHalf v28, v29, v30, v31, v8, v9 // batch=6,7

    uzp1 v4.4s, v0.4s, v1.4s
    uzp2 v5.4s, v0.4s, v1.4s
    uzp1 v6.4s, v12.4s, v13.4s
    uzp2 v7.4s, v12.4s, v13.4s
    uzp1 v0.4s, v14.4s, v15.4s
    uzp2 v1.4s, v14.4s, v15.4s
    uzp1 v2.4s, v8.4s, v9.4s
    uzp2 v3.4s, v8.4s, v9.4s
Tile8Dequant:
    ld1 {v16.8h}, [x20], #16  // zero
    ld1 {v17.8h}, [x21], #16  // bias
    ld1 {v12.8h}, [x22]  // sums
    // sum + (zero * sumx) + bias
    Dequant v4, v16, v17, v12, 0
    Dequant v5, v16, v17, v12, 1
    Dequant v6, v16, v17, v12, 2
    Dequant v7, v16, v17, v12, 3

    Dequant v0, v16, v17, v12, 4
    Dequant v1, v16, v17, v12, 5
    Dequant v2, v16, v17, v12, 6
    Dequant v3, v16, v17, v12, 7
    st1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x28], #64
    st1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x28], x14
    cmp x27, #1
    bge LoopDz_TILE_8
Tile8End:
    sub x6, x6, #8      // bach -= 8
    add x0, x0, #128     // dst += 8 * 8 * sizeof(float16_t)
    add x1, x1, #64     // src += 8 * 8 * sizeof(int8_t)
    add x11, x11, #16    // sum += 8 * sizeof(float16_t)
    add x12, x12, #16    // scale += 8 * sizeof(float16_t)
    b TILE_8

TILE_4:
    cmp x6, #4
    blt TILE_2
    mov x14, x4       // dst_step
    lsr x15, x4, #1   // src_step = dst_step / 2
    mov x27, x5 // dst_depth_quad
    mov x28, x0 // dst
    mov x7, x2 // weight
    // dequant info
    mov x19, x8 // alpha
    mov x20, x9 // zero
    mov x21, x10 // bias
LoopDz_TILE_4:
    // dequant info for batch
    mov x22, x11 // sums
    mov x23, x12 // scales
    mov x24, x1  // src
    mov x25, x7 // weight
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
    ld1 {v14.d}[0], [x23]  // scales
    ld1 {v15.8h}, [x19], #16  // alpha
LoopSz_TILE_4:
    // src    : 2 x [2 x 8] : v4-5
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 2 x 4 x [4] : v16-23
    ld1 {v8.16b, v9.16b}, [x25], #32    // weight
    ld1 {v4.16b, v5.16b}, [x24], x15   // src
    // int4 to int8: v0, v1, v2, v3
    ushr v0.16b, v8.16b, #4
    and v1.16b, v8.16b, v10.16b
    ushr v2.16b, v9.16b, #4
    and v3.16b, v9.16b, v10.16b

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
    add x7, x7, x13
    fcvtl v8.4s, v15.4h // oc:0-3
    fcvtl2 v9.4s, v15.8h // oc:4-7
    fcvtl v12.4s, v14.4h // scales: batch 0,1,2,3

    sub x27, x27, #1
    Int32ToFloat v16, v17, v18, v19
    Int32ToFloat v20, v21, v22, v23

    zip1 v0.4s, v12.4s, v12.4s // scales: batch 0,0,1,1
    zip2 v1.4s, v12.4s, v12.4s // scales: batch 2,2,3,3
    trn1 v4.2d, v8.2d, v8.2d // alpha: oc 0,1,0,1
    trn2 v5.2d, v8.2d, v8.2d // alpha: oc 2,3,2,3
    trn1 v6.2d, v9.2d, v9.2d // alpha: oc 4,5,4,5
    trn2 v7.2d, v9.2d, v9.2d // alpha: oc 6,7,6,7

    MulScale_New v16, v17, v18, v19, v0, v4, v5, v6, v7
    MulScale_New v20, v21, v22, v23, v1, v4, v5, v6, v7
    Float32ToHalf v16, v17, v18, v19, v0, v1 // (batch,oc) v12:(0,0)(0,1)(1,0)(1,1)(0,2)(0,3)(1,3)(1,2)
    Float32ToHalf v20, v21, v22, v23, v12, v13 // batch=2,3 v14:(2,0)(2,1)(3,0)(3,1)(2,2)(2,3)(3,3)(3,2)

    uzp1 v4.4s, v0.4s, v1.4s
    uzp2 v5.4s, v0.4s, v1.4s
    uzp1 v6.4s, v12.4s, v13.4s
    uzp2 v7.4s, v12.4s, v13.4s
Tile4Dequant:
    ld1 {v16.8h}, [x20], #16  // zero
    ld1 {v17.8h}, [x21], #16  // bias
    ld1 {v12.d}[0], [x22]  // sums
    // sum + (zero * sumx) + bias
    Dequant v4, v16, v17, v12, 0
    Dequant v5, v16, v17, v12, 1
    Dequant v6, v16, v17, v12, 2
    Dequant v7, v16, v17, v12, 3
    st1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x28], x14
    cmp x27, #1
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
    mov x27, x5 // dst_depth_quad
    mov x28, x0 // dst
    mov x7, x2 // weight
    // dequant info
    mov x19, x8 // alpha
    mov x20, x9 // zero
    mov x21, x10 // bias
LoopDz_TILE_2:
    mov x22, x11 // sums
    mov x23, x12 // scales
    mov x24, x1  // src
    mov x25, x7 // weight
    mov x26, x3  // src_depth_quad
    // init
    dup v16.4s, wzr
    dup v17.4s, wzr
    dup v18.4s, wzr
    dup v19.4s, wzr
    ld1 {v14.s}[0], [x23]  // scales
    ld1 {v15.8h}, [x19], #16  // alpha
LoopSz_TILE_2:
    // src    : 1 x [2 x 8] : v4
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 1 x 4 x [4] : v16-19
    ld1 {v8.16b, v9.16b}, [x25], #32    // weight
    ld1 {v4.16b}, [x24], x15   // src
    // int4 to int8: v0, v1, v2, v3
    ushr v0.16b, v8.16b, #4
    and v1.16b, v8.16b, v10.16b
    ushr v2.16b, v9.16b, #4
    and v3.16b, v9.16b, v10.16b

    .inst 0x4e80a490 // smmla v16.4s, v4.16b, v0.16b
    .inst 0x4e81a491 // smmla v17.4s, v4.16b, v1.16b
    .inst 0x4e82a492 // smmla v18.4s, v4.16b, v2.16b
    .inst 0x4e83a493 // smmla v19.4s, v4.16b, v3.16b
    subs x26, x26, #1
    bne LoopSz_TILE_2

LoopSzEnd_TILE_2:
    add x7, x7, x13
    fcvtl v8.4s, v15.4h // oc:0-3
    fcvtl2 v9.4s, v15.8h // oc:4-7
    fcvtl v12.4s, v14.4h // scales: batch 0,1
    sub x27, x27, #1
    Int32ToFloat v16, v17, v18, v19
    zip1 v0.4s, v12.4s, v12.4s // scales: batch 0,0,1,1
    trn1 v4.2d, v8.2d, v8.2d // alpha: oc 0,1,0,1
    trn2 v5.2d, v8.2d, v8.2d // alpha: oc 2,3,2,3
    trn1 v6.2d, v9.2d, v9.2d // alpha: oc 4,5,4,5
    trn2 v7.2d, v9.2d, v9.2d // alpha: oc 6,7,6,7
    MulScale_New v16, v17, v18, v19, v0, v4, v5, v6, v7
    Float32ToHalf v16, v17, v18, v19, v0, v1 // (batch,oc) v12:(0,0)(0,1)(1,0)(1,1)(0,2)(0,3)(1,3)(1,2)

    uzp1 v4.4s, v0.4s, v1.4s
    uzp2 v5.4s, v0.4s, v1.4s

Tile2Dequant:
    ld1 {v16.8h}, [x20], #16  // zero
    ld1 {v17.8h}, [x21], #16  // bias
    ld1 {v12.s}[0], [x22]  // sums
    // sum + (zero * sumx) + bias
    Dequant v4, v16, v17, v12, 0
    Dequant v5, v16, v17, v12, 1
    st1 {v4.8h, v5.8h}, [x28], x14
    cmp x27, #1
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
    mov x27, x5 // dst_depth_quad
    mov x28, x0 // dst
    mov x7, x2 // weight
    // dequant info
    mov x19, x8 // alpha
    mov x20, x9 // zero
    mov x21, x10 // bias
LoopDz_TILE_1:
    mov x22, x11 // sums
    mov x23, x12 // scales
    mov x24, x1  // src
    mov x25, x7 // weight
    mov x26, x3  // src_depth_quad
    // init
    dup v16.4s, wzr
    dup v17.4s, wzr
    dup v18.4s, wzr
    dup v19.4s, wzr
    ld1 {v14.d}[0], [x23]  // scales
    ld1 {v15.8h}, [x19], #16  // alpha
LoopSz_TILE_1:
    // src    : 1 x [1 x 8] : v4
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 1 x 4 x [2] : v16-v19
    prfm pldl1keep, [x25, #64]   // 预取下一次权重数据
    prfm pldl1keep, [x24, x15]   // 预取下一次源数据
    ld1 {v8.16b, v9.16b}, [x25], #32    // weight
    ld1 {v4.8b}, [x24], x15   // src
    // int4 to int8: v0, v1, v2, v3
    ushr v0.16b, v8.16b, #4
    and v1.16b, v8.16b, v10.16b
    ushr v2.16b, v9.16b, #4
    and v3.16b, v9.16b, v10.16b

    .inst 0x4e84a410 // smmla v16.4s, v0.16b, v4.16b
    .inst 0x4e84a431 // smmla v17.4s, v1.16b, v4.16b
    .inst 0x4e84a452 // smmla v18.4s, v2.16b, v4.16b
    .inst 0x4e84a473 // smmla v19.4s, v3.16b, v4.16b
    subs x26, x26, #1
    bne LoopSz_TILE_1

LoopSzEnd_TILE_1:
    add x7, x7, x13
    sub x27, x27, #1
    uzp1 v20.4s, v16.4s, v17.4s
    uzp1 v21.4s, v18.4s, v19.4s
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s
    // using float scale dequant for precison
    fcvtl v28.4s, v15.4h // oc:0-3
    fcvtl2 v29.4s, v15.8h // oc:4-7
    fcvtl v12.4s, v14.4h // scales: batch 0

    fmul v20.4s, v20.4s, v12.s[0]
    fmul v21.4s, v21.4s, v12.s[0]
    fmul v20.4s, v20.4s, v28.4s
    fmul v21.4s, v21.4s, v29.4s
    fcvtn v17.4h,  v20.4s
    fcvtn2 v17.8h, v21.4s
Tile1Dequant:

    ld1 {v1.8h}, [x20], #16  // zero
    ld1 {v2.8h}, [x21], #16  // bias
    ld1 {v3.h}[0], [x22]  // sums
    // alpha * sum + (zero * sumx) + bias
    fadd v2.8h, v2.8h, v17.8h
    fmla v2.8h, v1.8h, v3.h[0]
    st1 {v2.8h}, [x28], x14
    cmp x27, #1
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
