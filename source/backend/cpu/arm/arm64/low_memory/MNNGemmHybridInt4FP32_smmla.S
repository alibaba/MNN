//
//  MNNGemmHybridInt4FP32_smmla.S
//  MNN
//
//  Created by MNN on 2023/11/09.
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

.macro MulScale d0, d1, d2, d3, s, idx0, idx1
    fmul \d0\().4s, \d0\().4s, \s\().s[\idx0]
    fmul \d1\().4s, \d1\().4s, \s\().s[\idx0]
    fmul \d2\().4s, \d2\().4s, \s\().s[\idx1]
    fmul \d3\().4s, \d3\().4s, \s\().s[\idx1]
.endm

.macro Dequant c0, a0, z0, b0, s0, idx
    fmul \c0\().4s, \c0\().4s, \a0\().4s
    fmla \c0\().4s, \z0\().4s, \s0\().s[\idx]
    fadd \c0\().4s, \c0\().4s, \b0\().4s
.endm

asm_function MNNGemmHybridInt4FP32_smmla

//struct QuanPostTreatParameters {
//    const float* scale;
//    const int32_t* bias;
//    int32_t maxValue;
//    int32_t minValue;
//    int32_t useInt8;
//};

//void MNNGemmHybridInt4FP32_smmla(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realSize, float** param);


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

TILE_8:
    cmp x6, #8
    blt TILE_4
    sub x14, x4, #192       // dst_step
    lsr x15, x4, #2   // src_step = dst_step / 4
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

    // mask
    movi v10.16b, #15
LoopSz_TILE_8:
    // src    : 2 x [2 x 8] : v4-5
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 2 x 4 x [4] : v16-23
    //ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x25], #64    // weight
    ld1 {v8.16b, v9.16b}, [x25], #32    // weight
    ld1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x24], x15   // src
    // int4 to int8: v0, v1, v2, v3
    ushr v0.16b, v8.16b, #4
    and v1.16b, v8.16b, v10.16b
    ushr v2.16b, v9.16b, #4
    and v3.16b, v9.16b, v10.16b

    .inst 0x4e80a590 // smmla v16.4s, v12.16b, v0.16b
    .inst 0x4e81a591 // smmla v17.4s, v12.16b, v1.16b
    .inst 0x4e82a592 // smmla v18.4s, v12.16b, v2.16b
    .inst 0x4e83a593 // smmla v19.4s, v12.16b, v3.16b
    .inst 0x4e80a5b4 // smmla v20.4s, v13.16b, v0.16b
    .inst 0x4e81a5b5 // smmla v21.4s, v13.16b, v1.16b
    .inst 0x4e82a5b6 // smmla v22.4s, v13.16b, v2.16b
    .inst 0x4e83a5b7 // smmla v23.4s, v13.16b, v3.16b
    .inst 0x4e80a5d8 // smmla v24.4s, v14.16b, v0.16b
    .inst 0x4e81a5d9 // smmla v25.4s, v14.16b, v1.16b
    .inst 0x4e82a5da // smmla v26.4s, v14.16b, v2.16b
    .inst 0x4e83a5db // smmla v27.4s, v14.16b, v3.16b
    .inst 0x4e80a5fc // smmla v28.4s, v15.16b, v0.16b
    .inst 0x4e81a5fd // smmla v29.4s, v15.16b, v1.16b
    .inst 0x4e82a5fe // smmla v30.4s, v15.16b, v2.16b
    .inst 0x4e83a5ff // smmla v31.4s, v15.16b, v3.16b
    subs x26, x26, #1
    bne LoopSz_TILE_8

LoopSzEnd_TILE_8:
    add x7, x7, x13
    sub x27, x27, #1

    trn1 v0.2d, v16.2d, v17.2d // batch:0 oc:0-3
    trn1 v1.2d, v18.2d, v19.2d // batch:0 oc:4-7
    trn2 v2.2d, v16.2d, v17.2d // batch:1 oc:0-3
    trn2 v3.2d, v18.2d, v19.2d // batch:1 oc:4-7
    trn1 v4.2d, v20.2d, v21.2d // batch:2 oc:0-3
    trn1 v5.2d, v22.2d, v23.2d // batch:2 oc:4-7
    trn2 v6.2d, v20.2d, v21.2d // batch:3 oc:0-3
    trn2 v7.2d, v22.2d, v23.2d // batch:3 oc:4-7

    trn1 v8.2d, v24.2d, v25.2d // batch:0 oc:0-3
    trn1 v9.2d, v26.2d, v27.2d // batch:0 oc:4-7
    trn2 v10.2d, v24.2d, v25.2d // batch:1 oc:0-3
    trn2 v11.2d, v26.2d, v27.2d // batch:1 oc:4-7
    trn1 v12.2d, v28.2d, v29.2d // batch:2 oc:0-3
    trn1 v13.2d, v30.2d, v31.2d // batch:2 oc:4-7
    trn2 v14.2d, v28.2d, v29.2d // batch:3 oc:0-3
    trn2 v15.2d, v30.2d, v31.2d // batch:3 oc:4-7

    Int32ToFloat v0, v1, v2, v3
    Int32ToFloat v4, v5, v6, v7
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    // using float scale dequant for precison
    ld1 {v16.4s, v17.4s}, [x23]  // scales
    MulScale v0, v1, v2, v3, v16, 0, 1
    MulScale v4, v5, v6, v7, v16, 2, 3
    MulScale v8, v9, v10, v11, v17, 0, 1
    MulScale v12, v13, v14, v15, v17, 2, 3
Tile8Dequant:
    ld1 {v18.4s, v19.4s}, [x19], #32  // alpha
    ld1 {v20.4s, v21.4s}, [x20], #32  // zero
    ld1 {v22.4s, v23.4s}, [x21], #32  // bias
    ld1 {v24.4s, v25.4s}, [x22]  // sums
    // alpha * cusum + (zero * sums) + bias
    Dequant v0, v18, v20, v22, v24, 0 // Batch0
    Dequant v1, v19, v21, v23, v24, 0
    Dequant v2, v18, v20, v22, v24, 1 // Batch1
    Dequant v3, v19, v21, v23, v24, 1
    Dequant v4, v18, v20, v22, v24, 2 // Batch2
    Dequant v5, v19, v21, v23, v24, 2
    Dequant v6, v18, v20, v22, v24, 3 // Batch3
    Dequant v7, v19, v21, v23, v24, 3
    Dequant v8, v18, v20, v22, v25, 0 // Batch4
    Dequant v9, v19, v21, v23, v25, 0
    Dequant v10, v18, v20, v22, v25, 1 // Batch5
    Dequant v11, v19, v21, v23, v25, 1
    Dequant v12, v18, v20, v22, v25, 2 // Batch6
    Dequant v13, v19, v21, v23, v25, 2
    Dequant v14, v18, v20, v22, v25, 3 // Batch7
    Dequant v15, v19, v21, v23, v25, 3
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x28], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x28], #64
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x28], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x28], x14
    cmp x27, #1
    bge LoopDz_TILE_8
Tile8End:
    sub x6, x6, #8      // bach -= 8
    add x0, x0, #256     // dst += 8 * 8 * sizeof(float32_t)
    add x1, x1, #64     // src += 8 * 8 * sizeof(int8_t)
    add x11, x11, #32    // sum += 8 * sizeof(float32_t)
    add x12, x12, #32   // scale += 8 * sizeof(float32_t)
    b TILE_8

TILE_4:
    cmp x6, #4
    blt TILE_2
    mov x14, x4       // dst_step
    lsr x15, x4, #2   // src_step = dst_step / 4
    sub x14, x14, #64
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
    // mask
    movi v10.16b, #15
LoopSz_TILE_4:
    // src    : 2 x [2 x 8] : v4-5
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 2 x 4 x [4] : v16-23
    //ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x25], #64    // weight
    ld1 {v8.16b, v9.16b}, [x25], #32    // weight
    // int4 to int8: v0, v1, v2, v3
    ushr v0.16b, v8.16b, #4
    and v1.16b, v8.16b, v10.16b
    ushr v2.16b, v9.16b, #4
    and v3.16b, v9.16b, v10.16b
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
    add x7, x7, x13
    sub x27, x27, #1

    trn1 v24.2d, v16.2d, v17.2d // batch:0 oc:0-3
    trn1 v25.2d, v18.2d, v19.2d // batch:0 oc:4-7
    trn2 v26.2d, v16.2d, v17.2d // batch:1 oc:0-3
    trn2 v27.2d, v18.2d, v19.2d // batch:1 oc:4-7
    trn1 v28.2d, v20.2d, v21.2d // batch:2 oc:0-3
    trn1 v29.2d, v22.2d, v23.2d // batch:2 oc:4-7
    trn2 v30.2d, v20.2d, v21.2d // batch:3 oc:0-3
    trn2 v31.2d, v22.2d, v23.2d // batch:3 oc:4-7
    Int32ToFloat v24, v25, v26, v27
    Int32ToFloat v28, v29, v30, v31
    // using float scale dequant for precison
    ld1 {v5.4s}, [x23]  // scales
    MulScale v24, v25, v26, v27, v5, 0, 1
    MulScale v28, v29, v30, v31, v5, 2, 3
Tile4Dequant:
    ld1 {v0.4s, v1.4s}, [x19], #32  // alpha
    ld1 {v2.4s, v3.4s}, [x20], #32  // zero
    ld1 {v8.4s, v9.4s}, [x21], #32  // bias
    ld1 {v6.4s}, [x22]  // sums
    // alpha * cusum + (zero * sums) + bias
    Dequant v24, v0, v2, v8, v6, 0 // Batch0
    Dequant v25, v1, v3, v9, v6, 0
    Dequant v26, v0, v2, v8, v6, 1 // Batch1
    Dequant v27, v1, v3, v9, v6, 1
    Dequant v28, v0, v2, v8, v6, 2 // Batch2
    Dequant v29, v1, v3, v9, v6, 2
    Dequant v30, v0, v2, v8, v6, 3 // Batch3
    Dequant v31, v1, v3, v9, v6, 3
    st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x28], #64
    st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x28], x14
    cmp x27, #1
    bge LoopDz_TILE_4
Tile4End:
    sub x6, x6, #4      // bach -= 4
    add x0, x0, #128     // dst += 4 * 8 * sizeof(float32_t)
    add x1, x1, #32     // src += 4 * 8 * sizeof(int8_t)
    add x11, x11, #16    // sum += 4 * sizeof(float32_t)
    add x12, x12, #16   // scale += 4 * sizeof(float32_t)
    b TILE_4

TILE_2:
    cmp x6, #2
    blt TILE_1
    mov x14, x4       // dst_step
    lsr x15, x4, #2   // src_step = dst_step / 4
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
    // mask
    movi v14.16b, #15
LoopSz_TILE_2:
    // src    : 1 x [2 x 8] : v4
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 1 x 4 x [4] : v16-19
    //ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x25], #64    // weight
    ld1 {v8.16b, v9.16b}, [x25], #32    // weight
    // int4 to int8: v0, v1, v2, v3
    ushr v0.16b, v8.16b, #4
    and v1.16b, v8.16b, v14.16b
    ushr v2.16b, v9.16b, #4
    and v3.16b, v9.16b, v14.16b
    ld1 {v4.16b}, [x24], x15   // src
    .inst 0x4e80a490 // smmla v16.4s, v4.16b, v0.16b
    .inst 0x4e81a491 // smmla v17.4s, v4.16b, v1.16b
    .inst 0x4e82a492 // smmla v18.4s, v4.16b, v2.16b
    .inst 0x4e83a493 // smmla v19.4s, v4.16b, v3.16b
    subs x26, x26, #1
    bne LoopSz_TILE_2

LoopSzEnd_TILE_2:
    add x7, x7, x13
    sub x27, x27, #1
    trn1 v20.2d, v16.2d, v17.2d
    trn1 v21.2d, v18.2d, v19.2d
    trn2 v22.2d, v16.2d, v17.2d
    trn2 v23.2d, v18.2d, v19.2d
    Int32ToFloat v20, v21, v22, v23
    // using float scale dequant for precison
    ld1 {v5.d}[0], [x23]  // scales
    fmul v20.4s, v20.4s, v5.s[0]
    fmul v21.4s, v21.4s, v5.s[0]
    fmul v22.4s, v22.4s, v5.s[1]
    fmul v23.4s, v23.4s, v5.s[1]
Tile2Dequant:
    ld1 {v0.4s, v1.4s}, [x19], #32  // alpha
    ld1 {v2.4s, v3.4s}, [x20], #32  // zero
    ld1 {v8.4s, v9.4s}, [x21], #32  // bias
    ld1 {v10.d}[0], [x22]  // sums
    // alpha * sum + (zero * sumx) + bias
    Dequant v20, v0, v2, v8, v10, 0
    Dequant v21, v1, v3, v9, v10, 0
    Dequant v22, v0, v2, v8, v10, 1
    Dequant v23, v1, v3, v9, v10, 1
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x28], x14
    cmp x27, #1
    bge LoopDz_TILE_2
Tile2End:
    sub x6, x6, #2      // batch -= 2
    add x0, x0, #64     // dst += 2 * 8 * sizeof(float32_t)
    add x1, x1, #16     // dst += 2 * 8 * sizeof(int8_t)
    add x11, x11, #8    // sum += 2 * sizeof(float32_t)
    add x12, x12, #8    // scale += 2 * sizeof(float32_t)
    b TILE_2

TILE_1:
    cmp x6, #1
    blt End
    mov x14, x4       // dst_step
    lsr x15, x4, #2   // src_step = dst_step / 4, sizeof(float32_t)/4=sizeof(int8_t)
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
    // mask
    movi v14.16b, #15

LoopSz_TILE_1:
    // src    : 1 x [1 x 8] : v4
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 1 x 4 x [2] : v16-v19
    //ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x25], #64    // weight
    ld1 {v8.16b, v9.16b}, [x25], #32    // weight
    // int4 to int8: v0, v1, v2, v3
    ushr v0.16b, v8.16b, #4
    and v1.16b, v8.16b, v14.16b
    ushr v2.16b, v9.16b, #4
    and v3.16b, v9.16b, v14.16b
    ld1 {v4.8b}, [x24], x15   // src
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
    ld1 {v4.s}[0], [x23]  // scales
    fmul v20.4s, v20.4s, v4.s[0]
    fmul v21.4s, v21.4s, v4.s[0]
Tile1Dequant:
    ld1 {v0.4s, v1.4s}, [x19], #32  // alpha
    ld1 {v2.4s, v3.4s}, [x20], #32  // zero
    ld1 {v12.4s, v13.4s}, [x21], #32  // bias
    ld1 {v6.s}[0], [x22]  // sums
    // alpha * sum + (zero * sumx) + bias
    fmla v12.4s, v20.4s, v0.4s
    fmla v13.4s, v21.4s, v1.4s
    fmla v12.4s, v2.4s, v6.s[0]
    fmla v13.4s, v3.4s, v6.s[0]
    st1 {v12.4s, v13.4s}, [x28], x14
    cmp x27, #1
    bge LoopDz_TILE_1
Tile1End:
    sub x6, x6, #1      // batch -= 1
    add x0, x0, #32     // dst += 1 * 8 * sizeof(float32_t)
    add x1, x1, #8      // dst += 1 * 8 * sizeof(int8_t)
    add x11, x11, #4   // sum += 1 * sizeof(float32_t)
    add x12, x12, #4   // scale += 1 * sizeof(float32_t)
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