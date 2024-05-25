//
//  MNNGemmHybridInt8_smmla.S
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

.macro Float32ToHalf s0, s1, s2, s3, d0, d1
    fcvtn \d0\().4h,  \s0\().4s
    fcvtn2 \d0\().8h, \s1\().4s
    fcvtn \d1\().4h,  \s2\().4s
    fcvtn2 \d1\().8h, \s3\().4s
.endm

.macro Dequant c0, z0, b0, s0, idx
    fmla \c0\().8h, \z0\().8h, \s0\().h[\idx]
    fadd \c0\().8h, \c0\().8h, \b0\().8h
.endm

asm_function MNNGemmHybridInt8FP16_smmla

//struct QuanPostTreatParameters {
//    const float* scale;
//    const int32_t* bias;
//    int32_t maxValue;
//    int32_t minValue;
//    int32_t useInt8;
//};

//void MNNGemmHybridInt8_smmla(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realSize, float** param); 


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
lsl x13, x3, #6 // x13 = src_depth_quad * UNIT * UNIT_SRC / 1(int8) = src_depth_quad * 64  = src_depth_quad << 6
cmp x6, #1
beq TILE_EQ_1

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
LoopSz_TILE_8:
    // src    : 2 x [2 x 8] : v4-5
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 2 x 4 x [4] : v16-23
    ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x25], #64    // weight
    ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x24], x15   // src
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
    sub x27, x27, #1
    Int32ToFloat v16, v17, v18, v19
    Int32ToFloat v20, v21, v22, v23
    Int32ToFloat v24, v25, v26, v27
    Int32ToFloat v28, v29, v30, v31
    // using float scale dequant for precison
    trn1 v8.2d,  v16.2d, v17.2d // batch=0,oc:0-3
    trn1 v9.2d,  v18.2d, v19.2d // batch=0,oc:4-7
    trn2 v10.2d, v16.2d, v17.2d // batch=1,oc:0-3
    trn2 v11.2d, v18.2d, v19.2d // batch=1,oc:4-7
    trn1 v12.2d, v20.2d, v21.2d // batch=2,oc:0-3
    trn1 v13.2d, v22.2d, v23.2d // batch=2,oc:4-7
    trn2 v14.2d, v20.2d, v21.2d // batch=3,oc:0-3
    trn2 v15.2d, v22.2d, v23.2d // batch=3,oc:4-7

    trn1 v0.2d, v24.2d, v25.2d // batch=4,oc:0-3
    trn1 v1.2d, v26.2d, v27.2d // batch=4,oc:4-7
    trn2 v2.2d, v24.2d, v25.2d // batch=5,oc:0-3
    trn2 v3.2d, v26.2d, v27.2d // batch=5,oc:4-7
    trn1 v4.2d, v28.2d, v29.2d // batch=6,oc:0-3
    trn1 v5.2d, v30.2d, v31.2d // batch=6,oc:4-7
    trn2 v6.2d, v28.2d, v29.2d // batch=7,oc:0-3
    trn2 v7.2d, v30.2d, v31.2d // batch=7,oc:4-7

    ld1 {v16.8h}, [x23]  // scales
    ld1 {v17.8h}, [x19], #16  // alpha
    fcvtl v18.4s, v17.4h // oc:0-3
    fcvtl2 v19.4s, v17.8h // oc:4-7
    fcvtl v28.4s, v16.4h // scales: batch 0,1,2,3
    fcvtl2 v29.4s, v16.8h // scales: batch 4,5,6,7

    MulScale v8, v9, v10, v11, v28, 0, 1, v18, v19
    MulScale v12, v13, v14, v15, v28, 2, 3, v18, v19
    Float32ToHalf v8, v9, v10, v11, v20, v21 // batch=0,1
    Float32ToHalf v12, v13, v14, v15, v22, v23 // batch=2,3

    MulScale v0, v1, v2, v3, v29, 0, 1, v18, v19
    MulScale v4, v5, v6, v7, v29, 2, 3, v18, v19
    Float32ToHalf v0, v1, v2, v3, v24, v25 // batch=4,5
    Float32ToHalf v4, v5, v6, v7, v26, v27 // batch=6,7

Tile8Dequant:
    ld1 {v1.8h}, [x20], #16  // zero
    ld1 {v2.8h}, [x21], #16  // bias
    ld1 {v3.8h}, [x22]  // sums
    // sum + (zero * sumx) + bias
    Dequant v20, v1, v2, v3, 0
    Dequant v21, v1, v2, v3, 1
    Dequant v22, v1, v2, v3, 2
    Dequant v23, v1, v2, v3, 3

    Dequant v24, v1, v2, v3, 4
    Dequant v25, v1, v2, v3, 5
    Dequant v26, v1, v2, v3, 6
    Dequant v27, v1, v2, v3, 7
    st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x28], #64
    st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x28], x14
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
LoopSz_TILE_4:
    // src    : 2 x [2 x 8] : v4-5
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 2 x 4 x [4] : v16-23
    ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x25], #64    // weight
    ld1 {v4.16b, v5.16b}, [x24], x15   // src
    .inst 0x4e80a490 // smmla v16.4s, v4.16b, v0.16b // batch=0,1, oc=0,1
    .inst 0x4e81a491 // smmla v17.4s, v4.16b, v1.16b // batch=0,1, oc=2,3
    .inst 0x4e82a492 // smmla v18.4s, v4.16b, v2.16b // batch=0,1, oc=4,5
    .inst 0x4e83a493 // smmla v19.4s, v4.16b, v3.16b // batch=0,1, oc=6,7
    .inst 0x4e80a4b4 // smmla v20.4s, v5.16b, v0.16b // batch=2,3, oc=0,1
    .inst 0x4e81a4b5 // smmla v21.4s, v5.16b, v1.16b // batch=2,3, oc=2,3
    .inst 0x4e82a4b6 // smmla v22.4s, v5.16b, v2.16b // batch=2,3, oc=4,5
    .inst 0x4e83a4b7 // smmla v23.4s, v5.16b, v3.16b // batch=2,3, oc=6,7
    subs x26, x26, #1
    bne LoopSz_TILE_4

LoopSzEnd_TILE_4:
    add x7, x7, x13
    sub x27, x27, #1
    Int32ToFloat v16, v17, v18, v19
    Int32ToFloat v20, v21, v22, v23
    // using float scale dequant for precison
    ld1 {v4.d}[0], [x23]  // scales
    ld1 {v31.8h}, [x19], #16  // alpha
    fcvtl v29.4s, v31.4h // oc:0-3
    fcvtl2 v30.4s, v31.8h // oc:4-7
    trn1 v24.2d, v16.2d, v17.2d // batch=0,oc:0-3
    trn1 v25.2d, v18.2d, v19.2d // batch=0,oc:4-7
    trn2 v26.2d, v16.2d, v17.2d // batch=1,oc:0-3
    trn2 v27.2d, v18.2d, v19.2d // batch=1,oc:4-7
    trn1 v28.2d, v20.2d, v21.2d // batch=2,oc:0-3
    trn1 v6.2d, v22.2d, v23.2d  // batch=2,oc:4-7
    trn2 v7.2d, v20.2d, v21.2d  // batch=3,oc:0-3
    trn2 v8.2d, v22.2d, v23.2d  // batch=3,oc:4-7

    fcvtl v5.4s, v4.4h // scales: 4 batch

    MulScale v24, v25, v26, v27, v5, 0, 1, v29, v30
    MulScale v28, v6, v7, v8, v5, 2, 3, v29, v30
    Float32ToHalf v24, v25, v26, v27, v12, v13
    Float32ToHalf v28, v6, v7, v8, v14, v15
Tile4Dequant:
    ld1 {v1.8h}, [x20], #16  // zero
    ld1 {v2.8h}, [x21], #16  // bias
    ld1 {v3.d}[0], [x22]  // sums
    // sum + (zero * sumx) + bias
    Dequant v12, v1, v2, v3, 0
    Dequant v13, v1, v2, v3, 1
    Dequant v14, v1, v2, v3, 2
    Dequant v15, v1, v2, v3, 3
    st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x28], x14
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
LoopSz_TILE_2:
    // src    : 1 x [2 x 8] : v4
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 1 x 4 x [4] : v16-19
    ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x25], #64    // weight
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
    uzp1 v13.2d, v16.2d, v17.2d
    uzp1 v14.2d, v18.2d, v19.2d
    uzp2 v15.2d, v16.2d, v17.2d
    uzp2 v16.2d, v18.2d, v19.2d
    Int32ToFloat v13, v14, v15, v16
    // using float scale dequant for precison
    ld1 {v4.s}[0], [x23]  // scales
    ld1 {v0.8h}, [x19], #16  // alpha
    fcvtl v5.4s, v4.4h
    fcvtl v20.4s, v0.4h
    fcvtl2 v21.4s, v0.8h
    MulScale v13, v14, v15, v16, v5, 0, 1, v20, v21
    fcvtn v11.4h,  v13.4s
    fcvtn2 v11.8h, v14.4s
    fcvtn v12.4h,  v15.4s
    fcvtn2 v12.8h, v16.4s
Tile2Dequant:
    //ld1 {v0.8h}, [x19], #16  // alpha
    ld1 {v1.8h}, [x20], #16  // zero
    ld1 {v2.8h}, [x21], #16  // bias
    ld1 {v3.s}[0], [x22]  // sums
    // alpha * sum + (zero * sumx) + bias
    Dequant v11, v1, v2, v3, 0
    Dequant v12, v1, v2, v3, 1
    st1 {v11.8h, v12.8h}, [x28], x14
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
    ld1 {v29.8h}, [x20], #16  // zero
    ld1 {v30.8h}, [x21], #16  // bias
    ld1 {v8.h}[0], [x22]  // sums
    // init
    dup v16.4s, wzr
    dup v17.4s, wzr
    dup v18.4s, wzr
    dup v19.4s, wzr
    fmla v30.8h, v29.8h, v8.h[0] // bias + zero * sum

LoopSz_TILE_1:
    // src    : 1 x [1 x 8] : v4
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 1 x 4 x [2] : v16-v19
    ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x25], #64    // weight
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
    uzp1 v22.4s, v16.4s, v17.4s
    uzp1 v23.4s, v18.4s, v19.4s
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s
    // using float scale dequant for precison
    ld1 {v4.h}[0], [x23]  // scales
    ld1 {v0.8h}, [x19], #16  // alpha
    fcvtl v5.4s, v4.4h
    fcvtl v20.4s, v0.4h
    fcvtl2 v21.4s, v0.8h

    fmul v22.4s, v22.4s, v5.s[0]
    fmul v23.4s, v23.4s, v5.s[0]
    fmul v22.4s, v22.4s, v20.4s
    fmul v23.4s, v23.4s, v21.4s
    fcvtn v17.4h,  v22.4s
    fcvtn2 v17.8h, v23.4s
Tile1Dequant:
    // sum + (zero * sumx) + bias
    fadd v30.8h, v30.8h, v17.8h
    st1 {v30.8h}, [x28], x14
    cmp x27, #1
    bge LoopDz_TILE_1
Tile1End:
    sub x6, x6, #1      // batch -= 1
    add x0, x0, #16     // dst += 1 * 8 * sizeof(float16_t)
    add x1, x1, #8      // dst += 1 * 8 * sizeof(int8_t)
    add x11, x11, #2   // sum += 1 * sizeof(float16_t)
    add x12, x12, #2   // scale += 1 * sizeof(float16_t)
    b TILE_1
b End
TILE_EQ_1:

    mov x14, x4       // dst_step
    lsr x15, x4, #1   // src_step = dst_step / 2
    mov x27, x5 // dst_depth_quad
    mov x28, x0 // dst
    mov x7, x2 // weight
    // dequant info
    mov x19, x8 // alpha
    mov x20, x9 // zero
    mov x21, x10 // bias
LoopDz:
    mov x22, x11 // sums
    mov x23, x12 // scales
    mov x24, x1  // src
    mov x25, x7 // weight
    mov x26, x3  // src_depth_quad
    ld1 {v29.8h}, [x20], #16  // zero
    ld1 {v30.8h}, [x21], #16  // bias
    ld1 {v8.h}[0], [x22]  // sums
    // init
    dup v14.4s, wzr
    dup v15.4s, wzr
    dup v16.4s, wzr
    dup v17.4s, wzr
    dup v18.4s, wzr
    dup v19.4s, wzr
    dup v20.4s, wzr
    dup v21.4s, wzr
    fmla v30.8h, v29.8h, v8.h[0] // bias + zero * sum


L2:
cmp x26, #2
blt L1
LoopSz_2:
    ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x25], #64    // weight
    ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x25], #64
    ld1 {v8.16b}, [x24], #16  // src
    sub x26, x26, #2

    .inst 0x4e80a50e // smmla v14.4s, v8.16b, v0.16b // (N=0,OC=0) (N=0,OC=1) () ()
    .inst 0x4e81a50f // smmla v15.4s, v8.16b, v1.16b // (N=0,OC=2) (N=0,OC=3) () ()
    .inst 0x4e82a510 // smmla v16.4s, v8.16b, v2.16b // (N=0,OC=4) (N=0,OC=5) () ()
    .inst 0x4e83a511 // smmla v17.4s, v8.16b, v3.16b // (N=0,OC=6) (N=0,OC=7) () ()
    .inst 0x4e84a512 // smmla v18.4s, v8.16b, v4.16b
    .inst 0x4e85a513 // smmla v19.4s, v8.16b, v5.16b
    .inst 0x4e86a514 // smmla v20.4s, v8.16b, v6.16b
    .inst 0x4e87a515 // smmla v21.4s, v8.16b, v7.16b
    cmp x26, #2
    bge LoopSz_2
L1:
cmp x26, #1
blt LoopSzEnd
LoopSz_1:
    // src    : 1 x [1 x 8] : v4
    // weight : 4 x [2 x 8] : v0-3
    ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x25], #64    // weight
    ld1 {v4.8b}, [x24], x15   // src
    .inst 0x4e80a48e // smmla v14.4s, v4.16b, v0.16b
    .inst 0x4e81a48f // smmla v15.4s, v4.16b, v1.16b
    .inst 0x4e82a490 // smmla v16.4s, v4.16b, v2.16b
    .inst 0x4e83a491 // smmla v17.4s, v4.16b, v3.16b

    subs x26, x26, #1
    bne LoopSz_1

LoopSzEnd:
    add x7, x7, x13
    sub x27, x27, #1

    trn1 v26.2d, v14.2d, v15.2d
    trn1 v27.2d, v16.2d, v17.2d
    trn2 v28.2d, v18.2d, v19.2d
    trn2 v29.2d, v20.2d, v21.2d
    add v26.4s, v26.4s, v28.4s
    add v27.4s, v27.4s, v29.4s
    scvtf v26.4s, v26.4s
    scvtf v27.4s, v27.4s
    // using float scale dequant for precison
    ld1 {v4.h}[0], [x23]  // scales
    ld1 {v0.8h}, [x19], #16  // alpha
    fcvtl v5.4s, v4.4h
    fcvtl v20.4s, v0.4h
    fcvtl2 v21.4s, v0.8h

    fmul v26.4s, v26.4s, v5.s[0]
    fmul v27.4s, v27.4s, v5.s[0]
    fmul v26.4s, v26.4s, v20.4s
    fmul v27.4s, v27.4s, v21.4s
    fcvtn v17.4h,  v26.4s
    fcvtn2 v17.8h, v27.4s
Int8ToFP16:
    // sum + (zero * sumx) + bias
    fadd v30.8h, v30.8h, v17.8h
    st1 {v30.8h}, [x28], x14
    cmp x27, #1
    bge LoopDz

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