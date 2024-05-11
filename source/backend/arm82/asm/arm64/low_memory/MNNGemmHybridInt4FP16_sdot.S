//
//  MNNGemmHybridInt4FP16_sdot.S
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

asm_function MNNGemmHybridInt4FP16_sdot

//struct QuanPostTreatParameters {
//    const float* scale;
//    const int32_t* bias;
//    int32_t maxValue;
//    int32_t minValue;
//    int32_t useInt8;
//};

//void MNNGemmHybridInt4_sdot(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realSize, float** param);


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
ldr x14, [x7, #40]
Start:
lsl x13, x3, #5 // x13 = src_depth_quad * UNIT * UNIT_SRC / 2(int4) = src_depth_quad * 32  = src_depth_quad << 5
ld1 {v6.16b, v7.16b}, [x14]
// mask
movi v14.16b, #15
TILE_4:
    cmp x6, #4
    blt TILE_1
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
    dup v24.4s, wzr
    dup v25.4s, wzr
    dup v26.4s, wzr
    dup v27.4s, wzr
    dup v28.4s, wzr
    dup v29.4s, wzr
    dup v30.4s, wzr
    dup v31.4s, wzr
LoopSz_TILE_4:
    // src    : 2 x [2 x 8] : v4-5
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 2 x 4 x [4] : v16-23
    ld1 {v8.16b, v9.16b}, [x25], #32    // weight
    ld1 {v4.16b, v5.16b}, [x24], x15   // src
    // int4 to int8: v0, v1, v2, v3
    ushr v0.16b, v8.16b, #4
    and v1.16b, v8.16b, v14.16b
    ushr v2.16b, v9.16b, #4
    and v3.16b, v9.16b, v14.16b

    mov v10.d[0], v4.d[1]
    mov v10.d[1], v4.d[0]
    mov v11.d[1], v5.d[0]
    mov v11.d[0], v5.d[1]
    .inst 0x4e809490 // sdot v16.4s, v4.16b, v0.16b  // (0,0)x2 (1,1)x2
    .inst 0x4e809558 // sdot v24.4s, v10.16b, v0.16b // (1,0)x2 (0,1)x2
    .inst 0x4e819491 // sdot v17.4s, v4.16b, v1.16b  // (0,2) (1,3)
    .inst 0x4e819559 // sdot v25.4s, v10.16b, v1.16b // (1,2) (0,3)
    .inst 0x4e829492 // sdot v18.4s, v4.16b, v2.16b
    .inst 0x4e82955a // sdot v26.4s, v10.16b, v2.16b
    .inst 0x4e839493 // sdot v19.4s, v4.16b, v3.16b
    .inst 0x4e83955b // sdot v27.4s, v10.16b, v3.16b
    .inst 0x4e8094b4 // sdot v20.4s, v5.16b, v0.16b
    .inst 0x4e80957c // sdot v28.4s, v11.16b, v0.16b
    .inst 0x4e8194b5 // sdot v21.4s, v5.16b, v1.16b
    .inst 0x4e81957d // sdot v29.4s, v11.16b, v1.16b
    .inst 0x4e8294b6 // sdot v22.4s, v5.16b, v2.16b
    .inst 0x4e82957e // sdot v30.4s, v11.16b, v2.16b
    .inst 0x4e8394b7 // sdot v23.4s, v5.16b, v3.16b
    .inst 0x4e83957f // sdot v31.4s, v11.16b, v3.16b

    subs x26, x26, #1
    bne LoopSz_TILE_4

    addp v16.4s, v16.4s, v24.4s // (batch,oc)(0,0)(1,1)(1,0)(0,1)
    addp v17.4s, v17.4s, v25.4s // (0,2)(1,3)(1,2)(0,3)
    addp v18.4s, v18.4s, v26.4s // (0,4)(1,5)(1,4)(0,5)
    addp v19.4s, v19.4s, v27.4s // (0,6)(1,7)(1,6)(0,7)
    addp v20.4s, v20.4s, v28.4s
    addp v21.4s, v21.4s, v29.4s
    addp v22.4s, v22.4s, v30.4s
    addp v23.4s, v23.4s, v31.4s
    tbl v24.16b, {v16.16b, v17.16b}, v6.16b // batch=0,oc=0-3
    tbl v25.16b, {v16.16b, v17.16b}, v7.16b // batch=1,oc=0-3
    tbl v26.16b, {v18.16b, v19.16b}, v6.16b // batch=0,oc=4-7
    tbl v27.16b, {v18.16b, v19.16b}, v7.16b // batch=1,oc=4-7
    tbl v28.16b, {v20.16b, v21.16b}, v6.16b
    tbl v29.16b, {v20.16b, v21.16b}, v7.16b
    tbl v30.16b, {v22.16b, v23.16b}, v6.16b
    tbl v31.16b, {v22.16b, v23.16b}, v7.16b

LoopSzEnd_TILE_4:
    add x7, x7, x13
    sub x27, x27, #1
    Int32ToFloat v24, v25, v26, v27
    Int32ToFloat v28, v29, v30, v31
    // using float scale dequant for precison
    ld1 {v1.d}[0], [x23]  // scales 4 batch
    ld1 {v2.8h}, [x19], #16  // alpha

    fcvtl v3.4s, v2.4h // oc:0-3
    fcvtl2 v4.4s, v2.8h // oc:4-7
    fcvtl v5.4s, v1.4h // scales: 4 batch

    MulScale v24, v26, v25, v27, v5, 0, 1, v3, v4
    MulScale v28, v30, v29, v31, v5, 2, 3, v3, v4
    Float32ToHalf v24, v26, v25, v27, v10, v11
    Float32ToHalf v28, v30, v29, v31, v12, v13
Tile4Dequant:
    ld1 {v1.8h}, [x20], #16  // zero
    ld1 {v2.8h}, [x21], #16  // bias
    ld1 {v3.d}[0], [x22]  // sums
    // sum + (zero * sumx) + bias
    Dequant v10, v1, v2, v3, 0
    Dequant v11, v1, v2, v3, 1
    Dequant v12, v1, v2, v3, 2
    Dequant v13, v1, v2, v3, 3
    st1 {v10.8h, v11.8h, v12.8h, v13.8h}, [x28], x4
    cmp x27, #1
    bge LoopDz_TILE_4
Tile4End:
    sub x6, x6, #4      // bach -= 4
    add x0, x0, #64     // dst += 4 * 8 * sizeof(float16_t)
    add x1, x1, #32     // src += 4 * 8 * sizeof(int8_t)
    add x11, x11, #8    // sum += 4 * sizeof(float16_t)
    add x12, x12, #8    // scale += 4 * sizeof(float16_t)
    b TILE_4

TILE_1:
    cmp x6, #1
    blt End
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
    movi v24.4s, #0
    movi v25.4s, #0
    movi v26.4s, #0
    movi v27.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
LoopSz_TILE_1:
    // src    : 1 x [1 x 8] : v4
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 1 x 4 x [2] : v16-v19
    ld1 {v8.16b, v9.16b}, [x25], #32    // weight
    ld1 {v4.8b}, [x24], x15   // src
    // int4 to int8: v0, v1, v2, v3
    ushr v0.16b, v8.16b, #4
    and v1.16b, v8.16b, v14.16b
    ushr v2.16b, v9.16b, #4
    and v3.16b, v9.16b, v14.16b

    mov v29.d[0], v4.d[1]
    mov v29.d[1], v4.d[0]

    .inst 0x4e809498 // sdot v24.4s, v4.16b, v0.16b // (0,0)x2 (1,1)x2
    .inst 0x4e8097b9 // sdot v25.4s, v29.16b, v0.16b // (1,0)x2 (0,1)x2
    .inst 0x4e81949a // sdot v26.4s, v4.16b, v1.16b // (0,2)x2 (1,3)x2
    .inst 0x4e8197bb // sdot v27.4s, v29.16b, v1.16b // (1,2)x2 (0,3)x2
    .inst 0x4e82948a // sdot v10.4s, v4.16b, v2.16b // (0,4)x2 (1,5)x2
    .inst 0x4e8297ab // sdot v11.4s, v29.16b, v2.16b // (1,4)x2 (0,5)x2
    .inst 0x4e83948c // sdot v12.4s, v4.16b, v3.16b // (0,6)x2 (1,7)x2
    .inst 0x4e8397ad // sdot v13.4s, v29.16b, v3.16b // (1,6)x2 (0,7)x2

    subs x26, x26, #1
    bne LoopSz_TILE_1
    addp v16.4s, v24.4s, v25.4s
    addp v17.4s, v26.4s, v27.4s
    addp v18.4s, v10.4s, v11.4s
    addp v19.4s, v12.4s, v13.4s

LoopSzEnd_TILE_1:
    add x7, x7, x13
    sub x27, x27, #1
    tbl v24.16b, {v16.16b, v17.16b}, v6.16b
    tbl v20.16b, {v18.16b, v19.16b}, v6.16b

    scvtf v24.4s, v24.4s
    scvtf v20.4s, v20.4s
    // using float scale dequant for precison
    ld1 {v4.h}[0], [x23]  // scales
    ld1 {v0.8h}, [x19], #16  // alpha
    fcvtl v5.4s, v4.4h
    fcvtl v22.4s, v0.4h
    fcvtl2 v21.4s, v0.8h
    fmul v24.4s, v24.4s, v5.s[0]
    fmul v20.4s, v20.4s, v5.s[0]
    fmul v24.4s, v24.4s, v22.4s
    fmul v20.4s, v20.4s, v21.4s
    fcvtn v17.4h,  v24.4s
    fcvtn2 v17.8h, v20.4s
Tile1Dequant:
    ld1 {v1.8h}, [x20], #16  // zero
    ld1 {v2.8h}, [x21], #16  // bias
    ld1 {v3.h}[0], [x22]  // sums
    // sum + (zero * sumx) + bias
    fadd v2.8h, v2.8h, v17.8h
    fmla v2.8h, v1.8h, v3.h[0]
    st1 {v2.8h}, [x28], x4
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