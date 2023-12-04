//
//  MNNGemmHybridInt8FP32_smmla.S
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

asm_function MNNGemmHybridInt8FP32_smmla

//struct QuanPostTreatParameters {
//    const float* scale;
//    const int32_t* bias;
//    int32_t maxValue;
//    int32_t minValue;
//    int32_t useInt8;
//};

//void MNNGemmHybridInt8FP32_smmla(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realSize, float** param); 


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

TILE_4:
    cmp x6, #4
    blt TILE_2
    mov x14, x4       // dst_step
    lsr x15, x4, #2   // src_step = dst_step / 4
    sub x14, x14, #64
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
LoopSz_TILE_4:
    // src    : 2 x [2 x 8] : v4-5
    // weight : 4 x [2 x 8] : v0-3
    // dst    : 2 x 4 x [4] : v16-23
    ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x25], #64    // weight
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
    st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x17], #64
    st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x17], x14
    cmp x16, #1
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
    add x18, x18, x13
    sub x16, x16, #1
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
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x17], x14
    cmp x16, #1
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
    add x18, x18, x13
    sub x16, x16, #1
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
    ld1 {v10.4s, v11.4s}, [x21], #32  // bias
    ld1 {v8.s}[0], [x22]  // sums
    // alpha * sum + (zero * sumx) + bias
    fmla v10.4s, v20.4s, v0.4s
    fmla v11.4s, v21.4s, v1.4s
    fmla v10.4s, v2.4s, v8.s[0]
    fmla v11.4s, v3.4s, v8.s[0]
    st1 {v10.4s, v11.4s}, [x17], x14
    cmp x16, #1
    bge LoopDz_TILE_1
Tile1End:
    sub x6, x6, #1      // batch -= 1
    add x0, x0, #32     // dst += 1 * 8 * sizeof(float32_t)
    add x1, x1, #8      // src += 1 * 8 * sizeof(int8_t)
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