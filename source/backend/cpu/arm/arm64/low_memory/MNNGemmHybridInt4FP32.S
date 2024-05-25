//
//  MNNGemmHybridInt4_sdot.S
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

.macro MulScale d0, d1, d2, d3, s
    fmul \d0\().4s, \d0\().4s, \s\().s[0]
    fmul \d1\().4s, \d1\().4s, \s\().s[1]
    fmul \d2\().4s, \d2\().4s, \s\().s[2]
    fmul \d3\().4s, \d3\().4s, \s\().s[3]
.endm

.macro Dequant c0, a0, z0, b0, s0, idx
    fmul \c0\().4s, \c0\().4s, \a0\().4s
    fmla \c0\().4s, \z0\().4s, \s0\().s[\idx]
    fadd \c0\().4s, \c0\().4s, \b0\().4s
.endm

asm_function MNNGemmHybridInt4FP32

//struct QuanPostTreatParameters {
//    const float* scale;
//    const int32_t* bias;
//    int32_t maxValue;
//    int32_t minValue;
//    int32_t useInt8;
//};

//void MNNGemmHybridInt4FP32(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realSize, float** param);


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
lsl x13, x3, #3 // x13 = src_depth_quad * UNIT * UNIT_SRC / 2(int4) = src_depth_quad * 8  = src_depth_quad << 3

TILE_4:
    cmp x6, #4
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
LoopDz_TILE_4:
    // dequant info for batch
    mov x22, x11 // sums
    mov x23, x12 // scales
    mov x24, x1  // src
    mov x25, x7 // weight
    mov x26, x3  // src_depth_quad
    // init
    // batch=0,oc=0-3
    movi v10.4s, #0 //ic=0-3
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    // batch=1,oc=0-3
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    // batch=2,oc=0-3
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    // batch=3,oc=0-3
    movi v24.4s, #0
    movi v25.4s, #0
    movi v26.4s, #0
    movi v27.4s, #0
    // mask
    movi v14.16b, #15
    // offset
    movi v15.16b, #8
LoopSz_TILE_4:
    // src    : 4(batch) x [1 x 4] : v4
    // weight : 4(oc) x [1 x 4] : v0
    // dst    : 4 x 4 x [1] : v16-v19
    ld1 {v0.8b}, [x25], #8    // weight
    ld1 {v4.16b}, [x24], x15   // src
    // int4->int8
    ushr v8.16b, v0.16b, #4
    and v9.16b, v0.16b, v14.16b
    zip1 v0.16b, v8.16b, v9.16b

    Unit_TILE_4:
        sxtl v5.8h, v4.8b // src batch=0,1
        sxtl2 v6.8h, v4.16b // batch=2,3
        sxtl v1.8h, v0.8b // weight oc=0,1
        sxtl2 v2.8h, v0.16b // oc=2,3
        dup v28.2d, v1.d[0] // oc=0,0
        dup v29.2d, v1.d[1] // oc=1,1
        dup v30.2d, v2.d[0] // oc=2,2
        dup v31.2d, v2.d[1] // oc=3,3
        // batch=0
        smlal v10.4s, v5.4h, v28.4h
        smlal v11.4s, v5.4h, v29.4h
        smlal v12.4s, v5.4h, v30.4h
        smlal v13.4s, v5.4h, v31.4h
        // batch=1
        smlal2 v16.4s, v5.8h, v28.8h
        smlal2 v17.4s, v5.8h, v29.8h
        smlal2 v18.4s, v5.8h, v30.8h
        smlal2 v19.4s, v5.8h, v31.8h
        // batch=2
        smlal v20.4s, v6.4h, v28.4h
        smlal v21.4s, v6.4h, v29.4h
        smlal v22.4s, v6.4h, v30.4h
        smlal v23.4s, v6.4h, v31.4h
        // batch=3
        smlal2 v24.4s, v6.8h, v28.8h
        smlal2 v25.4s, v6.8h, v29.8h
        smlal2 v26.4s, v6.8h, v30.8h
        smlal2 v27.4s, v6.8h, v31.8h
    // .inst 0x4f84e010 // sdot v16.4s, v0.16b, v4.4b[0] // batch0
    // .inst 0x4fa4e011 // sdot v17.4s, v0.16b, v4.4b[1] // batch1
    // .inst 0x4f84e812 // sdot v18.4s, v0.16b, v4.4b[2] // batch2
    // .inst 0x4fa4e813 // sdot v19.4s, v0.16b, v4.4b[3] // batch3

    subs x26, x26, #1
    bne LoopSz_TILE_4

LoopSzEnd_TILE_4:
    // add 4 ic
    addp v10.4s, v10.4s, v11.4s
    addp v12.4s, v12.4s, v13.4s
    addp v16.4s, v16.4s, v17.4s
    addp v18.4s, v18.4s, v19.4s
    addp v20.4s, v20.4s, v21.4s
    addp v22.4s, v22.4s, v23.4s
    addp v24.4s, v24.4s, v25.4s
    addp v26.4s, v26.4s, v27.4s

    addp v10.4s, v10.4s, v12.4s // batch=0,oc=0-3
    addp v11.4s, v16.4s, v18.4s
    addp v12.4s, v20.4s, v22.4s
    addp v13.4s, v24.4s, v26.4s

    add x7, x7, x13
    sub x27, x27, #1
    Int32ToFloat v10, v11, v12, v13
    // Int32ToFloat v20, v21, v22, v23
    // using float scale dequant for precison
    ld1 {v5.4s}, [x23]  // scales, 4 batch,so 4 scale

    MulScale v10, v11, v12, v13, v5

Tile4Dequant:
    ld1 {v0.4s}, [x19], #16  // alpha
    ld1 {v1.4s}, [x20], #16  // zero
    ld1 {v2.4s}, [x21], #16  // bias
    ld1 {v3.4s}, [x22]  // sums
    // alpha * sum + (zero * sums) + bias
    Dequant v10, v0, v1, v2, v3, 0
    Dequant v11, v0, v1, v2, v3, 1
    Dequant v12, v0, v1, v2, v3, 2
    Dequant v13, v0, v1, v2, v3, 3
    st1 {v10.4s, v11.4s, v12.4s, v13.4s}, [x28], x14
    cmp x27, #1
    bge LoopDz_TILE_4
Tile4End:
    sub x6, x6, #4      // bach -= 4
    add x0, x0, #64     // dst += 4 * 4 * sizeof(float32_t)
    add x1, x1, #16     // src += 4 * 4 * sizeof(int8_t)
    add x11, x11, #16    // sum += 4 * sizeof(float32_t)
    add x12, x12, #16    // scale += 4 * sizeof(float32_t)
    b TILE_4

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
    // batch=0,oc=0-3
    movi v10.4s, #0 //ic=0-3
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    // mask
    movi v14.16b, #15
    // offset
    movi v15.16b, #8
LoopSz_TILE_1:
    // src    : 1(batch) x [1 x 4] : v4
    // weight : 4(oc) x [1 x 4] : v0
    // dst    : 1 x 4 x [1] : v16
    ld1 {v0.8b}, [x25], #8    // weight pack*pack*0.5
    ld1 {v4.s}[0], [x24], x15   // src
    // int4->int8
    ushr v8.16b, v0.16b, #4
    and v9.16b, v0.16b, v14.16b
    zip1 v0.16b, v8.16b, v9.16b

    Unit_TILE_1:
        sxtl v5.8h, v4.8b // src batch=0
        sxtl v1.8h, v0.8b // weight oc=0,1
        sxtl2 v2.8h, v0.16b // oc=2,3
        dup v28.2d, v1.d[0] // oc=0,0
        dup v29.2d, v1.d[1] // oc=1,1
        dup v30.2d, v2.d[0] // oc=2,2
        dup v31.2d, v2.d[1] // oc=3,3
        // batch=0
        smlal v10.4s, v5.4h, v28.4h
        smlal v11.4s, v5.4h, v29.4h
        smlal v12.4s, v5.4h, v30.4h
        smlal v13.4s, v5.4h, v31.4h

    //.inst 0x4f84e010 // sdot v16.4s, v0.16b, v4.4b[0]

    subs x26, x26, #1
    bne LoopSz_TILE_1

LoopSzEnd_TILE_1:
    // add 4 ic
    addp v10.4s, v10.4s, v11.4s
    addp v12.4s, v12.4s, v13.4s
    addp v16.4s, v10.4s, v12.4s
    add x7, x7, x13
    sub x27, x27, #1
    scvtf v16.4s, v16.4s
    // using float scale dequant for precison
    ld1 {v4.s}[0], [x23]  // scales
    fmul v16.4s, v16.4s, v4.s[0]
Tile1Dequant:
    ld1 {v0.4s}, [x19], #16  // alpha
    ld1 {v1.4s}, [x20], #16  // zero
    ld1 {v2.4s}, [x21], #16  // bias
    ld1 {v3.s}[0], [x22]  // sums
    // alpha * sum + (zero * sumx) + bias
    fmla v2.4s, v0.4s, v16.4s
    fmla v2.4s, v1.4s, v3.s[0]
    st1 {v2.4s}, [x28], x14
    cmp x27, #1
    bge LoopDz_TILE_1
Tile1End:
    subs x6, x6, #1      // batch -= 1
    add x0, x0, #16     // dst += 1 * 4 * sizeof(float32_t)
    add x1, x1, #4      // src += 1 * 4 * sizeof(int8_t)
    add x11, x11, #4   // sum += 1 * sizeof(float32_t)
    add x12, x12, #4   // scale += 1 * sizeof(float32_t)
    bne TILE_1

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