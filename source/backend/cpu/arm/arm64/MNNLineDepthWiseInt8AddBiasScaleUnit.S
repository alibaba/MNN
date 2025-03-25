//
//  MNNLineDepthWiseInt8AddBiasScaleUnit.S
//  MNN
//
//  Created by MNN on 2019/06/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNLineDepthWiseInt8AddBiasScaleUnit

//void MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters,
//                                          size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step,
//                                          size_t dilateY_step) {


//struct QuanPostTreatParameters {
//    const float* scale;
//    const int32_t* bias;
//    int32_t maxValue;
//    int32_t minValue;
//    float roundValuePos = 0.5f;
//    float roundValueNeg = -0.5f;
//};

// Auto Load:
// x0: dst*, x1: src*, x2: weight*, x3: parameters*
// x4: width, x5: src_w_step, x6: fw, x7: fh
// Load from sp
// x8: dilateX_step, x9: dilateY_step

ldr x8, [sp, #0]
ldr x9, [sp, #8]

str d14, [sp, #(-16 * 10)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x27, x28, [sp, #(16 * 4)]
stp x25, x26, [sp, #(16 * 5)]
stp x23, x24, [sp, #(16 * 6)]
stp x21, x22, [sp, #(16 * 7)]
stp x19, x20, [sp, #(16 * 8)]

ldr x19, [x3, #0]
ldr w11, [x3, #16]
dup v30.16b, w11 // max
ldr w11, [x3, #20]
dup v31.16b, w11 // min
ldr x3, [x3, #72]

mul x10, x6, x8
sub x9, x9, x10

L4:
cmp x4, #4
blt L2

// load scale
ld1 {v26.4s, v27.4s, v28.4s, v29.4s}, [x19]

mov x12, #4
mul x12, x5, x12
cmp x5, #16
bne L4Loop_NORMAL

L4Loop_NOSTRIDE:
    // load bias
    ld1 {v10.4s, v11.4s, v12.4s, v13.4s}, [x3]
    ld1 {v14.4s, v15.4s, v16.4s, v17.4s}, [x3]
    ld1 {v18.4s, v19.4s, v20.4s, v21.4s}, [x3]
    ld1 {v22.4s, v23.4s, v24.4s, v25.4s}, [x3]

    mov x13, x1
    mov x14, x2
    mov x10, x7
    L4LoopH_NOSTRIDE:
        mov x11, x6
        L4LoopW_NOSTRIDE:
            ld1 {v9.4s}, [x2], #16         // weight
            ld1 {v5.4s, v6.4s, v7.4s, v8.4s}, [x1], #64 // feature

            sxtl  v0.8h, v9.8b
            sxtl2 v9.8h, v9.16b

            sxtl  v1.8h, v5.8b
            sxtl2 v2.8h, v5.16b
            sxtl  v3.8h, v6.8b
            sxtl2 v4.8h, v6.16b
            sxtl  v5.8h, v7.8b
            sxtl2 v6.8h, v7.16b
            sxtl  v7.8h, v8.8b
            sxtl2 v8.8h, v8.16b

            smlal  v10.4s,  v0.4h, v1.4h
            smlal2 v11.4s,  v0.8h, v1.8h
            smlal  v12.4s,  v9.4h, v2.4h
            smlal2 v13.4s,  v9.8h, v2.8h
            smlal  v14.4s,  v0.4h, v3.4h
            smlal2 v15.4s,  v0.8h, v3.8h
            smlal  v16.4s,  v9.4h, v4.4h
            smlal2 v17.4s,  v9.8h, v4.8h
            smlal  v18.4s,  v0.4h, v5.4h
            smlal2 v19.4s,  v0.8h, v5.8h
            smlal  v20.4s,  v9.4h, v6.4h
            smlal2 v21.4s,  v9.8h, v6.8h
            smlal  v22.4s,  v0.4h, v7.4h
            smlal2 v23.4s,  v0.8h, v7.8h
            smlal  v24.4s,  v9.4h, v8.4h
            smlal2 v25.4s,  v9.8h, v8.8h

            sub x1, x1, x12
            add x1, x1, x8
            subs x11, x11, #1
            bne L4LoopW_NOSTRIDE
        L4LoopWEnd_NOSTRIDE:
        subs x10, x10, #1
        add x1, x1, x9
    bne L4LoopH_NOSTRIDE

    scvtf v10.4s, v10.4s
    scvtf v11.4s, v11.4s
    scvtf v12.4s, v12.4s
    scvtf v13.4s, v13.4s
    scvtf v14.4s, v14.4s
    scvtf v15.4s, v15.4s
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s
    scvtf v24.4s, v24.4s
    scvtf v25.4s, v25.4s

    fmul v10.4s, v10.4s, v26.4s
    fmul v11.4s, v11.4s, v27.4s
    fmul v12.4s, v12.4s, v28.4s
    fmul v13.4s, v13.4s, v29.4s
    fmul v14.4s, v14.4s, v26.4s
    fmul v15.4s, v15.4s, v27.4s
    fmul v16.4s, v16.4s, v28.4s
    fmul v17.4s, v17.4s, v29.4s
    fmul v18.4s, v18.4s, v26.4s
    fmul v19.4s, v19.4s, v27.4s
    fmul v20.4s, v20.4s, v28.4s
    fmul v21.4s, v21.4s, v29.4s
    fmul v22.4s, v22.4s, v26.4s
    fmul v23.4s, v23.4s, v27.4s
    fmul v24.4s, v24.4s, v28.4s
    fmul v25.4s, v25.4s, v29.4s

    fcvtas v10.4s, v10.4s
    fcvtas v11.4s, v11.4s
    fcvtas v12.4s, v12.4s
    fcvtas v13.4s, v13.4s
    fcvtas v14.4s, v14.4s
    fcvtas v15.4s, v15.4s
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s
    fcvtas v21.4s, v21.4s
    fcvtas v22.4s, v22.4s
    fcvtas v23.4s, v23.4s
    fcvtas v24.4s, v24.4s
    fcvtas v25.4s, v25.4s
    /*
    sqrshrn  v10.4h, v10.4s, #14
    sqrshrn2 v10.8h, v11.4s, #14
    sqrshrn  v12.4h, v12.4s, #14
    sqrshrn2 v12.8h, v13.4s, #14
    sqrshrn  v14.4h, v14.4s, #14
    sqrshrn2 v14.8h, v15.4s, #14
    sqrshrn  v16.4h, v16.4s, #14
    sqrshrn2 v16.8h, v17.4s, #14
    sqrshrn  v18.4h, v18.4s, #14
    sqrshrn2 v18.8h, v19.4s, #14
    sqrshrn  v20.4h, v20.4s, #14
    sqrshrn2 v20.8h, v21.4s, #14
    sqrshrn  v22.4h, v22.4s, #14
    sqrshrn2 v22.8h, v23.4s, #14
    sqrshrn  v24.4h, v24.4s, #14
    sqrshrn2 v24.8h, v25.4s, #14
    */

    sqxtn  v10.4h, v10.4s
    sqxtn2 v10.8h, v11.4s
    sqxtn  v12.4h, v12.4s
    sqxtn2 v12.8h, v13.4s
    sqxtn  v14.4h, v14.4s
    sqxtn2 v14.8h, v15.4s
    sqxtn  v16.4h, v16.4s
    sqxtn2 v16.8h, v17.4s
    sqxtn  v18.4h, v18.4s
    sqxtn2 v18.8h, v19.4s
    sqxtn  v20.4h, v20.4s
    sqxtn2 v20.8h, v21.4s
    sqxtn  v22.4h, v22.4s
    sqxtn2 v22.8h, v23.4s
    sqxtn  v24.4h, v24.4s
    sqxtn2 v24.8h, v25.4s

    sqxtn   v0.8b,  v10.8h
    sqxtn2  v0.16b, v12.8h
    sqxtn   v1.8b,  v14.8h
    sqxtn2  v1.16b, v16.8h
    sqxtn   v2.8b,  v18.8h
    sqxtn2  v2.16b, v20.8h
    sqxtn   v3.8b,  v22.8h
    sqxtn2  v3.16b, v24.8h

    smin v0.16b, v0.16b, v30.16b
    smax v0.16b, v0.16b, v31.16b
    smin v1.16b, v1.16b, v30.16b
    smax v1.16b, v1.16b, v31.16b
    smin v2.16b, v2.16b, v30.16b
    smax v2.16b, v2.16b, v31.16b
    smin v3.16b, v3.16b, v30.16b
    smax v3.16b, v3.16b, v31.16b

    st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64

    mov x2, x14
    add x1, x13, x12
    sub x4, x4, #4
    cmp x4, #4
    bge L4Loop_NOSTRIDE

b L4End

L4Loop_NORMAL:
    // load bias
    ld1 {v10.4s, v11.4s, v12.4s, v13.4s}, [x3]
    ld1 {v14.4s, v15.4s, v16.4s, v17.4s}, [x3]
    ld1 {v18.4s, v19.4s, v20.4s, v21.4s}, [x3]
    ld1 {v22.4s, v23.4s, v24.4s, v25.4s}, [x3]

    mov x13, x1
    mov x14, x2
    mov x10, x7
    L4LoopH_NORMAL:
        mov x11, x6
        L4LoopW_NORMAL:
            // weight
            ld1 {v9.4s}, [x2], #16
            // feature input
            ld1 {v5.4s}, [x1], x5
            ld1 {v6.4s}, [x1], x5
            ld1 {v7.4s}, [x1], x5
            ld1 {v8.4s}, [x1], x5
            sxtl  v0.8h, v9.8b
            sxtl2 v9.8h, v9.16b

            sxtl  v1.8h, v5.8b
            sxtl2 v2.8h, v5.16b
            sxtl  v3.8h, v6.8b
            sxtl2 v4.8h, v6.16b
            sxtl  v5.8h, v7.8b
            sxtl2 v6.8h, v7.16b
            sxtl  v7.8h, v8.8b
            sxtl2 v8.8h, v8.16b

            smlal  v10.4s,  v0.4h, v1.4h
            smlal2 v11.4s,  v0.8h, v1.8h
            smlal  v12.4s,  v9.4h, v2.4h
            smlal2 v13.4s,  v9.8h, v2.8h
            smlal  v14.4s,  v0.4h, v3.4h
            smlal2 v15.4s,  v0.8h, v3.8h
            smlal  v16.4s,  v9.4h, v4.4h
            smlal2 v17.4s,  v9.8h, v4.8h
            smlal  v18.4s,  v0.4h, v5.4h
            smlal2 v19.4s,  v0.8h, v5.8h
            smlal  v20.4s,  v9.4h, v6.4h
            smlal2 v21.4s,  v9.8h, v6.8h
            smlal  v22.4s,  v0.4h, v7.4h
            smlal2 v23.4s,  v0.8h, v7.8h
            smlal  v24.4s,  v9.4h, v8.4h
            smlal2 v25.4s,  v9.8h, v8.8h

            sub x1, x1, x12
            add x1, x1, x8
            subs x11, x11, #1
            bne L4LoopW_NORMAL
        L4LoopWEnd_NORMAL:
        subs x10, x10, #1
        add x1, x1, x9
        bne L4LoopH_NORMAL

    scvtf v10.4s, v10.4s
    scvtf v11.4s, v11.4s
    scvtf v12.4s, v12.4s
    scvtf v13.4s, v13.4s
    scvtf v14.4s, v14.4s
    scvtf v15.4s, v15.4s
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s
    scvtf v24.4s, v24.4s
    scvtf v25.4s, v25.4s

    fmul v10.4s, v10.4s, v26.4s
    fmul v11.4s, v11.4s, v27.4s
    fmul v12.4s, v12.4s, v28.4s
    fmul v13.4s, v13.4s, v29.4s
    fmul v14.4s, v14.4s, v26.4s
    fmul v15.4s, v15.4s, v27.4s
    fmul v16.4s, v16.4s, v28.4s
    fmul v17.4s, v17.4s, v29.4s
    fmul v18.4s, v18.4s, v26.4s
    fmul v19.4s, v19.4s, v27.4s
    fmul v20.4s, v20.4s, v28.4s
    fmul v21.4s, v21.4s, v29.4s
    fmul v22.4s, v22.4s, v26.4s
    fmul v23.4s, v23.4s, v27.4s
    fmul v24.4s, v24.4s, v28.4s
    fmul v25.4s, v25.4s, v29.4s

    fcvtas v10.4s, v10.4s
    fcvtas v11.4s, v11.4s
    fcvtas v12.4s, v12.4s
    fcvtas v13.4s, v13.4s
    fcvtas v14.4s, v14.4s
    fcvtas v15.4s, v15.4s
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s
    fcvtas v21.4s, v21.4s
    fcvtas v22.4s, v22.4s
    fcvtas v23.4s, v23.4s
    fcvtas v24.4s, v24.4s
    fcvtas v25.4s, v25.4s
    /*
    sqrshrn  v10.4h, v10.4s, #14
    sqrshrn2 v10.8h, v11.4s, #14
    sqrshrn  v12.4h, v12.4s, #14
    sqrshrn2 v12.8h, v13.4s, #14
    sqrshrn  v14.4h, v14.4s, #14
    sqrshrn2 v14.8h, v15.4s, #14
    sqrshrn  v16.4h, v16.4s, #14
    sqrshrn2 v16.8h, v17.4s, #14
    sqrshrn  v18.4h, v18.4s, #14
    sqrshrn2 v18.8h, v19.4s, #14
    sqrshrn  v20.4h, v20.4s, #14
    sqrshrn2 v20.8h, v21.4s, #14
    sqrshrn  v22.4h, v22.4s, #14
    sqrshrn2 v22.8h, v23.4s, #14
    sqrshrn  v24.4h, v24.4s, #14
    sqrshrn2 v24.8h, v25.4s, #14
    */
    sqxtn  v10.4h, v10.4s
    sqxtn2 v10.8h, v11.4s
    sqxtn  v12.4h, v12.4s
    sqxtn2 v12.8h, v13.4s
    sqxtn  v14.4h, v14.4s
    sqxtn2 v14.8h, v15.4s
    sqxtn  v16.4h, v16.4s
    sqxtn2 v16.8h, v17.4s
    sqxtn  v18.4h, v18.4s
    sqxtn2 v18.8h, v19.4s
    sqxtn  v20.4h, v20.4s
    sqxtn2 v20.8h, v21.4s
    sqxtn  v22.4h, v22.4s
    sqxtn2 v22.8h, v23.4s
    sqxtn  v24.4h, v24.4s
    sqxtn2 v24.8h, v25.4s

    sqxtn   v0.8b,  v10.8h
    sqxtn2  v0.16b, v12.8h
    sqxtn   v1.8b,  v14.8h
    sqxtn2  v1.16b, v16.8h
    sqxtn   v2.8b,  v18.8h
    sqxtn2  v2.16b, v20.8h
    sqxtn   v3.8b,  v22.8h
    sqxtn2  v3.16b, v24.8h

    smin v0.16b, v0.16b, v30.16b
    smax v0.16b, v0.16b, v31.16b
    smin v1.16b, v1.16b, v30.16b
    smax v1.16b, v1.16b, v31.16b
    smin v2.16b, v2.16b, v30.16b
    smax v2.16b, v2.16b, v31.16b
    smin v3.16b, v3.16b, v30.16b
    smax v3.16b, v3.16b, v31.16b
    smax v7.16b, v7.16b, v31.16b

    st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64

    mov x2, x14
    add x1, x13, x12
    sub x4, x4, #4
    cmp x4, #4
    bge L4Loop_NORMAL

L4End:
///////
L2:
cmp x4, #2
blt L1

// load scale
ld1 {v22.4s, v23.4s, v24.4s, v25.4s}, [x19]

mov x12, #2
mul x12, x5, x12
cmp x5, #16
bne L2Loop_NORMAL

L2Loop_NOSTRIDE:
    // load bias and scale
    ld1 {v13.4s, v14.4s, v15.4s, v16.4s}, [x3]
    ld1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x3]

    mov x13, x1
    mov x14, x2
    mov x10, x7
    L2LoopH_NOSTRIDE:
        mov x11, x6
        L2LoopW_NOSTRIDE:
            ld1 {v12.4s}, [x2], #16         // weight
            ld1 {v8.4s, v9.4s}, [x1], #32 // feature

            sxtl  v10.8h, v12.8b
            sxtl2 v11.8h, v12.16b
            sxtl  v0.8h, v8.8b
            sxtl2 v1.8h, v8.16b
            sxtl  v2.8h, v9.8b
            sxtl2 v3.8h, v9.16b

            smlal  v13.4s,  v10.4h, v0.4h
            smlal2 v14.4s,  v10.8h, v0.8h
            smlal  v15.4s,  v11.4h, v1.4h
            smlal2 v16.4s,  v11.8h, v1.8h
            smlal  v17.4s,  v10.4h, v2.4h
            smlal2 v18.4s,  v10.8h, v2.8h
            smlal  v19.4s,  v11.4h, v3.4h
            smlal2 v20.4s,  v11.8h, v3.8h

            sub x1, x1, x12
            add x1, x1, x8
            subs x11, x11, #1
            bne L2LoopW_NOSTRIDE
        L2LoopWEnd_NOSTRIDE:
        subs x10, x10, #1
        add x1, x1, x9
        bne L2LoopH_NOSTRIDE

    scvtf v13.4s, v13.4s
    scvtf v14.4s, v14.4s
    scvtf v15.4s, v15.4s
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s

    fmul v13.4s, v13.4s, v22.4s
    fmul v14.4s, v14.4s, v23.4s
    fmul v15.4s, v15.4s, v24.4s
    fmul v16.4s, v16.4s, v25.4s
    fmul v17.4s, v17.4s, v22.4s
    fmul v18.4s, v18.4s, v23.4s
    fmul v19.4s, v19.4s, v24.4s
    fmul v20.4s, v20.4s, v25.4s

    fcvtas v13.4s, v13.4s
    fcvtas v14.4s, v14.4s
    fcvtas v15.4s, v15.4s
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s
    /*
    sqrshrn  v13.4h, v13.4s, #14
    sqrshrn2 v13.8h, v14.4s, #14
    sqrshrn  v15.4h, v15.4s, #14
    sqrshrn2 v15.8h, v16.4s, #14
    sqrshrn  v17.4h, v17.4s, #14
    sqrshrn2 v17.8h, v18.4s, #14
    sqrshrn  v19.4h, v19.4s, #14
    sqrshrn2 v19.8h, v20.4s, #14
    */
    sqxtn  v13.4h, v13.4s
    sqxtn2 v13.8h, v14.4s
    sqxtn  v15.4h, v15.4s
    sqxtn2 v15.8h, v16.4s
    sqxtn  v17.4h, v17.4s
    sqxtn2 v17.8h, v18.4s
    sqxtn  v19.4h, v19.4s
    sqxtn2 v19.8h, v20.4s

    sqxtn   v0.8b,  v13.8h
    sqxtn2  v0.16b, v15.8h
    sqxtn   v1.8b,  v17.8h
    sqxtn2  v1.16b, v19.8h

    smin v0.16b, v0.16b, v30.16b
    smax v0.16b, v0.16b, v31.16b
    smin v1.16b, v1.16b, v30.16b
    smax v1.16b, v1.16b, v31.16b

    st1 {v0.16b, v1.16b}, [x0], #32

    mov x2, x14
    add x1, x13, x12
    sub x4, x4, #2
    cmp x4, #2
    bge L2Loop_NOSTRIDE

b L2End

L2Loop_NORMAL:
    // load bias
    ld1 {v13.4s, v14.4s, v15.4s, v16.4s}, [x3]
    ld1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x3]

    mov x13, x1
    mov x14, x2
    mov x10, x7
    L2LoopH_NORMAL:
        mov x11, x6
        L2LoopW_NORMAL:
            // weight
            ld1 {v12.4s}, [x2], #16
            // feature input
            ld1 {v8.4s}, [x1], x5
            ld1 {v9.4s}, [x1], x5
            sxtl  v10.8h, v12.8b
            sxtl2 v11.8h, v12.16b
            sxtl  v0.8h, v8.8b
            sxtl2 v1.8h, v8.16b
            sxtl  v2.8h, v9.8b
            sxtl2 v3.8h, v9.16b

            smlal  v13.4s,  v10.4h, v0.4h
            smlal2 v14.4s,  v10.8h, v0.8h
            smlal  v15.4s,  v11.4h, v1.4h
            smlal2 v16.4s,  v11.8h, v1.8h
            smlal  v17.4s,  v10.4h, v2.4h
            smlal2 v18.4s,  v10.8h, v2.8h
            smlal  v19.4s,  v11.4h, v3.4h
            smlal2 v20.4s,  v11.8h, v3.8h

            sub x1, x1, x12
            add x1, x1, x8
            subs x11, x11, #1
            bne L2LoopW_NORMAL
        L2LoopWEnd_NORMAL:
        subs x10, x10, #1
        add x1, x1, x9
        bne L2LoopH_NORMAL

    scvtf v13.4s, v13.4s
    scvtf v14.4s, v14.4s
    scvtf v15.4s, v15.4s
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s

    fmul v13.4s, v13.4s, v22.4s
    fmul v14.4s, v14.4s, v23.4s
    fmul v15.4s, v15.4s, v24.4s
    fmul v16.4s, v16.4s, v25.4s
    fmul v17.4s, v17.4s, v22.4s
    fmul v18.4s, v18.4s, v23.4s
    fmul v19.4s, v19.4s, v24.4s
    fmul v20.4s, v20.4s, v25.4s

    fcvtas v13.4s, v13.4s
    fcvtas v14.4s, v14.4s
    fcvtas v15.4s, v15.4s
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s
    /*
    sqrshrn  v13.4h, v13.4s, #14
    sqrshrn2 v13.8h, v14.4s, #14
    sqrshrn  v15.4h, v15.4s, #14
    sqrshrn2 v15.8h, v16.4s, #14
    sqrshrn  v17.4h, v17.4s, #14
    sqrshrn2 v17.8h, v18.4s, #14
    sqrshrn  v19.4h, v19.4s, #14
    sqrshrn2 v19.8h, v20.4s, #14
    */
    sqxtn  v13.4h, v13.4s
    sqxtn2 v13.8h, v14.4s
    sqxtn  v15.4h, v15.4s
    sqxtn2 v15.8h, v16.4s
    sqxtn  v17.4h, v17.4s
    sqxtn2 v17.8h, v18.4s
    sqxtn  v19.4h, v19.4s
    sqxtn2 v19.8h, v20.4s

    sqxtn   v0.8b,  v13.8h
    sqxtn2  v0.16b, v15.8h
    sqxtn   v1.8b,  v17.8h
    sqxtn2  v1.16b, v19.8h

    smin v0.16b, v0.16b, v30.16b
    smax v0.16b, v0.16b, v31.16b
    smin v1.16b, v1.16b, v30.16b
    smax v1.16b, v1.16b, v31.16b

    st1 {v0.16b, v1.16b}, [x0], #32

    mov x2, x14
    add x1, x13, x12
    sub x4, x4, #2
    cmp x4, #2
    bge L2Loop_NORMAL

L2End:

L1:
cmp x4, #1
blt End
ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x19]
L1Loop_NORMAL:
    ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x3]

    mov x13, x1
    mov x14, x2
    mov x10, x7
    L1LoopH_NORMAL:
        mov x11, x6
        L1LoopW_NORMAL:
            ld1 {v24.4s}, [x2], #16
            sxtl  v25.8h, v24.8b
            sxtl2 v26.8h, v24.16b
            ld1 {v0.4s}, [x1], x8
            sxtl  v1.8h, v0.8b
            sxtl2 v2.8h, v0.16b

            smlal  v16.4s, v25.4h, v1.4h
            smlal2 v17.4s, v25.8h, v1.8h
            smlal  v18.4s, v26.4h, v2.4h
            smlal2 v19.4s, v26.8h, v2.8h
            subs x11, x11, #1
            bne L1LoopW_NORMAL
        L1LoopWEnd_NORMAL:
    subs x10, x10, #1
    add x1, x1, x9
    bne L1LoopH_NORMAL

    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s

    fmul v16.4s, v16.4s, v20.4s
    fmul v17.4s, v17.4s, v21.4s
    fmul v18.4s, v18.4s, v22.4s
    fmul v19.4s, v19.4s, v23.4s

    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    /*
    sqrshrn  v16.4h, v16.4s, #14
    sqrshrn2 v16.8h, v17.4s, #14
    sqrshrn  v18.4h, v18.4s, #14
    sqrshrn2 v18.8h, v19.4s, #14
    */
    sqxtn  v16.4h, v16.4s
    sqxtn2 v16.8h, v17.4s
    sqxtn  v18.4h, v18.4s
    sqxtn2 v18.8h, v19.4s

    sqxtn v16.8b, v16.8h
    sqxtn2 v16.16b, v18.8h

    smin v16.16b, v16.16b, v30.16b
    smax v16.16b, v16.16b, v31.16b

    st1 {v16.4s}, [x0], #16

    mov x2, x14
    add x1, x13, x5
    subs x4, x4, #1
    bne L1Loop_NORMAL


End:
ldp x19, x20, [sp, #(16 * 8)]
ldp x21, x22, [sp, #(16 * 7)]
ldp x23, x24, [sp, #(16 * 6)]
ldp x25, x26, [sp, #(16 * 5)]
ldp x27, x28, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldr d14, [sp], #(16 * 10)
ret

#endif
