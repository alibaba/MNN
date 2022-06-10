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

stp d10, d11, [sp, #-32]!
stp d8,  d9,  [sp, #16]

ldr x10, [x3, #0]
ldr w11, [x3, #16]
dup v30.16b, w11 // max
ldr w11, [x3, #20]
dup v31.16b, w11 // min
ldr x3, [x3, #8]
ld1 {v6.4s}, [x3]
ld1 {v4.4s}, [x10]
mul x10, x6, x8
sub x9, x9, x10

L8:
cmp x4, #8
blt L4

mov x12, #8
mul x12, x5, x12
cmp x5, #4
bne L8Loop_NORMAL

L8Loop_NOSTRIDE:
    // load bias
    mov v16.16b, v6.16b
    mov v17.16b, v6.16b
    mov v18.16b, v6.16b
    mov v19.16b, v6.16b
    mov v20.16b, v6.16b
    mov v21.16b, v6.16b
    mov v22.16b, v6.16b
    mov v23.16b, v6.16b

    mov x13, x1
    mov x14, x2
    mov x10, x7
    L8LoopH_NOSTRIDE:
        mov x11, x6
        L8LoopW_NOSTRIDE:
            ld1 {v7.s}[0], [x2], #4
            ld1 {v24.4s, v25.4s}, [x1], #32
            dup v7.2s, v7.s[0]
            sxtl v7.8h, v7.8b
            subs x11, x11, #1
            sxtl v0.8h, v24.8b
            sxtl2 v1.8h, v24.16b
            sxtl v2.8h, v25.8b
            sxtl2 v3.8h, v25.16b
            smlal v16.4s, v7.4h, v0.4h
            smlal2 v17.4s, v7.8h, v0.8h
            smlal v18.4s, v7.4h, v1.4h
            smlal2 v19.4s, v7.8h, v1.8h
            smlal v20.4s, v7.4h, v2.4h
            smlal2 v21.4s, v7.8h, v2.8h
            smlal v22.4s, v7.4h, v3.4h
            smlal2 v23.4s, v7.8h, v3.8h

            sub x1, x1, x12
            add x1, x1, x8
            bne L8LoopW_NOSTRIDE
        L8LoopWEnd_NOSTRIDE:
        subs x10, x10, #1
        add x1, x1, x9
        bne L8LoopH_NOSTRIDE
    
    sub x4, x4, #8
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s
    
    fmul v16.4s, v16.4s, v4.4s
    fmul v17.4s, v17.4s, v4.4s
    fmul v18.4s, v18.4s, v4.4s
    fmul v19.4s, v19.4s, v4.4s
    fmul v20.4s, v20.4s, v4.4s
    fmul v21.4s, v21.4s, v4.4s
    fmul v22.4s, v22.4s, v4.4s
    fmul v23.4s, v23.4s, v4.4s
    
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s
    fcvtas v21.4s, v21.4s
    fcvtas v22.4s, v22.4s
    fcvtas v23.4s, v23.4s
    
    sqxtn v24.4h, v16.4s
    sqxtn2 v24.8h, v17.4s
    sqxtn v26.4h, v18.4s
    sqxtn2 v26.8h, v19.4s
    
    sqxtn v25.8b, v24.8h
    sqxtn2 v25.16b, v26.8h

    smin v25.16b, v25.16b, v30.16b
    smax v25.16b, v25.16b, v31.16b

    st1 {v25.4s}, [x0], #16
    
    sqxtn v24.4h, v20.4s
    sqxtn2 v24.8h, v21.4s
    sqxtn v26.4h, v22.4s
    sqxtn2 v26.8h, v23.4s

    sqxtn v25.8b, v24.8h
    sqxtn2 v25.16b, v26.8h
    mov x2, x14
    add x1, x13, x12
    cmp x4, #8
    smin v25.16b, v25.16b, v30.16b
    smax v25.16b, v25.16b, v31.16b
    st1 {v25.4s}, [x0], #16
    bge L8Loop_NOSTRIDE

b L8End

L8Loop_NORMAL:
    // load bias
    mov v16.16b, v6.16b
    mov v17.16b, v6.16b
    mov v18.16b, v6.16b
    mov v19.16b, v6.16b
    mov v20.16b, v6.16b
    mov v21.16b, v6.16b
    mov v22.16b, v6.16b
    mov v23.16b, v6.16b

    mov x13, x1
    mov x14, x2
    mov x10, x7
    L8LoopH_NORMAL:
        mov x11, x6
        L8LoopW_NORMAL:
            ld1 {v3.s}[0], [x2], #4
            dup v3.2s, v3.s[0]
            sxtl v3.8h, v3.8b
            ld1 {v0.s}[0], [x1], x5
            ld1 {v0.s}[1], [x1], x5
            subs x11, x11, #1
            sxtl v1.8h, v0.8b
            ld1 {v0.s}[2], [x1], x5
            smlal v16.4s, v3.4h, v1.4h
            ld1 {v0.s}[3], [x1], x5
            sxtl2 v2.8h, v0.16b
            smlal2 v17.4s, v3.8h, v1.8h
            ld1 {v0.s}[0], [x1], x5
            smlal v18.4s, v3.4h, v2.4h
            ld1 {v0.s}[1], [x1], x5
            ld1 {v0.s}[2], [x1], x5
            smlal2 v19.4s, v3.8h, v2.8h
            ld1 {v0.s}[3], [x1], x5
            sxtl v1.8h, v0.8b
            sxtl2 v2.8h, v0.16b
            smlal v20.4s, v3.4h, v1.4h
            smlal2 v21.4s, v3.8h, v1.8h
            smlal v22.4s, v3.4h, v2.4h
            smlal2 v23.4s, v3.8h, v2.8h

            sub x1, x1, x12
            add x1, x1, x8
            bne L8LoopW_NORMAL
        L8LoopWEnd_NORMAL:
        subs x10, x10, #1
        add x1, x1, x9
        bne L8LoopH_NORMAL
    
    sub x4, x4, #8
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s
    
    fmul v16.4s, v16.4s, v4.4s
    fmul v17.4s, v17.4s, v4.4s
    fmul v18.4s, v18.4s, v4.4s
    fmul v19.4s, v19.4s, v4.4s
    fmul v20.4s, v20.4s, v4.4s
    fmul v21.4s, v21.4s, v4.4s
    fmul v22.4s, v22.4s, v4.4s
    fmul v23.4s, v23.4s, v4.4s
    
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s
    fcvtas v21.4s, v21.4s
    fcvtas v22.4s, v22.4s
    fcvtas v23.4s, v23.4s
    
    sqxtn v24.4h, v16.4s
    sqxtn2 v24.8h, v17.4s
    sqxtn v26.4h, v18.4s
    sqxtn2 v26.8h, v19.4s
    
    sqxtn v25.8b, v24.8h
    sqxtn2 v25.16b, v26.8h

    smin v25.16b, v25.16b, v30.16b
    smax v25.16b, v25.16b, v31.16b

    st1 {v25.4s}, [x0], #16
    
    sqxtn v24.4h, v20.4s
    sqxtn2 v24.8h, v21.4s
    sqxtn v26.4h, v22.4s
    sqxtn2 v26.8h, v23.4s

    sqxtn v25.8b, v24.8h
    sqxtn2 v25.16b, v26.8h
    mov x2, x14
    add x1, x13, x12
    cmp x4, #8
    smin v25.16b, v25.16b, v30.16b
    smax v25.16b, v25.16b, v31.16b
    st1 {v25.4s}, [x0], #16
    bge L8Loop_NORMAL

L8End:

L4:
cmp x4, #4
blt L1

mov x12, #4
mul x12, x5, x12

L4Loop_NORMAL:
    mov v16.16b, v6.16b
    mov v17.16b, v6.16b
    mov v18.16b, v6.16b
    mov v19.16b, v6.16b
    
    mov x13, x1
    mov x14, x2
    mov x10, x7
    L4LoopH_NORMAL:
        mov x11, x6
        L4LoopW_NORMAL:
            ld1 {v3.s}[0], [x2], #4
            dup v3.2s, v3.s[0]
            sxtl v3.8h, v3.8b
            ld1 {v0.s}[0], [x1], x5
            ld1 {v0.s}[1], [x1], x5
            subs x11, x11, #1
            sxtl v1.8h, v0.8b
            ld1 {v0.s}[2], [x1], x5
            ld1 {v0.s}[3], [x1], x5
            sxtl2 v2.8h, v0.16b
            smlal v16.4s, v3.4h, v1.4h
            smlal2 v17.4s, v3.8h, v1.8h
            smlal v18.4s, v3.4h, v2.4h
            smlal2 v19.4s, v3.8h, v2.8h

            sub x1, x1, x12
            add x1, x1, x8
            bne L4LoopW_NORMAL
        L4LoopWEnd_NORMAL:
        subs x10, x10, #1
        add x1, x1, x9
        bne L4LoopH_NORMAL
    sub x4, x4, #4
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s

    fmul v16.4s, v16.4s, v4.4s
    fmul v17.4s, v17.4s, v4.4s
    fmul v18.4s, v18.4s, v4.4s
    fmul v19.4s, v19.4s, v4.4s
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s

    sqxtn v24.4h, v16.4s
    sqxtn2 v24.8h, v17.4s
    sqxtn v26.4h, v18.4s
    sqxtn2 v26.8h, v19.4s
    
    sqxtn v25.8b, v24.8h
    sqxtn2 v25.16b, v26.8h

    smin v25.16b, v25.16b, v30.16b
    smax v25.16b, v25.16b, v31.16b

    st1 {v25.4s}, [x0], #16

    mov x2, x14
    add x1, x13, x12
    cmp x4, #4
    bge L4Loop_NORMAL


L1:
cmp x4, #0
beq End

L1Loop:
    mov v0.16b, v6.16b
    mov x10, x7
    mov x13, x1
    mov x14, x2
    L1LoopH:
        mov x11, x6
        L1LoopW:
            ld1 {v1.s}[0], [x1], x8
            ld1 {v2.s}[0], [x2], #4
            sxtl v1.8h, v1.8b
            sxtl v2.8h, v2.8b
            smlal v0.4s, v1.4h, v2.4h
            subs x11, x11, #1
            bne L1LoopW
        subs x10, x10, #1
        add x1, x1, x9
        bne L1LoopH
    
    subs x4, x4, #1
    
    scvtf v0.4s, v0.4s
    fmul v0.4s, v0.4s, v4.4s
    fcvtas v1.4s, v0.4s
    sqxtn v2.4h, v1.4s
    sqxtn v3.8b, v2.8h
    smin v3.8b, v3.8b, v30.8b
    smax v3.8b, v3.8b, v31.8b
    mov x2, x14
    add x1, x13, x5
    st1 {v3.s}[0], [x0], #4
    bne L1Loop

End:
ldp d8,  d9,  [sp, #16]
ldp d10, d11, [sp], #32
ret

#endif
