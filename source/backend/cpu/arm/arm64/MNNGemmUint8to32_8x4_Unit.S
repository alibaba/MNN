//
//  MNNGemmUint8to32_8x4_Unit.S
//  MNN
//
//  Created by MNN on 2018/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmUint8to32_8x4_Unit
//void MNNGemmUint8to32_8x4_Unit(int32_t* dst, const uint8_t* src, const uint8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const int32_t* inputOffset);
//Auto: x0: dst, x1: src, x2:weight, x3: src_depth_quad
//x4: dst_step, x5: dst_depth_quad, x6: inputOffset

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

ldr w8, [x6, #0]
ldr w9, [x6, #4]
ldr w10, [x6, #8]
ldr w11, [x6, #12]
ldr w12, [x6, #16]
ldr w13, [x6, #20]

L6LoopDz:
    mov x6, x1
    subs x7, x3, #1
    ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [x2], #32
.macro START_TWO z0 z1 z2 z3 z4 z5 z6 z7
        ld1 {v2.8b, v3.8b}, [x1], #16
        umull v0.8h, v4.8b, v2.8b
        uaddlp \z0, v0.8h
        umull v1.8h, v4.8b, v3.8b
        uaddlp \z4, v1.8h
        umull v0.8h, v5.8b, v2.8b
        uaddlp \z1, v0.8h
        umull v1.8h, v5.8b, v3.8b
        uaddlp \z5, v1.8h
        umull v0.8h, v6.8b, v2.8b
        uaddlp \z2, v0.8h
        umull v1.8h, v6.8b, v3.8b
        uaddlp \z6, v1.8h
        umull v0.8h, v7.8b, v2.8b
        uaddlp \z3, v0.8h
        umull v1.8h, v7.8b, v3.8b
        uaddlp \z7, v1.8h
.endm
    START_TWO v8.4s, v9.4s, v10.4s, v11.4s, v12.4s, v13.4s, v14.4s, v15.4s
    START_TWO v16.4s, v17.4s, v18.4s, v19.4s, v20.4s, v21.4s, v22.4s, v23.4s
    START_TWO v24.4s, v25.4s, v26.4s, v27.4s, v28.4s, v29.4s, v30.4s, v31.4s
    beq L6LoopSzEnd
    L6LoopSz:
        ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [x2], #32
.macro COMPUTE_TWO z0 z1 z2 z3 z4 z5 z6 z7
        ld1 {v2.8b, v3.8b}, [x1], #16
        umull v0.8h, v4.8b, v2.8b
        uadalp \z0, v0.8h
        umull v1.8h, v4.8b, v3.8b
        uadalp \z4, v1.8h
        umull v0.8h, v5.8b, v2.8b
        uadalp \z1, v0.8h
        umull v1.8h, v5.8b, v3.8b
        uadalp \z5, v1.8h
        umull v0.8h, v6.8b, v2.8b
        uadalp \z2, v0.8h
        umull v1.8h, v6.8b, v3.8b
        uadalp \z6, v1.8h
        umull v0.8h, v7.8b, v2.8b
        uadalp \z3, v0.8h
        umull v1.8h, v7.8b, v3.8b
        uadalp \z7, v1.8h
.endm
        COMPUTE_TWO v8.4s, v9.4s, v10.4s, v11.4s, v12.4s, v13.4s, v14.4s, v15.4s
        COMPUTE_TWO v16.4s, v17.4s, v18.4s, v19.4s, v20.4s, v21.4s, v22.4s, v23.4s
        COMPUTE_TWO v24.4s, v25.4s, v26.4s, v27.4s, v28.4s, v29.4s, v30.4s, v31.4s
        subs x7, x7, #1
        bne L6LoopSz
    L6LoopSzEnd:

    dup v0.4s, w8
    dup v1.4s, w9
    dup v2.4s, w10
    dup v3.4s, w11
    dup v4.4s, w12
    dup v5.4s, w13
    addp v8.4s, v8.4s, v9.4s
    addp v10.4s, v10.4s, v11.4s
    addp v12.4s, v12.4s, v13.4s
    addp v14.4s, v14.4s, v15.4s
    addp v16.4s, v16.4s, v17.4s
    addp v18.4s, v18.4s, v19.4s
    addp v20.4s, v20.4s, v21.4s
    addp v22.4s, v22.4s, v23.4s
    addp v24.4s, v24.4s, v25.4s
    addp v26.4s, v26.4s, v27.4s
    addp v28.4s, v28.4s, v29.4s
    addp v30.4s, v30.4s, v31.4s

    addp v8.4s, v8.4s, v10.4s
    addp v9.4s, v12.4s, v14.4s
    addp v10.4s, v16.4s, v18.4s
    addp v11.4s, v20.4s, v22.4s
    addp v12.4s, v24.4s, v26.4s
    addp v13.4s, v28.4s, v30.4s

    sqsub v8.4s, v8.4s, v0.4s
    sqsub v9.4s, v9.4s, v1.4s
    sqsub v10.4s, v10.4s, v2.4s
    st1 {v8.4s, v9.4s}, [x0], #32
    sqsub v11.4s, v11.4s, v3.4s
    sqsub v12.4s, v12.4s, v4.4s
    st1 {v10.4s, v11.4s}, [x0], #32
    sqsub v13.4s, v13.4s, v5.4s
    st1 {v12.4s, v13.4s}, [x0], #32

    subs x5, x5, #1
    mov x1, x6
    bne L6LoopDz

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
