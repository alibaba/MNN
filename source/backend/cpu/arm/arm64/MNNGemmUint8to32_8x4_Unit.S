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
    ld1 {v2.8b, v3.8b}, [x1], #16
    umull v0.8h, v4.8b, v2.8b
    umull v1.8h, v4.8b, v3.8b
    uaddlp v8.4s, v0.8h
    uaddlp v12.4s, v1.8h
    umull v0.8h, v5.8b, v2.8b
    umull v1.8h, v5.8b, v3.8b
    uaddlp v9.4s, v0.8h
    uaddlp v13.4s, v1.8h
    umull v0.8h, v6.8b, v2.8b
    umull v1.8h, v6.8b, v3.8b
    uaddlp v10.4s, v0.8h
    uaddlp v14.4s, v1.8h
    umull v0.8h, v7.8b, v2.8b
    umull v1.8h, v7.8b, v3.8b
    uaddlp v11.4s, v0.8h
    ld1 {v2.8b, v3.8b}, [x1], #16
    uaddlp v15.4s, v1.8h
    umull v0.8h, v4.8b, v2.8b
    umull v1.8h, v4.8b, v3.8b
    uaddlp v16.4s, v0.8h
    uaddlp v20.4s, v1.8h
    umull v0.8h, v5.8b, v2.8b
    umull v1.8h, v5.8b, v3.8b
    uaddlp v17.4s, v0.8h
    uaddlp v21.4s, v1.8h
    umull v0.8h, v6.8b, v2.8b
    umull v1.8h, v6.8b, v3.8b
    uaddlp v18.4s, v0.8h
    uaddlp v22.4s, v1.8h
    umull v0.8h, v7.8b, v2.8b
    umull v1.8h, v7.8b, v3.8b
    uaddlp v19.4s, v0.8h
    ld1 {v2.8b, v3.8b}, [x1], #16
    uaddlp v23.4s, v1.8h

    umull v0.8h, v4.8b, v2.8b
    umull v1.8h, v4.8b, v3.8b
    uaddlp v24.4s, v0.8h
    uaddlp v28.4s, v1.8h
    umull v0.8h, v5.8b, v2.8b
    umull v1.8h, v5.8b, v3.8b
    uaddlp v25.4s, v0.8h
    uaddlp v29.4s, v1.8h
    umull v0.8h, v6.8b, v2.8b
    umull v1.8h, v6.8b, v3.8b
    uaddlp v26.4s, v0.8h
    uaddlp v30.4s, v1.8h
    umull v0.8h, v7.8b, v2.8b
    umull v1.8h, v7.8b, v3.8b
    uaddlp v27.4s, v0.8h
    cmp x7, #0
    uaddlp v31.4s, v1.8h
    beq L6LoopSzEnd
    L6LoopSz:
        ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [x2], #32
        ld1 {v2.8b, v3.8b}, [x1], #16
        umull v0.8h, v4.8b, v2.8b
        umull v1.8h, v4.8b, v3.8b
        uadalp v8.4s, v0.8h
        uadalp v12.4s, v1.8h
        umull v0.8h, v5.8b, v2.8b
        umull v1.8h, v5.8b, v3.8b
        uadalp v9.4s, v0.8h
        uadalp v13.4s, v1.8h
        umull v0.8h, v6.8b, v2.8b
        umull v1.8h, v6.8b, v3.8b
        uadalp v10.4s, v0.8h
        uadalp v14.4s, v1.8h
        umull v0.8h, v7.8b, v2.8b
        umull v1.8h, v7.8b, v3.8b
        uadalp v11.4s, v0.8h
        ld1 {v2.8b, v3.8b}, [x1], #16
        uadalp v15.4s, v1.8h
        umull v0.8h, v4.8b, v2.8b
        umull v1.8h, v4.8b, v3.8b
        uadalp v16.4s, v0.8h
        uadalp v20.4s, v1.8h
        umull v0.8h, v5.8b, v2.8b
        umull v1.8h, v5.8b, v3.8b
        uadalp v17.4s, v0.8h
        uadalp v21.4s, v1.8h
        umull v0.8h, v6.8b, v2.8b
        umull v1.8h, v6.8b, v3.8b
        uadalp v18.4s, v0.8h
        uadalp v22.4s, v1.8h
        umull v0.8h, v7.8b, v2.8b
        umull v1.8h, v7.8b, v3.8b
        uadalp v19.4s, v0.8h
        ld1 {v2.8b, v3.8b}, [x1], #16
        uadalp v23.4s, v1.8h

        umull v0.8h, v4.8b, v2.8b
        umull v1.8h, v4.8b, v3.8b
        uadalp v24.4s, v0.8h
        uadalp v28.4s, v1.8h
        umull v0.8h, v5.8b, v2.8b
        umull v1.8h, v5.8b, v3.8b
        uadalp v25.4s, v0.8h
        uadalp v29.4s, v1.8h
        umull v0.8h, v6.8b, v2.8b
        umull v1.8h, v6.8b, v3.8b
        uadalp v26.4s, v0.8h
        uadalp v30.4s, v1.8h
        umull v0.8h, v7.8b, v2.8b
        umull v1.8h, v7.8b, v3.8b
        uadalp v27.4s, v0.8h
        subs x7, x7, #1
        uadalp v31.4s, v1.8h

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

    subs x5, x5, #1
    st1 {v12.4s, v13.4s}, [x0], #32
    mov x1, x6
    bne L6LoopDz

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
