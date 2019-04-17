//
//  MNNFloat16_Common_4_MatMul.S
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNFloat16_Common_4_MatMul
//void MNNFloat16_Common_4_MatMul(int16_t* dst, const int16_t* src, const int16_t* weight, size_t icUnit, size_t ocUnit, size_t ocStep, size_t width);
//Auto: x0: dst, x1:src, x2:weight, x3:icUnit
//x4: ocUnit, x5:ocStep, x6: width

mov x12, #8 // 4*sizeof(__fp16)
mul x12, x6, x12

.macro COMPUTE z0 z1 z2 z3 z4 z5 z6 z7
        ld1 {v0.8h}, [x7], #16
        fmla \z0, v2.8h, v0.h[0]
        fmla \z1, v2.8h, v0.h[4]
        fmla \z0, v3.8h, v0.h[1]
        fmla \z1, v3.8h, v0.h[5]
        fmla \z0, v4.8h, v0.h[2]
        ld1 {v1.8h}, [x7], #16
        fmla \z1, v4.8h, v0.h[6]
        fmla \z0, v5.8h, v0.h[3]
        fmla \z1, v5.8h, v0.h[7]
        fmla \z2, v2.8h, v1.h[0]
        fmla \z3, v2.8h, v1.h[4]
        ld1 {v0.8h}, [x7], #16
        fmla \z2, v3.8h, v1.h[1]
        fmla \z3, v3.8h, v1.h[5]
        fmla \z2, v4.8h, v1.h[2]
        fmla \z3, v4.8h, v1.h[6]
        fmla \z2, v5.8h, v1.h[3]
        fmla \z3, v5.8h, v1.h[7]

        fmla \z4, v2.8h, v0.h[0]
        fmla \z5, v2.8h, v0.h[4]
        fmla \z4, v3.8h, v0.h[1]
        fmla \z5, v3.8h, v0.h[5]
        ld1 {v1.8h}, [x7], #16
        fmla \z4, v4.8h, v0.h[2]
        fmla \z5, v4.8h, v0.h[6]
        fmla \z4, v5.8h, v0.h[3]
        fmla \z5, v5.8h, v0.h[7]


        fmla \z6, v2.8h, v1.h[0]
        fmla \z7, v2.8h, v1.h[4]
        fmla \z6, v3.8h, v1.h[1]
        fmla \z7, v3.8h, v1.h[5]
        fmla \z6, v4.8h, v1.h[2]
        fmla \z7, v4.8h, v1.h[6]
        fmla \z6, v5.8h, v1.h[3]
        fmla \z7, v5.8h, v1.h[7]
.endm


L16:
cmp w6, #16
blt L8

mov x13, x2
mov x14, x0
mov x15, x1

mov x11, x4

LoopOzL16:
    movi v31.8h, #0
    movi v30.8h, #0
    movi v29.8h, #0
    movi v28.8h, #0
    movi v27.8h, #0
    movi v26.8h, #0
    movi v25.8h, #0
    movi v24.8h, #0

    movi v23.8h, #0
    movi v22.8h, #0
    movi v21.8h, #0
    movi v20.8h, #0
    movi v19.8h, #0
    movi v18.8h, #0
    movi v17.8h, #0
    movi v16.8h, #0

    mov x7, x1
    mov x8, x3
    mov x9, x14
    LoopSz16:
        ld1 {v2.8h, v3.8h}, [x2], #32
        ld1 {v4.8h, v5.8h}, [x2], #32

        COMPUTE v16.8h, v17.8h, v18.8h, v19.8h, v20.8h, v21.8h, v22.8h, v23.8h
        COMPUTE v24.8h, v25.8h, v26.8h, v27.8h, v28.8h, v29.8h, v30.8h, v31.8h

        sub x7, x7, #128
        add x7, x7, x12

        subs x8, x8, #1
        bne LoopSz16

    subs x11, x11, #1

    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x9], #64
    st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x9], #64
    st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x9], #64
    st1 {v28.8h, v29.8h, v30.8h, v31.8h}, [x9], #64

    add x14, x14, x5
    bne LoopOzL16

sub w6, w6, #16
add x0, x0, #256 // 16*8*sizeof(float16)
add x1, x1, #128 // 16*4*sizeof(float16)
mov x2, x13

L8:
cmp w6, #8
blt L1

mov x13, x2
mov x14, x0
mov x15, x1

mov x11, x4

LoopOzL8:
    movi v23.8h, #0
    movi v22.8h, #0
    movi v21.8h, #0
    movi v20.8h, #0
    movi v19.8h, #0
    movi v18.8h, #0
    movi v17.8h, #0
    movi v16.8h, #0

    mov x7, x1
    mov x8, x3
    mov x9, x14
    LoopSz8:
        ld1 {v2.8h, v3.8h}, [x2], #32
        ld1 {v4.8h, v5.8h}, [x2], #32

        COMPUTE v16.8h, v17.8h, v18.8h, v19.8h, v20.8h, v21.8h, v22.8h, v23.8h
        sub x7, x7, #64
        add x7, x7, x12

        subs x8, x8, #1
        bne LoopSz8

    subs x11, x11, #1

    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x9], #64
    st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x9], #64

    add x14, x14, x5
    bne LoopOzL8


add x0, x0, #128 // 8*8*sizeof(float16)
add x1, x1, #64 // 8*4*sizeof(float16)
mov x2, x13
sub w6, w6, #8

L1:
cmp w6, #0
beq End

LoopL1:
    mov x13, x2
    mov x14, x0
    mov x15, x1

    mov x11, x4

    LoopOzL1:
        movi v16.8h, #0
        movi v17.8h, #0

        mov x7, x1
        mov x8, x3
        mov x9, x14
        LoopSz1:
            ld1 {v2.8h, v3.8h}, [x2], #32
            ld1 {v4.8h, v5.8h}, [x2], #32

            ld1 {v0.4h}, [x7], x12
            fmla v16.8h, v2.8h, v0.h[0]
            fmla v17.8h, v3.8h, v0.h[1]
            fmla v16.8h, v4.8h, v0.h[2]
            fmla v17.8h, v5.8h, v0.h[3]

            subs x8, x8, #1
            bne LoopSz1
        fadd v16.8h, v16.8h, v17.8h
        subs x11, x11, #1
        st1 {v16.8h}, [x9], #16
        add x14, x14, x5
        bne LoopOzL1

    subs w6, w6, #1
    add x0, x0, #16 // 8*sizeof(float16)
    add x1, x1, #8 // 4*sizeof(float16)
    mov x2, x13
    bne LoopL1

End:

ret

#endif
