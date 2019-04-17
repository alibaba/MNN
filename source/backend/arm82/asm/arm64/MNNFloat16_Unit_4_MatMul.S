//
//  MNNFloat16_Unit_4_MatMul.S
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNFloat16_Unit_4_MatMul
//void MNNFloat16_Unit_4_MatMul(int16_t* dst, const int16_t* src, const int16_t* weight, size_t icUnit, size_t ocUnit, size_t ocStep);
//Auto: x0: dst, x1:src, x2:weight, x3:icUnit
//x4: ocUnit, x5:ocStep
sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
.macro START z0 z1 z2 z3 z4 z5 z6 z7
        ld1 {v0.8h}, [x6], #16
        fmul \z0, v2.8h, v0.h[0]
        fmul \z1, v2.8h, v0.h[4]
        fmla \z0, v3.8h, v0.h[1]
        fmla \z1, v3.8h, v0.h[5]
        fmla \z0, v4.8h, v0.h[2]
        ld1 {v1.8h}, [x6], #16
        fmla \z1, v4.8h, v0.h[6]
        fmla \z0, v5.8h, v0.h[3]
        fmla \z1, v5.8h, v0.h[7]
        fmul \z2, v2.8h, v1.h[0]
        fmul \z3, v2.8h, v1.h[4]
        ld1 {v0.8h}, [x6], #16
        fmla \z2, v3.8h, v1.h[1]
        fmla \z3, v3.8h, v1.h[5]
        fmla \z2, v4.8h, v1.h[2]
        fmla \z3, v4.8h, v1.h[6]
        fmla \z2, v5.8h, v1.h[3]
        fmla \z3, v5.8h, v1.h[7]

        fmul \z4, v2.8h, v0.h[0]
        fmul \z5, v2.8h, v0.h[4]
        fmla \z4, v3.8h, v0.h[1]
        fmla \z5, v3.8h, v0.h[5]
        ld1 {v1.8h}, [x6], #16
        fmla \z4, v4.8h, v0.h[2]
        fmla \z5, v4.8h, v0.h[6]
        fmla \z4, v5.8h, v0.h[3]
        fmla \z5, v5.8h, v0.h[7]

        fmul \z6, v2.8h, v1.h[0]
        fmul \z7, v2.8h, v1.h[4]
        fmla \z6, v3.8h, v1.h[1]
        fmla \z7, v3.8h, v1.h[5]
        fmla \z6, v4.8h, v1.h[2]
        fmla \z7, v4.8h, v1.h[6]
        fmla \z6, v5.8h, v1.h[3]
        fmla \z7, v5.8h, v1.h[7]
.endm

LoopOz:
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

    movi v15.8h, #0
    movi v14.8h, #0
    movi v13.8h, #0
    movi v12.8h, #0
    movi v11.8h, #0
    movi v10.8h, #0
    movi v9.8h, #0
    movi v8.8h, #0

    mov x6, x1
    subs x7, x3, #1

    ld1 {v2.8h, v3.8h}, [x2], #32
    ld1 {v4.8h, v5.8h}, [x2], #32
    START v8.8h, v9.8h, v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h
    START v16.8h, v17.8h, v18.8h, v19.8h, v20.8h, v21.8h, v22.8h, v23.8h
    START v24.8h, v25.8h, v26.8h, v27.8h, v28.8h, v29.8h, v30.8h, v31.8h

    beq LoopSzEnd

    LoopSz:
        ld1 {v2.8h, v3.8h}, [x2], #32
        ld1 {v4.8h, v5.8h}, [x2], #32

.macro COMPUTE z0 z1 z2 z3 z4 z5 z6 z7
        ld1 {v0.8h}, [x6], #16
        fmla \z0, v2.8h, v0.h[0]
        fmla \z1, v2.8h, v0.h[4]
        fmla \z0, v3.8h, v0.h[1]
        fmla \z1, v3.8h, v0.h[5]
        fmla \z0, v4.8h, v0.h[2]
        ld1 {v1.8h}, [x6], #16
        fmla \z1, v4.8h, v0.h[6]
        fmla \z0, v5.8h, v0.h[3]
        fmla \z1, v5.8h, v0.h[7]
        fmla \z2, v2.8h, v1.h[0]
        fmla \z3, v2.8h, v1.h[4]
        ld1 {v0.8h}, [x6], #16
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
        ld1 {v1.8h}, [x6], #16
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
        COMPUTE v8.8h, v9.8h, v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h
        COMPUTE v16.8h, v17.8h, v18.8h, v19.8h, v20.8h, v21.8h, v22.8h, v23.8h
        COMPUTE v24.8h, v25.8h, v26.8h, v27.8h, v28.8h, v29.8h, v30.8h, v31.8h


        subs x7, x7, #1
        bne LoopSz

    LoopSzEnd:

    subs x4, x4, #1
    mov x8, x0

    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x8], #64
    st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x8], #64
    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x8], #64
    st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x8], #64
    st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x8], #64
    add x0, x0, x5
    st1 {v28.8h, v29.8h, v30.8h, v31.8h}, [x8], #64

    bne LoopOz

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

ret
#endif
