//
//  MNNScaleAndAddBias.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNScaleAndAddBias
//void MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber, size_t biasNumber)

//Auto: x0:dst, x1:src, x2:bias, x3:alpha, x4:planeNumber, x5:biasNumber

cmp x4, #0
beq BSEnd

cmp x5, #0
beq BSEnd

BSLoopZ:
    mov x6, x4
    ld1 {v31.4s}, [x2], #16
    ld1 {v30.4s}, [x3], #16
    cmp x6, #7
    ble BSLoopP1
    BSLoopP8:
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
        fmul v0.4s, v0.4s, v30.4s
        fmul v1.4s, v1.4s, v30.4s
        fmul v2.4s, v2.4s, v30.4s
        fmul v3.4s, v3.4s, v30.4s
        ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
        fadd v0.4s, v0.4s, v31.4s
        fadd v1.4s, v1.4s, v31.4s
        fadd v2.4s, v2.4s, v31.4s
        fadd v3.4s, v3.4s, v31.4s
        fmul v4.4s, v4.4s, v30.4s
        fmul v5.4s, v5.4s, v30.4s
        fmul v6.4s, v6.4s, v30.4s
        fmul v7.4s, v7.4s, v30.4s
        st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
        sub x6, x6, #8
        fadd v4.4s, v4.4s, v31.4s
        fadd v5.4s, v5.4s, v31.4s
        fadd v6.4s, v6.4s, v31.4s
        fadd v7.4s, v7.4s, v31.4s
        st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
        cmp x6, #8
        bge BSLoopP8

    cmp x6, #0
    beq BSLoopPEnd

    BSLoopP1:
        ld1 {v0.4s}, [x1], #16
        fmul v0.4s, v0.4s, v30.4s
        fadd v0.4s, v0.4s, v31.4s
        st1 {v0.4s}, [x0], #16
        subs x6, x6, #1
        bne BSLoopP1
    BSLoopPEnd:

    subs x5, x5, #1
    bne BSLoopZ


BSEnd:


ret


#endif
