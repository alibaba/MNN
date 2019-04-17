//
//  MNNAddBiasRelu6.S
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNAddBiasRelu6
//void MNNAddBiasRelu6(float* dst, const float* bias, int planeNumber, int biasNumber)
//x0:dst, x1:bias, x2:planeNumber, x3:biasNumber
cmp x3, #0
beq BiasReluEnd

cmp x2, #0
beq BiasReluEnd

movi v22.4s, #0
movi v21.4s, #6
scvtf v21.4s, v21.4s

ReluLoopBias:
    ld1 {v23.4s}, [x1], #16
    
    mov x4, x2

    ReluBiasReluL4:
    cmp x4, #3
    ble BiasReluL1
    ReluLoop4:
        mov x5, x0
        ld1 {v0.4s, v1.4s}, [x5], #32
        fadd v0.4s, v0.4s, v23.4s
        fadd v1.4s, v1.4s, v23.4s
        ld1 {v2.4s, v3.4s}, [x5]
        fmax v0.4s, v0.4s, v22.4s
        fmax v1.4s, v1.4s, v22.4s
        fmin v0.4s, v0.4s, v21.4s
        fmin v1.4s, v1.4s, v21.4s
        fadd v2.4s, v2.4s, v23.4s
        st1 {v0.4s, v1.4s}, [x0], #32
        fmax v2.4s, v2.4s, v22.4s
        fadd v3.4s, v3.4s, v23.4s
        fmin v2.4s, v2.4s, v21.4s
        fmax v3.4s, v3.4s, v22.4s
        fmin v3.4s, v3.4s, v21.4s
        st1 {v2.4s, v3.4s}, [x0], #32
        sub x4, x4, #4
        cmp x4, #4
        bge ReluLoop4

    BiasReluL1:
    cmp x4, #0
    beq EndReluLoopPlane
    ReluLoop1:
        ld1 {v0.4s}, [x0]
        fadd v0.4s, v0.4s, v23.4s
        fmax v0.4s, v0.4s, v22.4s
        fmin v0.4s, v0.4s, v21.4s
        subs x4, x4, #1
        st1 {v0.4s}, [x0], #16
        bne ReluLoop1
    
    EndReluLoopPlane:

    subs x3, x3, #1
    bne ReluLoopBias


BiasReluEnd:
ret

#endif
