//
//  MNNTranspose32Bit4x4.S
//  MNN
//
//  Created by MNN on 2020/09/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNTranspose32Bit4x4
//void MNNTranspose32Bit4x4(int32_t* dstO, const int32_t* srcO, int* dim)
//Auto: r0: dstO, r1:srcO, r2: dim

push {r4-r8, lr} // avoid to touch platform-register r-9
ldr r4, [r2, #0]
ldr r5, [r2, #4]
ldr r6, [r2, #8]
ldr r7, [r2, #12]

// r4, r5 -> wC4, hC4
lsr r4, r4, #2
lsr r5, r5, #2

// r6, r7 -> srcStride * sizeof(float), dstStride * sizeof(float)
lsl r6, r6, #2
lsl r7, r7, #2


LoopY:
    mov r2, r4
    mov r8, r0
    mov lr, r1
    LoopX:
        vld1.32 {q0}, [r1], r6
        vld1.32 {q1}, [r1], r6
        vld1.32 {q2}, [r1], r6
        vld1.32 {q3}, [r1], r6

        vtrn.32 d0, d2
        vtrn.32 d1, d3
        vtrn.32 d4, d6
        vtrn.32 d5, d7

        vswp d1, d4
        vswp d3, d6

        mov r12, r0

        vst1.32 {q0}, [r12], r7
        vst1.32 {q1}, [r12], r7
        vst1.32 {q2}, [r12], r7
        vst1.32 {q3}, [r12], r7

        add r0, r0, #16 // 4 * sizeof(float)

        subs r2, r2, #1
        bne LoopX


    lsl r12, r7, #2
    subs r5, r5, #1
    add r1, lr, #16 // 4 * sizeof(float)
    add r0, r8, r12
    bne LoopY

End:

pop {r4-r8, pc}

#endif
#endif
