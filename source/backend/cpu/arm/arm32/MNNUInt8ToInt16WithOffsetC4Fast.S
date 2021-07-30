//
//  MNNUInt8ToInt16WithOffsetC4Fast.S
//  MNN
//
//  Created by MNN on 2018/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNUInt8ToInt16WithOffsetC4Fast

//void MNNUInt8ToInt16WithOffsetC4Fast(int16_t* dst, const uint8_t* src, size_t zeroPoint, size_t sizeQuad, size_t depthQuad, size_t dstZStep, size_t srcZStep);

//Auto: r0:dst, r1:src, r2: zeroPoint, r3: sizeQuad

//Load from sp: r4: depthQuad, r5: dstStride, r6:srcStride

push {r4-r8, lr} // avoid to touch platform-register r-9
ldr r4, [sp, #24]
ldr r5, [sp, #28]
ldr r6, [sp, #32]

vdup.u8 d30, r2

LoopZ:
    mov r7, r0
    mov r8, r1
    mov lr, r3

    L8:
    cmp r3, #8
    blt L1

    LoopL8:
    vld1.32 {q0}, [r1]!

    vsubl.u8 q8, d0, d30
    vld1.32 {q1}, [r1]!
    vsubl.u8 q9, d1, d30
    vst1.32 {q8, q9}, [r0]!
    vsubl.u8 q10, d2, d30
    vsubl.u8 q11, d3, d30
    vst1.32 {q10, q11}, [r0]!

    sub r3, r3, #8
    cmp r3, #8
    bge LoopL8


    L1:
    cmp r3, #0
    beq End

    LoopL1:
    vld1.32 {d0[0]}, [r1]!
    vsubl.u8 q0, d0, d30

    vst1.32 {d0}, [r0]!

    subs r3, r3, #1
    bne LoopL1



    End:

    subs r4, r4, #1
    add r0, r7, r5
    add r1, r8, r6
    mov r3, lr
    bne LoopZ

pop {r4-r8, pc}

#endif
#endif
