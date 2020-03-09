//
//  MNNMatrixCopyUnit.S
//  MNN
//
//  Created by MNN on 2020/01/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNMatrixCopyUnit
//void MNNMatrixCopyUnit(float* C, const float* A, size_t cStride, size_t aStride, size_t height)

//Auto: r0: C, r1:A, r2:cStride
//r3:aStride, r4:height

push {r4, lr}
ldr r4, [sp, #8]

mov r12, #4 //sizeof(float)
mul r2, r12, r2
mul r3, r12, r3
sub r2, r2, #128
sub r3, r3, #128

subs r4, r4, #1
// Unit = 8 for armv7a

vldm r1!, {d16-d23}

beq LoopYEnd

LoopY:
    vst1.64 {q8}, [r0]!
    PLD [r1, #0xC0]
    vldm r1!, {d24-d31}
    vst1.64 {q9}, [r0]!
    vst1.64 {q10}, [r0]!
    vst1.64 {q11}, [r0]!
    vst1.64 {q12}, [r0]!
    add r1, r1, r3
    vst1.64 {q13}, [r0]!
    PLD [r1, #0xC0]
    vldm r1!, {d16-d23}
    vst1.64 {q14}, [r0]!
    vst1.64 {q15}, [r0]!
    add r0, r0, r2
    subs r4, r4, #1
    bne LoopY

LoopYEnd:

vst1.64 {q8, q9}, [r0]!
vld1.64 {q12, q13}, [r1]!
vst1.64 {q10, q11}, [r0]!
vld1.64 {q14, q15}, [r1]

vst1.64 {q12, q13}, [r0]!
vst1.64 {q14, q15}, [r0]

pop {r4, pc}

#endif
#endif
