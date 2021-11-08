//
//  MNNLoadU8AndSum.S
//  MNN
//
//  Created by MNN on 2018/11/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNLoadU8AndSum
//void MNNLoadU8AndSum(int32_t* inputSum, int8_t* colAddr, const uint8_t* inputOrigin, size_t srcZStep, size_t icDiv8, size_t realDstCount, size_t filter_offset)

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9

//Auto: r0:inputSum, r1:colAddr, r2:inputOrigin, r3:srcZStep

//Load from sp:
//r4: icDiv8, r5: realDstCount, r6:filter_offset
ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]
vdup.i32 d30, r6
vmov.i8 d31, #128

mov r11, #32//SRC_UNIT*DST_XUNIT
LoopCount:
    mov r12, r4
    mov r7, r2
    mov r8, r1
    vmov.i32 q8, #0
    vmov.i32 q9, #0
    LoopSz:
        subs r12, r12, #1
        vld1.32 {d0[0]}, [r7], r3
        vld1.32 {d0[1]}, [r7], r3
        vld1.32 {d1[0]}, [r7], r3
        vld1.32 {d1[1]}, [r7], r3

        vsubl.u8 q1, d0, d31
        vsubl.u8 q2, d1, d31

        vmovn.s16 d6, q1
        vmovn.s16 d7, q2

        vst1.8 {q3}, [r8], r11
        vpadal.s16 q8, q1
        vpadal.s16 q9, q2

        bne LoopSz
    vpadd.s32 d16, d16, d17
    vpadd.s32 d18, d18, d19
    vpadd.s32 d16, d16, d18
    vmul.s32 d16, d16, d30
    vpadd.s32 d16, d16, d16


    subs r5, r5, #1
    vst1.32 {d16[0]}, [r0]!
    add r2, r2, #4//UNIT
    add r1, r1, #16//SRC_UNIT
    bne LoopCount


pop {r4-r8, r10, r11, pc}
#endif
#endif
