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
//void MNNLoadU8AndSum(int32_t* inputSum, uint8_t* colAddr, const uint8_t* inputOrigin, size_t srcZStep, size_t icDiv8, size_t realDstCount, size_t filter_offset)

push {r4-r11, lr}
//Auto: r0:inputSum, r1:colAddr, r2:inputOrigin, r3:srcZStep

//Load from sp:
//r4: icDiv8, r5: realDstCount, r6:filter_offset
ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r6, [sp, #44]
vdup.i32 d28, r6

mov r11, #16
LoopCount:
    mov r12, r4
    mov r7, r2
    mov r8, r1
    vmov.i32 q15, #0
    vmov.i32 q12, #0
    LoopSz:
        subs r12, r12, #1
        vld1.32 {d0[0]}, [r7], r3
        vld1.32 {d0[1]}, [r7], r3

        vmovl.u8 q1, d0
        vst1.8 {d0}, [r8], r11
        vpadal.u16 q15, q1
        vmlal.u16 q12, d2, d3

        bne LoopSz
    vpadd.u32 d30, d30, d31
    vpadd.u32 d24, d24, d25
    vmla.u32 d24, d30, d28
    vpadd.u32 d24, d24, d24


    subs r5, r5, #1
    vst1.32 {d24[0]}, [r0]!
    add r2, r2, #4
    add r1, r1, #8
    bne LoopCount


pop {r4-r11, pc}
#endif
#endif
