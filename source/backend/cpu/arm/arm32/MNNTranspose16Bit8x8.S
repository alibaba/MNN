//
//  MNNTranspose16Bit8x8.S
//  MNN
//
//  Created by MNN on 2023/11/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNTranspose16Bit8x8
//void MNNTranspose16Bit8x8(int16_t* dstO, const int16_t* srcO, int* dim)
//Auto: r0: dstO, r1:srcO, r2: dim

push {r4-r8, lr} // avoid to touch platform-register r-9
ldr r4, [r2, #0]
ldr r5, [r2, #4]
ldr r6, [r2, #8]
ldr r7, [r2, #12]

// r4, r5 -> wC8, hC8
lsr r4, r4, #3
lsr r5, r5, #3

// r6, r7 -> srcStride * sizeof(half), dstStride * sizeof(half)
lsl r6, r6, #1
lsl r7, r7, #1


LoopY:
    mov r2, r4
    mov r8, r0
    mov lr, r1
    LoopX:
        /*
        after vld1.16
        [ 0,  1,  2,  3,  4,  5,  6,  7]
        [ 8,  9, 10, 11, 12, 13, 14, 15]
        [16, 17, 18, 19, 20, 21, 22, 23]
        [24, 25, 26, 27, 28, 29, 30, 31]
        [32, 33, 34, 35, 36, 37, 38, 39]
        [40, 41, 42, 43, 44, 45, 46, 47]
        [48, 49, 50, 51, 52, 53, 54, 55]
        [56, 57, 58, 59, 60, 61, 62, 63]
        */
        vld1.16 {q0}, [r1], r6
        vld1.16 {q1}, [r1], r6
        vld1.16 {q2}, [r1], r6
        vld1.16 {q3}, [r1], r6
        vld1.16 {q4}, [r1], r6
        vld1.16 {q5}, [r1], r6
        vld1.16 {q6}, [r1], r6
        vld1.16 {q7}, [r1], r6

        /*
        after vtrn.16
        [ 0,  8,  2, 10,  4, 12,  6, 14]
        [ 1,  9,  3, 11,  5, 13,  7, 15]
        [16, 24, 18, 26, 20, 28, 22, 30]
        [17, 25, 19, 27, 21, 29, 23, 31]
        [32, 40, 34, 42, 36, 44, 38, 46]
        [33, 41, 35, 43, 37, 45, 39, 47]
        [48, 56, 50, 58, 52, 60, 54, 62]
        [49, 57, 51, 59, 53, 61, 55, 63]
        */
        vtrn.16 q0, q1
        vtrn.16 q2, q3
        vtrn.16 q4, q5
        vtrn.16 q6, q7

        /*
        after vtrn.32
        [ 0,  8, 16, 24,  4, 12,  20, 28]
        [ 1,  9, 17, 25,  5, 13, 21, 29]
        [ 2, 10, 18, 26,  6, 14, 22, 30]
        [ 3, 11, 19, 27,  7, 15, 23, 31]
        [32, 40, 48, 56, 36, 44, 52, 60]
        [33, 41, 49, 57, 37, 45, 53, 61]
        [34, 42, 50, 58, 38, 46, 54, 62]
        [35, 43, 51, 59, 39, 47, 55, 63]
        */
        vtrn.32 q0, q2
        vtrn.32 q1, q3
        vtrn.32 q4, q6
        vtrn.32 q5, q7

        /*
        after vswp
        [ 0,  8, 16, 24, 32, 40, 48, 56]
        [ 1,  9, 17, 25, 33, 41, 49, 57]
        [ 2, 10, 18, 26, 34, 42, 50, 58]
        [ 3, 11, 19, 27, 35, 43, 51, 59]
        [ 4, 12, 20, 28, 36, 44, 52, 60]
        [ 5, 13, 21, 29, 37, 45, 53, 61]
        [ 6, 14, 22, 30, 38, 46, 54, 62]
        [ 7, 15, 23, 31, 39, 47, 55, 63]
        */
        vswp d1, d8
        vswp d3, d10
        vswp d5, d12
        vswp d7, d14

        mov r12, r0

        vst1.16 {q0}, [r12], r7
        vst1.16 {q1}, [r12], r7
        vst1.16 {q2}, [r12], r7
        vst1.16 {q3}, [r12], r7
        vst1.16 {q4}, [r12], r7
        vst1.16 {q5}, [r12], r7
        vst1.16 {q6}, [r12], r7
        vst1.16 {q7}, [r12], r7

        add r0, r0, #16 // 4 * sizeof(float)

        subs r2, r2, #1
        bne LoopX


    lsl r12, r7, #3
    subs r5, r5, #1
    add r1, lr, #16 // 8 * sizeof(half)
    add r0, r8, r12
    bne LoopY

End:

pop {r4-r8, pc}

#endif
#endif
