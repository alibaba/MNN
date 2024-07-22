//
//  NEON_MNNPackC4ForMatMul_A_BF16.S
//  MNN
//
//  Created by MNN on 2021/02/21.
//  Copyright © 2018-2021 Alibaba Group Holding Limited
//
#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
asm_function NEON_MNNPackC4ForMatMul_A_BF16
// treate float pointer as int16_t*
//void NEON_MNNPackC4ForMatMul_A_BF16(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el)
//Auto: r0: dest, r1:sourceGroup, r2: info, r3:el
push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9
ldr r10, [r2, #0] // number
ldr r4, [r2, #4] // eReal
ldr r11, [r2, #8] // eDest
ldr r6, [r2, #12] // xOffset
// xOffset -> xOffset * 4 * sizeof(float)
// eReal -> eReal * 4 * sizeof(float)
// eDest -> eDest * sizeof(float)
mov r12, #2 // sizeof(int16_t)
mov lr, #8  // sizeof(int16_t) * 4
mul r4, lr, r4
mul r11, r12, r11
mul r6, lr, r6

LoopNumber:
ldr r5, [r3, #4] // l
ldr r8, [r3, #8] // eOffset
ldr r7, [r3, #12] // lOffset

push {r0, r1}
ldr r1, [r1, #0]

// Compute dest ptr: r0 = r0 + eOffset * sizeof(float) + lOffset * eDest * sizeof(float)
; mov lr, #2 //sizeof(int16_t)
mul r7, r11, r7
mul r8, r12, r8
add r0, r0, r7
add r0, r0, r8

ldr r2, [r3, #0] // e

Body:
cmp r2, #12
bne Right
    cmp r5, #4
    blt LoopEL3
    LoopL4:
        mov r2, r1
.macro MAIN_TRANSPOSE
        vld1.16 {d16}, [r1], r6 // load size: 4 * sizeof(int16_t)
        vld1.16 {d19}, [r1], r6
        vld1.16 {d22}, [r1], r6
        vld1.16 {d25}, [r1], r6
        vld1.16 {d17}, [r1], r6
        vld1.16 {d20}, [r1], r6
        vld1.16 {d23}, [r1], r6
        vld1.16 {d26}, [r1], r6
        vld1.16 {d18}, [r1], r6
        vld1.16 {d21}, [r1], r6
        vld1.16 {d24}, [r1], r6
        vld1.16 {d27}, [r1], r6

        // transpose each 4 16-bit elements in 2 d_n vectors, by transpose 16-bit and scale up transpose 32-bit.
        vtrn.16 d16, d19
        vtrn.16 d22, d25
        // vswp d0[2-3], d2[0-1]
        // vswp d1[2-3], d3[0-1]
        // swap half of 64-bit is equal to transpose in 32-bit unit.
        vtrn.32 d16, d22
        vtrn.32 d19, d25

        vtrn.16 d17, d20
        vtrn.16 d23, d26
        vtrn.32 d17, d23
        vtrn.32 d20, d26

        vtrn.16 d18, d21
        vtrn.16 d24, d27
        vtrn.32 d18, d24
        vtrn.32 d21, d27
        // after transpose from 12x4 to 4x12, memory layout is
        // +-------+------+------+
        // | d16...|d17...|d18...|
        // +-------+------+------+
        // | d19...|d20...|d21...|
        // +-------+------+------+
        // | d22...|d23...|d24...|
        // +-------+------+------+
        // | d25...|d26...|d27...|
        // +-------+------+------+
.endm
        MAIN_TRANSPOSE

        vstm r0!, {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27} // store at one time: 12 * 4 * sizeof(int16_t)

        add r1, r2, r4
        sub r5, r5, #4
        cmp r5, #4
        bge LoopL4

    LoopEL3:
    cmp r5, #3
    blt LoopEL2
        MAIN_TRANSPOSE

        vstm r0!, {d16, d17, d18, d19, d20, d21, d22, d23, d24}

        b LoopEEnd

    LoopEL2:
    cmp r5, #2
    blt LoopEL1
        MAIN_TRANSPOSE

        vstm r0!, {d16, d17, d18, d19, d20, d21}

        b LoopEEnd

    LoopEL1:
    cmp r5, #0
    beq LoopEEnd
        MAIN_TRANSPOSE

        vstm r0!, {d16, d17, d18}

    LoopEEnd:

b End


Right:

LoopE1:
    mov lr, r5
    mov r7, r1
    mov r8, r0
    cmp r5, #4
    blt LoopE1L3
    LoopE1L4:
        vld1.16 {d0}, [r1], r4
        vst1.16 {d0[0]}, [r0], r11
        vst1.16 {d0[1]}, [r0], r11
        vst1.16 {d0[2]}, [r0], r11
        vst1.16 {d0[3]}, [r0], r11
        sub r5, r5, #4
        cmp r5, #4
        bge LoopE1L4

    LoopE1L3:
    cmp r5, #3
    blt LoopE1L2
        vld1.16 {d0}, [r1], r4
        vst1.16 {d0[0]}, [r0], r11
        vst1.16 {d0[1]}, [r0], r11
        vst1.16 {d0[2]}, [r0], r11

        sub r5, r5, #3

    LoopE1L2:
    cmp r5, #2
    blt LoopE1L1
        vld1.16 {d0}, [r1], r4
        vst1.16 {d0[0]}, [r0], r11
        vst1.16 {d0[1]}, [r0], r11
        sub r5, r5, #2

    LoopE1L1:
    cmp r5, #1
    blt LoopE1End
        vld1.16 {d0[0]}, [r1], r4
        vst1.16 {d0[0]}, [r0], r11

    LoopE1End:

    subs r2, r2, #1
    add r0, r8, r12 // !!!! caution : sizeof(int16_t)
    add r1, r7, r6
    mov r5, lr
    bne LoopE1

End:

pop {r0, r1}
subs r10, r10, #1

// x3 is (const int32_t* el), this array size of 4. as a result for next struct element,
// address added by 4 * sizeof(int32_t)
add r3, r3, #16

// x1 is (const int16_t** sourceGroup), even though data content is int16_t,
// the element in sourceGroup in 'int16_t*', as a result for next struct element,
// value added by sizeof(void*)
add r1, r1, #4

bne LoopNumber

pop {r4-r8, r10, r11, pc}

#endif
#endif
