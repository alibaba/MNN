//
//  MNNWinogradOptFunctionInt8.S
//  MNN
//
//  Created by MNN on 2021/09/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/*
 some compiler don't support vreinterpret_xx_xx, inline asm will generate different code by different compilers. So write raw asm code here
 ARM82:
 4(UNIT) -> 4(SRC_UNIT)
 1xN ~ _sourceTransXPack4x4Int8
 Nx1 ~ _sourceTransYPack4x4Int8
 NxN ~ _sourceTrans2Pack4x4Int8
 ARM32/64/SSE: 4(UNIT) -> 16(SRC_UNIT)
 1xN ~ _sourceTransXPack4x16Int8
 Nx1 ~ _sourceTransYPack4x16Int8
 NxM ~ _sourceTrans2Pack4x16Int8
 */

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.macro LOAD_4X4 ch, addr, stepY
vld4.32 {d0[\ch], d1[\ch], d2[\ch], d3[\ch]}, [\addr], \stepY
vld4.32 {d4[\ch], d5[\ch], d6[\ch], d7[\ch]}, [\addr], \stepY
vld4.32 {d8[\ch], d9[\ch], d10[\ch], d11[\ch]}, [\addr], \stepY
vld4.32 {d12[\ch], d13[\ch], d14[\ch], d15[\ch]}, [\addr], \stepY
.endm

.macro LOADY_C4 d0, d1, d2, d3, addr, stepX, stepZ, t0
mov \t0, \addr
vld1.32 {\d0[0]}, [\t0], \stepX
vld1.32 {\d1[0]}, [\t0], \stepX
vld1.32 {\d2[0]}, [\t0], \stepX
vld1.32 {\d3[0]}, [\t0]
add \t0, \addr, \stepZ
vld1.32 {\d0[1]}, [\t0], \stepX
vld1.32 {\d1[1]}, [\t0], \stepX
vld1.32 {\d2[1]}, [\t0], \stepX
vld1.32 {\d3[1]}, [\t0]
.endm

.macro TRANS s0, s1, s2, s3, d0, d1, d2, d3
vsub.i8 \d0, \s0, \s2
vadd.i8 \d1, \s1, \s2
vsub.i8 \d2, \s2, \s1
vsub.i8 \d3, \s3, \s1
.endm

.macro SAVE_C4 ch, s0, s1, s2, s3, addr, stepX, stepZ
vst1.32 {\s0[\ch]}, [\addr], \stepX
vst1.32 {\s1[\ch]}, [\addr], \stepX
vst1.32 {\s2[\ch]}, [\addr], \stepX
vst1.32 {\s3[\ch]}, [\addr], \stepX
.endm

.macro SAVE s0, s1, s2, s3, addr, step
vst1.32 {\s0}, [\addr], \step
vst1.32 {\s1}, [\addr], \step
vst1.32 {\s2}, [\addr], \step
vst1.32 {\s3}, [\addr], \step
.endm

#ifdef ENABLE_ARMV82
asm_function _sourceTransXPack4x4Int8
/*
 void _sourceTransXPack4x4Int8(const int8_t* srcStart, int8_t* dstStart, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countCUnit, size_t xC, size_t unit)
r0: srcStart, r1: dstStart, r2:srcZStep, r3: dstXStep, r4: dstZStep, r5: countCUnit, r6: xC, r7: unit
 */
push {r4-r8, r10-r11, lr}
add sp, sp, #32
ldm sp, {r4-r7}
sub sp, sp, #32

lsl r5, r5, #1 // CUnit: 16 -> 8
lsl r7, r7, #2 // unit * 4
Loop_X_4x4_countC4:
    mov r8, r6
    mov r10, r0
    mov r11, r1
    Loop_X_4x4_xC:
        mov r12, r10
        vld4.32 {d0[0], d1[0], d2[0], d3[0]}, [r12], r2
        vld4.32 {d0[1], d1[1], d2[1], d3[1]}, [r12], r2
        TRANS d0, d1, d2, d3, d4, d5, d6, d7
        mov r12, r11
        SAVE_C4 0, d4, d5, d6, d7, r12, r3
        add r12, r11, r4
        SAVE_C4 1, d4, d5, d6, d7, r12, r3
        add r10, r10, r7
        add r11, r11, #4
        subs r8, r8, #1
        bne Loop_X_4x4_xC
    add r0, r0, r2, LSL #1
    add r1, r1, r4, LSL #1
    subs r5, r5, #1
    bne Loop_X_4x4_countC4
pop {r4-r8, r10-r11, pc}

asm_function _sourceTransYPack4x4Int8
/*
 void _sourceTransYPack4x4Int8(const int8_t* srcStart, int8_t* dstStart, size_t srcXStep, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countCUnit, size_t xC)
 r0: srcStart, r1: dstStart, r2: srcXStep, r3: srcZStep, r4: dstXStep, r5: dstZStep, r6: countCUnit, r7: xC
 */
push {r4-r8, r10-r11, lr}
add sp, sp, #32
ldm sp, {r4-r7}
sub sp, sp, #32

lsl r6, r6, #1 // CUnit: 16 -> 8
Loop_Y_4x4_countC4:
    mov r8, r7
    mov r10, r0
    mov r11, r1
    Loop_Y_4x4_xC:
        LOADY_C4 d0, d1, d2, d3, r10, r2, r3, r12
        TRANS d0, d1, d2, d3, d4, d5, d6, d7
        mov r12, r11
        SAVE_C4 0, d4, d5, d6, d7, r12, r4
        add r12, r11, r5
        SAVE_C4 1, d4, d5, d6, d7, r12, r4
        add r10, r10, #4
        add r11, r11, #4
        subs r8, r8, #1
        bne Loop_Y_4x4_xC
    add r0, r0, r3, LSL #1
    add r1, r1, r5, LSL #1
    subs r6, r6, #1
    bne Loop_Y_4x4_countC4
pop {r4-r8, r10-r11, pc}

asm_function _sourceTrans2Pack4x4Int8
/*
 void _sourceTrans2Pack4x4Int8(const int8_t* srcStart, int8_t* dstStart, size_t srcYStep, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countCUnit, size_t xC, size_t unit)
 r0: srcStart, r1: dstStart, r2: srcYStep, r3: srcZStep, r4: dstXStep, r5: dstZStep, r6: countCUnit, r7: xC, r8: unit
 */
push {r4-r8, r10-r11, lr}
add sp, sp, #32
ldm sp, {r4-r8}
sub sp, sp, #32

vpush {q4-q7}

lsl r6, r6, #1 // CUnit: 16 -> 8
lsl r8, r8, #2 // unit * 4
Loop_XY_4x4_countC4:
    mov r10, r7
    mov r11, r0
    mov r12, r1
    Loop_XY_4x4_xC:
        mov lr, r11
        LOAD_4X4 0, lr, r2
        add lr, r11, r3
        LOAD_4X4 0, lr, r2
        TRANS d0, d1, d2, d3, d16, d17, d18, d19
        TRANS d4, d5, d6, d7, d20, d21, d22, d23
        TRANS d8, d9, d10, d11, d24, d25, d26, d27
        TRANS d12, d13, d14, d15, d28, d29, d30, d31
        TRANS d16, d20, d24, d28, d0, d4, d8, d12
        TRANS d17, d21, d25, d29, d1, d5, d9, d13
        TRANS d18, d22, d26, d30, d2, d6, d10, d14
        TRANS d19, d23, d27, d31, d3, d7, d11, d15
        mov lr, r12
        SAVE_C4 0, d0, d1, d2, d3, lr, r4
        SAVE_C4 0, d4, d5, d6, d7, lr, r4
        SAVE_C4 0, d8, d9, d10, d11, lr, r4
        SAVE_C4 0, d12, d13, d14, d15, lr, r4
        add lr, r12, r5
        SAVE_C4 1, d0, d1, d2, d3, lr, r4
        SAVE_C4 1, d4, d5, d6, d7, lr, r4
        SAVE_C4 1, d8, d9, d10, d11, lr, r4
        SAVE_C4 1, d12, d13, d14, d15, lr, r4
        add r11, r11, r8
        add r12, r12, #4
        subs r10, r10, #1
        bne Loop_XY_4x4_xC
    add r0, r0, r3, LSL #1
    add r1, r1, r5, LSL #1
    subs r6, r6, #1
    bne Loop_XY_4x4_countC4
vpop {q4-q7}
pop {r4-r8, r10-r11, pc}

#endif // ENABLE_ARMV82

asm_function _sourceTransXPack4x16Int8
/*
 void _sourceTransXPack4x16Int8(const int8_t* srcStart, int8_t* dstStart, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countCUnit, size_t xC, size_t unit)
 r0: srcStart, r1: dstStart, r2:srcZStep, r3: dstXStep, r4: dstZStep, r5: countCUnit, r6: xC, r7: unit
 */
push {r4-r8, r10-r11, lr}
add sp, sp, #32
ldm sp, {r4-r7}
sub sp, sp, #32

lsl r7, r7, #2 // unit * 4
Loop_X_4x16_countC4:
    mov r8, r6
    mov r10, r0
    mov r11, r1
    Loop_X_4x16_xC:
        mov r12, r10
        vld4.32 {d0[0], d2[0], d4[0], d6[0]}, [r12], r2
        vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [r12], r2
        vld4.32 {d1[0], d3[0], d5[0], d7[0]}, [r12], r2
        vld4.32 {d1[1], d3[1], d5[1], d7[1]}, [r12]
        TRANS q0, q1, q2, q3, q8, q9, q10, q11
        mov r12, r11
        SAVE q8, q9, q10, q11, r12, r3
        add r10, r10, r7
        add r11, r11, #16
        subs r8, r8, #1
        bne Loop_X_4x16_xC
    add r0, r0, r2, LSL #2
    add r1, r1, r4
    subs r5, r5, #1
    bne Loop_X_4x16_countC4
pop {r4-r8, r10-r11, pc}

asm_function _sourceTransYPack4x16Int8
/*
 void _sourceTransYPack4x16Int8(const int8_t* srcStart, int8_t* dstStart, size_t srcXStep, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countCUnit, size_t xC)
 r0: srcStart, r1: dstStart, r2: srcXStep, r3: srcZStep, r4: dstXStep, r5: dstZStep, r6: countCUnit, r7: xC
 */
push {r4-r8, r10-r11, lr}
add sp, sp, #32
ldm sp, {r4-r7}
sub sp, sp, #32

Loop_Y_4x16_countC4:
    mov r8, r7
    mov r10, r0
    mov r11, r1
    Loop_Y_4x16_xC:
        LOADY_C4 d0, d2, d4, d6, r10, r2, r3, r12
        add lr, r10, r3, LSL #1
        LOADY_C4 d1, d3, d5, d7, lr, r2, r3, r12
        TRANS q0, q1, q2, q3, q8, q9, q10, q11
        mov r12, r11
        SAVE q8, q9, q10, q11, r12, r4
        add r10, r10, #4
        add r11, r11, #16
        subs r8, r8, #1
        bne Loop_Y_4x16_xC
    add r0, r0, r3, LSL #2
    add r1, r1, r5
    subs r6, r6, #1
    bne Loop_Y_4x16_countC4
pop {r4-r8, r10-r11, pc}

asm_function _sourceTrans2Pack4x16Int8
/*
 void _sourceTrans2Pack4x16Int8(const int8_t* srcStart, int8_t* dstStart, size_t srcYStep, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countCUnit, size_t xC, size_t unit)
 r0: srcStart, r1: dstStart, r2: srcYStep, r3: srcZStep, r4: dstXStep, r5: dstZStep, r6: countCUnit, r7: xC, r8: unit
 */
push {r4-r8, r10-r11, lr}
add sp, sp, #32
ldm sp, {r4-r8}
sub sp, sp, #32
vpush {q4-q7}

lsl r8, r8, #2 // unit * 4
Loop_XY_4x16_countC4:
    mov r10, r7
    mov r11, r0
    mov r12, r1
    Loop_XY_4x16_xC:
        mov lr, r11
        LOAD_4X4 0, lr, r2
        add lr, r11, r3
        LOAD_4X4 1, lr, r2
        TRANS d0, d1, d2, d3, d16, d17, d18, d19
        TRANS d4, d5, d6, d7, d20, d21, d22, d23
        TRANS d8, d9, d10, d11, d24, d25, d26, d27
        TRANS d12, d13, d14, d15, d28, d29, d30, d31
        add r11, r11, r3, LSL #1
        TRANS d16, d20, d24, d28, d0, d4, d8, d12
        TRANS d17, d21, d25, d29, d1, d5, d9, d13
        TRANS d18, d22, d26, d30, d2, d6, d10, d14
        TRANS d19, d23, d27, d31, d3, d7, d11, d15
        mov lr, r12
        SAVE d0, d1, d2, d3, lr, r4
        SAVE d4, d5, d6, d7, lr, r4
        SAVE d8, d9, d10, d11, lr, r4
        SAVE d12, d13, d14, d15, lr, r4

        mov lr, r11
        LOAD_4X4 0, lr, r2
        add lr, r11, r3
        LOAD_4X4 1, lr, r2
        TRANS d0, d1, d2, d3, d16, d17, d18, d19
        TRANS d4, d5, d6, d7, d20, d21, d22, d23
        TRANS d8, d9, d10, d11, d24, d25, d26, d27
        TRANS d12, d13, d14, d15, d28, d29, d30, d31
        sub r11, r11, r3, LSL #1
        TRANS d16, d20, d24, d28, d0, d4, d8, d12
        TRANS d17, d21, d25, d29, d1, d5, d9, d13
        TRANS d18, d22, d26, d30, d2, d6, d10, d14
        TRANS d19, d23, d27, d31, d3, d7, d11, d15
        add lr, r12, #8
        SAVE d0, d1, d2, d3, lr, r4
        SAVE d4, d5, d6, d7, lr, r4
        SAVE d8, d9, d10, d11, lr, r4
        SAVE d12, d13, d14, d15, lr, r4

        add r11, r11, r8
        add r12, r12, #16
        subs r10, r10, #1
        bne Loop_XY_4x16_xC
    add r0, r0, r3, LSL #2
    add r1, r1, r5
    subs r6, r6, #1
    bne Loop_XY_4x16_countC4

vpop {q4-q7}
pop {r4-r8, r10-r11, pc}

#endif
#endif
