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

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.macro LOADX_C4 d0, d1, d2, d3, addr, stepZ
ld4 {\d0\().s, \d1\().s, \d2\().s, \d3\().s}[0], [\addr], \stepZ
ld4 {\d0\().s, \d1\().s, \d2\().s, \d3\().s}[1], [\addr], \stepZ
ld4 {\d0\().s, \d1\().s, \d2\().s, \d3\().s}[2], [\addr], \stepZ
ld4 {\d0\().s, \d1\().s, \d2\().s, \d3\().s}[3], [\addr]
.endm

.macro LOADY_C4 d0, d1, d2, d3, addr, stepX, stepZ, t0, t1, t2
add \t0, \addr, \stepX
add \t1, \t0, \stepX
add \t2, \t1, \stepX
ld1 {\d0\().s}[0], [\addr], \stepZ
ld1 {\d1\().s}[0], [\t0], \stepZ
ld1 {\d2\().s}[0], [\t1], \stepZ
ld1 {\d3\().s}[0], [\t2], \stepZ
ld1 {\d0\().s}[1], [\addr], \stepZ
ld1 {\d1\().s}[1], [\t0], \stepZ
ld1 {\d2\().s}[1], [\t1], \stepZ
ld1 {\d3\().s}[1], [\t2], \stepZ
ld1 {\d0\().s}[2], [\addr], \stepZ
ld1 {\d1\().s}[2], [\t0], \stepZ
ld1 {\d2\().s}[2], [\t1], \stepZ
ld1 {\d3\().s}[2], [\t2], \stepZ
ld1 {\d0\().s}[3], [\addr], \stepZ
ld1 {\d1\().s}[3], [\t0], \stepZ
ld1 {\d2\().s}[3], [\t1], \stepZ
ld1 {\d3\().s}[3], [\t2], \stepZ
sub \addr, \addr, \stepZ, LSL #2
.endm

.macro TRANS s0, s1, s2, s3, d0, d1, d2, d3
sub \d0\().16b, \s0\().16b, \s2\().16b
add \d1\().16b, \s1\().16b, \s2\().16b
sub \d2\().16b, \s2\().16b, \s1\().16b
sub \d3\().16b, \s3\().16b, \s1\().16b
.endm

.macro SAVE_C4 s0, s1, s2, s3, addr, stepX, stepZ, t0, t1, t2
add \t0, \addr, \stepX
add \t1, \t0, \stepX
add \t2, \t1, \stepX
st1 {\s0\().s}[0], [\addr], \stepZ
st1 {\s0\().s}[1], [\addr], \stepZ
st1 {\s0\().s}[2], [\addr], \stepZ
st1 {\s0\().s}[3], [\addr], \stepZ
sub \addr, \addr, \stepZ, LSL #2
st1 {\s1\().s}[0], [\t0], \stepZ
st1 {\s1\().s}[1], [\t0], \stepZ
st1 {\s1\().s}[2], [\t0], \stepZ
st1 {\s1\().s}[3], [\t0], \stepZ
st1 {\s2\().s}[0], [\t1], \stepZ
st1 {\s2\().s}[1], [\t1], \stepZ
st1 {\s2\().s}[2], [\t1], \stepZ
st1 {\s2\().s}[3], [\t1], \stepZ
st1 {\s3\().s}[0], [\t2], \stepZ
st1 {\s3\().s}[1], [\t2], \stepZ
st1 {\s3\().s}[2], [\t2], \stepZ
st1 {\s3\().s}[3], [\t2], \stepZ
.endm

.macro SAVE_C16 s0, s1, s2, s3, addr, step
st1 {\s0\().4s}, [\addr], \step
st1 {\s1\().4s}, [\addr], \step
st1 {\s2\().4s}, [\addr], \step
st1 {\s3\().4s}, [\addr], \step
.endm

#ifdef ENABLE_ARMV82
asm_function _sourceTransXPack4x4Int8
/*
 void _sourceTransXPack4x4Int8(const int8_t* srcStart, int8_t* dstStart, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countCUnit, size_t xC, size_t unit)
x0: srcStart, x1: dstStart, x2:srcZStep, x3: dstXStep, x4: dstZStep, x5: countCUnit, x6: xC, x7: unit
 */
lsl x7, x7, #2 // unit * 4
lsl x8, x2, #2 // srcZStep * 4
lsl x9, x4, #2 // dstZStep * 4
Loop_X_4x4_countC4:
    mov x10, x6
    mov x11, x0
    mov x12, x1
    Loop_X_4x4_xC:
        mov x13, x11
        LOADX_C4 v0, v1, v2, v3, x13, x2
        TRANS v0, v1, v2, v3, v4, v5, v6, v7
        SAVE_C4 v4, v5, v6, v7, x12, x3, x4, x13, x14, x15
        add x11, x11, x7
        add x12, x12, #4
        subs x10, x10, #1
        bne Loop_X_4x4_xC
    add x0, x0, x8
    add x1, x1, x9
    subs x5, x5, #1
    bne Loop_X_4x4_countC4
ret

asm_function _sourceTransYPack4x4Int8
/*
 void _sourceTransYPack4x4Int8(const int8_t* srcStart, int8_t* dstStart, size_t srcXStep, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countCUnit, size_t xC)
 x0: srcStart, x1: dstStart, x2: srcXStep, x3: srcZStep, x4: dstXStep, x5: dstZStep, x6: countCUnit, x7: xC
 */
lsl x8, x3, #2 // srcZStep * 4
lsl x9, x5, #2 // dstZStep * 4
Loop_Y_4x4_countC4:
    mov x10, x7
    mov x11, x0
    mov x12, x1
    Loop_Y_4x4_xC:
        LOADY_C4 v0, v1, v2, v3, x11, x2, x3, x13, x14, x15
        TRANS v0, v1, v2, v3, v4, v5, v6, v7
        SAVE_C4 v4, v5, v6, v7, x12, x4, x5, x13, x14, x15
        add x11, x11, #4
        add x12, x12, #4
        subs x10, x10, #1
        bne Loop_Y_4x4_xC
    add x0, x0, x8
    add x1, x1, x9
    subs x6, x6, #1
    bne Loop_Y_4x4_countC4
ret

asm_function _sourceTrans2Pack4x4Int8
/*
 void _sourceTrans2Pack4x4Int8(const int8_t* srcStart, int8_t* dstStart, size_t srcYStep, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countCUnit, size_t xC, size_t unit)
 x0: srcStart, x1: dstStart, x2: srcYStep, x3: srcZStep, x4: dstXStep, x5: dstZStep, x6: countCUnit, x7: xC, x8: unit
 */
ldr x8, [sp]
lsl x8, x8, #2 // unit * 4
lsl x9, x3, #2 // srcZStep * 4
lsl x10, x5, #2 // dstZStep * 4
stp x19, x20, [sp, #-16]!
Loop_XY_4x4_countC4:
    mov x11, x7
    mov x12, x0
    mov x13, x1
    Loop_XY_4x4_xC:
        mov x14, x12
        add x15, x12, x2
        add x19, x15, x2
        add x20, x19, x2
        LOADX_C4 v0, v1, v2, v3, x14, x3
        LOADX_C4 v4, v5, v6, v7, x15, x3
        LOADX_C4 v8, v9, v10, v11, x19, x3
        LOADX_C4 v12, v13, v14, v15, x20, x3
        TRANS v0, v1, v2, v3, v16, v17, v18, v19
        TRANS v4, v5, v6, v7, v20, v21, v22, v23
        TRANS v8, v9, v10, v11, v24, v25, v26, v27
        TRANS v12, v13, v14, v15, v28, v29, v30, v31
        TRANS v16, v20, v24, v28, v0, v4, v8, v12
        TRANS v17, v21, v25, v29, v1, v5, v9, v13
        TRANS v18, v22, v26, v30, v2, v6, v10, v14
        TRANS v19, v23, v27, v31, v3, v7, v11, v15
        SAVE_C4 v0, v1, v2, v3, x13, x4, x5, x15, x19, x20
        add x14, x13, x4, LSL #2
        SAVE_C4 v4, v5, v6, v7, x14, x4, x5, x15, x19, x20
        add x14, x14, x4, LSL #2
        SAVE_C4 v8, v9, v10, v11, x14, x4, x5, x15, x19, x20
        add x14, x14, x4, LSL #2
        SAVE_C4 v12, v13, v14, v15, x14, x4, x5, x15, x19, x20
        add x12, x12, x8
        add x13, x13, #4
        subs x11, x11, #1
        bne Loop_XY_4x4_xC
    add x0, x0, x9
    add x1, x1, x10
    subs x6, x6, #1
    bne Loop_XY_4x4_countC4
ldp x19, x20, [sp], #16
ret

#endif // ENABLE_ARMV82

asm_function _sourceTransXPack4x16Int8
/*
 void _sourceTransXPack4x16Int8(const int8_t* srcStart, int8_t* dstStart, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countCUnit, size_t xC, size_t unit)
 x0: srcStart, x1: dstStart, x2:srcZStep, x3: dstXStep, x4: dstZStep, x5: countCUnit, x6: xC, x7: unit
 */
lsl x7, x7, #2 // unit * 4
lsl x8, x2, #2 // srcZStep * 4
Loop_X_4x16_countC4:
    mov x10, x6
    mov x11, x0
    mov x12, x1
    Loop_X_4x16_xC:
        mov x13, x11
        LOADX_C4 v0, v1, v2, v3, x13, x2
        TRANS v0, v1, v2, v3, v4, v5, v6, v7
        mov x13, x12
        SAVE_C16 v4, v5, v6, v7, x13, x3
        add x11, x11, x7
        add x12, x12, #16
        subs x10, x10, #1
        bne Loop_X_4x16_xC
    add x0, x0, x8
    add x1, x1, x4
    subs x5, x5, #1
    bne Loop_X_4x16_countC4
ret

asm_function _sourceTransYPack4x16Int8
/*
 void _sourceTransYPack4x16Int8(const int8_t* srcStart, int8_t* dstStart, size_t srcXStep, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countCUnit, size_t xC)
 x0: srcStart, x1: dstStart, x2: srcXStep, x3: srcZStep, x4: dstXStep, x5: dstZStep, x6: countCUnit, x7: xC
 */
lsl x8, x3, #2 // srcZStep * 4
Loop_Y_4x16_countC4:
    mov x10, x7
    mov x11, x0
    mov x12, x1
    Loop_Y_4x16_xC:
        LOADY_C4 v0, v1, v2, v3, x11, x2, x3, x13, x14, x15
        TRANS v0, v1, v2, v3, v4, v5, v6, v7
        mov x13, x12
        SAVE_C16 v4, v5, v6, v7, x13, x4
        add x11, x11, #4
        add x12, x12, #16
        subs x10, x10, #1
        bne Loop_Y_4x16_xC
    add x0, x0, x8
    add x1, x1, x5
    subs x6, x6, #1
    bne Loop_Y_4x16_countC4
ret

asm_function _sourceTrans2Pack4x16Int8
/*
 void _sourceTrans2Pack4x16Int8(const int8_t* srcStart, int8_t* dstStart, size_t srcYStep, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countCUnit, size_t xC, size_t unit)
 x0: srcStart, x1: dstStart, x2: srcYStep, x3: srcZStep, x4: dstXStep, x5: dstZStep, x6: countCUnit, x7: xC, x8: unit
 */
ldr x8, [sp]
lsl x8, x8, #2 // unit * 4
lsl x9, x3, #2 // srcZStep * 4
stp x19, x20, [sp, #-16]!
Loop_XY_4x16_countC4:
    mov x11, x7
    mov x12, x0
    mov x13, x1
    Loop_XY_4x16_xC:
        mov x14, x12
        add x15, x12, x2
        add x19, x15, x2
        add x20, x19, x2
        LOADX_C4 v0, v1, v2, v3, x14, x3
        LOADX_C4 v4, v5, v6, v7, x15, x3
        LOADX_C4 v8, v9, v10, v11, x19, x3
        LOADX_C4 v12, v13, v14, v15, x20, x3
        TRANS v0, v1, v2, v3, v16, v17, v18, v19
        TRANS v4, v5, v6, v7, v20, v21, v22, v23
        TRANS v8, v9, v10, v11, v24, v25, v26, v27
        TRANS v12, v13, v14, v15, v28, v29, v30, v31
        TRANS v16, v20, v24, v28, v0, v4, v8, v12
        TRANS v17, v21, v25, v29, v1, v5, v9, v13
        TRANS v18, v22, v26, v30, v2, v6, v10, v14
        TRANS v19, v23, v27, v31, v3, v7, v11, v15
        mov x10, x13
        SAVE_C16 v0, v1, v2, v3, x10, x4
        SAVE_C16 v4, v5, v6, v7, x10, x4
        SAVE_C16 v8, v9, v10, v11, x10, x4
        SAVE_C16 v12, v13, v14, v15, x10, x4
        add x12, x12, x8
        add x13, x13, #16
        subs x11, x11, #1
        bne Loop_XY_4x16_xC
    add x0, x0, x9
    add x1, x1, x5
    subs x6, x6, #1
    bne Loop_XY_4x16_countC4
ldp x19, x20, [sp], #16
ret

#endif // __aarch64__
