//
//  DynamicQuanInput_ARM82.S
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.macro SCALE_TO_FLOAT_8 s0, s1, s2, s3, s4, s5, s6, s7, z0
    fmul \s0\().8h, \s0\().8h, \z0\().8h
    fmul \s1\().8h, \s1\().8h, \z0\().8h
    fmul \s2\().8h, \s2\().8h, \z0\().8h
    fmul \s3\().8h, \s3\().8h, \z0\().8h
    fmul \s4\().8h, \s4\().8h, \z0\().8h
    fmul \s5\().8h, \s5\().8h, \z0\().8h
    fmul \s6\().8h, \s6\().8h, \z0\().8h
    fmul \s7\().8h, \s7\().8h, \z0\().8h
.endm

.macro SCALE_TO_FLOAT_4 s0, s1, s2, s3, z0
    fmul \s0\().8h, \s0\().8h, \z0\().8h
    fmul \s1\().8h, \s1\().8h, \z0\().8h
    fmul \s2\().8h, \s2\().8h, \z0\().8h
    fmul \s3\().8h, \s3\().8h, \z0\().8h
.endm

.macro ADD_ZEROPOINT_8 s0, s1, s2, s3, s4, s5, s6, s7, z0
    fadd \s0\().8h, \s0\().8h, \z0\().8h
    fadd \s1\().8h, \s1\().8h, \z0\().8h
    fadd \s2\().8h, \s2\().8h, \z0\().8h
    fadd \s3\().8h, \s3\().8h, \z0\().8h
    fadd \s4\().8h, \s4\().8h, \z0\().8h
    fadd \s5\().8h, \s5\().8h, \z0\().8h
    fadd \s6\().8h, \s6\().8h, \z0\().8h
    fadd \s7\().8h, \s7\().8h, \z0\().8h
.endm

.macro ADD_ZEROPOINT_4 s0, s1, s2, s3, z0
    fadd \s0\().8h, \s0\().8h, \z0\().8h
    fadd \s1\().8h, \s1\().8h, \z0\().8h
    fadd \s2\().8h, \s2\().8h, \z0\().8h
    fadd \s3\().8h, \s3\().8h, \z0\().8h
.endm

.macro FLOAT_TO_INT_8 s0, s1, s2, s3, s4, s5, s6, s7
    fcvtas \s0\().8h, \s0\().8h
    fcvtas \s1\().8h, \s1\().8h
    fcvtas \s2\().8h, \s2\().8h
    fcvtas \s3\().8h, \s3\().8h
    fcvtas \s4\().8h, \s4\().8h
    fcvtas \s5\().8h, \s5\().8h
    fcvtas \s6\().8h, \s6\().8h
    fcvtas \s7\().8h, \s7\().8h
.endm

.macro FLOAT_TO_INT_4 s0, s1, s2, s3
    fcvtas \s0\().8h, \s0\().8h
    fcvtas \s1\().8h, \s1\().8h
    fcvtas \s2\().8h, \s2\().8h
    fcvtas \s3\().8h, \s3\().8h
.endm

.macro INT16_TO_INT8_8 s0, s1, s2, s3, s4, s5, s6, s7, d0, d1, d2, d3
    sqxtn \d0\().8b, \s0\().8h
    sqxtn2 \d0\().16b, \s1\().8h
    sqxtn \d1\().8b, \s2\().8h
    sqxtn2 \d1\().16b, \s3\().8h
    sqxtn \d2\().8b, \s4\().8h
    sqxtn2 \d2\().16b, \s5\().8h
    sqxtn \d3\().8b, \s6\().8h
    sqxtn2 \d3\().16b, \s7\().8h
.endm

.macro INT16_TO_INT8_4 s0, s1, s2, s3, d0, d1
    sqxtn \d0\().8b, \s0\().8h
    sqxtn2 \d0\().16b, \s1\().8h
    sqxtn \d1\().8b, \s2\().8h
    sqxtn2 \d1\().16b, \s3\().8h
.endm


/*
Note: Only used in dynamic quant,so do not need compare min max!
1. Quant Float16 to Int8;
2. Pack data from C8 to C4 for Im2Col fixed unit=4
 */
asm_function DynamicQuanInputAndReorder_ARM82
//void DynamicQuanInputAndReorder_ARM82(const float* src, int8_t* dst, size_t planeSize, float* scale, size_t aMin, size_t aMax, size_t zeroPoint, size_t ocQuad, size_t offset);
//x0:src, x1:dst, x2:planeSize, x3:scale, x4:aMin, x5:aMax, x6:zeroPoint, x7:ocQuad, x8:offset
ldr x8, [sp, #0] // plane*4
stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

ld1 {v29.s}[0], [x3] // Load scale
// copy zero point
dup v30.4s, w6
fcvtn v31.4h, v29.4s
scvtf v30.4s, v30.4s

add x13, x8, x8

dup v31.8h, v31.h[0]
fcvtn v30.4h, v30.4s
dup v30.8h, v30.h[0]

mov x9, x1 // first N*4
add x10, x1, x8 // seconde N*4
mov x14, x2 // Reserve planeSize

Outter_Channel_Loop:
cmp x7, #1
blt End

mov x11, x9 // flag address
mov x12, x10

FL28: // N loop
cmp x2, #28
blt FL20

FLLoop28: // N=28

ChannleLoop_28:
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x0], #64
ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64
ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], #64
ld1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64
ld1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x0], #64

SCALE_TO_FLOAT_8 v0, v1, v2, v3, v4, v5, v6, v7, v31
SCALE_TO_FLOAT_8 v8, v9, v10, v11, v12, v13, v14, v15, v31
SCALE_TO_FLOAT_8 v16, v17, v18, v19, v20, v21, v22, v23, v31
SCALE_TO_FLOAT_4 v24, v25, v26, v27, v31
sub x2, x2, #28
ADD_ZEROPOINT_8 v0, v1, v2, v3, v4, v5, v6, v7, v30
ADD_ZEROPOINT_8 v8, v9, v10, v11, v12, v13, v14, v15, v30
ADD_ZEROPOINT_8 v16, v17, v18, v19, v20, v21, v22, v23, v30
ADD_ZEROPOINT_4 v24, v25, v26, v27, v30

FLOAT_TO_INT_8 v0, v1, v2, v3, v4, v5, v6, v7
FLOAT_TO_INT_8 v8, v9, v10, v11, v12, v13, v14, v15
FLOAT_TO_INT_8 v16, v17, v18, v19, v20, v21, v22, v23
FLOAT_TO_INT_4 v24, v25, v26, v27
cmp x2, #28
INT16_TO_INT8_8 v0, v1, v2, v3, v4, v5, v6, v7, v28, v29, v0, v1
INT16_TO_INT8_8 v8, v9, v10, v11, v12, v13, v14, v15, v2, v3, v4, v5
//st1 {v28.16b, v29.16b}, [x1], #32
INT16_TO_INT8_8 v16, v17, v18, v19, v20, v21, v22, v23, v6, v7, v8, v9
//st1 {v0.16b, v1.16b}, [x1], #32
INT16_TO_INT8_4 v24, v25, v26, v27, v10, v11

// Reorder c8->c4, 0,..27 means plane index
uzp1 v12.4s, v28.4s, v29.4s // 0 0 1 1 x 2 2 3 3 -> 0 1 2 3
uzp1 v13.4s, v0.4s, v1.4s // 4 4 5 5 x 6 6 7 7 -> 4 5 6 7
uzp1 v14.4s, v2.4s, v3.4s // 8 8 9 9 x 10 10 11 11 -> 8 9 10 11
uzp1 v15.4s, v4.4s, v5.4s // 12 12 13 13 x 14 14 15 15 -> 12 13 14 15
uzp1 v16.4s, v6.4s, v7.4s // 16 16 17 17 x 18 18 19 19 -> 16 17 18 19
uzp1 v17.4s, v8.4s, v9.4s // 20 20 21 21 x 22 22 23 23 -> 20 21 22 23
uzp1 v18.4s, v10.4s, v11.4s // 24 24 25 25 x 26 26 27 27 -> 24 25 26 27
uzp2 v19.4s, v28.4s, v29.4s
uzp2 v20.4s, v0.4s, v1.4s
uzp2 v21.4s, v2.4s, v3.4s
uzp2 v22.4s, v4.4s, v5.4s
uzp2 v23.4s, v6.4s, v7.4s
uzp2 v24.4s, v8.4s, v9.4s
uzp2 v25.4s, v10.4s, v11.4s

st1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x11], #64
st1 {v16.16b, v17.16b, v18.16b}, [x11], #48
st1 {v19.16b, v20.16b, v21.16b, v22.16b}, [x12], #64
st1 {v23.16b, v24.16b, v25.16b}, [x12], #48

bge FLLoop28

FL24:
cmp x2, #24
blt FL20

FLLoop24:
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x0], #64
ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64
ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], #64
ld1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64

SCALE_TO_FLOAT_8 v0, v1, v2, v3, v4, v5, v6, v7, v31
SCALE_TO_FLOAT_8 v8, v9, v10, v11, v12, v13, v14, v15, v31
SCALE_TO_FLOAT_8 v16, v17, v18, v19, v20, v21, v22, v23, v31
sub x2, x2, #24
ADD_ZEROPOINT_8 v0, v1, v2, v3, v4, v5, v6, v7, v30
ADD_ZEROPOINT_8 v8, v9, v10, v11, v12, v13, v14, v15, v30
ADD_ZEROPOINT_8 v16, v17, v18, v19, v20, v21, v22, v23, v30

FLOAT_TO_INT_8 v0, v1, v2, v3, v4, v5, v6, v7
FLOAT_TO_INT_8 v8, v9, v10, v11, v12, v13, v14, v15
FLOAT_TO_INT_8 v16, v17, v18, v19, v20, v21, v22, v23
cmp x2, #24
INT16_TO_INT8_8 v0, v1, v2, v3, v4, v5, v6, v7, v24, v25, v26, v27
INT16_TO_INT8_8 v8, v9, v10, v11, v12, v13, v14, v15, v28, v29, v0, v1
INT16_TO_INT8_8 v16, v17, v18, v19, v20, v21, v22, v23, v2, v3, v4, v5

// Reorder c8->c4
uzp1 v6.4s, v24.4s, v25.4s // 0 0 1 1 x 2 2 3 3 -> 0 1 2 3
uzp1 v7.4s, v26.4s, v27.4s // 4 4 5 5 x 6 6 7 7 -> 4 5 6 7
uzp1 v8.4s, v28.4s, v29.4s // 8 8 9 9 x 10 10 11 11 -> 8 9 10 11
uzp1 v9.4s, v0.4s, v1.4s // 12 12 13 13 x 14 14 15 15 -> 12 13 14 15
uzp1 v10.4s, v2.4s, v3.4s // 16 16 17 17 x 18 18 19 19 -> 16 17 18 19
uzp1 v11.4s, v4.4s, v5.4s // 20 20 21 21 x 22 22 23 23 -> 20 21 22 23
uzp2 v12.4s, v24.4s, v25.4s
uzp2 v13.4s, v26.4s, v27.4s
uzp2 v14.4s, v28.4s, v29.4s
uzp2 v15.4s, v0.4s, v1.4s
uzp2 v16.4s, v2.4s, v3.4s
uzp2 v17.4s, v4.4s, v5.4s

st1 {v6.16b, v7.16b, v8.16b, v9.16b}, [x11], #64
st1 {v10.16b, v11.16b}, [x11], #32
st1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x12], #64
st1 {v16.16b, v17.16b}, [x12], #32

bge FLLoop24

FL20:
cmp x2, #20
blt FL12

FLLoop20:
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x0], #64
ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64
ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], #64

SCALE_TO_FLOAT_8 v0, v1, v2, v3, v4, v5, v6, v7, v31
SCALE_TO_FLOAT_8 v8, v9, v10, v11, v12, v13, v14, v15, v31
SCALE_TO_FLOAT_4 v16, v17, v18, v19, v31
sub x2, x2, #20
ADD_ZEROPOINT_8 v0, v1, v2, v3, v4, v5, v6, v7, v30
ADD_ZEROPOINT_8 v8, v9, v10, v11, v12, v13, v14, v15, v30
ADD_ZEROPOINT_4 v16, v17, v18, v19, v30

FLOAT_TO_INT_8 v0, v1, v2, v3, v4, v5, v6, v7
FLOAT_TO_INT_8 v8, v9, v10, v11, v12, v13, v14, v15
FLOAT_TO_INT_4 v16, v17, v18, v19
cmp x2, #20
INT16_TO_INT8_8 v0, v1, v2, v3, v4, v5, v6, v7, v24, v25, v26, v27
INT16_TO_INT8_8 v8, v9, v10, v11, v12, v13, v14, v15, v21, v22, v23, v28
INT16_TO_INT8_4 v16, v17, v18, v19, v29, v20

// Reorder c8->c4
uzp1 v0.4s, v24.4s, v25.4s // 0 0 1 1 x 2 2 3 3 -> 0 1 2 3
uzp1 v1.4s, v26.4s, v27.4s // 4 4 5 5 x 6 6 7 7 -> 4 5 6 7
uzp1 v2.4s, v21.4s, v22.4s // 8 8 9 9 x 10 10 11 11 -> 8 9 10 11
uzp1 v3.4s, v23.4s, v28.4s // 12 12 13 13 x 14 14 15 15 -> 12 13 14 15
uzp1 v4.4s, v29.4s, v20.4s // 16 16 17 17 x 18 18 19 19 -> 16 17 18 19
uzp2 v5.4s, v24.4s, v25.4s
uzp2 v6.4s, v26.4s, v27.4s
uzp2 v7.4s, v21.4s, v22.4s
uzp2 v8.4s, v23.4s, v28.4s
uzp2 v9.4s, v29.4s, v20.4s

st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x11], #64
st1 {v4.16b}, [x11], #16
st1 {v5.16b, v6.16b, v7.16b, v8.16b}, [x12], #64
st1 {v9.16b}, [x12], #16

bge FLLoop20

FL16:
cmp x2, #16
blt FL12

FLLoop16:
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x0], #64
ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64

SCALE_TO_FLOAT_8 v0, v1, v2, v3, v4, v5, v6, v7, v31
SCALE_TO_FLOAT_8 v8, v9, v10, v11, v12, v13, v14, v15, v31
sub x2, x2, #16
ADD_ZEROPOINT_8 v0, v1, v2, v3, v4, v5, v6, v7, v30
ADD_ZEROPOINT_8 v8, v9, v10, v11, v12, v13, v14, v15, v30

FLOAT_TO_INT_8 v0, v1, v2, v3, v4, v5, v6, v7
FLOAT_TO_INT_8 v8, v9, v10, v11, v12, v13, v14, v15
cmp x2, #16
INT16_TO_INT8_8 v0, v1, v2, v3, v4, v5, v6, v7, v24, v25, v26, v27
INT16_TO_INT8_8 v8, v9, v10, v11, v12, v13, v14, v15, v20, v21, v22, v23

// Reorder c8->c4
uzp1 v16.4s, v24.4s, v25.4s // 0 0 1 1 x 2 2 3 3 -> 0 1 2 3
uzp1 v17.4s, v26.4s, v27.4s // 4 4 5 5 x 6 6 7 7 -> 4 5 6 7
uzp1 v18.4s, v20.4s, v21.4s // 8 8 9 9 x 10 10 11 11 -> 8 9 10 11
uzp1 v19.4s, v22.4s, v23.4s // 12 12 13 13 x 14 14 15 15 -> 12 13 14 15

uzp2 v0.4s, v24.4s, v25.4s
uzp2 v1.4s, v26.4s, v27.4s
uzp2 v2.4s, v20.4s, v21.4s
uzp2 v3.4s, v22.4s, v23.4s

st1 {v16.16b, v17.16b, v18.16b, v19.16b}, [x11], #64
st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x12], #64

bge FLLoop16

FL12:
cmp x2, #12
blt FL8

FLLoop12:
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x0], #64
ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64

SCALE_TO_FLOAT_8 v0, v1, v2, v3, v4, v5, v6, v7, v31
SCALE_TO_FLOAT_4 v8, v9, v10, v11, v31
sub x2, x2, #12
ADD_ZEROPOINT_8 v0, v1, v2, v3, v4, v5, v6, v7, v30
ADD_ZEROPOINT_4 v8, v9, v10, v11, v30

FLOAT_TO_INT_8 v0, v1, v2, v3, v4, v5, v6, v7
FLOAT_TO_INT_4 v8, v9, v10, v11
cmp x2, #12
INT16_TO_INT8_8 v0, v1, v2, v3, v4, v5, v6, v7, v24, v25, v26, v27
INT16_TO_INT8_4 v8, v9, v10, v11, v20, v21

// Reorder c8->c4
uzp1 v12.4s, v24.4s, v25.4s // 0 0 1 1 x 2 2 3 3 -> 0 1 2 3
uzp2 v16.4s, v24.4s, v25.4s
uzp1 v13.4s, v26.4s, v27.4s // 4 4 5 5 x 6 6 7 7 -> 4 5 6 7
uzp2 v17.4s, v26.4s, v27.4s
uzp1 v14.4s, v20.4s, v21.4s // 8 8 9 9 x 10 10 11 11 -> 8 9 10 11
uzp2 v18.4s, v20.4s, v21.4s

st1 {v12.16b, v13.16b, v14.16b}, [x11], #48
st1 {v16.16b, v17.16b, v18.16b}, [x12], #48

bge FLLoop12

FL8:
cmp x2, #8
blt FL4

FLLoop8:
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x0], #64
sub x2, x2, #8
SCALE_TO_FLOAT_8 v0, v1, v2, v3, v4, v5, v6, v7, v31
ADD_ZEROPOINT_8 v0, v1, v2, v3, v4, v5, v6, v7, v30
cmp x2, #8
FLOAT_TO_INT_8 v0, v1, v2, v3, v4, v5, v6, v7
INT16_TO_INT8_8 v0, v1, v2, v3, v4, v5, v6, v7, v24, v25, v26, v27

// Reorder c8->c4
uzp1 v12.4s, v24.4s, v25.4s // 0 0 1 1 x 2 2 3 3 -> 0 1 2 3
uzp2 v19.4s, v24.4s, v25.4s
uzp1 v13.4s, v26.4s, v27.4s // 4 4 5 5 x 6 6 7 7 -> 4 5 6 7
uzp2 v20.4s, v26.4s, v27.4s

st1 {v12.16b, v13.16b}, [x11], #32
st1 {v19.16b, v20.16b}, [x12], #32

bge FLLoop8

FL4:
cmp x2, #4
blt FL1

FLLoop4:
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
sub x2, x2, #4
SCALE_TO_FLOAT_4 v0, v1, v2, v3, v31
ADD_ZEROPOINT_4 v0, v1, v2, v3, v30
cmp x2, #4
FLOAT_TO_INT_4 v0, v1, v2, v3
INT16_TO_INT8_4 v0, v1, v2, v3, v24, v25

// Reorder c8->c4
uzp1 v12.4s, v24.4s, v25.4s // 0 0 1 1 x 2 2 3 3 -> 0 1 2 3
uzp2 v19.4s, v24.4s, v25.4s

st1 {v12.16b}, [x11], #16
st1 {v19.16b}, [x12], #16
//st1 {v24.16b, v25.16b}, [x1], #32

bge FLLoop4

FL1:
cmp x2, #0
ble FLEnd

FLLoop1:
ld1 {v0.8h}, [x0], #16
fmul v0.8h, v0.8h, v31.8h
fadd v0.8h, v0.8h, v30.8h
sub x2, x2, #1

fcvtas v0.8h, v0.8h
sqxtn v0.8b, v0.8h

cmp x2, #1
st1 {v0.s}[0], [x11], #4
st1 {v0.s}[1], [x12], #4

bge FLLoop1

FLEnd:
sub x7, x7, #1
add x9, x9, x13
add x10, x10, x13
mov x2, x14
b Outter_Channel_Loop

End:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret
#endif
