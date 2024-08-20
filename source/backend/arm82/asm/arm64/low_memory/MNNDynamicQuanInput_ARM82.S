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
 */
asm_function DynamicQuanInput_ARM82
//void DynamicQuanInput_ARM82(const float* src, int8_t* dst, size_t sizeQuad, float* scale, size_t aMin, size_t aMax, size_t zeroPoint);
//x0:src, x1:dst, x2:sizeQuad, x3:scale, x4:aMin, x5:aMax, x6:zeroPoint
stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

ld1 {v29.s}[0], [x3] // Load scale
// copy zero point
dup v30.4s, w6
fcvtn v31.4h, v29.4s
scvtf v30.4s, v30.4s

dup v31.8h, v31.h[0]
fcvtn v30.4h, v30.4s
dup v30.8h, v30.h[0]

FL28:
cmp x2, #28
blt FL24

FLLoop28:
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
st1 {v28.16b, v29.16b}, [x1], #32
INT16_TO_INT8_8 v16, v17, v18, v19, v20, v21, v22, v23, v6, v7, v8, v9
st1 {v0.16b, v1.16b}, [x1], #32
INT16_TO_INT8_4 v24, v25, v26, v27, v10, v11

st1 {v2.16b, v3.16b, v4.16b, v5.16b}, [x1], #64
st1 {v6.16b, v7.16b, v8.16b, v9.16b}, [x1], #64
st1 {v10.16b, v11.16b}, [x1], #32

bge FLLoop28

FL24:
cmp x2, #24
blt FL16

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
INT16_TO_INT8_8 v8, v9, v10, v11, v12, v13, v14, v15, v0, v1, v2, v3
INT16_TO_INT8_8 v16, v17, v18, v19, v20, v21, v22, v23, v4, v5, v6, v7

st1 {v24.16b, v25.16b, v26.16b, v27.16b}, [x1], #64
st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
st1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x1], #64

bge FLLoop24

FL16:
cmp x2, #16
blt FL8

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
INT16_TO_INT8_8 v8, v9, v10, v11, v12, v13, v14, v15, v0, v1, v2, v3

st1 {v24.16b, v25.16b, v26.16b, v27.16b}, [x1], #64
st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64

bge FLLoop16

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
st1 {v24.16b, v25.16b, v26.16b, v27.16b}, [x1], #64

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
st1 {v24.16b, v25.16b}, [x1], #32

bge FLLoop4

FL1:
cmp x2, #0
beq FLEnd

FLLoop1:
ld1 {v0.8h}, [x0], #16
fmul v0.8h, v0.8h, v31.8h
fadd v0.8h, v0.8h, v30.8h

fcvtas v0.8h, v0.8h
sqxtn v0.8b, v0.8h

st1 {v0.d}[0], [x1], #8

subs x2, x2, #1
bne FLLoop1

FLEnd:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret
#endif
