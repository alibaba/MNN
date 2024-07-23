//
//  MNNDynamicUpdateConvBiasScale.S
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.macro Round z0, z1, z2, z3
    fcvtzs \z0\().4s, \z0\().4s
    fcvtzs \z1\().4s, \z1\().4s
    fcvtzs \z2\().4s, \z2\().4s
    fcvtzs \z3\().4s, \z3\().4s
.endm

.macro MUL_CONSTANT s0, s1, s2, s3, z0
    fmul \s0\().4s, \s0\().4s, \z0\().4s
    fmul \s1\().4s, \s1\().4s, \z0\().4s
    fmul \s2\().4s, \s2\().4s, \z0\().4s
    fmul \s3\().4s, \s3\().4s, \z0\().4s
.endm

.macro DIV4 s0, s1, s2, s3, z0, z1, z2, z3
    fdiv \s0\().4s, \s0\().4s, \z0\().4s 
    fdiv \s1\().4s, \s1\().4s, \z1\().4s 
    fdiv \s2\().4s, \s2\().4s, \z2\().4s 
    fdiv \s3\().4s, \s3\().4s, \z3\().4s 
.endm

.macro SUB4 s0, s1, s2, s3, z0, z1, z2, z3
    fsub \s0\().4s, \s0\().4s, \z0\().4s 
    fsub \s1\().4s, \s1\().4s, \z1\().4s 
    fsub \s2\().4s, \s2\().4s, \z2\().4s 
    fsub \s3\().4s, \s3\().4s, \z3\().4s 
.endm

.macro Float32ToHalf s0, s1, s2, s3, d0, d1
    fcvtn \d0\().4h,  \s0\().4s
    fcvtn2 \d0\().8h, \s1\().4s
    fcvtn \d1\().4h,  \s2\().4s
    fcvtn2 \d1\().8h, \s3\().4s
.endm

/*
Note: Only used in dynamic quant,so do not need compare min max!
 */
asm_function MNNDynamicUpdateConvBiasScale
//MNNDynamicUpdateConvBiasScale(biasFloat.data(), scaleFloat.data(), biasfp32, weightDequantScale,
//inputScale, weightKernelSum, inputZero, UP_DIV(output->channel(), 4), alphaSize)
//x0:biasFloat, x1:scaleFloat, x2:biasfp32, x3:weightDequantScale, x4:inputScale, x5:weightKernelSum, x6:inputZero, x7:ocQuad
//Load from sp: x9: scaleSize

ldr x9, [sp, #0]
stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

ld1r {v31.4s}, [x4] // input dequant scale
ld1r {v30.4s}, [x6] // input dequant zero:fp32 zero

lsr x9, x9, #2
// fuse scale

SCALE_L24:
cmp x9, #24
blt SCALE_L16

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x3], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x3], #64
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x3], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x3], #64
ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x3], #64
ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x3], #64
MUL_CONSTANT v0, v1, v2, v3, v31 // w_scale *  x_scale
MUL_CONSTANT v4, v5, v6, v7, v31
MUL_CONSTANT v8, v9, v10, v11, v31
MUL_CONSTANT v12, v13, v14, v15, v31
MUL_CONSTANT v16, v17, v18, v19, v31
MUL_CONSTANT v20, v21, v22, v23, v31
sub x9, x9, #24
st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x1], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x1], #64
st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64
st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x1], #64
b SCALE_L24

SCALE_L16:
cmp x9, #16
blt SCALE_L8

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x3], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x3], #64
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x3], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x3], #64
MUL_CONSTANT v0, v1, v2, v3, v31 // w_scale *  x_scale
MUL_CONSTANT v4, v5, v6, v7, v31
MUL_CONSTANT v8, v9, v10, v11, v31
MUL_CONSTANT v12, v13, v14, v15, v31
sub x9, x9, #16
st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x1], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x1], #64
b SCALE_L16

SCALE_L8:
cmp x9, #8
blt SCALE_L4

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x3], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x3], #64
MUL_CONSTANT v0, v1, v2, v3, v31 // w_scale *  x_scale
MUL_CONSTANT v4, v5, v6, v7, v31
sub x9, x9, #8
st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
b SCALE_L8

SCALE_L4:
cmp x9, #4
blt SCALE_L1

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x3], #64
MUL_CONSTANT v0, v1, v2, v3, v31 // w_scale *  x_scale
sub x9, x9, #4
st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
b SCALE_L4

SCALE_L1:
cmp x9, #1
blt BIAS_L8

ld1 {v0.4s}, [x3], #16
fmul v0.4s, v0.4s, v31.4s
sub x9, x9, #1
st1 {v0.4s}, [x1], #16
b SCALE_L1

// Bias:
BIAS_L16:
cmp x7, #16
blt BIAS_L8

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x2], #64 // oldbias
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x2], #64
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x2], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x2], #64
ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x5], #64 // weightKernelSum
ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x5], #64
ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x5], #64

sub x7, x7, #16

MUL_CONSTANT v16, v17, v18, v19, v30 // w_sum * x_zero
MUL_CONSTANT v20, v21, v22, v23, v30 // w_sum * x_zero
MUL_CONSTANT v24, v25, v26, v27, v30 // w_sum * x_zero

SUB4 v0, v1, v2, v3, v16, v17, v18, v19
SUB4 v4, v5, v6, v7, v20, v21, v22, v23
ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x5], #64
SUB4 v8, v9, v10, v11, v24, v25, v26, v27
MUL_CONSTANT v16, v17, v18, v19, v30 // w_sum * x_zero
SUB4 v12, v13, v14, v15, v16, v17, v18, v19

st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64 // bias float
st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], #64
b BIAS_L16

BIAS_L8:
cmp x7, #8
blt BIAS_L4

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x2], #64 // oldbias
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x2], #64
ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x5], #64 // weightKernelSum
ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x5], #64
sub x7, x7, #8

MUL_CONSTANT v16, v17, v18, v19, v30 // w_sum * x_zero
MUL_CONSTANT v20, v21, v22, v23, v30 // w_sum * x_zero
SUB4 v0, v1, v2, v3, v16, v17, v18, v19
SUB4 v4, v5, v6, v7, v20, v21, v22, v23
st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64 // bias float
st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
b BIAS_L8

BIAS_L4:
cmp x7, #4
blt BIAS_L1

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x2], #64 // oldbias
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x5], #64 // weightKernelSum
sub x7, x7, #4

MUL_CONSTANT v8, v9, v10, v11, v30 // w_sum * x_zero
SUB4 v0, v1, v2, v3, v8, v9, v10, v11
st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
b BIAS_L4

BIAS_L1:
cmp x7, #1
blt End
ld1 {v0.4s}, [x2], #16 // oldbias
ld1 {v4.4s}, [x5], #16 // weightKernelSum
sub x7, x7, #1
fmul v4.4s, v4.4s, v30.4s // w_sum * x_zero
fsub v0.4s, v0.4s, v4.4s // oldbias - w_sum * x_zero
st1 {v0.4s}, [x0], #16
b BIAS_L1

End:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret
#endif
