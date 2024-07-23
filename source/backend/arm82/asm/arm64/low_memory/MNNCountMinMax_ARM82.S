//
//  MNNAbsMaxFP16.S
//  MNN
//
//  Created by MNN on 2023/10/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

.macro MaxMin_4 s0, s1, s2, s3, z0, z1, z2, z3 // z0,z1:max z2,z3:min
    fmax \z0\().8h, \s0\().8h, \s1\().8h
    fmax \z1\().8h, \s2\().8h, \s3\().8h
    fmin \z2\().8h, \s0\().8h, \s1\().8h
    fmin \z3\().8h, \s2\().8h, \s3\().8h

    fmax \z0\().8h, \z0\().8h, \z1\().8h
    fmin \z2\().8h, \z2\().8h, \z3\().8h
.endm

.macro Max_6 s0, s1, s2, s3, s4, s5, z0
    fmax \s0\().8h, \s0\().8h, \s4\().8h
    fmax \s1\().8h, \s1\().8h, \s5\().8h
    fmax \s2\().8h, \s2\().8h, \s3\().8h

    fmax \s0\().8h, \s0\().8h, \s1\().8h
    fmax \z0\().8h, \z0\().8h, \s2\().8h

    fmax \z0\().8h, \z0\().8h, \s0\().8h
.endm

.macro Min_6 s0, s1, s2, s3, s4, s5, z0
    fmin \s0\().8h, \s0\().8h, \s4\().8h
    fmin \s1\().8h, \s1\().8h, \s5\().8h
    fmin \s2\().8h, \s2\().8h, \s3\().8h

    fmin \s0\().8h, \s0\().8h, \s1\().8h
    fmin \z0\().8h, \z0\().8h, \s2\().8h

    fmin \z0\().8h, \z0\().8h, \s0\().8h
.endm

.macro Max_5 s0, s1, s2, s3, s4, z0
    fmax \s0\().8h, \s0\().8h, \s3\().8h
    fmax \s1\().8h, \s1\().8h, \s4\().8h
    fmax \z0\().8h, \s2\().8h, \z0\().8h

    fmax \s0\().8h, \s0\().8h, \s1\().8h
    fmax \z0\().8h, \z0\().8h, \s0\().8h

.endm

.macro Min_5 s0, s1, s2, s3, s4, z0
    fmin \s0\().8h, \s0\().8h, \s3\().8h
    fmin \s1\().8h, \s1\().8h, \s4\().8h
    fmin \z0\().8h, \s2\().8h, \z0\().8h

    fmin \s0\().8h, \s0\().8h, \s1\().8h
    fmin \z0\().8h, \z0\().8h, \s0\().8h
.endm

.macro Max_4 s0, s1, s2, s3, z0
    fmax \s0\().8h, \s0\().8h, \s2\().8h
    fmax \s1\().8h, \s1\().8h, \s3\().8h
    fmax \z0\().8h, \s0\().8h, \z0\().8h
    fmax \z0\().8h, \z0\().8h, \s1\().8h

.endm

.macro Min_4 s0, s1, s2, s3, z0
    fmin \s0\().8h, \s0\().8h, \s2\().8h
    fmin \s1\().8h, \s1\().8h, \s3\().8h
    fmin \z0\().8h, \s0\().8h, \z0\().8h
    fmin \z0\().8h, \z0\().8h, \s1\().8h
.endm

.macro Max_3 s0, s1, s2, z0
    fmax \s0\().8h, \s0\().8h, \s2\().8h
    fmax \z0\().8h, \s1\().8h, \z0\().8h
    fmax \z0\().8h, \s0\().8h, \z0\().8h

.endm

.macro Min_3 s0, s1, s2, z0
    fmin \s0\().8h, \s0\().8h, \s2\().8h
    fmin \z0\().8h, \s1\().8h, \z0\().8h
    fmin \z0\().8h, \s0\().8h, \z0\().8h
.endm

.macro Reduce_Max_Min s0, s1
    // 8->4
    fmaxp \s0\().8h, \s0\().8h, \s0\().8h
    fminp \s1\().8h, \s1\().8h, \s1\().8h
    // 4->2
    fmaxp \s0\().8h, \s0\().8h, \s0\().8h
    fminp \s1\().8h, \s1\().8h, \s1\().8h
    // 2->1
    fmaxp \s0\().8h, \s0\().8h, \s0\().8h
    fminp \s1\().8h, \s1\().8h, \s1\().8h
.endm


//void CountMinMaxValue_FP16(float* source, float* minVal, float* maxVal, size_t sizeQuad)
asm_function CountMinMaxValue_FP16

// x0: source, x1:minVal, x2:maxVal, x3:size
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

Start:
ld1 {v31.8h}, [x0], #16
sub x3, x3, #1
mov v30.8h, v31.8h // v30:min v31:max


TILE_24:
cmp x3, #24
blt TILE_20

ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x0], #64
ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64
ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], #64
ld1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64

MaxMin_4 v0, v1, v2, v3, v24, v25, v26, v27 // v24:max, v26:min
MaxMin_4 v4, v5, v6, v7, v28, v29, v0, v1 // v28:max, v0:min
MaxMin_4 v8, v9, v10, v11, v2, v3, v25, v27 // v2:max, v25:min
MaxMin_4 v12, v13, v14, v15, v4, v5, v6, v7 // v4:max, v6:min
MaxMin_4 v16, v17, v18, v19, v1, v3, v10, v27 // v1:max, v10:min
MaxMin_4 v20, v21, v22, v23, v12, v13, v14, v15 // v12:max, v14:min

Max_6 v1, v2, v4, v12, v24, v28, v31
Min_6 v0, v6, v10, v14, v26, v25, v30

sub x3, x3, #24
b TILE_24

TILE_20:
cmp x3, #20
blt TILE_16

ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x0], #64
ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64
ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], #64

MaxMin_4 v0, v1, v2, v3, v24, v25, v26, v27 // v24:max, v26:min
MaxMin_4 v4, v5, v6, v7, v20, v21, v22, v23 // v20:max, v22:min
MaxMin_4 v8, v9, v10, v11, v0, v1, v2, v3 // v0:max, v2:min
MaxMin_4 v12, v13, v14, v15, v4, v5, v6, v7 // v4:max, v6:min
MaxMin_4 v16, v17, v18, v19, v25, v27, v21, v23 // v25:max, v21:min

Max_5 v0, v4, v20, v25, v24, v31
Min_5 v2, v6, v21, v22, v26, v30

sub x3, x3, #20
b TILE_20

TILE_16:
cmp x3, #16
blt TILE_12

ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x0], #64
ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64

MaxMin_4 v0, v1, v2, v3, v24, v25, v26, v27 // v24:max, v26:min
MaxMin_4 v4, v5, v6, v7, v20, v21, v22, v23 // v20:max, v22:min
MaxMin_4 v8, v9, v10, v11, v16, v17, v18, v19 // v16:max, v18:min
MaxMin_4 v12, v13, v14, v15, v0, v1, v2, v3 // v0:max, v2:min

Max_4 v0, v16, v20, v24, v31
Min_4 v2, v18, v22, v26, v30

sub x3, x3, #16
b TILE_16

TILE_12:
cmp x3, #12
blt TILE_8

ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x0], #64
ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64

MaxMin_4 v0, v1, v2, v3, v24, v25, v26, v27 // v24:max, v26:min
MaxMin_4 v4, v5, v6, v7, v20, v21, v22, v23 // v20:max, v22:min
MaxMin_4 v8, v9, v10, v11, v16, v17, v18, v19 // v16:max, v18:min

Max_3 v16, v20, v24, v31
Min_3 v18, v22, v26, v30

sub x3, x3, #12
b TILE_12

TILE_8:
cmp x3, #8
blt TILE_4

ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x0], #64

MaxMin_4 v0, v1, v2, v3, v24, v25, v26, v27 // v24:max, v26:min
MaxMin_4 v4, v5, v6, v7, v20, v21, v22, v23 // v20:max, v22:min

fmax v24.8h, v24.8h, v20.8h
fmin v26.8h, v26.8h, v22.8h
fmax v31.8h, v31.8h, v24.8h
fmin v30.8h, v30.8h, v26.8h

sub x3, x3, #8
b TILE_8

TILE_4:
cmp x3, #4
blt TILE_2

ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64

MaxMin_4 v0, v1, v2, v3, v24, v25, v26, v27 // v24:max, v26:min

fmax v31.8h, v31.8h, v24.8h
fmin v30.8h, v30.8h, v26.8h

sub x3, x3, #4
b TILE_4

TILE_2:
cmp x3, #2
blt TILE_1

ld1 {v0.8h, v1.8h}, [x0], #32

fmax v2.8h, v0.8h, v1.8h
fmin v3.8h, v0.8h, v1.8h

fmax v31.8h, v31.8h, v2.8h
fmin v30.8h, v30.8h, v3.8h

sub x3, x3, #2
b TILE_2

TILE_1:
cmp x3, #1
blt End

ld1 {v0.8h}, [x0], #16

fmax v31.8h, v31.8h, v0.8h
fmin v30.8h, v30.8h, v0.8h

sub x3, x3, #1
b TILE_1

End:
Reduce_Max_Min v31, v30
//fcvtl v30.4s, v30.4h
//fcvtl v31.4s, v31.4h
st1 {v30.h}[0], [x1]
st1 {v31.h}[1], [x2]

ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif
