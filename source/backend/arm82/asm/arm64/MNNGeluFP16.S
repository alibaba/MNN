//
//  MNNGeluFP16.S
//  MNN
//
//  Created by MNN on 2024/2/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

.macro DupFp32ToFp16 z0, z1
    dup v0.4s, \z0
    fcvtn \z1\().4h, v0.4s
    fcvtn2 \z1\().8h, v0.4s
.endm

asm_function MNNGeluFP16
//void MNNGeluFP16(FLOAT16* dst, const FLOAT16* src, size_t size, float* parameters);

//Auto Load:
//x0:dst, x1:src, x2:size, x3: parameters

stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

cmp x2, #0
beq GeluEnd

ldr w4, [x3, #0]       // w4, 0.044715f
ldr w5, [x3, #4]       // w5, 0.79788458f
ldr w6, [x3, #8]       // w6, 13513.5
ldr w7, [x3, #12]      // w7, 1732.5
ldr w8, [x3, #16]      // w8, 37.8
ldr w9, [x3, #20]      // w9, 6237.f
ldr w10, [x3, #24]     // w10, 315.f
ldr w11, [x3, #28]     // w11, 2.8
ldr w12, [x3, #32]     // w12, 1/half_scale

DupFp32ToFp16 w4, v15
DupFp32ToFp16 w5, v14
DupFp32ToFp16 w6, v13
DupFp32ToFp16 w7, v12
DupFp32ToFp16 w8, v11
DupFp32ToFp16 w9, v10
DupFp32ToFp16 w10, v9
DupFp32ToFp16 w11, v8
DupFp32ToFp16 w12, v7

.inst 0x4f00fe9f // fmov v31.8h, #5.0
.inst 0x4f04fe9e // fmov v30.8h, #-5.0
.inst 0x4f03fe1d // fmov v29.8h, #1.0
.inst 0x4f07fe1c // fmov v28.8h, #-1.0
.inst 0x4f03fc1b // fmov v27.8h, #0.5

GeluZLoop:

ld1 {v0.8h}, [x1], #16   // v0, v1: fp16x8

// value = (x + 0.044715 * x^3) * 0.79788458
fmul v1.8h, v0.8h, v0.8h
fmul v2.8h, v1.8h, v0.8h
fmul v2.8h, v2.8h, v15.8h
fadd v2.8h, v2.8h, v0.8h
fmul v2.8h, v2.8h, v14.8h

// clip: -5 ~ 5
fmin v2.8h, v2.8h, v31.8h
fmax v2.8h, v2.8h, v30.8h

// tanh(value) start
// x2 = value * value
fmul v3.8h, v2.8h, v2.8h
// a = value * (135135 + x2 * (17325 + x2 * (378 + x2 * 1.0/half_scale)));
fmul v4.8h, v3.8h, v7.8h
fadd v4.8h, v4.8h, v11.8h
fmul v4.8h, v4.8h, v3.8h
fadd v4.8h, v4.8h, v12.8h
fmul v4.8h, v4.8h, v3.8h
fadd v4.8h, v4.8h, v13.8h
fmul v4.8h, v4.8h, v2.8h
// b = 135135 + x2 * (62370 + x2 * (3150 + x2 * 28));
fmul v5.8h, v3.8h, v8.8h
fadd v5.8h, v5.8h, v9.8h
fmul v5.8h, v5.8h, v3.8h
fadd v5.8h, v5.8h, v10.8h
fmul v5.8h, v5.8h, v3.8h
fadd v5.8h, v5.8h, v13.8h
// a / b
fdiv v6.8h, v4.8h, v5.8h

// border case: -1 ~ 1
fmin v6.8h, v6.8h, v29.8h
fmax v6.8h, v6.8h, v28.8h
// tanh(value) end

fadd v6.8h, v6.8h, v29.8h
fmul v6.8h, v6.8h, v0.8h
fmul v6.8h, v6.8h, v27.8h

st1 {v6.8h}, [x0], #16

subs x2, x2, #1
bne GeluZLoop

GeluEnd:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret
#endif
