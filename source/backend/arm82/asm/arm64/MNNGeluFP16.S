//
//  MNNGelu.S
//  MNN
//
//  Created by MNN on 2023/2/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

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
ldr w6, [x3, #8]       // w6, 378.f
ldr w7, [x3, #12]      // w7, 17325.f
ldr w8, [x3, #16]      // w8, 135135.f
ldr w9, [x3, #20]      // w9, 28.f
ldr w10, [x3, #24]     // w10, 3150.f
ldr w11, [x3, #28]     // w11, 62370.f


dup v15.8h, w4        // v15: [0.044715f]x4
dup v14.8h, w5        // v14: [0.79788458f]x4
dup v13.8h, w6        // v13: [378.f]x4
dup v12.8h, w7        // v12: [17325.f]x4
dup v11.8h, w8        // v11: [135135.f]x4
dup v10.8h, w9        // v10: [28.f]x4
dup v9.8h, w10        // v9: [3150.f]x4
dup v8.8h, w11        // v8: [62370.f]x4

mov w4, #5.0
mov w5, #-5.0

GeluZLoop:

ld1 {v0.8h, v1.8h}, [x1], #32   // v0, v1: fp32x4

fmul v2.8h, v0.8h, v0.8h
fmul v3.8h, v1.8h, v1.8h
fmul v2.8h, v2.8h, v0.8h
fmul v3.8h, v3.8h, v1.8h

fmul v2.8h, v2.8h, v15.8h
fadd v2.8h, v2.8h, v0.8h
fmul v3.8h, v3.8h, v15.8h
fadd v3.8h, v3.8h, v1.8h

fmul v2.8h, v2.8h, v14.8h
fmul v3.8h, v3.8h, v14.8h

dup v6.8h, w5
dup v7.8h, w4
fmin v2.8h, v2.8h, v7.8h
fmin v3.8h, v3.8h, v7.8h
fmax v2.8h, v2.8h, v6.8h
fmax v3.8h, v3.8h, v6.8h

// tanh(value)
fmul v4.8h, v2.8h, v2.8h     // q4: value*value
fmul v5.8h, v3.8h, v3.8h     // q5: value*value
// a
fadd v6.8h, v4.8h, v13.8h
fadd v7.8h, v5.8h, v13.8h
fmul v6.8h, v6.8h, v4.8h
fmul v7.8h, v7.8h, v5.8h
fadd v6.8h, v6.8h, v12.8h
fadd v7.8h, v7.8h, v12.8h
fmul v6.8h, v6.8h, v4.8h
fmul v7.8h, v7.8h, v5.8h
fadd v6.8h, v6.8h, v11.8h
fadd v7.8h, v7.8h, v11.8h
fmul v6.8h, v6.8h, v2.8h
fmul v7.8h, v7.8h, v3.8h
//b
fmul v2.8h, v4.8h, v10.8h
fmul v3.8h, v5.8h, v10.8h
fadd v2.8h, v2.8h, v9.8h
fadd v3.8h, v3.8h, v9.8h
fmul v2.8h, v2.8h, v4.8h
fmul v3.8h, v3.8h, v5.8h
fadd v2.8h, v2.8h, v8.8h
fadd v3.8h, v3.8h, v8.8h
fmul v2.8h, v2.8h, v4.8h
fmul v3.8h, v3.8h, v5.8h
fadd v2.8h, v2.8h, v11.8h
fadd v3.8h, v3.8h, v11.8h
//a/b
fdiv v6.8h, v6.8h, v2.8h
fdiv v7.8h, v7.8h, v3.8h
// border case
fmov v2.8h, #1.0
fmov v3.8h, #-1.0
fmov v4.8h, #0.5
fmin v6.8h, v6.8h, v2.8h
fmin v7.8h, v7.8h, v2.8h
fmax v6.8h, v6.8h, v3.8h
fmax v7.8h, v7.8h, v3.8h
// tanh(value)

fadd v6.8h, v6.8h, v2.8h
fadd v7.8h, v7.8h, v2.8h
fmul v6.8h, v6.8h, v0.8h
fmul v7.8h, v7.8h, v1.8h
fmul v6.8h, v6.8h, v4.8h
fmul v7.8h, v7.8h, v4.8h

st1 {v6.8h, v7.8h}, [x0], #32

subs x2, x2, #1
bne GeluZLoop

GeluEnd:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret
#endif
