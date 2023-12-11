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

asm_function NEON_MNNGelu_BF16
//void NEON_MNNGelu_BF16(float* dst, const float* src, size_t size, float* parameters);

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


dup v15.4s, w4        // v15: [0.044715f]x4
dup v14.4s, w5        // v14: [0.79788458f]x4
dup v13.4s, w6        // v13: [378.f]x4
dup v12.4s, w7        // v12: [17325.f]x4
dup v11.4s, w8        // v11: [135135.f]x4
dup v10.4s, w9        // v10: [28.f]x4
dup v9.4s, w10        // v9: [3150.f]x4
dup v8.4s, w11        // v8: [62370.f]x4

mov w4, #5
mov w5, #-5

dup v30.4s, w4
dup v31.4s, w5

scvtf v30.4s, v30.4s
scvtf v31.4s, v31.4s

GeluZLoop:

ld1 {v0.4h, v1.4h}, [x1], #16   // v0, v1: 4xint16_t

shll v0.4s, v0.4h, #16
shll v1.4s, v1.4h, #16

fmul v2.4s, v0.4s, v0.4s
fmul v3.4s, v1.4s, v1.4s
fmul v2.4s, v2.4s, v0.4s
fmul v3.4s, v3.4s, v1.4s

fmul v2.4s, v2.4s, v15.4s
fadd v2.4s, v2.4s, v0.4s
fmul v3.4s, v3.4s, v15.4s
fadd v3.4s, v3.4s, v1.4s

fmul v2.4s, v2.4s, v14.4s
fmul v3.4s, v3.4s, v14.4s

fmax v2.4s, v31.4s, v2.4s
fmax v3.4s, v31.4s, v3.4s
fmin v2.4s, v30.4s, v2.4s
fmin v3.4s, v30.4s, v3.4s

// tanh(value)
fmul v4.4s, v2.4s, v2.4s     // q4: value*value
fmul v5.4s, v3.4s, v3.4s     // q5: value*value
// a
fadd v6.4s, v4.4s, v13.4s
fadd v7.4s, v5.4s, v13.4s
fmul v6.4s, v6.4s, v4.4s
fmul v7.4s, v7.4s, v5.4s
fadd v6.4s, v6.4s, v12.4s
fadd v7.4s, v7.4s, v12.4s
fmul v6.4s, v6.4s, v4.4s
fmul v7.4s, v7.4s, v5.4s
fadd v6.4s, v6.4s, v11.4s
fadd v7.4s, v7.4s, v11.4s
fmul v6.4s, v6.4s, v2.4s
fmul v7.4s, v7.4s, v3.4s
//b
fmul v2.4s, v4.4s, v10.4s
fmul v3.4s, v5.4s, v10.4s
fadd v2.4s, v2.4s, v9.4s
fadd v3.4s, v3.4s, v9.4s
fmul v2.4s, v2.4s, v4.4s
fmul v3.4s, v3.4s, v5.4s
fadd v2.4s, v2.4s, v8.4s
fadd v3.4s, v3.4s, v8.4s
fmul v2.4s, v2.4s, v4.4s
fmul v3.4s, v3.4s, v5.4s
fadd v2.4s, v2.4s, v11.4s
fadd v3.4s, v3.4s, v11.4s
//a/b
fdiv v6.4s, v6.4s, v2.4s
fdiv v7.4s, v7.4s, v3.4s
// border case
fmov v2.4s, #1.0
fmov v3.4s, #-1.0
fmov v4.4s, #0.5
fmin v6.4s, v6.4s, v2.4s
fmin v7.4s, v7.4s, v2.4s
fmax v6.4s, v6.4s, v3.4s
fmax v7.4s, v7.4s, v3.4s
// tanh(value)

fadd v6.4s, v6.4s, v2.4s
fadd v7.4s, v7.4s, v2.4s
fmul v6.4s, v6.4s, v0.4s
fmul v7.4s, v7.4s, v1.4s
fmul v6.4s, v6.4s, v4.4s
fmul v7.4s, v7.4s, v4.4s

shrn v6.4h, v6.4s, #16
shrn v7.4h, v7.4s, #16
st1 {v6.4h, v7.4h}, [x0], #16

subs x2, x2, #1
bne GeluZLoop

GeluEnd:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret
#endif
