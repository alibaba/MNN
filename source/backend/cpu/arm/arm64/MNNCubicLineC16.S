//  MNNCubicLineC16.S
//  MNN
//
//  Created by MNN on 2019/01/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5
asm_function MNNCubicLineC16
// void MNNCubicLineC16(int8_t* dst, const float* A, const float* B, const float* C, const float* D, float* t, int8_t* zeroPoint,
//                     size_t number, ssize_t minValue, ssize_t maxValue);

// Auto load:
// x0: dst, x1: A, x2: B, x3: C, x4: D, x5: t, x6: zeroPoint, x7: number
// Load from sp: x8: minValue, x9: maxValue
ldr x8, [sp, #0]
ldr x9, [sp, #8]
stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

cmp x7, #0
beq END

ldr w5, [x5, #0]
fmov s1, #1.0
ld1r {v0.8b}, [x6]

dup v31.4s, w5    // v31: t
fmov s30, #1.0
fsub s30, s30, s31  // 1-t

fmul s29, s31, s31  // t^2
fmul s28, s30, s30  // (1-t)^2
fmul s27, s31, s29  // t^3
fmul s26, s28, s30  // (1-t)^3

fmov s25, #-2.25
fmov s24, #1.25
fmul s27, s27, s24
fmul s26, s26, s24
fmla s27, s25, v29.s[0]
fmla s26, s25, v28.s[0]
fadd s27, s27, s1     // bo   
fadd s26, s26, s1     // c0 

dup v3.4s, v27.s[0]    // b0
dup v29.4s, v26.s[0]   // c0

fadd s23, s31, s1         // t_a
fmul s22, s23, s23   // t_a^2
fmul s21, s22, s23   // t_a^3
fadd s20, s30, s1         // t_b
fmul s19, s20, s20  // t_b^2
fmul s18, s19, s20  // t_b^3
fmov s31, #-0.75
fmov s30, #3.75
fmov s24, #-6.0
fmov s25, #3.0

fmul s21, s21, s31
fmul s18, s18, s31
fmla s21, s22, v30.s[0]
fmla s18, s19, v30.s[0]
fmla s21, s23, v24.s[0]
fmla s18, s20, v24.s[0]
fadd s21, s25, s21    // a0
fadd s18, s25, s18      // d0
dup v30.4s, v21.s[0]       // a0
dup v31.4s, v18.s[0]       // d0

L1Loop:

ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
ld1 {v11.4s, v12.4s, v13.4s, v14.4s}, [x2], #64
ld1 {v18.4s, v19.4s, v20.4s, v21.4s}, [x3], #64
ld1 {v25.4s, v26.4s, v27.4s, v28.4s}, [x4], #64

fmul v4.4s,  v4.4s, v30.s[0]
fmul v5.4s,  v5.4s, v30.s[0]
fmul v6.4s,  v6.4s, v30.s[0]
fmul v7.4s,  v7.4s, v30.s[0]
fmla v4.4s, v11.4s, v3.s[0]
fmla v5.4s, v12.4s, v3.s[0]
fmla v6.4s, v13.4s, v3.s[0]
fmla v7.4s, v14.4s, v3.s[0]
fmla v4.4s, v18.4s, v29.s[0]
fmla v5.4s, v19.4s, v29.s[0]
fmla v6.4s, v20.4s, v29.s[0]
fmla v7.4s, v21.4s, v29.s[0]
fmla v4.4s, v25.4s, v31.s[0]
fmla v5.4s, v26.4s, v31.s[0]
fmla v6.4s, v27.4s, v31.s[0]
fmla v7.4s, v28.4s, v31.s[0]

fcvtas v4.4s, v4.4s 
fcvtas v5.4s, v5.4s
fcvtas v6.4s, v6.4s
fcvtas v7.4s, v7.4s

dup v18.16b, w8
dup v19.16b, w9

sqxtn  v4.4h, v4.4s
sqxtn2 v4.8h, v5.4s
sqxtn  v6.4h, v6.4s
sqxtn2 v6.8h, v7.4s

saddw v4.8h, v4.8h, v0.8b
saddw v6.8h, v6.8h, v0.8b

sqxtn  v10.8b, v4.8h
sqxtn2 v10.16b, v6.8h

smin v10.16b, v10.16b, v19.16b
smax v10.16b, v10.16b, v18.16b

st1 {v10.16b}, [x0], #16

sub x7, x7, #1
cmp x7, #1
bge L1Loop

END:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret

#endif
