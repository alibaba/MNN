//  MNNCubicSampleC16.S
//  MNN
//
//  Created by MNN on 2019/01/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5
asm_function MNNCubicSampleC16
// void MNNCubicSampleC16(const int8_t* src, float* dst, int32_t* position, const float* factor, int8_t* zeroPoint, size_t number)

// Auto load:
// x0: src, x1: dst, x2: position, x3: factor, x4: zeroPoint, x5: number

stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

cmp x5, #0
beq END

mov w15, #16
uxtw x15, w15

L1Loop:
ld1 {v31.s}[0], [x3], #4

fmov s1, #1.0

dup v31.4s, v31.s[0]    // v31: t
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
fadd s27, s27, s1       // bo   
fadd s26, s26, s1       // c0 

dup v3.4s, v27.s[0]     // b0
dup v29.4s, v26.s[0]    // c0

fadd s23, s31, s1       // t_a
fmul s22, s23, s23      // t_a^2
fmul s21, s22, s23      // t_a^3
fadd s20, s30, s1       // t_b
fmul s19, s20, s20      // t_b^2
fmul s18, s19, s20      // t_b^3
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

ldr w7, [x2, #0]
ldr w8, [x2, #4]
ldr w9, [x2, #8]
ldr w10, [x2, #12]
add x2, x2, #16
uxtw x7, w7
uxtw x8, w8
uxtw x9, w9
uxtw x10, w10

ld1r {v11.8b}, [x4]

mul x7, x7, x15
mul x8, x8, x15
mul x9, x9, x15
mul x10, x10, x15
add x7, x0, x7
add x8, x0, x8
add x9, x0, x9
add x10,x0, x10

ld1 {v0.16b}, [x7]
ld1 {v8.16b}, [x8]
ld1 {v15.16b}, [x9]
ld1 {v22.16b}, [x10]

sxtl v1.8h, v0.8b        // v1: int16x8_t
sxtl2 v2.8h, v0.16b
sxtl v9.8h, v8.8b
sxtl2 v10.8h, v8.16b
sxtl v16.8h, v15.8b
sxtl2 v17.8h, v15.16b
sxtl v23.8h, v22.8b
sxtl2 v24.8h, v22.16b

ssubw v1.8h, v1.8h, v11.8b
ssubw v2.8h, v2.8h, v11.8b
ssubw v9.8h, v9.8h, v11.8b
ssubw v10.8h, v10.8h, v11.8b
ssubw v16.8h, v16.8h, v11.8b
ssubw v17.8h, v17.8h, v11.8b
ssubw v23.8h, v23.8h, v11.8b
ssubw v24.8h, v24.8h, v11.8b

sxtl v4.4s, v1.4h
sxtl2 v5.4s, v1.8h
sxtl v6.4s, v2.4h
sxtl2 v7.4s, v2.8h
sxtl v11.4s, v9.4h
sxtl2 v12.4s, v9.8h
sxtl v13.4s, v10.4h
sxtl2 v14.4s, v10.8h

sxtl  v18.4s, v16.4h
sxtl2 v19.4s, v16.8h
sxtl  v20.4s, v17.4h
sxtl2 v21.4s, v17.8h
sxtl  v25.4s, v23.4h
sxtl2 v26.4s, v23.8h
sxtl  v27.4s, v24.4h
sxtl2 v28.4s, v24.8h

scvtf v4.4s, v4.4s    // A
scvtf v5.4s, v5.4s
scvtf v6.4s, v6.4s
scvtf v7.4s, v7.4s
scvtf v11.4s, v11.4s  // B
scvtf v12.4s, v12.4s
scvtf v13.4s, v13.4s
scvtf v14.4s, v14.4s
scvtf v18.4s, v18.4s  // C
scvtf v19.4s, v19.4s
scvtf v20.4s, v20.4s
scvtf v21.4s, v21.4s
scvtf v25.4s, v25.4s  // D
scvtf v26.4s, v26.4s
scvtf v27.4s, v27.4s
scvtf v28.4s, v28.4s

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
st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64

sub x5, x5, #1
cmp x5, #1
bge L1Loop

END:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret

#endif
