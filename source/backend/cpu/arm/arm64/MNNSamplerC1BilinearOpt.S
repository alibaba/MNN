//
//  MNNSamplerC1BilinearOpt.S
//  MNN
//
//  Created by MNN on 2018/11/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
//void MNNSamplerC1BilinearOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t iw, size_t ih, size_t yStride);
asm_function MNNSamplerC1BilinearOpt

//Auto: x0:source, x1:dest, x2:points, x3:count
//Load: w4: xMax, w5: yMax, x6:yStride

movi v23.4s, #0
//v22.4s: ih-1, v23.4s:iw-1

ld1 {v30.2s, v31.2s}, [x2]
L8:
cmp x3, #8
blt L1

dup v25.2d, x0
dup v24.4s, w6
dup v22.4s, w5
dup v21.4s, w4

ucvtf v22.4s, v22.4s
ucvtf v21.4s, v21.4s

movi v3.2s, #4
scvtf v3.2s, v3.2s 
fmul v3.2s, v3.2s, v31.2s
dup v26.4s, v3.s[0]
dup v27.4s, v3.s[1]

fadd v2.2s, v30.2s, v31.2s
mov v28.s[0], v30.s[0]
fadd v3.2s, v2.2s, v31.2s
mov v29.s[0], v30.s[1]
mov v28.s[1], v2.s[0]
mov v29.s[1], v2.s[1]
mov v28.s[2], v3.s[0]
fadd v2.2s, v3.2s, v31.2s
mov v29.s[2], v3.s[1]
mov v28.s[3], v2.s[0]
mov v29.s[3], v2.s[1]


LoopL8:
    sub x3, x3, #8
    mov v4.16b, v28.16b
    mov v5.16b, v29.16b
    //v4, v6: x , v5, v7: y
    fmin v4.4s, v4.4s, v21.4s
    fadd v6.4s, v28.4s, v26.4s
    fadd v7.4s, v29.4s, v27.4s
    fmax v4.4s, v4.4s, v23.4s
    fmin v5.4s, v5.4s, v22.4s
    fadd v28.4s, v6.4s, v26.4s
    fadd v29.4s, v7.4s, v27.4s
    fmin v6.4s, v6.4s, v21.4s
    fmin v7.4s, v7.4s, v22.4s
    fmax v5.4s, v5.4s, v23.4s
    fmax v6.4s, v6.4s, v23.4s
    fmax v7.4s, v7.4s, v23.4s

.macro COMPUTE_FOUR
    fcvtms v16.4s, v4.4s
    fcvtps v17.4s, v4.4s
    frintm v30.4s, v4.4s
    fcvtms v18.4s, v5.4s
    fcvtps v19.4s, v5.4s
    frintm v31.4s, v5.4s

    //v0: xf, v1:yf
    fabd v0.4s, v30.4s, v4.4s
    fabd v1.4s, v5.4s, v31.4s

    //v16, v17, v18, v19 -> x00, x01, x10, x11
    mul v30.4s, v18.4s, v24.4s
    mul v31.4s, v19.4s, v24.4s
    add v18.4s, v31.4s, v16.4s
    add v19.4s, v31.4s, v17.4s
    add v16.4s, v16.4s, v30.4s
    add v17.4s, v17.4s, v30.4s

    //Load
    uxtl v31.2d, v16.2s
    uxtl2 v30.2d, v16.4s
    uqadd v31.2d, v25.2d, v31.2d
    uqadd v30.2d, v25.2d, v30.2d
    mov x7, v31.d[0]
    ld1 {v2.b}[0], [x7]
    mov x7, v31.d[1]
    ld1 {v2.b}[1], [x7]
    mov x7, v30.d[0]
    ld1 {v2.b}[2], [x7]
    mov x7, v30.d[1]
    ld1 {v2.b}[3], [x7]

    uxtl v31.2d, v17.2s
    uxtl2 v30.2d, v17.4s
    uqadd v31.2d, v25.2d, v31.2d
    uqadd v30.2d, v25.2d, v30.2d
    mov x7, v31.d[0]
    ld1 {v2.b}[4], [x7]
    mov x7, v31.d[1]
    ld1 {v2.b}[5], [x7]
    mov x7, v30.d[0]
    ld1 {v2.b}[6], [x7]
    mov x7, v30.d[1]
    ld1 {v2.b}[7], [x7]

    uxtl v31.2d, v18.2s
    uxtl2 v30.2d, v18.4s
    uqadd v31.2d, v25.2d, v31.2d
    uqadd v30.2d, v25.2d, v30.2d
    mov x7, v31.d[0]
    ld1 {v3.b}[0], [x7]
    mov x7, v31.d[1]
    ld1 {v3.b}[1], [x7]
    mov x7, v30.d[0]
    ld1 {v3.b}[2], [x7]
    mov x7, v30.d[1]
    ld1 {v3.b}[3], [x7]

    uxtl v31.2d, v19.2s
    uxtl2 v30.2d, v19.4s
    uqadd v31.2d, v25.2d, v31.2d
    uqadd v30.2d, v25.2d, v30.2d
    mov x7, v31.d[0]
    ld1 {v3.b}[4], [x7]
    mov x7, v31.d[1]
    ld1 {v3.b}[5], [x7]
    mov x7, v30.d[0]
    ld1 {v3.b}[6], [x7]
    mov x7, v30.d[1]
    ld1 {v3.b}[7], [x7]

    uxtl v2.8h, v2.8b
    uxtl v3.8h, v3.8b
    uxtl v16.4s, v2.4h
    uxtl2 v17.4s, v2.8h
    uxtl v18.4s, v3.4h
    uxtl2 v19.4s, v3.8h

    ucvtf v16.4s, v16.4s
    ucvtf v17.4s, v17.4s
    ucvtf v18.4s, v18.4s
    ucvtf v19.4s, v19.4s

    fsub v17.4s, v17.4s, v16.4s
    fsub v19.4s, v19.4s, v18.4s
    fmla v16.4s, v17.4s, v0.4s
    fmla v18.4s, v19.4s, v0.4s

    fsub v18.4s, v18.4s, v16.4s
    fmla v16.4s, v18.4s, v1.4s

    fcvtzs v0.4s, v16.4s

    uqxtn v0.4h, v0.4s
    uqxtn v0.8b, v0.8h

    st1 {v0.s}[0], [x1], #4

.endm
    COMPUTE_FOUR
    mov v4.16b, v6.16b
    mov v5.16b, v7.16b
    COMPUTE_FOUR

    cmp x3, #8
    bge LoopL8

mov v30.s[0], v28.s[0]
mov v30.s[1], v29.s[0]

L1:
cmp x3, #0
beq End


//int limit
mov v21.s[0], w4
mov v21.s[1], w5

//float limit
ucvtf v22.2s, v21.2s

LoopL1:
    mov v0.8b, v30.8b
    fmin v0.2s, v0.2s, v22.2s
    fmax v0.2s, v0.2s, v23.2s
    fadd v30.2s, v30.2s, v31.2s

    //d1:x0y0, d2:x1y1
    fcvtms v1.2s, v0.2s // floor and turn int
    frintm v3.2s, v0.2s // floor
    fcvtps v2.2s, v0.2s // ceil and turn int

    //d16-d17 source pixels
    mov w8, v1.s[1]
    mov w7, v1.s[0]
    mov w9, v2.s[0]
    //v0:factor
    fabd v5.2s, v0.2s, v3.2s
    smull x8, w8, w6
    add x8, x8, x0
    sxtw x7, w7
    mov v6.s[0], v5.s[1]
    sxtw x9, w9
    add x10, x8, x7
    ld1 {v4.b}[0], [x10]

    add x10, x8, x9
    ld1 {v4.b}[1], [x10]

    mov w8, v2.s[1]
    umull x8, w8, w6
    add x8, x8, x0
    add x10, x8, x7
    ld1 {v4.b}[2], [x10]

    add x10, x8, x9
    ld1 {v4.b}[3], [x10]

    uxtl v4.8h, v4.8b
    uxtl v0.4s, v4.4h
    ucvtf v0.4s, v0.4s
    mov s1, v0.s[1]
    mov s2, v0.s[2]
    mov s3, v0.s[3]

    fsub s1, s1, s0
    fsub s3, s3, s2
    fmul s1, s1, s5
    fmul s3, s3, s5
    fadd s0, s0, s1
    fadd s2, s2, s3

    fsub s2, s2, s0
    fmul s2, s2, s6
    fadd s0, s0, s2

    fcvtzs s0, s0
    uqxtn v4.4h, v0.4s
    uqxtn v4.8b, v4.8h

    st1 {v4.b}[0], [x1], #1
    subs x3, x3, #1
    bne LoopL1

End:

ret

#endif
