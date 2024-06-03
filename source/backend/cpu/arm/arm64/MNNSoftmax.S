//
//  MNNSoftmax.S
//  MNN
//
//  Created by MNN on 2021/07/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

//void MNNSoftmax(float* dest, const float* source, size_t countC8)
asm_function MNNSoftmax
    stp x21, x22, [sp, #-32]!
    stp x19, x20, [sp, #16]
    sxtw    x8, w2
    lsr     w9, w2, #2
    and     x8, x8, #-4
    cbz     w9, Loop_5
    ldr     q0, [x1]
    cmp     w9, #1
    b.eq    Loop_4
    add     x10, x1, #16
    sub     x11, x9, #1
Loop_3:
    ldr     q1, [x10], #16
    subs    x11, x11, #1
    fmax    v0.4s, v0.4s, v1.4s
    b.ne    Loop_3
Loop_4:
    mov     s1, v0.s[1]
    fcmp    s0, s1
    mov     s2, v0.s[2]
    fcsel   s1, s0, s1, gt
    fcmp    s1, s2
    fcsel   s1, s1, s2, gt
    mov     s0, v0.s[3]
    fcmp    s1, s0
    fcsel   s0, s1, s0, gt
    cmp     w8, w2
    b.lo    Loop_6
    b       Loop_8
Loop_5:
    fmov    s0, wzr
    cmp     w8, w2
    b.hs    Loop_8
Loop_6:
    add     x10, x1, w8, sxtw #2
    mov     w11, w8
Loop_7:
    ldr     s1, [x10], #4
    add     w11, w11, #1
    fcmp    s0, s1
    fcsel   s0, s0, s1, gt
    cmp     w11, w2
    b.lo    Loop_7
Loop_8:
    cbz     w9, Loop_12
    mov     w10, #-1028784128
    mov     w12, #43579
    dup     v4.4s, w10
    mov     w10, #29208
    mov     w11, #1118699520
    movk    w12, #16312, lsl #16
    movk    w10, #48945, lsl #16
    dup     v5.4s, w11
    mov     w11, #34953
    dup     v6.4s, w12
    mov     w12, #43691
    dup     v7.4s, w10
    mov     w10, #43691
    movk    w11, #15368, lsl #16
    movk    w12, #15658, lsl #16
    movk    w10, #15914, lsl #16
    dup     v2.4s, v0.s[0]
    movi    v1.16b, #0
    fmov    v3.4s, #1.0
    dup     v16.4s, w11
    dup     v17.4s, w12
    dup     v18.4s, w10
    movi    v19.4s, #63, lsl #24
    mov     x10, x9
    mov     x11, x1
    mov     x12, x0
Loop_10:
    ldr     q20, [x11], #16
    subs    x10, x10, #1
    fsub    v20.4s, v20.4s, v2.4s
    fmax    v20.4s, v20.4s, v4.4s
    fmin    v20.4s, v20.4s, v5.4s
    fmul    v21.4s, v20.4s, v6.4s
    fcvtzs  v21.4s, v21.4s
    scvtf   v22.4s, v21.4s
    fmla    v20.4s, v22.4s, v7.4s
    fmul    v22.4s, v20.4s, v16.4s
    fadd    v22.4s, v22.4s, v17.4s
    fmul    v22.4s, v20.4s, v22.4s
    fadd    v22.4s, v22.4s, v18.4s
    fmul    v22.4s, v20.4s, v22.4s
    fadd    v22.4s, v22.4s, v19.4s
    fmul    v22.4s, v20.4s, v22.4s
    fadd    v22.4s, v22.4s, v3.4s
    shl     v21.4s, v21.4s, #23
    fmul    v20.4s, v20.4s, v22.4s
    add     v21.4s, v21.4s, v3.4s
    fadd    v20.4s, v20.4s, v3.4s
    fmul    v20.4s, v20.4s, v21.4s
    fadd    v1.4s, v1.4s, v20.4s
    str     q20, [x12], #16
    b.ne    Loop_10
    dup     v2.4s, v1.s[1]
    dup     v3.4s, v1.s[2]
    fadd    v2.4s, v1.4s, v2.4s
    fadd    v2.4s, v3.4s, v2.4s
    dup     v1.4s, v1.s[3]
    fadd    v1.4s, v1.4s, v2.4s
    b       Loop_13
Loop_12:
    fmov    s1, wzr
Loop_13:
    cmp     w8, w2
    fmov    s2, #1.0
    b.hs    Loop_16
    lsl     x21, x8, #2
    mov     w12, #29208
    mov     w14, #34953
    mov     w15, #43691
    mov     w19, #43691
    mov     w10, #-1028784128
    mov     w11, #1118699520
    movk    w12, #16177, lsl #16
    mov     w13, #1065353216
    movk    w14, #15368, lsl #16
    movk    w15, #15658, lsl #16
    movk    w19, #15914, lsl #16
    add     x20, x1, x21
    add     x21, x0, x21
    fmov    s3, #0.5
    mov     w1, w8
Loop_15:
    ldr     s4, [x20], #4
    fmov    s5, w10
    fmov    s6, w11
    fmov    s7, w12
    fsub    s4, s4, s0
    fmaxnm  s4, s4, s5
    fminnm  s4, s4, s6
    fdiv    s6, s4, s7
    fcvtzs  w3, s6
    scvtf   s6, w3
    fmul    s6, s6, s7
    fmov    s5, w14
    fsub    s4, s4, s6
    fmov    s7, w15
    fmul    s5, s4, s5
    fadd    s5, s5, s7
    fmov    s6, w19
    fmul    s5, s4, s5
    fadd    s5, s5, s6
    fmul    s5, s4, s5
    fadd    s5, s5, s3
    fmul    s5, s4, s5
    fadd    s5, s5, s2
    add     w3, w13, w3, lsl #23
    fmul    s4, s4, s5
    fmov    s7, w3
    fadd    s4, s4, s2
    add     w1, w1, #1
    fmul    s4, s4, s7
    cmp     w1, w2
    str     s4, [x21], #4
    fadd    s1, s1, s4
    b.lo    Loop_15
Loop_16:
    fdiv    s0, s2, s1
    cbz     w9, Loop_19
    dup     v1.4s, v0.s[0]
    mov     x10, x0
Loop_18:
    ldr     q2, [x10]
    subs    x9, x9, #1
    fmul    v2.4s, v1.4s, v2.4s
    str     q2, [x10], #16
    b.ne    Loop_18
Loop_19:
    cmp     w8, w2
    b.hs    Loop_22
    add     x9, x0, w8, sxtw #2
    Loop_21:
    ldr     s1, [x9]
    add     w8, w8, #1
    cmp     w8, w2
    fmul    s1, s0, s1
    str     s1, [x9], #4
    b.lo    Loop_21
Loop_22:
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp], #32
    ret
#endif

