//
//  MNNRouteSigmoid.S
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.align 4
.LcInvLog2:
.rept 4
    .float 1.4426950408889634
.endr
.LcLog2:
.rept 4
    .float 0.6931471805599453
.endr
.LcQuarter:
.rept 4
    .float 0.25
.endr
.LcHalf:
.rept 4
    .float 0.5
.endr
.LcSixth:
.rept 4
    .float 0.16666666666666666
.endr
.LcP24:
.rept 4
    .float 0.041666666666666664
.endr
.LcP120:
.rept 4
    .float 0.008333333333333333
.endr
.LcOne:
.rept 4
    .float 1.0
.endr
.LcLimit:
.rept 4
    .float 87.0
.endr
.LcNegLim:
.rept 4
    .float -87.0
.endr

asm_function MNNRouteSigmoid
//void MNNRouteSigmoid(const float* src, const float* bias, float* probs, float* scores, int rowSize, int numRows)
//x0:src, x1:bias, x2:probs, x3:scores, w4:rowSize, w5:numRows
sub sp, sp, #64
stp d8, d9, [sp, #0]
stp d10, d11, [sp, #16]
stp d12, d13, [sp, #32]
stp d14, d15, [sp, #48]
cbz w5, .Lexit
mov x6, x0
mov x7, x1
mov x8, x2
mov x9, x3
.Lrow:
mov w10, w4
mov x11, x6
mov x12, x7
mov x13, x8
mov x14, x9
.Lquad16:
cmp w10, #16
b.lt .Lquad4
ld1 {v0.4s}, [x11], #16
ld1 {v8.4s}, [x11], #16
ld1 {v16.4s}, [x11], #16
ld1 {v24.4s}, [x11], #16
    fneg v0.4s, v0.4s
    ldr q7, .LcNegLim
    fmax v0.4s, v0.4s, v7.4s
    ldr q7, .LcLimit
    fmin v0.4s, v0.4s, v7.4s
    ldr q7, .LcInvLog2
    fmul v1.4s, v0.4s, v7.4s
    fcvtzs v1.4s, v1.4s
    movi v2.4s, #127
    add v2.4s, v2.4s, v1.4s
    shl v2.4s, v2.4s, #23
    scvtf v1.4s, v1.4s
    ldr q7, .LcLog2
    fmls v0.4s, v1.4s, v7.4s
    ldr q7, .LcQuarter
    fmul v0.4s, v0.4s, v7.4s
    ldr q3, .LcP120
    fmul v3.4s, v3.4s, v0.4s
    ldr q7, .LcP24
    fadd v3.4s, v3.4s, v7.4s
    fmul v3.4s, v3.4s, v0.4s
    ldr q7, .LcSixth
    fadd v3.4s, v3.4s, v7.4s
    fmul v3.4s, v3.4s, v0.4s
    ldr q7, .LcHalf
    fadd v3.4s, v3.4s, v7.4s
    fmul v3.4s, v3.4s, v0.4s
    fmov v4.4s, 1.0
    fadd v3.4s, v3.4s, v4.4s
    fmul v3.4s, v3.4s, v0.4s
    fadd v3.4s, v3.4s, v4.4s
    fmul v3.4s, v3.4s, v3.4s
    fmul v3.4s, v3.4s, v3.4s
    fmul v2.4s, v2.4s, v3.4s
    fadd v2.4s, v2.4s, v4.4s
    fmov v5.4s, 1.0
    fdiv v5.4s, v5.4s, v2.4s
    fneg v8.4s, v8.4s
    ldr q15, .LcNegLim
    fmax v8.4s, v8.4s, v15.4s
    ldr q15, .LcLimit
    fmin v8.4s, v8.4s, v15.4s
    ldr q15, .LcInvLog2
    fmul v9.4s, v8.4s, v15.4s
    fcvtzs v9.4s, v9.4s
    movi v10.4s, #127
    add v10.4s, v10.4s, v9.4s
    shl v10.4s, v10.4s, #23
    scvtf v9.4s, v9.4s
    ldr q15, .LcLog2
    fmls v8.4s, v9.4s, v15.4s
    ldr q15, .LcQuarter
    fmul v8.4s, v8.4s, v15.4s
    ldr q11, .LcP120
    fmul v11.4s, v11.4s, v8.4s
    ldr q15, .LcP24
    fadd v11.4s, v11.4s, v15.4s
    fmul v11.4s, v11.4s, v8.4s
    ldr q15, .LcSixth
    fadd v11.4s, v11.4s, v15.4s
    fmul v11.4s, v11.4s, v8.4s
    ldr q15, .LcHalf
    fadd v11.4s, v11.4s, v15.4s
    fmul v11.4s, v11.4s, v8.4s
    fmov v12.4s, 1.0
    fadd v11.4s, v11.4s, v12.4s
    fmul v11.4s, v11.4s, v8.4s
    fadd v11.4s, v11.4s, v12.4s
    fmul v11.4s, v11.4s, v11.4s
    fmul v11.4s, v11.4s, v11.4s
    fmul v10.4s, v10.4s, v11.4s
    fadd v10.4s, v10.4s, v12.4s
    fmov v13.4s, 1.0
    fdiv v13.4s, v13.4s, v10.4s
    fneg v16.4s, v16.4s
    ldr q23, .LcNegLim
    fmax v16.4s, v16.4s, v23.4s
    ldr q23, .LcLimit
    fmin v16.4s, v16.4s, v23.4s
    ldr q23, .LcInvLog2
    fmul v17.4s, v16.4s, v23.4s
    fcvtzs v17.4s, v17.4s
    movi v18.4s, #127
    add v18.4s, v18.4s, v17.4s
    shl v18.4s, v18.4s, #23
    scvtf v17.4s, v17.4s
    ldr q23, .LcLog2
    fmls v16.4s, v17.4s, v23.4s
    ldr q23, .LcQuarter
    fmul v16.4s, v16.4s, v23.4s
    ldr q19, .LcP120
    fmul v19.4s, v19.4s, v16.4s
    ldr q23, .LcP24
    fadd v19.4s, v19.4s, v23.4s
    fmul v19.4s, v19.4s, v16.4s
    ldr q23, .LcSixth
    fadd v19.4s, v19.4s, v23.4s
    fmul v19.4s, v19.4s, v16.4s
    ldr q23, .LcHalf
    fadd v19.4s, v19.4s, v23.4s
    fmul v19.4s, v19.4s, v16.4s
    fmov v20.4s, 1.0
    fadd v19.4s, v19.4s, v20.4s
    fmul v19.4s, v19.4s, v16.4s
    fadd v19.4s, v19.4s, v20.4s
    fmul v19.4s, v19.4s, v19.4s
    fmul v19.4s, v19.4s, v19.4s
    fmul v18.4s, v18.4s, v19.4s
    fadd v18.4s, v18.4s, v20.4s
    fmov v21.4s, 1.0
    fdiv v21.4s, v21.4s, v18.4s
    fneg v24.4s, v24.4s
    ldr q31, .LcNegLim
    fmax v24.4s, v24.4s, v31.4s
    ldr q31, .LcLimit
    fmin v24.4s, v24.4s, v31.4s
    ldr q31, .LcInvLog2
    fmul v25.4s, v24.4s, v31.4s
    fcvtzs v25.4s, v25.4s
    movi v26.4s, #127
    add v26.4s, v26.4s, v25.4s
    shl v26.4s, v26.4s, #23
    scvtf v25.4s, v25.4s
    ldr q31, .LcLog2
    fmls v24.4s, v25.4s, v31.4s
    ldr q31, .LcQuarter
    fmul v24.4s, v24.4s, v31.4s
    ldr q27, .LcP120
    fmul v27.4s, v27.4s, v24.4s
    ldr q31, .LcP24
    fadd v27.4s, v27.4s, v31.4s
    fmul v27.4s, v27.4s, v24.4s
    ldr q31, .LcSixth
    fadd v27.4s, v27.4s, v31.4s
    fmul v27.4s, v27.4s, v24.4s
    ldr q31, .LcHalf
    fadd v27.4s, v27.4s, v31.4s
    fmul v27.4s, v27.4s, v24.4s
    fmov v28.4s, 1.0
    fadd v27.4s, v27.4s, v28.4s
    fmul v27.4s, v27.4s, v24.4s
    fadd v27.4s, v27.4s, v28.4s
    fmul v27.4s, v27.4s, v27.4s
    fmul v27.4s, v27.4s, v27.4s
    fmul v26.4s, v26.4s, v27.4s
    fadd v26.4s, v26.4s, v28.4s
    fmov v29.4s, 1.0
    fdiv v29.4s, v29.4s, v26.4s
st1 {v5.4s}, [x13], #16
st1 {v13.4s}, [x13], #16
st1 {v21.4s}, [x13], #16
st1 {v29.4s}, [x13], #16
ld1 {v6.4s}, [x12], #16
ld1 {v14.4s}, [x12], #16
ld1 {v22.4s}, [x12], #16
ld1 {v30.4s}, [x12], #16
fadd v5.4s, v5.4s, v6.4s
fadd v13.4s, v13.4s, v14.4s
fadd v21.4s, v21.4s, v22.4s
fadd v29.4s, v29.4s, v30.4s
st1 {v5.4s}, [x14], #16
st1 {v13.4s}, [x14], #16
st1 {v21.4s}, [x14], #16
st1 {v29.4s}, [x14], #16
sub w10, w10, #16
b .Lquad16
.Lquad4:
cmp w10, #4
b.lt .Ltail
ld1 {v0.4s}, [x11], #16
    fneg v0.4s, v0.4s
    ldr q7, .LcNegLim
    fmax v0.4s, v0.4s, v7.4s
    ldr q7, .LcLimit
    fmin v0.4s, v0.4s, v7.4s
    ldr q7, .LcInvLog2
    fmul v1.4s, v0.4s, v7.4s
    fcvtzs v1.4s, v1.4s
    movi v2.4s, #127
    add v2.4s, v2.4s, v1.4s
    shl v2.4s, v2.4s, #23
    scvtf v1.4s, v1.4s
    ldr q7, .LcLog2
    fmls v0.4s, v1.4s, v7.4s
    ldr q7, .LcQuarter
    fmul v0.4s, v0.4s, v7.4s
    ldr q3, .LcP120
    fmul v3.4s, v3.4s, v0.4s
    ldr q7, .LcP24
    fadd v3.4s, v3.4s, v7.4s
    fmul v3.4s, v3.4s, v0.4s
    ldr q7, .LcSixth
    fadd v3.4s, v3.4s, v7.4s
    fmul v3.4s, v3.4s, v0.4s
    ldr q7, .LcHalf
    fadd v3.4s, v3.4s, v7.4s
    fmul v3.4s, v3.4s, v0.4s
    fmov v4.4s, 1.0
    fadd v3.4s, v3.4s, v4.4s
    fmul v3.4s, v3.4s, v0.4s
    fadd v3.4s, v3.4s, v4.4s
    fmul v3.4s, v3.4s, v3.4s
    fmul v3.4s, v3.4s, v3.4s
    fmul v2.4s, v2.4s, v3.4s
    fadd v2.4s, v2.4s, v4.4s
    fmov v5.4s, 1.0
    fdiv v5.4s, v5.4s, v2.4s
st1 {v5.4s}, [x13], #16
ld1 {v6.4s}, [x12], #16
fadd v5.4s, v5.4s, v6.4s
st1 {v5.4s}, [x14], #16
sub w10, w10, #4
b .Lquad4
.Ltail:
cbz w10, .Lnext
.Ltail_loop:
ld1 {v0.s}[0], [x11], #4
    fneg v0.4s, v0.4s
    ldr q7, .LcNegLim
    fmax v0.4s, v0.4s, v7.4s
    ldr q7, .LcLimit
    fmin v0.4s, v0.4s, v7.4s
    ldr q7, .LcInvLog2
    fmul v1.4s, v0.4s, v7.4s
    fcvtzs v1.4s, v1.4s
    movi v2.4s, #127
    add v2.4s, v2.4s, v1.4s
    shl v2.4s, v2.4s, #23
    scvtf v1.4s, v1.4s
    ldr q7, .LcLog2
    fmls v0.4s, v1.4s, v7.4s
    ldr q7, .LcQuarter
    fmul v0.4s, v0.4s, v7.4s
    ldr q3, .LcP120
    fmul v3.4s, v3.4s, v0.4s
    ldr q7, .LcP24
    fadd v3.4s, v3.4s, v7.4s
    fmul v3.4s, v3.4s, v0.4s
    ldr q7, .LcSixth
    fadd v3.4s, v3.4s, v7.4s
    fmul v3.4s, v3.4s, v0.4s
    ldr q7, .LcHalf
    fadd v3.4s, v3.4s, v7.4s
    fmul v3.4s, v3.4s, v0.4s
    fmov v4.4s, 1.0
    fadd v3.4s, v3.4s, v4.4s
    fmul v3.4s, v3.4s, v0.4s
    fadd v3.4s, v3.4s, v4.4s
    fmul v3.4s, v3.4s, v3.4s
    fmul v3.4s, v3.4s, v3.4s
    fmul v2.4s, v2.4s, v3.4s
    fadd v2.4s, v2.4s, v4.4s
    fmov v5.4s, 1.0
    fdiv v5.4s, v5.4s, v2.4s
st1 {v5.s}[0], [x13], #4
ld1 {v6.s}[0], [x12], #4
fadd v5.4s, v5.4s, v6.4s
st1 {v5.s}[0], [x14], #4
subs w10, w10, #1
b.ne .Ltail_loop
.Lnext:
add x6, x6, w4, uxtw #2
add x8, x8, w4, uxtw #2
add x9, x9, w4, uxtw #2
mov x7, x1
subs w5, w5, #1
b.ne .Lrow
.Lexit:
ldp d8, d9, [sp, #0]
ldp d10, d11, [sp, #16]
ldp d12, d13, [sp, #32]
ldp d14, d15, [sp, #48]
add sp, sp, #64
ret

#endif