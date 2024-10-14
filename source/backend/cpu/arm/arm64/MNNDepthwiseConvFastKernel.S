//
//  MNNDepthwiseConvFastKernel.S
//  MNN
//
//  Created by MNN on 2024/09/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//


#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNDepthwiseConvFastKernel

// void MNNDepthwiseConvFastKernel(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
//                                    size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
//                                    size_t srcHStep, size_t dstHStep);
//Auto Load:
//x0:dst, x1:src, x2:weight, x3:width, x4:src_w_step=pack*1, x5:fw, x6:fh, x7:dilate_x_step

//Load From sp:
//x8:dilate_y_step, x15: height, x10: srcHStep, x11:dstHStep, x12:bias, x13: minmax
ldr x8, [sp, #0]
ldr x15, [sp, #8]
ldr x10, [sp, #16]
ldr x11, [sp, #24]
ldr x12, [sp, #32]
ldr x13, [sp, #40]

stp d14, d15, [sp, #(-16 * 9)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]
stp x27, x28, [sp, #(16 * 6)]
stp x25, x26, [sp, #(16 * 7)]
stp x23, x24, [sp, #(16 * 8)]

lsl x4, x4, #2   // src_w_step*sizeof(float)
lsl x7, x7, #2   // dilate_x_step*sizeof(float)
lsl x8, x8, #2   // dilate_y_step*sizeof(float)
lsl x23, x10, #2 // srcHStep*sizeof(float)
lsl x24, x11, #2 // dstHStep*sizeof(float)
mov x20, x12     // bias
mov x26, x13     // min
add x27, x13, #4 // max

//dilate_y_step -> dilate_y_step - fw*dilate_x_step
mul x9, x5, x7
sub x8, x8, x9
mov x25, x3 // width
.macro assign_bias x0, x1, x2, x3, bv
    mov \x0\().16b, \bv\().16b
    mov \x1\().16b, \bv\().16b
    mov \x2\().16b, \bv\().16b
    mov \x3\().16b, \bv\().16b
.endm

.macro compare_min_max x0, x1, x2, x3, xmin, xmax
    fmax \x0\().4s, \x0\().4s, \xmin\().4s
    fmax \x1\().4s, \x1\().4s, \xmin\().4s
    fmax \x2\().4s, \x2\().4s, \xmin\().4s
    fmax \x3\().4s, \x3\().4s, \xmin\().4s
    fmin \x0\().4s, \x0\().4s, \xmax\().4s
    fmin \x1\().4s, \x1\().4s, \xmax\().4s
    fmin \x2\().4s, \x2\().4s, \xmax\().4s
    fmin \x3\().4s, \x3\().4s, \xmax\().4s
.endm

LoopDY:
//mov x23, x10
//mov x24, x11
mov x21, x0
mov x22, x1

L16:
cmp x3, #16
blt L8

mov x12, #-176
mov x19, #256

L16Loop:
    ld1 {v8.4s}, [x20] // load bias
    assign_bias v16, v17, v18, v19, v8
    assign_bias v20, v21, v22, v23, v8
    assign_bias v24, v25, v26, v27, v8
    assign_bias v28, v29, v30, v31, v8

    mov x13, x1
    mov x14, x2
    mov x9, x6
    L16LoopH:
        mov x10, x5
        L16LoopW:
            ld1 {v8.4s}, [x2], #16
            ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
            ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
            ld1 {v9.4s, v10.4s, v11.4s, v12.4s}, [x1], #64
            subs x10, x10, #1
            fmla v16.4s, v8.4s, v0.4s
            fmla v17.4s, v8.4s, v1.4s
            fmla v18.4s, v8.4s, v2.4s
            fmla v19.4s, v8.4s, v3.4s

            fmla v20.4s, v8.4s, v4.4s
            fmla v21.4s, v8.4s, v5.4s
            fmla v22.4s, v8.4s, v6.4s
            fmla v23.4s, v8.4s, v7.4s

            ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], x12

            fmla v24.4s, v8.4s, v9.4s
            fmla v25.4s, v8.4s, v10.4s
            fmla v26.4s, v8.4s, v11.4s
            fmla v27.4s, v8.4s, v12.4s

            fmla v28.4s, v8.4s, v0.4s
            fmla v29.4s, v8.4s, v1.4s
            fmla v30.4s, v8.4s, v2.4s
            fmla v31.4s, v8.4s, v3.4s

            bne L16LoopW
        subs x9, x9, #1
        add x1, x1, x8
        bne L16LoopH
    ld1r {v10.4s}, [x26] // min
    ld1r {v11.4s}, [x27] // max
    sub x3, x3, #16
    compare_min_max v16, v17, v18, v19, v10, v11
    compare_min_max v20, v21, v22, v23, v10, v11 
    compare_min_max v24, v25, v26, v27, v10, v11
    compare_min_max v28, v29, v30, v31, v10, v11
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    add x1, x13, x19 // 16 * pack * sizeof(float)
    cmp x3, #16
    mov x2, x14
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
    st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], #64
    st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x0], #64
    bge L16Loop


L8:
ld1r {v10.4s}, [x26] // min
ld1r {v11.4s}, [x27] // max
ld1 {v24.4s}, [x20] // load bias
cmp x3, #7
ble L4

mov x12, #-48
mov x19, #128


L8Loop:
    assign_bias v16, v17, v18, v19, v24
    assign_bias v20, v21, v22, v23, v24

    mov x13, x1
    mov x14, x2
    mov x9, x6
    L8LoopH:
        mov x10, x5
        L8LoopW:
            ld1 {v8.4s}, [x2], #16
            ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
            ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], x12
            subs x10, x10, #1
            fmla v16.4s, v8.4s, v0.4s
            fmla v17.4s, v8.4s, v1.4s
            fmla v18.4s, v8.4s, v2.4s
            fmla v19.4s, v8.4s, v3.4s

            fmla v20.4s, v8.4s, v4.4s
            fmla v21.4s, v8.4s, v5.4s
            fmla v22.4s, v8.4s, v6.4s
            fmla v23.4s, v8.4s, v7.4s

            bne L8LoopW
        subs x9, x9, #1
        add x1, x1, x8
        bne L8LoopH

    compare_min_max v16, v17, v18, v19, v10, v11
    compare_min_max v20, v21, v22, v23, v10, v11
    sub x3, x3, #8
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    add x1, x13, x19 // 8 * pack * sizeof(float)
    mov x2, x14
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64


L4:
cmp x3, #4
ble L1

mov x12, #16
mov x19, #64

L4Loop:
    assign_bias v16, v17, v18, v19, v24

    mov x13, x1
    mov x14, x2
    mov x9, x6
    L4LoopH:
        mov x10, x5
        L4LoopW:
            ld1 {v8.4s}, [x2], #16
            ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], x12
            subs x10, x10, #1
            fmla v16.4s, v8.4s, v0.4s
            fmla v17.4s, v8.4s, v1.4s
            fmla v18.4s, v8.4s, v2.4s
            fmla v19.4s, v8.4s, v3.4s

            bne L4LoopW
        subs x9, x9, #1
        add x1, x1, x8
        bne L4LoopH

    compare_min_max v16, v17, v18, v19, v10, v11
    sub x3, x3, #4
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    add x1, x13, x19
    mov x2, x14

L1:
cmp x3, #0
beq End

mov x19, #16

L1Loop:
    ld1 {v16.4s}, [x20] // assign bias

    mov x13, x1
    mov x14, x2
    mov x9, x6
    L1LoopH:
        mov x10, x5
        L1LoopW:
            ld1 {v8.4s}, [x2], #16
            ld1 {v0.4s}, [x1], #16
            subs x10, x10, #1
            fmla v16.4s, v8.4s, v0.4s

            bne L1LoopW
        subs x9, x9, #1
        add x1, x1, x8
        bne L1LoopH

    subs x3, x3, #1
    fmax v16.4s, v16.4s, v10.4s
    fmin v16.4s, v16.4s, v11.4s
    st1 {v16.4s}, [x0], #16
    add x1, x13, x4
    mov x2, x14
    bne L1Loop


End:

//mov x10, x23
//mov x11, x24
//mov x0, x21
//mov x1, x22
mov x3, x25

subs x15, x15, #1
add x0, x21, x24
add x1, x22, x23
bne LoopDY

ldp x23, x24, [sp, #(16 * 8)]
ldp x25, x26, [sp, #(16 * 7)]
ldp x27, x28, [sp, #(16 * 6)]
ldp x19, x20, [sp, #(16 * 5)]
ldp x21, x22, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 9)
ret
//MNNConvRunForLineDepthwise End

#endif
