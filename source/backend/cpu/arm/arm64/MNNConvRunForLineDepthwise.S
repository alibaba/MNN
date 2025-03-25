//
//  MNNConvRunForLineDepthwise.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvRunForLineDepthwise
//void MNNConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
//                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height, size_t srcHStep, size_t dstHStep)

//Auto Load:
//x0:dst, x1:src, x2:weight, x3:width, x4:src_w_setup, x5:fw, x6:fh, x7:dilate_x_step

//Load From sp:
//x8:dilate_y_step, x15: height, x10: srcHStep, x11:dstHStep
ldr x8, [sp, #0]
ldr x15, [sp, #8]
ldr x10, [sp, #16]
ldr x11, [sp, #24]
ldr x12, [sp, #32]
ldr x13, [sp, #40]

stp d8, d9, [sp, #(-16 * 3)]!
stp d10, d11, [sp, #(16 * 2)]
stp x19, x20, [sp, #(16 * 1)]

mov x9, #4
mul x4, x9, x4
mul x7, x9, x7
mul x8, x9, x8
mul x10, x9, x10
mul x11, x9, x11

ld1 {v8.4s}, [x12] // bias
ld1r {v10.4s}, [x13], #4 // min
ld1r {v11.4s}, [x13]

//dilate_y_step -> dilate_y_step - fw*dilate_x_step
mul x9, x5, x7
sub x8, x8, x9

.macro assign_bias x0, x1, x2, x3
    mov \x0\().16b, v8.16b
    mov \x1\().16b, v8.16b
    mov \x2\().16b, v8.16b
    mov \x3\().16b, v8.16b
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
mov v4.d[0], x10
mov v4.d[1], x11
mov v5.d[0], x0
mov v5.d[1], x1
mov v6.d[0], x3

L16:
cmp x3, #16
blt L8

mov x12, #16
mul x12, x4, x12

L16Loop:
    assign_bias v16, v17, v18, v19
    assign_bias v20, v21, v22, v23
    assign_bias v24, v25, v26, v27
    assign_bias v28, v29, v30, v31

    mov x13, x1
    mov x14, x2
    mov x9, x6
    L16LoopH:
        mov x10, x5
        L16LoopW:
            ld1 {v7.4s}, [x2], #16
            ld1 {v0.4s}, [x1], x4
            subs x10, x10, #1
            ld1 {v1.4s}, [x1], x4
            fmla v16.4s, v7.4s, v0.4s
            fmla v17.4s, v7.4s, v1.4s
            ld1 {v2.4s}, [x1], x4
            ld1 {v3.4s}, [x1], x4
            fmla v18.4s, v7.4s, v2.4s
            fmla v19.4s, v7.4s, v3.4s
            ld1 {v0.4s}, [x1], x4
            ld1 {v1.4s}, [x1], x4
            fmla v20.4s, v7.4s, v0.4s
            fmla v21.4s, v7.4s, v1.4s
            ld1 {v2.4s}, [x1], x4
            ld1 {v3.4s}, [x1], x4
            fmla v22.4s, v7.4s, v2.4s
            fmla v23.4s, v7.4s, v3.4s

            ld1 {v0.4s}, [x1], x4
            ld1 {v1.4s}, [x1], x4
            fmla v24.4s, v7.4s, v0.4s
            fmla v25.4s, v7.4s, v1.4s
            ld1 {v2.4s}, [x1], x4
            ld1 {v3.4s}, [x1], x4
            fmla v26.4s, v7.4s, v2.4s
            fmla v27.4s, v7.4s, v3.4s
            ld1 {v0.4s}, [x1], x4
            ld1 {v1.4s}, [x1], x4
            fmla v28.4s, v7.4s, v0.4s
            fmla v29.4s, v7.4s, v1.4s
            ld1 {v2.4s}, [x1], x4
            ld1 {v3.4s}, [x1], x4
            fmla v30.4s, v7.4s, v2.4s
            fmla v31.4s, v7.4s, v3.4s
            sub x1, x1, x12
            add x1, x1, x7

            bne L16LoopW
        subs x9, x9, #1
        add x1, x1, x8
        bne L16LoopH

    sub x3, x3, #16
    compare_min_max v16, v17, v18, v19, v10, v11
    compare_min_max v20, v21, v22, v23, v10, v11
    compare_min_max v24, v25, v26, v27, v10, v11
    compare_min_max v28, v29, v30, v31, v10, v11
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    add x1, x13, x12
    cmp x3, #16
    mov x2, x14
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
    st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], #64
    st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x0], #64
    bge L16Loop


L8:
cmp x3, #7
ble L4

mov x12, #8
mul x12, x4, x12

L8Loop:
    assign_bias v16, v17, v18, v19
    assign_bias v20, v21, v22, v23

    mov x13, x1
    mov x14, x2
    mov x9, x6
    L8LoopH:
        mov x10, x5
        L8LoopW:
            ld1 {v3.4s}, [x2], #16
            ld1 {v0.4s}, [x1], x4
            subs x10, x10, #1
            fmla v16.4s, v3.4s, v0.4s
            ld1 {v1.4s}, [x1], x4
            fmla v17.4s, v3.4s, v1.4s
            ld1 {v0.4s}, [x1], x4
            fmla v18.4s, v0.4s, v3.4s
            ld1 {v1.4s}, [x1], x4
            fmla v19.4s, v1.4s, v3.4s
            ld1 {v0.4s}, [x1], x4
            fmla v20.4s, v0.4s, v3.4s
            ld1 {v1.4s}, [x1], x4
            fmla v21.4s, v1.4s, v3.4s
            ld1 {v0.4s}, [x1], x4
            fmla v22.4s, v0.4s, v3.4s
            ld1 {v1.4s}, [x1], x4
            fmla v23.4s, v1.4s, v3.4s

            sub x1, x1, x12
            add x1, x1, x7

            bne L8LoopW
        subs x9, x9, #1
        add x1, x1, x8
        bne L8LoopH

    compare_min_max v16, v17, v18, v19, v10, v11
    compare_min_max v20, v21, v22, v23, v10, v11
    sub x3, x3, #8
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    add x1, x13, x12
    mov x2, x14
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64


L4:
cmp x3, #4
ble L1

mov x12, #4
mul x12, x4, x12

L4Loop:
    assign_bias v16, v17, v18, v19

    mov x13, x1
    mov x14, x2
    mov x9, x6
    L4LoopH:
        mov x10, x5
        L4LoopW:
            ld1 {v3.4s}, [x2], #16
            ld1 {v0.4s}, [x1], x4
            subs x10, x10, #1
            fmla v16.4s, v3.4s, v0.4s
            ld1 {v1.4s}, [x1], x4
            fmla v17.4s, v3.4s, v1.4s
            ld1 {v0.4s}, [x1], x4
            fmla v18.4s, v0.4s, v3.4s
            ld1 {v1.4s}, [x1], x4
            fmla v19.4s, v1.4s, v3.4s

            sub x1, x1, x12
            add x1, x1, x7

            bne L4LoopW
        subs x9, x9, #1
        add x1, x1, x8
        bne L4LoopH

    compare_min_max v16, v17, v18, v19, v10, v11
    sub x3, x3, #4
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    add x1, x13, x12
    mov x2, x14

L1:
cmp x3, #0
beq End

L1Loop:
    mov v0.16b, v8.16b
    mov x9, x6
    mov x11, x1
    mov x12, x2
    L1LoopH:
        mov x10, x5
        L1LoopW:
            ld1 {v1.4s}, [x1], x7
            ld1 {v2.4s}, [x2], #16
            fmla v0.4s, v1.4s, v2.4s
            subs x10, x10, #1
            bne L1LoopW
        subs x9, x9, #1
        add x1, x1, x8
        bne L1LoopH

    subs x3, x3, #1
    fmax v0.4s, v0.4s, v10.4s
    fmin v0.4s, v0.4s, v11.4s
    st1 {v0.4s}, [x0], #16
    mov x2, x12
    add x1, x11, x4
    bne L1Loop


End:

mov x10, v4.d[0]
mov x11, v4.d[1]
mov x0, v5.d[0]
mov x1, v5.d[1]
mov x3, v6.d[0]

subs x15, x15, #1
add x0, x0, x11
add x1, x1, x10
bne LoopDY

ldp x19, x20, [sp, #(16 * 1)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d8, d9, [sp], #(16 * 3)
ret
//MNNConvRunForLineDepthwise End

#endif
