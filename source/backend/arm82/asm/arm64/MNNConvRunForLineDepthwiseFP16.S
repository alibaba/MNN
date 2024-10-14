//
//  MNNConvRunForLineDepthwiseFP16.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvRunForLineDepthwiseFP16
//void MNNConvRunForLineDepthwiseFP16(FLOAT16* dst, const FLOAT16* src, const FLOAT16* weight, size_t width, size_t src_w_setup,
//                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height, size_t srcHStep, size_t dstHStep,
//                                const float* bias, float* parameters)

//Auto Load:
//x0:dst, x1:src, x2:weight, x3:width, x4:src_w_setup, x5:fw, x6:fh, x7:dilate_x_step

//Load From sp:
//x8:dilate_y_step, x15: height, x10: srcHStep, x11:dstHStep, x12:bias, x13:parameters
ldr x8, [sp, #0]
ldr x15, [sp, #8]
ldr x10, [sp, #16]
ldr x11, [sp, #24]
ldr x12, [sp, #32]
ldr x13, [sp, #40]

stp d8, d9, [sp, #(-16 * 3)]!
stp d10, d11, [sp, #(16 * 2)]
stp x19, x20, [sp, #(16 * 1)]

mov x9, #2 // sizeof(FLOAT16)
mul x4, x9, x4
mul x7, x9, x7
mul x8, x9, x8
mul x10, x9, x10
mul x11, x9, x11

ld1 {v8.8h}, [x12] // bias
ld1r {v10.8h}, [x13], #2 // min
ld1r {v11.8h}, [x13]

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
    fmax \x0\().8h, \x0\().8h, \xmin\().8h
    fmax \x1\().8h, \x1\().8h, \xmin\().8h
    fmax \x2\().8h, \x2\().8h, \xmin\().8h
    fmax \x3\().8h, \x3\().8h, \xmin\().8h
    fmin \x0\().8h, \x0\().8h, \xmax\().8h
    fmin \x1\().8h, \x1\().8h, \xmax\().8h
    fmin \x2\().8h, \x2\().8h, \xmax\().8h
    fmin \x3\().8h, \x3\().8h, \xmax\().8h
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

mov x19, #16
mul x19, x4, x19

L16Loop:
    assign_bias v16, v17, v18, v19
    assign_bias v20, v21, v22, v23
    assign_bias v24, v25, v26, v27
    assign_bias v28, v29, v30, v31

    mov x20, x1
    mov x14, x2
    mov x9, x6
    L16LoopH:
        mov x10, x5
        L16LoopW:
            ld1 {v7.8h}, [x2], #16
            ld1 {v0.8h}, [x1], x4
            subs x10, x10, #1
            ld1 {v1.8h}, [x1], x4
            fmla v16.8h, v7.8h, v0.8h
            fmla v17.8h, v7.8h, v1.8h
            ld1 {v2.8h}, [x1], x4
            ld1 {v3.8h}, [x1], x4
            fmla v18.8h, v7.8h, v2.8h
            fmla v19.8h, v7.8h, v3.8h
            ld1 {v0.8h}, [x1], x4
            ld1 {v1.8h}, [x1], x4
            fmla v20.8h, v7.8h, v0.8h
            fmla v21.8h, v7.8h, v1.8h
            ld1 {v2.8h}, [x1], x4
            ld1 {v3.8h}, [x1], x4
            fmla v22.8h, v7.8h, v2.8h
            fmla v23.8h, v7.8h, v3.8h

            ld1 {v0.8h}, [x1], x4
            ld1 {v1.8h}, [x1], x4
            fmla v24.8h, v7.8h, v0.8h
            fmla v25.8h, v7.8h, v1.8h
            ld1 {v2.8h}, [x1], x4
            ld1 {v3.8h}, [x1], x4
            fmla v26.8h, v7.8h, v2.8h
            fmla v27.8h, v7.8h, v3.8h
            ld1 {v0.8h}, [x1], x4
            ld1 {v1.8h}, [x1], x4
            fmla v28.8h, v7.8h, v0.8h
            fmla v29.8h, v7.8h, v1.8h
            ld1 {v2.8h}, [x1], x4
            ld1 {v3.8h}, [x1], x4
            fmla v30.8h, v7.8h, v2.8h
            fmla v31.8h, v7.8h, v3.8h
            sub x1, x1, x19
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
    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], #64
    add x1, x20, x19
    cmp x3, #16
    mov x2, x14
    st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64
    st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x0], #64
    st1 {v28.8h, v29.8h, v30.8h, v31.8h}, [x0], #64
    bge L16Loop


L8:
cmp x3, #7
ble L4

mov x19, #8
mul x19, x4, x19

L8Loop:
    assign_bias v16, v17, v18, v19
    assign_bias v20, v21, v22, v23

    mov x20, x1
    mov x14, x2
    mov x9, x6
    L8LoopH:
        mov x10, x5
        L8LoopW:
            ld1 {v3.8h}, [x2], #16
            ld1 {v0.8h}, [x1], x4
            subs x10, x10, #1
            fmla v16.8h, v3.8h, v0.8h
            ld1 {v1.8h}, [x1], x4
            fmla v17.8h, v3.8h, v1.8h
            ld1 {v0.8h}, [x1], x4
            fmla v18.8h, v0.8h, v3.8h
            ld1 {v1.8h}, [x1], x4
            fmla v19.8h, v1.8h, v3.8h
            ld1 {v0.8h}, [x1], x4
            fmla v20.8h, v0.8h, v3.8h
            ld1 {v1.8h}, [x1], x4
            fmla v21.8h, v1.8h, v3.8h
            ld1 {v0.8h}, [x1], x4
            fmla v22.8h, v0.8h, v3.8h
            ld1 {v1.8h}, [x1], x4
            fmla v23.8h, v1.8h, v3.8h

            sub x1, x1, x19
            add x1, x1, x7

            bne L8LoopW
        subs x9, x9, #1
        add x1, x1, x8
        bne L8LoopH

    compare_min_max v16, v17, v18, v19, v10, v11
    compare_min_max v20, v21, v22, v23, v10, v11

    sub x3, x3, #8
    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], #64
    add x1, x20, x19
    mov x2, x14
    st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64


L4:
cmp x3, #4
ble L1

mov x19, #4
mul x19, x4, x19

L4Loop:
    assign_bias v16, v17, v18, v19

    mov x20, x1
    mov x14, x2
    mov x9, x6
    L4LoopH:
        mov x10, x5
        L4LoopW:
            ld1 {v3.8h}, [x2], #16
            ld1 {v0.8h}, [x1], x4
            subs x10, x10, #1
            fmla v16.8h, v3.8h, v0.8h
            ld1 {v1.8h}, [x1], x4
            fmla v17.8h, v3.8h, v1.8h
            ld1 {v0.8h}, [x1], x4
            fmla v18.8h, v0.8h, v3.8h
            ld1 {v1.8h}, [x1], x4
            fmla v19.8h, v1.8h, v3.8h

            sub x1, x1, x19
            add x1, x1, x7

            bne L4LoopW
        subs x9, x9, #1
        add x1, x1, x8
        bne L4LoopH

    compare_min_max v16, v17, v18, v19, v10, v11
    sub x3, x3, #4
    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], #64
    add x1, x20, x19
    mov x2, x14

L1:
cmp x3, #0
beq End

L1Loop:
    mov v0.16b, v8.16b
    mov x9, x6
    mov x11, x1
    mov x19, x2
    L1LoopH:
        mov x10, x5
        L1LoopW:
            ld1 {v1.8h}, [x1], x7
            ld1 {v2.8h}, [x2], #16
            fmla v0.8h, v1.8h, v2.8h
            subs x10, x10, #1
            bne L1LoopW
        subs x9, x9, #1
        add x1, x1, x8
        bne L1LoopH

    subs x3, x3, #1
    fmax v0.8h, v0.8h, v10.8h
    fmin v0.8h, v0.8h, v11.8h
    st1 {v0.8h}, [x0], #16
    mov x2, x19
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

#endif
