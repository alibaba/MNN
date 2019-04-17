//
//  MNNConvRunForLineDepthWiseInt8.S
//  MNN
//
//  Created by MNN on 2018/09/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvRunForLineDepthWiseInt8
//void MNNConvRunForLineDepthWiseInt8(float* dst, const int8_t* src, const int8_t* weight, size_t width, size_t src_w_setup, size_t fw, size_t fh, 
//size_t dilate_x_step, size_t dilate_y_step, const float* alpha_z)

//Auto Load:
//x0:dst, x1:src, x2:weight, x3:width
//x4:src_w_setup, x5:fw, x6:fh, x7:dilate_x_step

//Load from sp:
//x8:dilate_y_step, x9:alpha_z
ldr x8, [sp, #0]
ldr x9, [sp, #8]


ld1 {v4.4s}, [x9]
//dilate_y_step -> dilate_y_step - fw*dilate_x_step
mul x9, x5, x7
sub x8, x8, x9

L8:
cmp x3, #8
blt L4

mov x12, #8
mul x12, x4, x12

L8Loop:
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0

    mov x13, x1
    mov x14, x2
    mov x9, x6
    L8LoopH:
        mov x10, x5
        L8LoopW:
            ld1 {v3.s}[0], [x2], #4
            dup v3.2s, v3.s[0]
            sxtl v3.8h, v3.8b
            ld1 {v0.s}[0], [x1], x4
            ld1 {v0.s}[1], [x1], x4
            subs x10, x10, #1
            sxtl v1.8h, v0.8b
            ld1 {v0.s}[2], [x1], x4
            ld1 {v0.s}[3], [x1], x4
            sxtl2 v2.8h, v0.16b
            smlal v16.4s, v3.4h, v1.4h
            ld1 {v0.s}[0], [x1], x4
            smlal2 v17.4s, v3.8h, v1.8h
            ld1 {v0.s}[1], [x1], x4
            ld1 {v0.s}[2], [x1], x4
            ld1 {v0.s}[3], [x1], x4
            smlal v18.4s, v3.4h, v2.4h
            sxtl v1.8h, v0.8b
            smlal2 v19.4s, v3.8h, v2.8h
            sxtl2 v2.8h, v0.16b
            smlal v20.4s, v3.4h, v1.4h
            smlal2 v21.4s, v3.8h, v1.8h
            smlal v22.4s, v3.4h, v2.4h
            smlal2 v23.4s, v3.8h, v2.8h

            sub x1, x1, x12
            add x1, x1, x7

            bne L8LoopW
        L8LoopWEnd:
        subs x9, x9, #1
        add x1, x1, x8
        bne L8LoopH

    sub x3, x3, #8
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    fmul v16.4s, v16.4s, v4.4s
    fmul v17.4s, v17.4s, v4.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    st1 {v16.4s, v17.4s}, [x0], #32
    fmul v18.4s, v18.4s, v4.4s
    fmul v19.4s, v19.4s, v4.4s
    fmul v20.4s, v20.4s, v4.4s
    scvtf v21.4s, v21.4s
    st1 {v18.4s, v19.4s}, [x0], #32
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s
    fmul v21.4s, v21.4s, v4.4s
    fmul v22.4s, v22.4s, v4.4s
    fmul v23.4s, v23.4s, v4.4s
    st1 {v20.4s, v21.4s}, [x0], #32

    mov x1, x13
    mov x2, x14
    add x1, x1, x12
    cmp x3, #8
    st1 {v22.4s, v23.4s}, [x0], #32
    bge L8Loop


L4:
cmp x3, #4
blt L1

mov x12, #4
mul x12, x4, x12

L4Loop:
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0

    mov x13, x1
    mov x14, x2
    mov x9, x6
    L4LoopH:
        mov x10, x5
        L4LoopW:
            ld1 {v3.s}[0], [x2], #4
            dup v3.2s, v3.s[0]
            sxtl v3.8h, v3.8b
            ld1 {v0.s}[0], [x1], x4
            ld1 {v0.s}[1], [x1], x4
            subs x10, x10, #1
            sxtl v1.8h, v0.8b
            ld1 {v0.s}[2], [x1], x4
            ld1 {v0.s}[3], [x1], x4
            sxtl2 v2.8h, v0.16b
            smlal v16.4s, v3.4h, v1.4h
            smlal2 v17.4s, v3.8h, v1.8h
            smlal v18.4s, v3.4h, v2.4h
            smlal2 v19.4s, v3.8h, v2.8h

            sub x1, x1, x12
            add x1, x1, x7

            bne L4LoopW
        L4LoopWEnd:
        subs x9, x9, #1
        add x1, x1, x8
        bne L4LoopH

    sub x3, x3, #4
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    fmul v16.4s, v16.4s, v4.4s
    fmul v17.4s, v17.4s, v4.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    st1 {v16.4s, v17.4s}, [x0], #32
    fmul v18.4s, v18.4s, v4.4s
    fmul v19.4s, v19.4s, v4.4s
    st1 {v18.4s, v19.4s}, [x0], #32

    mov x1, x13
    mov x2, x14
    add x1, x1, x12
    cmp x3, #4
    bge L4Loop


L1:
cmp x3, #0
beq End

L1Loop:
    movi v0.4s, #0
    mov x9, x6
    mov x11, x1
    mov x12, x2
    L1LoopH:
        mov x10, x5
        L1LoopW:
            ld1 {v1.s}[0], [x1], x7
            ld1 {v2.s}[0], [x2], #4
            sxtl v1.8h, v1.8b
            sxtl v2.8h, v2.8b
            smlal v0.4s, v1.4h, v2.4h
            subs x10, x10, #1
            bne L1LoopW
        subs x9, x9, #1
        add x1, x1, x8
        bne L1LoopH

    subs x3, x3, #1

    scvtf v0.4s, v0.4s
    fmul v0.4s, v0.4s, v4.4s
    st1 {v0.4s}, [x0], #16
    mov x2, x12
    add x1, x11, x4
    bne L1Loop


End:


ret

#endif
