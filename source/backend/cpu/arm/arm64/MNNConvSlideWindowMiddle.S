//
//  MNNConvSlideWindowMiddle.S
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvSlideWindowMiddle
//void MNNConvSlideWindowMiddle(float* dst, const float* src, const float* weight, size_t width, size_t src_w_step, size_t src_depth_quad, size_t src_depth_step, size_t fw, size_t fh, size_t dilate_x_step, size_t dilate_y_step)

//Auto Load:
//x0:dst, x1:src, x2:weight, x3:width, x4:src_w_step, x5:src_depth_quad, x6: src_depth_step, x7:fw

//Load from sp
//x8:fh, x9:dilate_x_step, x10:dilate_y_step
ldr x8, [sp, #0]
ldr x9, [sp, #8]
ldr x10, [sp, #16]

//step multi by sizeof(float)
mov x12, #4
mul x10, x12, x10
mul x9, x12, x9
mul x6, x12, x6
mul x4, x12, x4

//src_depth_step -> src_depth_step - fh*dilate_y_step
mul x12, x8, x10
sub x6, x6, x12

//dilate_y_step -> dilate_y_step-fw*dilate_x_step
mul x12, x7, x9
sub x10, x10, x12

sub sp, sp, #16
st1 {v8.2d}, [sp], #16

L16:
cmp x3, #15
ble L8

mov x14, #16
mul x14, x4, x14

L16Loop:
    mov x11, x1
    mov x12, x2
    mov x13, x5
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    movi v24.4s, #0
    movi v25.4s, #0
    movi v26.4s, #0
    movi v27.4s, #0
    movi v28.4s, #0
    movi v29.4s, #0
    movi v30.4s, #0
    movi v31.4s, #0
    L16LoopZ:
        mov v8.d[0], x8
        L16LoopFY:
            mov v8.d[1], x7
            L16LoopFX:
                ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x2], #64

                ld1 {v0.4s}, [x1], x4
                fmla v16.4s, v4.4s, v0.s[0]

                ld1 {v1.4s}, [x1], x4
                fmla v16.4s, v5.4s, v0.s[1]
                fmla v17.4s, v4.4s, v1.s[0]

                ld1 {v2.4s}, [x1], x4
                fmla v17.4s, v5.4s, v1.s[1]
                fmla v16.4s, v6.4s, v0.s[2]
                fmla v17.4s, v6.4s, v1.s[2]
                fmla v18.4s, v5.4s, v2.s[1]

                ld1 {v3.4s}, [x1], x4
                fmla v16.4s, v7.4s, v0.s[3]
                fmla v18.4s, v6.4s, v2.s[2]
                fmla v19.4s, v4.4s, v3.s[0]
                fmla v18.4s, v7.4s, v2.s[3]
                fmla v19.4s, v5.4s, v3.s[1]
                fmla v17.4s, v7.4s, v1.s[3]
                fmla v19.4s, v6.4s, v3.s[2]

                ld1 {v0.4s}, [x1], x4
                fmla v18.4s, v4.4s, v2.s[0]
                fmla v19.4s, v7.4s, v3.s[3]

                ld1 {v1.4s}, [x1], x4
                fmla v20.4s, v4.4s, v0.s[0]
                fmla v21.4s, v4.4s, v1.s[0]
                fmla v20.4s, v5.4s, v0.s[1]
                fmla v21.4s, v6.4s, v1.s[2]

                ld1 {v2.4s}, [x1], x4
                fmla v20.4s, v6.4s, v0.s[2]
                fmla v22.4s, v4.4s, v2.s[0]
                fmla v21.4s, v5.4s, v1.s[1]
                fmla v22.4s, v5.4s, v2.s[1]

                ld1 {v3.4s}, [x1], x4
                fmla v21.4s, v7.4s, v1.s[3]
                fmla v23.4s, v4.4s, v3.s[0]
                fmla v22.4s, v7.4s, v2.s[3]
                fmla v23.4s, v6.4s, v3.s[2]
                fmla v20.4s, v7.4s, v0.s[3]

                ld1 {v0.4s}, [x1], x4
                fmla v22.4s, v6.4s, v2.s[2]
                fmla v23.4s, v5.4s, v3.s[1]

                fmla v24.4s, v7.4s, v0.s[3]
                fmla v23.4s, v7.4s, v3.s[3]

                ld1 {v1.4s}, [x1], x4
                fmla v24.4s, v5.4s, v0.s[1]
                fmla v25.4s, v4.4s, v1.s[0]

                ld1 {v2.4s}, [x1], x4
                fmla v24.4s, v4.4s, v0.s[0]
                fmla v25.4s, v5.4s, v1.s[1]
                
                fmla v26.4s, v5.4s, v2.s[1]
                fmla v25.4s, v7.4s, v1.s[3]
                fmla v26.4s, v7.4s, v2.s[3]

                ld1 {v3.4s}, [x1], x4
                fmla v25.4s, v6.4s, v1.s[2]
                fmla v24.4s, v6.4s, v0.s[2]
                
                fmla v27.4s, v4.4s, v3.s[0]
                fmla v26.4s, v4.4s, v2.s[0]
                fmla v27.4s, v5.4s, v3.s[1]

                ld1 {v0.4s}, [x1], x4
                fmla v27.4s, v7.4s, v3.s[3]
                fmla v26.4s, v6.4s, v2.s[2]
                fmla v27.4s, v6.4s, v3.s[2]

                fmla v28.4s, v4.4s, v0.s[0]

                ld1 {v1.4s}, [x1], x4
                fmla v28.4s, v5.4s, v0.s[1]
                fmla v29.4s, v4.4s, v1.s[0]

                ld1 {v2.4s}, [x1], x4
                fmla v28.4s, v7.4s, v0.s[3]
                fmla v29.4s, v5.4s, v1.s[1]
                fmla v30.4s, v4.4s, v2.s[0]
                fmla v29.4s, v6.4s, v1.s[2]

                ld1 {v3.4s}, [x1], x4
                fmla v29.4s, v7.4s, v1.s[3]
                fmla v28.4s, v6.4s, v0.s[2]
                fmla v31.4s, v4.4s, v3.s[0]
                fmla v30.4s, v5.4s, v2.s[1]
                fmla v31.4s, v5.4s, v3.s[1]
                fmla v30.4s, v6.4s, v2.s[2]
                fmla v31.4s, v6.4s, v3.s[2]
                subs x7, x7, #1
                fmla v30.4s, v7.4s, v2.s[3]
                sub x1, x1, x14
                fmla v31.4s, v7.4s, v3.s[3]

                add x1, x1, x9
                bne L16LoopFX
            subs x8, x8, #1
            mov x7, v8.d[1]
            add x1, x1, x10
            bne L16LoopFY
        subs x5, x5, #1
        mov x8, v8.d[0]
        add x1, x1, x6
        bne L16LoopZ
    sub x3, x3, #16
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    add x1, x11, x14
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
    mov x2, x12
    cmp x3, #16
    st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], #64
    mov x5, x13
    st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x0], #64
    bge L16Loop


L8:
cmp x3, #7
ble L4

mov x14, #8
mul x14, x4, x14

L8Loop:
    mov x11, x1
    mov x12, x2
    mov x13, x5
    movi v0.4s, #0
    movi v1.4s, #0
    movi v2.4s, #0
    movi v3.4s, #0
    movi v4.4s, #0
    movi v5.4s, #0
    movi v6.4s, #0
    movi v7.4s, #0
    L8LoopZ:
        mov v27.d[0], x8
        L8LoopFY:
            mov v27.d[1], x7
            L8LoopFX:
                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                ld1 {v16.4s}, [x1], x4
                fmla v0.4s, v28.4s, v16.s[0]
                ld1 {v17.4s}, [x1], x4
                fmla v0.4s, v29.4s, v16.s[1]
                fmla v0.4s, v30.4s, v16.s[2]
                ld1 {v18.4s}, [x1], x4
                fmla v1.4s, v28.4s, v17.s[0]
                ld1 {v19.4s}, [x1], x4
                fmla v1.4s, v29.4s, v17.s[1]
                fmla v2.4s, v28.4s, v18.s[0]
                ld1 {v20.4s}, [x1], x4
                fmla v3.4s, v28.4s, v19.s[0]
                ld1 {v21.4s}, [x1], x4
                fmla v4.4s, v28.4s, v20.s[0]
                ld1 {v22.4s}, [x1], x4
                fmla v5.4s, v28.4s, v21.s[0]
                ld1 {v23.4s}, [x1], x4
                fmla v6.4s, v28.4s, v22.s[0]
                fmla v7.4s, v28.4s, v23.s[0]

                fmla v2.4s, v29.4s, v18.s[1]
                fmla v3.4s, v29.4s, v19.s[1]
                fmla v4.4s, v29.4s, v20.s[1]
                fmla v5.4s, v29.4s, v21.s[1]
                fmla v6.4s, v29.4s, v22.s[1]
                fmla v7.4s, v29.4s, v23.s[1]

                fmla v1.4s, v30.4s, v17.s[2]
                fmla v2.4s, v30.4s, v18.s[2]
                fmla v3.4s, v30.4s, v19.s[2]
                fmla v4.4s, v30.4s, v20.s[2]
                fmla v5.4s, v30.4s, v21.s[2]
                fmla v6.4s, v30.4s, v22.s[2]
                fmla v7.4s, v30.4s, v23.s[2]

                fmla v0.4s, v31.4s, v16.s[3]
                fmla v1.4s, v31.4s, v17.s[3]
                fmla v2.4s, v31.4s, v18.s[3]
                fmla v3.4s, v31.4s, v19.s[3]
                fmla v4.4s, v31.4s, v20.s[3]
                fmla v5.4s, v31.4s, v21.s[3]
                fmla v6.4s, v31.4s, v22.s[3]
                fmla v7.4s, v31.4s, v23.s[3]

                sub x1, x1, x14
                subs x7, x7, #1
                add x1, x1, x9
                bne L8LoopFX
            subs x8, x8, #1
            mov x7, v27.d[1]
            add x1, x1, x10
            bne L8LoopFY
        subs x5, x5, #1
        mov x8, v27.d[0]
        add x1, x1, x6
        bne L8LoopZ
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
    add x1, x11, x14
    mov x2, x12
    mov x5, x13
    sub x3, x3, #8
    cmp x3, #8
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
    bge L8Loop



L4:
cmp x3, #3
ble L1

mov x14, #4
mul x14, x4, x14

mov x11, x1
mov x12, x2
mov x13, x5
movi v0.4s, #0
movi v1.4s, #0
movi v2.4s, #0
movi v3.4s, #0
L4LoopZ:
mov v27.d[0], x8
L4LoopFY:
mov v27.d[1], x7
L4LoopFX:
ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
ld1 {v16.4s}, [x1], x4
fmla v0.4s, v28.4s, v16.s[0]
ld1 {v17.4s}, [x1], x4
fmla v0.4s, v29.4s, v16.s[1]
fmla v1.4s, v29.4s, v17.s[1]
ld1 {v18.4s}, [x1], x4
fmla v1.4s, v28.4s, v17.s[0]
fmla v0.4s, v30.4s, v16.s[2]
ld1 {v19.4s}, [x1], x4

fmla v2.4s, v28.4s, v18.s[0]
fmla v3.4s, v28.4s, v19.s[0]

fmla v2.4s, v29.4s, v18.s[1]
fmla v3.4s, v29.4s, v19.s[1]

fmla v1.4s, v30.4s, v17.s[2]
fmla v2.4s, v30.4s, v18.s[2]
fmla v3.4s, v30.4s, v19.s[2]

fmla v0.4s, v31.4s, v16.s[3]
fmla v1.4s, v31.4s, v17.s[3]
fmla v2.4s, v31.4s, v18.s[3]
fmla v3.4s, v31.4s, v19.s[3]

sub x1, x1, x14
subs x7, x7, #1
add x1, x1, x9
bne L4LoopFX
subs x8, x8, #1
mov x7, v27.d[1]
add x1, x1, x10
bne L4LoopFY
subs x5, x5, #1
mov x8, v27.d[0]
add x1, x1, x6
bne L4LoopZ
st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
add x1, x11, x14
mov x2, x12
mov x5, x13
sub x3, x3, #4

L1:
cmp x3, #0
ble End

L1Loop:
    mov x11, x1
    mov x12, x2
    mov x13, x5
    movi v0.4s, #0
    movi v1.4s, #0
    L1LoopZ:
        mov x14, x8
        L1LoopFY:
            mov x15, x7
            L1LoopFX:
                ld1 {v3.4s}, [x1], x9
                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64

                fmla v0.4s, v28.4s, v3.s[0]
                fmla v1.4s, v29.4s, v3.s[1]
                fmla v0.4s, v30.4s, v3.s[2]
                fmla v1.4s, v31.4s, v3.s[3]
                subs x7, x7, #1
                bne L1LoopFX
            subs x8, x8, #1
            mov x7, x15
            add x1, x1, x10
            bne L1LoopFY
        subs x5, x5, #1
        mov x8, x14
        add x1, x1, x6
        bne L1LoopZ

    fadd v0.4s, v0.4s, v1.4s
    add x1, x11, x4
    mov x2, x12
    mov x5, x13
    subs x3, x3, #1
    st1 {v0.4s}, [x0], #16
    bne L1Loop

End:


sub sp, sp, #16
ld1 {v8.2d}, [sp], #16

ret

#endif
