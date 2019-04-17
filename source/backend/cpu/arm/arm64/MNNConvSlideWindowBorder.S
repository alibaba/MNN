//
//  MNNConvSlideWindowBorder.S
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvSlideWindowBorder
//void MNNConvSlideWindowBorder(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t src_depth_step, size_t fw, size_t fh, size_t weight_y_step, size_t weight_z_step, size_t dilate_x_step, size_t dilate_y_step)

//Default
//x0:dst, x1:src, x2:weight, x3:src_depth_quad, x4:src_depth_step, x5:fw, x6:fh, x7:weight_y_step

movi v0.4s, #0
movi v1.4s, #0
cmp x6, #0
beq EndUnit
cmp x5, #0
beq EndUnit

//Load from sp
//x8:weight_z_step, x9:dilate_x_step, x10:dilate_y_step
ldr x8, [sp, #0]
ldr x9, [sp, #8]
ldr x10, [sp, #16]

//multi by sizeof(float)
mov x11, #4
mul x4, x11, x4
mul x7, x11, x7
mul x8, x11, x8
mul x9, x11, x9
mul x10, x11, x10

//weight_z_step -> weight_z_step - fh*weight_y_step
mul x11, x7, x6
sub x8, x8, x11

//weight_y_step -> weight_y_step - fw*16*sizeof(float)
mov x11, #64
mul x11, x5, x11
sub x7, x7, x11

//src_depth_step -> src_depth_step - fh*dilate_y_step
mul x11, x6, x10
sub x4, x4, x11

//dilate_y_step -> dilate_y_step - fw*dilate_x_step
mul x11, x9, x5
sub x10, x10, x11

LoopZ:
mov x11, x6
LoopFY:
mov x12, x5
LoopFX:
ld1 {v3.4s}, [x1], x9
ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
fmla v0.4s, v28.4s, v3.s[0]
fmla v1.4s, v29.4s, v3.s[1]
fmla v0.4s, v30.4s, v3.s[2]
fmla v1.4s, v31.4s, v3.s[3]
subs x5, x5, #1
bne LoopFX
subs x6, x6, #1
add x1, x10, x1
add x2, x2, x7
mov x5, x12
bne LoopFY
mov x6, x11
subs x3, x3, #1
add x1, x1, x4
add x2, x2, x8
bne LoopZ
fadd v0.4s, v0.4s, v1.4s

EndUnit:
st1 {v0.4s}, [x0]


ret

#endif
