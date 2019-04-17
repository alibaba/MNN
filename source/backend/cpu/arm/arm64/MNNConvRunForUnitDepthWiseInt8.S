//
//  MNNConvRunForUnitDepthWiseInt8.S
//  MNN
//
//  Created by MNN on 2018/09/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvRunForUnitDepthWiseInt8

//void MNNConvRunForUnitDepthWiseInt8(float* dst, const int8_t* src, const int8_t* weight, size_t fw, size_t fh, size_t weight_y_step, size_t dilate_x_step,
// size_t dilate_y_step, const float* scale)

//Default
//x0:dst, x1:src, x2:weight, x3:fw
//x4:fh, x5:weight_y_step, x6:dilate_x_step, x7:dilate_y_step

//Load from sp: x8:scale
ldr x8, [sp, #0]

ld1 {v23.4s}, [x8]

movi v0.4s, #0

cmp x3, #0
beq EndUnit

cmp x4, #0
beq EndUnit


//weight_y_step -> weight_y_step - fw*4*sizeof(int8_t)
mov x10, #4
mul x10, x3, x10
sub x5, x5, x10

//dilate_y_step -> dilate_y_step - fw*dilate_x_step
mul x10, x6, x3
sub x7, x7, x10
LoopFY:
    mov x12, x3
    LoopFX:
        ld1 {v2.s}[0], [x1], x6
        ld1 {v16.s}[0], [x2], #4
        sxtl v2.8h, v2.8b
        sxtl v16.8h, v16.8b
        smlal v0.4s, v2.4h, v16.4h

        subs x12, x12, #1
        bne LoopFX
    subs x4, x4, #1
    add x1, x7, x1
    add x2, x2, x5
    bne LoopFY

scvtf v1.4s, v0.4s
fmul v0.4s, v1.4s, v23.4s


EndUnit:
st1 {v0.4s}, [x0]


ret

#endif
