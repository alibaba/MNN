//
//  MNNDeconvRunForUnitDepthWise.S
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNDeconvRunForUnitDepthWise
//void MNNDeconvRunForUnitDepthWise(const float* dst, float* src, const float* weight, size_t fw, size_t fh, size_t weight_y_step, size_t dilate_x_step, size_t dilate_y_step)
//Default
//x0:dst, x1:sxc, x2:weight, x3:fw
//x4:fh, x5:weight_y_step, x6:dilate_x_step, x7:dilate_y_step

cmp x3, #0
beq EndUnit

cmp x4, #0
beq EndUnit

//multi by sizeof(float)
mov x12, #4
mul x5, x12, x5
mul x6, x12, x6
mul x7, x12, x7

//weight_y_step -> weight_y_step - fw*4*sizeof(float)
mov x12, #16
mul x12, x3, x12
sub x5, x5, x12

//dilate_y_step -> dilate_y_step - fw*dilate_x_step
mul x12, x6, x3
sub x7, x7, x12

ld1 {v0.4s}, [x0]

LoopFY:
    mov x12, x3
    LoopFX:
        ld1 {v4.4s}, [x2], #16
        ld1 {v1.4s}, [x1]
        fmla v1.4s, v4.4s, v0.4s
        st1 {v1.4s}, [x1], x6
        subs x12, x12, #1
        bne LoopFX

    add x1, x7, x1
    subs x4, x4, #1
    add x2, x2, x5
    bne LoopFY

EndUnit:

ret

#endif
