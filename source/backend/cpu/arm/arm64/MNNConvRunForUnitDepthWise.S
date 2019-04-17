//
//  MNNConvRunForUnitDepthWise.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvRunForUnitDepthWise
//void MNNConvRunForUnitDepthWise(float* dst, const float* src, const float* weight, size_t fw, size_t fh, size_t weight_y_step, size_t dilate_x_step, size_t dilate_y_step)

//Auto: x0:dst, x1:src, x2:weight, x3:fw
//x4:fh, x5:weight_y_step, x6:dilate_x_step, x7:dilate_y_step

cmp x3, #0
movi v0.4s, #0
beq UnitEnd
cmp x4, #0
beq UnitEnd

mov x9, #4
mul x5, x9, x5
mul x6, x9, x6
mul x7, x9, x7

//dilate_y_step -> dilate_y_step - dilate_x_step*fw
mul x9, x3, x6
sub x7, x7, x9

//weight_y_step -> weight_y_step - 4*sizeof(float)*fw
mov x9, #16
mul x9, x3, x9
sub x5, x5, x9


UnitLoopH:
mov x9, x3
UnitLoopW:
ld1 {v1.4s}, [x1], x6
ld1 {v2.4s}, [x2], #16
fmla v0.4s, v1.4s, v2.4s
subs x9, x9, #1
bne UnitLoopW
subs x4, x4, #1
add x1, x1, x7
add x2, x2, x5
bne UnitLoopH


UnitEnd:

st1 {v0.4s}, [x0]

ret

#endif
