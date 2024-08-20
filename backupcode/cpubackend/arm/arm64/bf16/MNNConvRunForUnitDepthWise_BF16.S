//
//  NEON_MNNConvRunForUnitDepthWise_BF16.S
//  MNN
//
//  Created by MNN on 2021/03/09.
//  Copyright © 2018-2021 Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function NEON_MNNConvRunForUnitDepthWise_BF16
//void NEON_MNNConvRunForUnitDepthWise_BF16(float* dst, const float* src, const float* weight, size_t fw, size_t fh, size_t weight_y_step, size_t dilate_x_step, size_t dilate_y_step)

//Auto: x0:dst, x1:src, x2:weight, x3:fw
//x4:fh, x5:weight_y_step, x6:dilate_x_step, x7:dilate_y_step

cmp x3, #0
movi v0.4s, #0
beq UnitEnd
cmp x4, #0
beq UnitEnd

mov x9, #2
mul x5, x9, x5 // x5(weight_y_step in byte) = sizeof(int16_t) * weight_y_step
mul x6, x9, x6 // x6(dilate_x_step in byte) = sizeof(int16_t) * dilate_x_step
mul x7, x9, x7 // x7(dilate_y_step in byte) = sizeof(int16_t) * dilate_y_step

//dilate_y_step -> dilate_y_step - dilate_x_step*fw
mul x9, x3, x6
sub x7, x7, x9 // because x1 has already been auto-increased at 'ld1 {v1.4h}, [x1], x6', here we should rewind by x6*fw

//weight_y_step -> weight_y_step - 4*sizeof(int16_t)*fw
mov x9, #8
mul x9, x3, x9
sub x5, x5, x9


UnitLoopH:
mov x9, x3
UnitLoopW:
ld1 {v1.4h}, [x1], x6
ld1 {v2.4h}, [x2], #8 // 4 * sizeof(int16_t)
shll v1.4s, v1.4h, #16
shll v2.4s, v2.4h, #16

fmla v0.4s, v1.4s, v2.4s
subs x9, x9, #1
bne UnitLoopW
subs x4, x4, #1
add x1, x1, x7
add x2, x2, x5
bne UnitLoopH


UnitEnd:
shrn v0.4h, v0.4s, #16
st1 {v0.4h}, [x0]

ret

#endif
