//
//  MNNDepthWiseInt8AddBiasScaleUnit.S
//  MNN
//
//  Created by MNN on 2019/06/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNDepthWiseInt8AddBiasScaleUnit

// void MNNDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, 
//      const int32_t* bias, size_t fw, size_t fh, size_t weight_y_step, size_t dilateX_step, 
//      size_t dilateY_step, const float* scale)

// Auto
// x0: dst, x1: src, x2: weight, x3: bias
// x4: fw, x5: fh, x6: weight_y_step, x7: dilateX_step
// Load from sp
// x8: dilateY_step, x9: scale
ldr x8, [sp, #0]
ldr x9, [sp, #8]

ld1 {v0.4s}, [x3]

cmp x4, #0
beq EndUnit

cmp x5, #0
beq EndUnit


mov x10, #4
mul x10, x4, x10
sub x6, x6, x10

mul x10, x7, x4
sub x8, x8, x10

LoopFy:
    mov x12, x4
    LoopFx:
        ld1 {v2.s}[0], [x1], x7
        ld1 {v16.s}[0], [x2], #4
        sxtl v2.8h, v2.8b
        sxtl v16.8h, v16.8b

        smlal v0.4s, v2.4h, v16.4h
        subs x12, x12, #1
        bne LoopFx
    subs x5, x5, #1
    add x1, x8, x1
    add x2, x2, x6
    bne LoopFy

EndUnit:

scvtf v1.4s, v0.4s
ld1 {v23.4s}, [x9]
fmul v2.4s, v1.4s, v23.4s
fcvtzs v0.4s, v2.4s

sqxtn v1.4h, v0.4s
sqxtn v2.8b, v1.8h

st1 {v2.s}[0], [x0]

ret
#endif