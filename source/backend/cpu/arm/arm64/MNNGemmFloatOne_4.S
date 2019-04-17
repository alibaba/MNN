//
//  MNNGemmFloatOne_4.S
//  MNN
//
//  Created by MNN on 2019/02/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmFloatOne_4
//void MNNGemmFloatOne_4(float* dst, const float* src, const float* weight, size_t src_depth_quad,
//                            size_t dst_step, size_t dst_depth_quad, size_t weight_depth_offset)

//Auto Load:
//x0:dst, x1:src, x2:weight, x3: src_depth_quad
//x4:dst_step, x5:dst_depth_quad, x6:weight_depth_offset

//step multi by sizeof(float)
mov x12, #4
mul x4, x12, x4
mul x6, x12, x6

//x11: weight_dz_step
mov x12, #64 //16*sizeof(float)
mul x11, x12, x3
add x11, x6, x11

mov x7, x3
mov x10, x1

LoopDz:
mov x8, x0
mov x12, x2

L1:
cmp x3, #0
beq LZEnd

ld1 {v0.4s}, [x1], #16
ld1 {v16.4s, v17.4s}, [x2], #32
fmul v2.4s, v16.4s, v0.s[0]
ld1 {v18.4s, v19.4s}, [x2], #32
subs x3, x3, #1
fmul v3.4s, v17.4s, v0.s[1]
beq L1LoopZEnd
L1LoopZ:
    ld1 {v16.4s, v17.4s}, [x2], #32
    fmla v2.4s, v18.4s, v0.s[2]
    fmla v3.4s, v19.4s, v0.s[3]
    ld1 {v0.4s}, [x1], #16
    fmla v2.4s, v16.4s, v0.s[0]
    ld1 {v18.4s, v19.4s}, [x2], #32
    fmla v3.4s, v17.4s, v0.s[1]
    subs x3, x3, #1
    bne L1LoopZ
L1LoopZEnd:
fmla v2.4s, v18.4s, v0.s[2]
fmla v3.4s, v19.4s, v0.s[3]

fadd v0.4s, v2.4s, v3.4s
st1 {v0.4s}, [x8], #16

LZEnd:

subs x5, x5, #1
add x0, x0, x4
mov x1, x10
add x2, x12, x11
mov x3, x7
bne LoopDz

ret

#endif
