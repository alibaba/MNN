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

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
//step multi by sizeof(float)
mov x12, #4
mul x4, x12, x4
mul x6, x12, x6

mov x12, #64 //16*sizeof(float)
mul x9, x12, x3
add x9, x6, x9

cmp x5, #3
blt L1_L1LoopDz

add x7, x2, x9
add x8, x2, x9, LSL #1

L1_L12LoopDz:
mov x11, x1
mov x13, x3

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x2], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x8], #64
ld1 {v12.4s}, [x11], #16
fmul v13.4s, v0.4s, v12.s[0]
fmul v14.4s, v4.4s, v12.s[0]
fmul v15.4s, v8.4s, v12.s[0]
fmul v16.4s, v1.4s, v12.s[1]
fmul v17.4s, v5.4s, v12.s[1]
fmul v18.4s, v9.4s, v12.s[1]
fmul v19.4s, v2.4s, v12.s[2]
fmul v20.4s, v6.4s, v12.s[2]
fmul v21.4s, v10.4s, v12.s[2]
fmul v22.4s, v3.4s, v12.s[3]
fmul v23.4s, v7.4s, v12.s[3]
fmul v24.4s, v11.4s, v12.s[3]
subs x13, x13, #1
beq L1_L12LoopZEnd

L1_L12LoopZ:
    prfm pldl1keep, [x2, #256]
    prfm pldl1keep, [x7, #256]
    prfm pldl1keep, [x8, #256]
    prfm pldl1keep, [x11, #128]

    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x2], #64
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x8], #64
    ld1 {v12.4s}, [x11], #16
    fmla v13.4s, v0.4s, v12.s[0]
    fmla v14.4s, v4.4s, v12.s[0]
    fmla v15.4s, v8.4s, v12.s[0]
    fmla v16.4s, v1.4s, v12.s[1]
    fmla v17.4s, v5.4s, v12.s[1]
    fmla v18.4s, v9.4s, v12.s[1]
    fmla v19.4s, v2.4s, v12.s[2]
    fmla v20.4s, v6.4s, v12.s[2]
    fmla v21.4s, v10.4s, v12.s[2]
    fmla v22.4s, v3.4s, v12.s[3]
    fmla v23.4s, v7.4s, v12.s[3]
    fmla v24.4s, v11.4s, v12.s[3]
    subs x13, x13, #1
    bne L1_L12LoopZ

L1_L12LoopZEnd:
fadd v13.4s, v13.4s, v16.4s
fadd v14.4s, v14.4s, v17.4s
fadd v15.4s, v15.4s, v18.4s
fadd v19.4s, v19.4s, v22.4s
fadd v20.4s, v20.4s, v23.4s
fadd v21.4s, v21.4s, v24.4s
fadd v13.4s, v13.4s, v19.4s
fadd v14.4s, v14.4s, v20.4s
fadd v15.4s, v15.4s, v21.4s
st1 {v13.4s}, [x0], x4
st1 {v14.4s}, [x0], x4
st1 {v15.4s}, [x0], x4
add x2, x2, x6
add x7, x7, x6
add x8, x8, x6
add x2, x2, x9, LSL #1
add x7, x7, x9, LSL #1
add x8, x8, x9, LSL #1
subs x5, x5, #3
beq End
cmp x5, #3
bge L1_L12LoopDz

L1_L1LoopDz:
mov x11, x1
subs x13, x3, #1
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x2], #64
ld1 {v4.4s}, [x11], #16
fmul v5.4s, v0.4s, v4.s[0]
fmul v6.4s, v1.4s, v4.s[1]
fmul v7.4s, v2.4s, v4.s[2]
fmul v8.4s, v3.4s, v4.s[3]
beq L1_L1LoopZEnd

L1_L1LoopZ:
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x2], #64
    ld1 {v4.4s}, [x11], #16
    fmla v5.4s, v0.4s, v4.s[0]
    fmla v6.4s, v1.4s, v4.s[1]
    fmla v7.4s, v2.4s, v4.s[2]
    fmla v8.4s, v3.4s, v4.s[3]
    subs x13, x13, #1
    bne L1_L1LoopZ

L1_L1LoopZEnd:
fadd v5.4s, v5.4s, v6.4s
fadd v7.4s, v7.4s, v8.4s
fadd v5.4s, v5.4s, v7.4s
st1 {v5.4s}, [x0], x4
add x2, x2, x6
subs x5, x5, #1
bne L1_L1LoopDz

End:

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
