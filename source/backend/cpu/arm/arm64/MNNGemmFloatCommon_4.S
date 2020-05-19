//
//  MNNGemmFloatCommon_4.S
//  MNN
//
//  Created by MNN on 2018/03/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmFloatCommon_4
//void MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad,
//                            size_t dst_step, size_t dst_depth_quad, size_t width, size_t weight_depth_offset)

//Auto Load:
//x0:dst, x1:src, x2:weight, x3:src_depth_quad, x4:dst_step, x5:dst_depth_quad, x6: width, x7: weight_depth_offset

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
//step multi by sizeof(float)
mov x12, #4
mul x4, x12, x4
mul x7, x12, x7

//x8: src_z_step
mov x12, #16
mul x8, x12, x6

//x9: weight_z_step
mov x12, #64
mul x9, x12, x3
add x9, x7, x9

cmp x6, #4
blt L2

L4:
mov x10, x0
mov x12, x2
mov x14, x5
add x15, x7, x3, LSL #6
add x9, x12, x15
add x15, x9, x15

cmp x5, #3
blt L4_L4LoopDz

L4_L12LoopDz:
mov x11, x1
mov x13, x3

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x9], #64
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x15], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x11], x8
fmul v16.4s, v0.4s, v12.s[0]
fmul v17.4s, v0.4s, v13.s[0]
fmul v18.4s, v0.4s, v14.s[0]
fmul v19.4s, v0.4s, v15.s[0]
fmul v20.4s, v4.4s, v12.s[0]
fmul v21.4s, v4.4s, v13.s[0]
fmul v22.4s, v4.4s, v14.s[0]
fmul v23.4s, v4.4s, v15.s[0]
fmul v24.4s, v8.4s, v12.s[0]
fmul v25.4s, v8.4s, v13.s[0]
fmul v26.4s, v8.4s, v14.s[0]
fmul v27.4s, v8.4s, v15.s[0]

subs x13, x13, #1
beq L4_L12LoopZEnd

L4_L12LoopZ:
    prfm pldl1keep, [x12, #64]
    prfm pldl1keep, [x9, #64]
    prfm pldl1keep, [x15, #64]
    prfm pldl1keep, [x11, x8]

    fmla v16.4s, v1.4s, v12.s[1]
    fmla v17.4s, v1.4s, v13.s[1]
    fmla v18.4s, v1.4s, v14.s[1]
    fmla v19.4s, v1.4s, v15.s[1]
    fmla v20.4s, v5.4s, v12.s[1]
    fmla v21.4s, v5.4s, v13.s[1]
    fmla v22.4s, v5.4s, v14.s[1]
    fmla v23.4s, v5.4s, v15.s[1]
    fmla v24.4s, v9.4s, v12.s[1]
    fmla v25.4s, v9.4s, v13.s[1]
    fmla v26.4s, v9.4s, v14.s[1]
    fmla v27.4s, v9.4s, v15.s[1]

    fmla v16.4s, v2.4s, v12.s[2]
    fmla v17.4s, v2.4s, v13.s[2]
    fmla v18.4s, v2.4s, v14.s[2]
    fmla v19.4s, v2.4s, v15.s[2]
    fmla v20.4s, v6.4s, v12.s[2]
    fmla v21.4s, v6.4s, v13.s[2]
    fmla v22.4s, v6.4s, v14.s[2]
    fmla v23.4s, v6.4s, v15.s[2]
    fmla v24.4s, v10.4s, v12.s[2]
    fmla v25.4s, v10.4s, v13.s[2]
    fmla v26.4s, v10.4s, v14.s[2]
    fmla v27.4s, v10.4s, v15.s[2]

    fmla v16.4s, v3.4s, v12.s[3]
    fmla v17.4s, v3.4s, v13.s[3]
    fmla v18.4s, v3.4s, v14.s[3]
    fmla v19.4s, v3.4s, v15.s[3]
    fmla v20.4s, v7.4s, v12.s[3]
    fmla v21.4s, v7.4s, v13.s[3]
    fmla v22.4s, v7.4s, v14.s[3]
    fmla v23.4s, v7.4s, v15.s[3]
    fmla v24.4s, v11.4s, v12.s[3]
    fmla v25.4s, v11.4s, v13.s[3]
    fmla v26.4s, v11.4s, v14.s[3]
    fmla v27.4s, v11.4s, v15.s[3]

    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x9], #64
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x15], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x11], x8

    fmla v16.4s, v0.4s, v12.s[0]
    fmla v17.4s, v0.4s, v13.s[0]
    fmla v18.4s, v0.4s, v14.s[0]
    fmla v19.4s, v0.4s, v15.s[0]
    fmla v20.4s, v4.4s, v12.s[0]
    fmla v21.4s, v4.4s, v13.s[0]
    fmla v22.4s, v4.4s, v14.s[0]
    fmla v23.4s, v4.4s, v15.s[0]
    fmla v24.4s, v8.4s, v12.s[0]
    fmla v25.4s, v8.4s, v13.s[0]
    fmla v26.4s, v8.4s, v14.s[0]
    fmla v27.4s, v8.4s, v15.s[0]

    subs x13, x13, #1
    bne L4_L12LoopZ

L4_L12LoopZEnd:

fmla v16.4s, v1.4s, v12.s[1]
fmla v17.4s, v1.4s, v13.s[1]
fmla v18.4s, v1.4s, v14.s[1]
fmla v19.4s, v1.4s, v15.s[1]
fmla v20.4s, v5.4s, v12.s[1]
fmla v21.4s, v5.4s, v13.s[1]
fmla v22.4s, v5.4s, v14.s[1]
fmla v23.4s, v5.4s, v15.s[1]
fmla v24.4s, v9.4s, v12.s[1]
fmla v25.4s, v9.4s, v13.s[1]
fmla v26.4s, v9.4s, v14.s[1]
fmla v27.4s, v9.4s, v15.s[1]

fmla v16.4s, v2.4s, v12.s[2]
fmla v17.4s, v2.4s, v13.s[2]
fmla v18.4s, v2.4s, v14.s[2]
fmla v19.4s, v2.4s, v15.s[2]
fmla v20.4s, v6.4s, v12.s[2]
fmla v21.4s, v6.4s, v13.s[2]
fmla v22.4s, v6.4s, v14.s[2]
fmla v23.4s, v6.4s, v15.s[2]
fmla v24.4s, v10.4s, v12.s[2]
fmla v25.4s, v10.4s, v13.s[2]
fmla v26.4s, v10.4s, v14.s[2]
fmla v27.4s, v10.4s, v15.s[2]

fmla v16.4s, v3.4s, v12.s[3]
fmla v17.4s, v3.4s, v13.s[3]
fmla v18.4s, v3.4s, v14.s[3]
fmla v19.4s, v3.4s, v15.s[3]
fmla v20.4s, v7.4s, v12.s[3]
fmla v21.4s, v7.4s, v13.s[3]
fmla v22.4s, v7.4s, v14.s[3]
fmla v23.4s, v7.4s, v15.s[3]
fmla v24.4s, v11.4s, v12.s[3]
fmla v25.4s, v11.4s, v13.s[3]
fmla v26.4s, v11.4s, v14.s[3]
fmla v27.4s, v11.4s, v15.s[3]

st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10], x4
st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], x4
st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10], x4
add x15, x7, x3, LSL #6
add x12, x12, x7
add x12, x12, x15, LSL #1
add x9, x12, x15
add x15, x9, x15
subs x14, x14, #3
beq L4End
cmp x14, #3
bge L4_L12LoopDz

L4_L4LoopDz:
mov x11, x1
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x11], x8
fmul v8.4s, v0.4s, v4.s[0]
fmul v9.4s, v0.4s, v5.s[0]
fmul v10.4s, v0.4s, v6.s[0]
fmul v11.4s, v0.4s, v7.s[0]
fmul v12.4s, v1.4s, v4.s[1]
fmul v13.4s, v1.4s, v5.s[1]
fmul v14.4s, v1.4s, v6.s[1]
fmul v15.4s, v1.4s, v7.s[1]
fmul v16.4s, v2.4s, v4.s[2]
fmul v17.4s, v2.4s, v5.s[2]
fmul v18.4s, v2.4s, v6.s[2]
fmul v19.4s, v2.4s, v7.s[2]
fmul v20.4s, v3.4s, v4.s[3]
fmul v21.4s, v3.4s, v5.s[3]
fmul v22.4s, v3.4s, v6.s[3]
fmul v23.4s, v3.4s, v7.s[3]
subs x13, x3, #1
beq L4_L4LoopZEnd

L4_L4LoopZ:
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x11], x8
    fmla v8.4s, v0.4s, v4.s[0]
    fmla v9.4s, v0.4s, v5.s[0]
    fmla v10.4s, v0.4s, v6.s[0]
    fmla v11.4s, v0.4s, v7.s[0]
    fmla v12.4s, v1.4s, v4.s[1]
    fmla v13.4s, v1.4s, v5.s[1]
    fmla v14.4s, v1.4s, v6.s[1]
    fmla v15.4s, v1.4s, v7.s[1]
    fmla v16.4s, v2.4s, v4.s[2]
    fmla v17.4s, v2.4s, v5.s[2]
    fmla v18.4s, v2.4s, v6.s[2]
    fmla v19.4s, v2.4s, v7.s[2]
    fmla v20.4s, v3.4s, v4.s[3]
    fmla v21.4s, v3.4s, v5.s[3]
    fmla v22.4s, v3.4s, v6.s[3]
    fmla v23.4s, v3.4s, v7.s[3]
    subs x13, x13, #1
    bne L4_L4LoopZ

L4_L4LoopZEnd:
fadd v8.4s, v8.4s, v12.4s
fadd v9.4s, v9.4s, v13.4s
fadd v10.4s, v10.4s, v14.4s
fadd v11.4s, v11.4s, v15.4s
fadd v16.4s, v16.4s, v20.4s
fadd v17.4s, v17.4s, v21.4s
fadd v18.4s, v18.4s, v22.4s
fadd v19.4s, v19.4s, v23.4s
fadd v8.4s, v8.4s, v16.4s
fadd v9.4s, v9.4s, v17.4s
fadd v10.4s, v10.4s, v18.4s
fadd v11.4s, v11.4s, v19.4s
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], x4
add x12, x12, x7
subs x14, x14, #1
bne L4_L4LoopDz

L4End:
add x0, x0, #64
add x1, x1, #64
sub x6, x6, #4
cmp x6, #4
bge L4

L2:
cmp x6, #2
blt L1
sub x6, x6, #2
mov x10, x0
mov x12, x2
mov x14, x5
cmp x5, #3
blt L2_L2LoopDz
add x15, x7, x3, LSL #6
add x9, x12, x15
add x15, x9, x15

L2_L12LoopDz:
mov x11, x1
mov x13, x3

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x9], #64
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x15], #64
ld1 {v12.4s, v13.4s}, [x11], x8
fmul v14.4s, v0.4s, v12.s[0]
fmul v15.4s, v0.4s, v13.s[0]
fmul v20.4s, v1.4s, v12.s[1]
fmul v21.4s, v1.4s, v13.s[1]
fmul v16.4s, v4.4s, v12.s[0]
fmul v17.4s, v4.4s, v13.s[0]
fmul v22.4s, v5.4s, v12.s[1]
fmul v23.4s, v5.4s, v13.s[1]
lsl x8, x8, #2
fmul v18.4s, v8.4s, v12.s[0]
fmul v19.4s, v8.4s, v13.s[0]
fmul v24.4s, v9.4s, v12.s[1]
fmul v25.4s, v9.4s, v13.s[1]
subs x13, x13, #1
beq L2_L12LoopZEnd

L2_L12LoopZ:
    prfm pldl1keep, [x12, #256]
    prfm pldl1keep, [x9, #256]
    prfm pldl1keep, [x15, #256]
    prfm pldl1keep, [x11, x8]

    fmla v14.4s, v2.4s, v12.s[2]
    fmla v15.4s, v2.4s, v13.s[2]
    fmla v20.4s, v3.4s, v12.s[3]
    fmla v21.4s, v3.4s, v13.s[3]
    fmla v16.4s, v6.4s, v12.s[2]
    fmla v17.4s, v6.4s, v13.s[2]
    fmla v22.4s, v7.4s, v12.s[3]
    fmla v23.4s, v7.4s, v13.s[3]
    lsr x8, x8, #2
    fmla v18.4s, v10.4s, v12.s[2]
    fmla v19.4s, v10.4s, v13.s[2]
    fmla v24.4s, v11.4s, v12.s[3]
    fmla v25.4s, v11.4s, v13.s[3]

    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x9], #64
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x15], #64
    ld1 {v12.4s, v13.4s}, [x11], x8
    fmla v14.4s, v0.4s, v12.s[0]
    fmla v15.4s, v0.4s, v13.s[0]
    fmla v16.4s, v4.4s, v12.s[0]
    fmla v17.4s, v4.4s, v13.s[0]
    fmla v18.4s, v8.4s, v12.s[0]
    fmla v19.4s, v8.4s, v13.s[0]
    fmla v20.4s, v1.4s, v12.s[1]
    fmla v21.4s, v1.4s, v13.s[1]
    lsl x8, x8, #2
    fmla v22.4s, v5.4s, v12.s[1]
    fmla v23.4s, v5.4s, v13.s[1]
    fmla v24.4s, v9.4s, v12.s[1]
    fmla v25.4s, v9.4s, v13.s[1]

    subs x13, x13, #1
    bne L2_L12LoopZ

L2_L12LoopZEnd:
fmla v14.4s, v2.4s, v12.s[2]
fmla v15.4s, v2.4s, v13.s[2]
fmla v16.4s, v6.4s, v12.s[2]
fmla v17.4s, v6.4s, v13.s[2]
fmla v18.4s, v10.4s, v12.s[2]
fmla v19.4s, v10.4s, v13.s[2]
fmla v20.4s, v3.4s, v12.s[3]
fmla v21.4s, v3.4s, v13.s[3]
lsr x8, x8, #2
fmla v22.4s, v7.4s, v12.s[3]
fmla v23.4s, v7.4s, v13.s[3]
fmla v24.4s, v11.4s, v12.s[3]
fmla v25.4s, v11.4s, v13.s[3]
fadd v14.4s, v14.4s, v20.4s
fadd v15.4s, v15.4s, v21.4s
fadd v16.4s, v16.4s, v22.4s
fadd v17.4s, v17.4s, v23.4s
fadd v18.4s, v18.4s, v24.4s
fadd v19.4s, v19.4s, v25.4s
st1 {v14.4s, v15.4s}, [x10], x4
st1 {v16.4s, v17.4s}, [x10], x4
st1 {v18.4s, v19.4s}, [x10], x4
add x15, x7, x3, LSL #6
add x12, x12, x7
add x12, x12, x15, LSL #1
add x9, x12, x15
add x15, x9, x15
subs x14, x14, #3
beq L2End
cmp x14, #3
bge L2_L12LoopDz

L2_L2LoopDz:
mov x11, x1
subs x13, x3, #1
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
ld1 {v4.4s, v5.4s}, [x11], x8
fmul v6.4s, v0.4s, v4.s[0]
fmul v7.4s, v0.4s, v5.s[0]
fmul v8.4s, v1.4s, v4.s[1]
fmul v9.4s, v1.4s, v5.s[1]
fmul v10.4s, v2.4s, v4.s[2]
fmul v11.4s, v2.4s, v5.s[2]
fmul v12.4s, v3.4s, v4.s[3]
fmul v13.4s, v3.4s, v5.s[3]
beq L2_L2LoopZEnd

L2_L2LoopZ:
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
    ld1 {v4.4s, v5.4s}, [x11], x8
    fmla v6.4s, v0.4s, v4.s[0]
    fmla v7.4s, v0.4s, v5.s[0]
    fmla v8.4s, v1.4s, v4.s[1]
    fmla v9.4s, v1.4s, v5.s[1]
    fmla v10.4s, v2.4s, v4.s[2]
    fmla v11.4s, v2.4s, v5.s[2]
    fmla v12.4s, v3.4s, v4.s[3]
    fmla v13.4s, v3.4s, v5.s[3]
    subs x13, x13, #1
    bne L2_L2LoopZ

L2_L2LoopZEnd:
fadd v6.4s, v6.4s, v8.4s
fadd v7.4s, v7.4s, v9.4s
fadd v10.4s, v10.4s, v12.4s
fadd v11.4s, v11.4s, v13.4s
fadd v6.4s, v6.4s, v10.4s
fadd v7.4s, v7.4s, v11.4s
st1 {v6.4s, v7.4s}, [x10], x4
add x12, x12, x7
subs x14, x14, #1
bne L2_L2LoopDz

L2End:
add x0, x0, #32
add x1, x1, #32

L1:
lsl x15, x8, #1
#lsl x15, x8, #2
cmp x6, #1
blt End
mov x10, x0
mov x12, x2
mov x14, x5
cmp x5, #3
blt L1_L1LoopDz
add x15, x7, x3, LSL #6
add x9, x12, x15
add x15, x9, x15

L1_L12LoopDz:
mov x11, x1
mov x13, x3

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x9], #64
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x15], #64
ld1 {v12.4s}, [x11], x8
fmul v13.4s, v0.4s, v12.s[0]
fmul v14.4s, v4.4s, v12.s[0]
fmul v15.4s, v8.4s, v12.s[0]
fmul v16.4s, v1.4s, v12.s[1]
fmul v17.4s, v5.4s, v12.s[1]
fmul v18.4s, v9.4s, v12.s[1]
fmul v19.4s, v2.4s, v12.s[2]
fmul v20.4s, v6.4s, v12.s[2]
lsl x8, x8, #2
fmul v21.4s, v10.4s, v12.s[2]
fmul v22.4s, v3.4s, v12.s[3]
fmul v23.4s, v7.4s, v12.s[3]
fmul v24.4s, v11.4s, v12.s[3]
subs x13, x13, #1
beq L1_L12LoopZEnd

L1_L12LoopZ:
    prfm pldl1keep, [x12, #256]
    prfm pldl1keep, [x9, #256]
    prfm pldl1keep, [x15, #256]
    prfm pldl1keep, [x11, x8]

    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x9], #64
    lsr x8, x8, #2
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x15], #64
    ld1 {v12.4s}, [x11], x8
    fmla v13.4s, v0.4s, v12.s[0]
    fmla v14.4s, v4.4s, v12.s[0]
    fmla v15.4s, v8.4s, v12.s[0]
    fmla v16.4s, v1.4s, v12.s[1]
    fmla v17.4s, v5.4s, v12.s[1]
    fmla v18.4s, v9.4s, v12.s[1]
    fmla v19.4s, v2.4s, v12.s[2]
    fmla v20.4s, v6.4s, v12.s[2]
    lsl x8, x8, #2
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
lsr x8, x8, #2
fadd v20.4s, v20.4s, v23.4s
fadd v21.4s, v21.4s, v24.4s
fadd v13.4s, v13.4s, v19.4s
fadd v14.4s, v14.4s, v20.4s
fadd v15.4s, v15.4s, v21.4s
st1 {v13.4s}, [x10], x4
st1 {v14.4s}, [x10], x4
st1 {v15.4s}, [x10], x4
add x15, x7, x3, LSL #6
add x12, x12, x7
add x12, x12, x15, LSL #1
add x9, x12, x15
add x15, x9, x15
subs x14, x14, #3
beq End
cmp x14, #3
bge L1_L12LoopDz

L1_L1LoopDz:
mov x11, x1
subs x13, x3, #1
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
ld1 {v4.4s}, [x11], x8
fmul v5.4s, v0.4s, v4.s[0]
fmul v6.4s, v1.4s, v4.s[1]
fmul v7.4s, v2.4s, v4.s[2]
fmul v8.4s, v3.4s, v4.s[3]
beq L1_L1LoopZEnd

L1_L1LoopZ:
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x12], #64
    ld1 {v4.4s}, [x11], x8
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
st1 {v5.4s}, [x10], x4
add x12, x12, x7
subs x14, x14, #1
bne L1_L1LoopDz

End:

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
