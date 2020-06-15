//
//  MNNGemmFloatUnit_4.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmFloatUnit_4
//void MNNGemmFloatUnit_4(float* dstOrigin, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t weight_depth_offset)

//Auto
//x0: dst, x1:src, x2:weight, x3:src_depth_quad

//x4:dst_step, x5:dst_depth_quad, x6: weight_depth_offset

mov x12, #4 //sizeof(float)
mul x4, x12, x4
mul x6, x12, x6
add x11, x6, x3, LSL #6

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

cmp x5, #2
blt LoopDzExtra

LoopDz:
mov x8, x1
subs x9, x3, #1

ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x2]
add x2, x2, x11
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x8], #64
fmul v16.4s, v8.4s, v0.s[0]
fmul v17.4s, v8.4s, v1.s[0]
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x8], #64
fmul v18.4s, v8.4s, v2.s[0]
fmul v19.4s, v8.4s, v3.s[0]
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x2], #64
sub x2, x2, x11
fmul v20.4s, v8.4s, v4.s[0]
fmul v21.4s, v8.4s, v5.s[0]
fmul v22.4s, v8.4s, v6.s[0]
fmul v23.4s, v8.4s, v7.s[0]
fmul v24.4s, v12.4s, v0.s[0]
fmul v25.4s, v12.4s, v1.s[0]
fmul v26.4s, v12.4s, v2.s[0]
fmul v27.4s, v12.4s, v3.s[0]
fmul v28.4s, v12.4s, v4.s[0]
fmul v29.4s, v12.4s, v5.s[0]
fmul v30.4s, v12.4s, v6.s[0]
fmul v31.4s, v12.4s, v7.s[0]

beq L8LoopZEnd
L8LoopZ:
    add x2, x2, #128
    prfm pldl1keep, [x2]
    prfm pldl1keep, [x2, x11]
    sub x2, x2, #128
    prfm pldl1keep, [x8, #128]
    prfm pldl1keep, [x8, #192]

    fmla v16.4s, v9.4s, v0.s[1]
    fmla v17.4s, v9.4s, v1.s[1]
    fmla v18.4s, v9.4s, v2.s[1]
    fmla v19.4s, v9.4s, v3.s[1]
    fmla v20.4s, v9.4s, v4.s[1]
    fmla v21.4s, v9.4s, v5.s[1]
    fmla v22.4s, v9.4s, v6.s[1]
    fmla v23.4s, v9.4s, v7.s[1]
    fmla v24.4s, v13.4s, v0.s[1]
    fmla v25.4s, v13.4s, v1.s[1]
    fmla v26.4s, v13.4s, v2.s[1]
    fmla v27.4s, v13.4s, v3.s[1]
    fmla v28.4s, v13.4s, v4.s[1]
    fmla v29.4s, v13.4s, v5.s[1]
    fmla v30.4s, v13.4s, v6.s[1]
    fmla v31.4s, v13.4s, v7.s[1]

    fmla v16.4s, v10.4s, v0.s[2]
    fmla v17.4s, v10.4s, v1.s[2]
    fmla v18.4s, v10.4s, v2.s[2]
    fmla v19.4s, v10.4s, v3.s[2]
    fmla v20.4s, v10.4s, v4.s[2]
    fmla v21.4s, v10.4s, v5.s[2]
    fmla v22.4s, v10.4s, v6.s[2]
    fmla v23.4s, v10.4s, v7.s[2]
    fmla v24.4s, v14.4s, v0.s[2]
    fmla v25.4s, v14.4s, v1.s[2]
    fmla v26.4s, v14.4s, v2.s[2]
    fmla v27.4s, v14.4s, v3.s[2]
    fmla v28.4s, v14.4s, v4.s[2]
    fmla v29.4s, v14.4s, v5.s[2]
    fmla v30.4s, v14.4s, v6.s[2]
    fmla v31.4s, v14.4s, v7.s[2]

    fmla v16.4s, v11.4s, v0.s[3]
    fmla v17.4s, v11.4s, v1.s[3]
    fmla v18.4s, v11.4s, v2.s[3]
    fmla v19.4s, v11.4s, v3.s[3]
    fmla v20.4s, v11.4s, v4.s[3]
    fmla v21.4s, v11.4s, v5.s[3]
    fmla v22.4s, v11.4s, v6.s[3]
    fmla v23.4s, v11.4s, v7.s[3]
    fmla v24.4s, v15.4s, v0.s[3]
    fmla v25.4s, v15.4s, v1.s[3]
    fmla v26.4s, v15.4s, v2.s[3]
    fmla v27.4s, v15.4s, v3.s[3]
    fmla v28.4s, v15.4s, v4.s[3]
    fmla v29.4s, v15.4s, v5.s[3]
    fmla v30.4s, v15.4s, v6.s[3]
    fmla v31.4s, v15.4s, v7.s[3]

    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x2]
    add x2, x2, x11
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x8], #64
    fmla v16.4s, v8.4s, v0.s[0]
    fmla v17.4s, v8.4s, v1.s[0]
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x8], #64
    fmla v18.4s, v8.4s, v2.s[0]
    fmla v19.4s, v8.4s, v3.s[0]
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x2], #64
    sub x2, x2, x11
    fmla v20.4s, v8.4s, v4.s[0]
    fmla v21.4s, v8.4s, v5.s[0]
    fmla v22.4s, v8.4s, v6.s[0]
    fmla v23.4s, v8.4s, v7.s[0]
    fmla v24.4s, v12.4s, v0.s[0]
    fmla v25.4s, v12.4s, v1.s[0]
    fmla v26.4s, v12.4s, v2.s[0]
    fmla v27.4s, v12.4s, v3.s[0]
    fmla v28.4s, v12.4s, v4.s[0]
    fmla v29.4s, v12.4s, v5.s[0]
    fmla v30.4s, v12.4s, v6.s[0]
    fmla v31.4s, v12.4s, v7.s[0]

    subs x9, x9, #1
    bne L8LoopZ

L8LoopZEnd:
fmla v16.4s, v9.4s, v0.s[1]
fmla v17.4s, v9.4s, v1.s[1]
fmla v18.4s, v9.4s, v2.s[1]
fmla v19.4s, v9.4s, v3.s[1]
fmla v20.4s, v9.4s, v4.s[1]
fmla v21.4s, v9.4s, v5.s[1]
fmla v22.4s, v9.4s, v6.s[1]
fmla v23.4s, v9.4s, v7.s[1]
fmla v24.4s, v13.4s, v0.s[1]
fmla v25.4s, v13.4s, v1.s[1]
fmla v26.4s, v13.4s, v2.s[1]
fmla v27.4s, v13.4s, v3.s[1]
fmla v28.4s, v13.4s, v4.s[1]
fmla v29.4s, v13.4s, v5.s[1]
fmla v30.4s, v13.4s, v6.s[1]
fmla v31.4s, v13.4s, v7.s[1]

fmla v16.4s, v10.4s, v0.s[2]
fmla v17.4s, v10.4s, v1.s[2]
fmla v18.4s, v10.4s, v2.s[2]
fmla v19.4s, v10.4s, v3.s[2]
fmla v20.4s, v10.4s, v4.s[2]
fmla v21.4s, v10.4s, v5.s[2]
fmla v22.4s, v10.4s, v6.s[2]
fmla v23.4s, v10.4s, v7.s[2]
fmla v24.4s, v14.4s, v0.s[2]
fmla v25.4s, v14.4s, v1.s[2]
fmla v26.4s, v14.4s, v2.s[2]
fmla v27.4s, v14.4s, v3.s[2]
fmla v28.4s, v14.4s, v4.s[2]
fmla v29.4s, v14.4s, v5.s[2]
fmla v30.4s, v14.4s, v6.s[2]
fmla v31.4s, v14.4s, v7.s[2]

mov x12, x0

fmla v16.4s, v11.4s, v0.s[3]
fmla v17.4s, v11.4s, v1.s[3]
fmla v18.4s, v11.4s, v2.s[3]
fmla v19.4s, v11.4s, v3.s[3]
fmla v20.4s, v11.4s, v4.s[3]
fmla v21.4s, v11.4s, v5.s[3]
fmla v22.4s, v11.4s, v6.s[3]
fmla v23.4s, v11.4s, v7.s[3]
fmla v24.4s, v15.4s, v0.s[3]
fmla v25.4s, v15.4s, v1.s[3]
fmla v26.4s, v15.4s, v2.s[3]
fmla v27.4s, v15.4s, v3.s[3]
fmla v28.4s, v15.4s, v4.s[3]
st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
fmla v29.4s, v15.4s, v5.s[3]
st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
fmla v30.4s, v15.4s, v6.s[3]
add x0, x12, x4
st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], #64
add x2, x2, x11
fmla v31.4s, v15.4s, v7.s[3]
add x2, x2, x6
sub x5, x5, #2
st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x0], #64
add x0, x12, x4, LSL #1

cmp x5, #1
blt LoopDzEnd
bgt LoopDz

LoopDzExtra:

mov w11, #0
mov x8, x1
mov x9, x3
dup v16.4s, w11
dup v17.4s, w11
dup v18.4s, w11
dup v19.4s, w11
dup v20.4s, w11
dup v21.4s, w11
dup v22.4s, w11
dup v23.4s, w11

L4LoopZ:
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x2], #64
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x8], #64
    fmla v16.4s, v8.4s, v0.s[0]
    fmla v17.4s, v8.4s, v1.s[0]
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x8], #64
    fmla v18.4s, v8.4s, v2.s[0]
    fmla v19.4s, v8.4s, v3.s[0]
    fmla v20.4s, v8.4s, v4.s[0]
    fmla v21.4s, v8.4s, v5.s[0]
    fmla v22.4s, v8.4s, v6.s[0]
    fmla v23.4s, v8.4s, v7.s[0]

    fmla v16.4s, v9.4s, v0.s[1]
    fmla v17.4s, v9.4s, v1.s[1]
    fmla v18.4s, v9.4s, v2.s[1]
    fmla v19.4s, v9.4s, v3.s[1]
    fmla v20.4s, v9.4s, v4.s[1]
    fmla v21.4s, v9.4s, v5.s[1]
    fmla v22.4s, v9.4s, v6.s[1]
    fmla v23.4s, v9.4s, v7.s[1]

    fmla v16.4s, v10.4s, v0.s[2]
    fmla v17.4s, v10.4s, v1.s[2]
    fmla v18.4s, v10.4s, v2.s[2]
    fmla v19.4s, v10.4s, v3.s[2]
    fmla v20.4s, v10.4s, v4.s[2]
    fmla v21.4s, v10.4s, v5.s[2]
    fmla v22.4s, v10.4s, v6.s[2]
    fmla v23.4s, v10.4s, v7.s[2]

    fmla v16.4s, v11.4s, v0.s[3]
    fmla v17.4s, v11.4s, v1.s[3]
    fmla v18.4s, v11.4s, v2.s[3]
    fmla v19.4s, v11.4s, v3.s[3]
    fmla v20.4s, v11.4s, v4.s[3]
    fmla v21.4s, v11.4s, v5.s[3]
    fmla v22.4s, v11.4s, v6.s[3]
    fmla v23.4s, v11.4s, v7.s[3]

    subs x9, x9, #1
    bne L4LoopZ

st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64

LoopDzEnd:
sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

ret
#endif
