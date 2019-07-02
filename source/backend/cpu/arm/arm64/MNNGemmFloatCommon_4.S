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

//Load from sp
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
mov x13, x3
add x9, x7, x9
mov v24.d[0], x6

LoopDz:
mov x10, x0
mov x14, x1
mov x15, x2

.macro START_TWO z0 z1
ld1 {v0.4s}, [x1], #16
fmul \z0, v2.4s, v0.s[0]
ld1 {v1.4s}, [x1], #16
fmla \z0, v3.4s, v0.s[1]
fmul \z1, v2.4s, v1.s[0]
fmla \z0, v4.4s, v0.s[2]
fmla \z1, v3.4s, v1.s[1]
fmla \z0, v5.4s, v0.s[3]
fmla \z1, v4.4s, v1.s[2]
fmla \z1, v5.4s, v1.s[3]
.endm

.macro COMPUTE_TWO z0 z1
ld1 {v0.4s}, [x1], #16
fmla \z0, v2.4s, v0.s[0]
ld1 {v1.4s}, [x1], #16
fmla \z0, v3.4s, v0.s[1]
fmla \z1, v2.4s, v1.s[0]
fmla \z0, v4.4s, v0.s[2]
fmla \z1, v3.4s, v1.s[1]
fmla \z0, v5.4s, v0.s[3]
fmla \z1, v4.4s, v1.s[2]
fmla \z1, v5.4s, v1.s[3]
.endm

L8:
cmp x6, #7
ble L4


L8Loop:
    mov x11, x1
    mov x12, x2
    ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
    
    fmul v16.4s, v28.4s, v0.s[0]
    fmul v17.4s, v28.4s, v1.s[0]
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
    fmul v18.4s, v28.4s, v2.s[0]
    fmul v19.4s, v28.4s, v3.s[0]
    fmul v20.4s, v28.4s, v4.s[0]
    fmul v21.4s, v28.4s, v5.s[0]
    fmul v22.4s, v28.4s, v6.s[0]
    fmul v23.4s, v28.4s, v7.s[0]

    fmla v16.4s, v29.4s, v0.s[1]
    fmla v17.4s, v29.4s, v1.s[1]
    fmla v18.4s, v29.4s, v2.s[1]
    fmla v19.4s, v29.4s, v3.s[1]
    fmla v20.4s, v29.4s, v4.s[1]
    fmla v21.4s, v29.4s, v5.s[1]
    fmla v22.4s, v29.4s, v6.s[1]
    fmla v23.4s, v29.4s, v7.s[1]

    fmla v16.4s, v30.4s, v0.s[2]
    fmla v17.4s, v30.4s, v1.s[2]
    fmla v18.4s, v30.4s, v2.s[2]
    fmla v19.4s, v30.4s, v3.s[2]
    fmla v20.4s, v30.4s, v4.s[2]
    fmla v21.4s, v30.4s, v5.s[2]
    fmla v22.4s, v30.4s, v6.s[2]
    fmla v23.4s, v30.4s, v7.s[2]

    fmla v16.4s, v31.4s, v0.s[3]
    fmla v17.4s, v31.4s, v1.s[3]
    fmla v18.4s, v31.4s, v2.s[3]
    fmla v19.4s, v31.4s, v3.s[3]
    fmla v20.4s, v31.4s, v4.s[3]
    fmla v21.4s, v31.4s, v5.s[3]
    fmla v22.4s, v31.4s, v6.s[3]
    fmla v23.4s, v31.4s, v7.s[3]

    subs x3, x3, #1

    beq L8LoopEnd

    L8LoopZ:
        sub x1, x1, #128
        add x1, x1, x8
        ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
        
        fmla v16.4s, v28.4s, v0.s[0]
        fmla v17.4s, v28.4s, v1.s[0]
        ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
        fmla v18.4s, v28.4s, v2.s[0]
        fmla v19.4s, v28.4s, v3.s[0]
        fmla v20.4s, v28.4s, v4.s[0]
        fmla v21.4s, v28.4s, v5.s[0]
        fmla v22.4s, v28.4s, v6.s[0]
        fmla v23.4s, v28.4s, v7.s[0]

        fmla v16.4s, v29.4s, v0.s[1]
        fmla v17.4s, v29.4s, v1.s[1]
        fmla v18.4s, v29.4s, v2.s[1]
        fmla v19.4s, v29.4s, v3.s[1]
        fmla v20.4s, v29.4s, v4.s[1]
        fmla v21.4s, v29.4s, v5.s[1]
        fmla v22.4s, v29.4s, v6.s[1]
        fmla v23.4s, v29.4s, v7.s[1]

        fmla v16.4s, v30.4s, v0.s[2]
        fmla v17.4s, v30.4s, v1.s[2]
        fmla v18.4s, v30.4s, v2.s[2]
        fmla v19.4s, v30.4s, v3.s[2]
        fmla v20.4s, v30.4s, v4.s[2]
        fmla v21.4s, v30.4s, v5.s[2]
        fmla v22.4s, v30.4s, v6.s[2]
        fmla v23.4s, v30.4s, v7.s[2]

        fmla v16.4s, v31.4s, v0.s[3]
        fmla v17.4s, v31.4s, v1.s[3]
        fmla v18.4s, v31.4s, v2.s[3]
        fmla v19.4s, v31.4s, v3.s[3]
        fmla v20.4s, v31.4s, v4.s[3]
        fmla v21.4s, v31.4s, v5.s[3]
        fmla v22.4s, v31.4s, v6.s[3]
        fmla v23.4s, v31.4s, v7.s[3]

        subs x3, x3, #1
        bne L8LoopZ

    L8LoopEnd:
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    add x1, x11, #128
    mov x2, x12
    sub x6, x6, #8
    mov x3, x13
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
    cmp x6, #8
    bge L8Loop



L4:
cmp x6, #3
ble L1

mov x11, x1
mov x12, x2

ld1 {v2.4s, v3.4s, v4.4s, v5.4s}, [x2], #64

START_TWO v16.4s, v17.4s
START_TWO v18.4s, v19.4s
subs x3, x3, #1
beq L4LoopZEnd

L4LoopZ:
    sub x1, x1, #64
    add x1, x1, x8
    ld1 {v2.4s, v3.4s, v4.4s, v5.4s}, [x2], #64
    COMPUTE_TWO v16.4s, v17.4s
    COMPUTE_TWO v18.4s, v19.4s

    subs x3, x3, #1
    bne L4LoopZ

L4LoopZEnd:
st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
add x1, x11, #64
mov x2, x12
mov x3, x13
sub x6, x6, #4

L1:
cmp x6, #0
ble End

L1Loop:
    mov x11, x1
    mov x12, x2
    ld1 {v0.4s}, [x1], x8
    ld1 {v2.4s, v3.4s, v4.4s, v5.4s}, [x2], #64
    fmul v16.4s, v2.4s, v0.s[0]
    subs x3, x3, #1
    fmul v17.4s, v3.4s, v0.s[1]

    beq L1LoopZEnd

    L1LoopZ:
        fmla v16.4s, v4.4s, v0.s[2]
        fmla v17.4s, v5.4s, v0.s[3]

        ld1 {v2.4s, v3.4s, v4.4s, v5.4s}, [x2], #64
        ld1 {v0.4s}, [x1], x8
        fmla v16.4s, v2.4s, v0.s[0]
        fmla v17.4s, v3.4s, v0.s[1]

        subs x3, x3, #1
        bne L1LoopZ
    L1LoopZEnd:

    fmla v16.4s, v4.4s, v0.s[2]
    fmla v17.4s, v5.4s, v0.s[3]

    fadd v16.4s, v16.4s, v17.4s
    add x1, x11, #16
    mov x2, x12
    mov x3, x13
    subs x6, x6, #1
    st1 {v16.4s}, [x0], #16
    bne L1Loop

End:

subs x5, x5, #1
add x0, x10, x4
mov x1, x14
add x2, x15, x9
mov x6, v24.d[0]
bne LoopDz

ret

#endif
