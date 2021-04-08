//
//  MNNInt8ScaleToFloat.S
//  MNN
//
//  Created by MNN on 2019/06/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNInt8ScaleToFloat

// void MNNInt8ScaleToFloat(float* dst, 
//    const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint)

// Auto Load:
// x0: dst*, x1: src*, x2: scale*, x3: size, x4: zeroPoint

// copy zero point
mov v28.s[0], w4
mov v28.s[1], w4
mov v28.s[2], w4
mov v28.s[3], w4
scvtf v28.4s, v28.4s

cmp x3, #0
beq End

ld1 {v16.4s}, [x2]

L4:
cmp x3, #4
blt L1

L4Loop:
    ld1 {v17.16b}, [x1], #16
    sub x3, x3, #4
    sxtl v18.8h, v17.8b
    sxtl2 v19.8h, v17.16b
    
    sxtl v0.4s, v18.4h
    sxtl2 v1.4s, v18.8h
    sxtl v2.4s, v19.4h
    sxtl2 v3.4s, v19.8h
    scvtf v4.4s, v0.4s
    scvtf v5.4s, v1.4s
    scvtf v6.4s, v2.4s
    fsub v4.4s, v4.4s, v28.4s
    fsub v5.4s, v5.4s, v28.4s
    fmul v0.4s, v4.4s, v16.4s
    fmul v1.4s, v5.4s, v16.4s
    scvtf v7.4s, v3.4s
    fsub v6.4s, v6.4s, v28.4s
    fmul v2.4s, v6.4s, v16.4s
    st1 {v0.4s, v1.4s}, [x0], #32
    fsub v7.4s, v7.4s, v28.4s
    fmul v3.4s, v7.4s, v16.4s
    cmp x3, #4
    st1 {v2.4s, v3.4s}, [x0], #32
    bge L4Loop
L1:
cmp x3, #0
beq End

L1Loop:
    ld1 {v17.s}[0], [x1], #4
    subs x3, x3, #1
    sxtl v0.8h, v17.8b
    sxtl v1.4s, v0.4h
    scvtf v2.4s, v1.4s
    fsub v2.4s, v2.4s, v28.4s
    fmul v1.4s, v2.4s, v16.4s
    st1 {v1.4s}, [x0], #16

    bne L1Loop

End:

ret
#endif