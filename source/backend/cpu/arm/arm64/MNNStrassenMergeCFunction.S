//
//  MNNStrassenMergeCFunction.S
//  MNN
//
//  Created by MNN on 2019/02/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNStrassenMergeCFunction
//void MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22,
//      float* xAddr, size_t cStride, size_t eSub, size_t hSub) {

//Auto: x0: c11, x1:c12, x2:c21, x3:c22
//x4: xAddr, x5: cStride, x6: eSub, x7: hSub

//x5 -> cExtraOffset
mov x12, #4 //sizeof(float)
mul x5, x12, x5
mov x11, #16
mul x11, x6, x11
sub x5, x5, x11

LoopY:
    XL1:
    subs x12, x6, #1
    ld1 {v16.4s}, [x4], #16//x
    ld1 {v20.4s}, [x1]//c12
    ld1 {v17.4s}, [x2]//c21
    fadd v20.4s, v20.4s, v16.4s
    ld1 {v18.4s}, [x3]//c22
    fadd v17.4s, v17.4s, v20.4s
    beq XEnd

    LoopXL1:
        fadd v20.4s, v20.4s, v18.4s
        st1 {v17.4s}, [x2], #16
        ld1 {v19.4s}, [x0], #16 //c11
        fadd v18.4s, v18.4s, v17.4s
        ld1 {v16.4s}, [x4], #16//x
        fadd v20.4s, v20.4s, v19.4s
        st1 {v18.4s}, [x3], #16
        st1 {v20.4s}, [x1], #16

        ld1 {v20.4s}, [x1]//c12
        ld1 {v17.4s}, [x2]//c21
        fadd v20.4s, v20.4s, v16.4s
        ld1 {v18.4s}, [x3]//c22
        fadd v17.4s, v17.4s, v20.4s

        subs x12, x12, #1
        bne LoopXL1

    XEnd:

    fadd v20.4s, v20.4s, v18.4s
    st1 {v17.4s}, [x2], #16
    ld1 {v19.4s}, [x0], #16 //c11
    fadd v18.4s, v18.4s, v17.4s
    fadd v20.4s, v20.4s, v19.4s
    st1 {v18.4s}, [x3], #16
    st1 {v20.4s}, [x1], #16

    add x0, x0, x5
    add x1, x1, x5
    add x2, x2, x5
    add x3, x3, x5


    subs x7, x7, #1
    bne LoopY

ret

#endif
