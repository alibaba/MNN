//
//  MNNBilinearProcC1.S
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNBilinearProcC1
//void MNNBilinearProcC1(const unsigned char *sample, unsigned char* dst, const int16_t* xFactor, const int16_t* yFactor, size_t w);

//Auto: x0:sample, x1:dst, x2:xFactor, x3:yFactor, x4:w

ld1 {v31.s}[0], [x3]
//Now x3 is no used
dup v30.4h, v31.h[0]
dup v31.4h, v31.h[1]

L8:
cmp x4, #8
blt End

LoopL8:
    ld4 {v4.8b, v5.8b, v6.8b, v7.8b}, [x0], #32
    ld2 {v0.8h, v1.8h}, [x2], #32//q0, q1
    //(x00,x01) -> (y0)
    uxtl v2.8h, v4.8b
    uxtl v3.8h, v5.8b
    umull v16.4s, v2.4h, v0.4h
    umull2 v17.4s, v2.8h, v0.8h
    umlal v16.4s, v3.4h, v1.4h
    umlal2 v17.4s, v3.8h, v1.8h

    uqshrn v18.4h, v16.4s, #4
    uqshrn v19.4h, v17.4s, #4

    //(x10,x11) -> (y1)
    uxtl v2.8h, v6.8b
    uxtl v3.8h, v7.8b
    umull v16.4s, v2.4h, v0.4h
    umull2 v17.4s, v2.8h, v0.8h
    umlal v16.4s, v3.4h, v1.4h
    umlal2 v17.4s, v3.8h, v1.8h

    uqshrn v20.4h, v16.4s, #4
    uqshrn v21.4h, v17.4s, #4

    //(y0,y1) -> dst
    umull v16.4s, v18.4h, v30.4h
    umull v17.4s, v19.4h, v30.4h
    umlal v16.4s, v20.4h, v31.4h
    umlal v17.4s, v21.4h, v31.4h

    uqshrn v2.4h, v16.4s, #16
    uqshrn2 v2.8h, v17.4s, #16

    uqrshrn v0.8b, v2.8h, #2

    st1 {v0.8b}, [x1], #8


    sub x4, x4, #8
    cmp x4, #8
    bge LoopL8


End:


ret
#endif
