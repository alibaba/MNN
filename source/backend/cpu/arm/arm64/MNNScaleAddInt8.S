//
//  MNNScaleAddInt8.S
//  MNN
//
//  Created by MNN on 2019/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNScaleAddInt8
// MNNScaleAddInt8(int8_t* dst, const int8_t* src0, const int8_t* src1, 
//  const float* scale0, const float* scale1, const float* outputScale, const size_t size)
// x0: dst, x1:src0, x2:src1, x3:scale0, x4:scale1, x5:outputScale, x6:size

cmp x6, #0
beq End

ld1 {v29.4s}, [x3]
ld1 {v30.4s}, [x4]
ld1 {v31.4s}, [x5]

L4:
cmp x6, #4
blt L1

L4Loop:
    ld1 {v27.16b}, [x1], #16
    sub x6, x6, #4
    ld1 {v28.16b}, [x2], #16
    sxtl v16.8h, v27.8b
    sxtl2 v17.8h, v27.16b
    sxtl v22.8h, v28.8b
    sxtl2 v23.8h, v28.16b
    
    sxtl v18.4s, v16.4h
    sxtl2 v19.4s, v16.8h
    sxtl v20.4s, v17.4h
    sxtl2 v21.4s, v17.8h
    sxtl v24.4s, v22.4h
    sxtl2 v25.4s, v22.8h
    sxtl v26.4s, v23.4h
    sxtl2 v27.4s, v23.8h

    scvtf v0.4s, v18.4s
    scvtf v1.4s, v19.4s
    scvtf v2.4s, v20.4s
    scvtf v3.4s, v21.4s
    scvtf v4.4s, v24.4s
    scvtf v5.4s, v25.4s
    scvtf v6.4s, v26.4s
    scvtf v7.4s, v27.4s

    fmul v0.4s, v0.4s, v29.4s
    fmul v1.4s, v1.4s, v29.4s
    fmul v2.4s, v2.4s, v29.4s
    fmul v3.4s, v3.4s, v29.4s
    fmul v4.4s, v4.4s, v30.4s
    fmul v5.4s, v5.4s, v30.4s
    fmul v6.4s, v6.4s, v30.4s
    fmul v7.4s, v7.4s, v30.4s

    fadd v0.4s, v0.4s, v4.4s
    fadd v1.4s, v1.4s, v5.4s
    fadd v2.4s, v2.4s, v6.4s
    fadd v3.4s, v3.4s, v7.4s

    fmul v16.4s, v0.4s, v31.4s
    fmul v17.4s, v1.4s, v31.4s
    fmul v18.4s, v2.4s, v31.4s
    fmul v19.4s, v3.4s, v31.4s

    fcvtzs v20.4s, v16.4s
    fcvtzs v21.4s, v17.4s
    fcvtzs v22.4s, v18.4s
    fcvtzs v23.4s, v19.4s

    sqxtn v0.4h, v20.4s
    sqxtn2 v0.8h, v21.4s
    sqxtn v1.4h, v22.4s
    sqxtn2 v1.8h, v23.4s

    sqxtn v2.8b, v0.8h
    sqxtn v3.8b, v1.8h

    st1 {v2.8b}, [x0], #8
    cmp x6, #4
    st1 {v3.8b}, [x0], #8
    bge L4Loop

L1:
cmp x6, #0
beq End

L1Loop:
    ld1 {v27.s}[0], [x1], #4
    subs x6, x6, #1
    ld1 {v28.s}[0], [x2], #4

    sxtl v16.8h, v27.8b
    sxtl v18.8h, v28.8b
    sxtl v17.4s, v16.4h
    sxtl v19.4s, v18.4h

    scvtf v0.4s, v17.4s
    scvtf v2.4s, v19.4s
    fmul v1.4s, v0.4s, v29.4s
    fmul v3.4s, v2.4s, v30.4s

    fadd v4.4s, v1.4s, v3.4s
    fmul v0.4s, v4.4s, v31.4s

    fcvtzs v5.4s, v0.4s
    sqxtn v6.4h, v5.4s
    sqxtn v7.8b, v6.8h
    st1 {v7.s}[0], [x0], #4

    bne L1Loop
End:

ret

#endif