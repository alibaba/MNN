//
//  MNNBinaryMinInt8.S
//  MNN
//
//  Created by MNN on 2019/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNBinaryMinInt8
// MNNBinaryMinInt8(int8_t* dst, const int8_t* src0, const int8_t* src1,
//  const float* scale0, const float* scale1, const float* outputScale, const size_t size, size_t needBroadcast)
// x0: dst, x1:src0, x2:src1, x3:scale0, x4:scale1, x5:outputScale, x6:size, x7: needBroadcast

cmp x6, #0
beq End

ld1 {v29.4s}, [x3] // scale0
ld1 {v30.4s}, [x4] // scale1
ld1 {v31.4s}, [x5] // outputScale

cmp x6, #8
bge L8Loop
cmp x6, #4
bge L4Loop
blt L1

L8Loop:
    cmp x7, #0
    beq L8NeedBroadcast0
    cmp x7, #1
    beq L8NeedBroadcast1

    L8NotNeedBroadcast:
    ld1 {v27.16b}, [x1], #16 // input0[0]: int8x16
    ld1 {v28.16b}, [x2], #16 // input1[0]: int8x16
    ld1 {v14.16b}, [x1], #16 // input0[1]: int8x16
    ld1 {v15.16b}, [x2], #16 // input1[1]: int8x16
    b L8Compute

    L8NeedBroadcast0:
    ld1 {v27.b}[0], [x1]
    dup v27.16b, v27.b[0]
    ld1 {v14.b}[0], [x1]
    dup v14.16b, v14.b[0]
    ld1 {v28.16b}, [x2], #16 // input0[1]: int8x16
    ld1 {v15.16b}, [x2], #16 // input1[1]: int8x16
    b L8Compute

    L8NeedBroadcast1:
    ld1 {v28.b}[0], [x2]
    dup v28.16b, v28.b[0]
    ld1 {v15.b}[0], [x2]
    dup v15.16b, v15.b[0]
    ld1 {v27.16b}, [x1], #16 // input0[1]: int8x16
    ld1 {v14.16b}, [x1], #16 // input1[1]: int8x16
    b L8Compute

    L8Compute:
    sxtl v16.8h, v27.8b      // v16:       int16x8
    sxtl2 v17.8h, v27.16b    // v17:       int16x8
    sxtl v22.8h, v28.8b
    sxtl2 v23.8h, v28.16b

    sxtl v10.8h, v14.8b      // v10:       int16x8
    sxtl2 v11.8h, v14.16b    // v11:       int16x8
    sxtl v12.8h, v15.8b
    sxtl2 v13.8h, v15.16b
    
    sxtl  v18.4s, v16.4h      // v18:       int32x4
    sxtl2 v19.4s, v16.8h
    sxtl  v20.4s, v17.4h
    sxtl2 v21.4s, v17.8h
    sxtl  v24.4s, v22.4h
    sxtl2 v25.4s, v22.8h
    sxtl  v26.4s, v23.4h
    sxtl2 v27.4s, v23.8h

    sxtl  v2.4s, v10.4h      // v18:       int32x4
    sxtl2 v3.4s, v10.8h
    sxtl  v4.4s, v11.4h
    sxtl2 v5.4s, v11.8h
    sxtl  v6.4s, v12.4h
    sxtl2 v7.4s, v12.8h
    sxtl  v8.4s, v13.4h
    sxtl2 v9.4s, v13.8h

    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s
    scvtf v24.4s, v24.4s
    scvtf v25.4s, v25.4s
    scvtf v26.4s, v26.4s
    scvtf v27.4s, v27.4s

    scvtf v2.4s, v2.4s
    scvtf v3.4s, v3.4s
    scvtf v4.4s, v4.4s
    scvtf v5.4s, v5.4s
    scvtf v6.4s, v6.4s
    scvtf v7.4s, v7.4s
    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s

    fmul v2.4s, v2.4s, v29.4s
    fmul v3.4s, v3.4s, v29.4s
    fmul v4.4s, v4.4s, v29.4s
    fmul v5.4s, v5.4s, v29.4s
    fmul v6.4s, v6.4s, v30.4s
    fmul v7.4s, v7.4s, v30.4s
    fmul v8.4s, v8.4s, v30.4s
    fmul v9.4s, v9.4s, v30.4s

    fmul v18.4s, v18.4s, v29.4s
    fmul v19.4s, v19.4s, v29.4s
    fmul v20.4s, v20.4s, v29.4s
    fmul v21.4s, v21.4s, v29.4s
    fmul v24.4s, v24.4s, v30.4s
    fmul v25.4s, v25.4s, v30.4s
    fmul v26.4s, v26.4s, v30.4s
    fmul v27.4s, v27.4s, v30.4s

    fmin v2.4s, v2.4s, v6.4s
    fmin v3.4s, v3.4s, v7.4s
    fmin v4.4s, v4.4s, v8.4s
    fmin v5.4s, v5.4s, v9.4s

    fmin v18.4s, v18.4s, v24.4s
    fmin v19.4s, v19.4s, v25.4s
    fmin v20.4s, v20.4s, v26.4s
    fmin v21.4s, v21.4s, v27.4s

    fmul v2.4s, v2.4s, v31.4s
    fmul v3.4s, v3.4s, v31.4s
    fmul v4.4s, v4.4s, v31.4s
    fmul v5.4s, v5.4s, v31.4s
    fmul v18.4s, v18.4s, v31.4s
    fmul v19.4s, v19.4s, v31.4s
    fmul v20.4s, v20.4s, v31.4s
    fmul v21.4s, v21.4s, v31.4s

    fcvtzs v18.4s, v18.4s
    fcvtzs v19.4s, v19.4s
    fcvtzs v20.4s, v20.4s
    fcvtzs v21.4s, v21.4s
    fcvtzs v2.4s, v2.4s
    fcvtzs v3.4s, v3.4s
    fcvtzs v4.4s, v4.4s
    fcvtzs v5.4s, v5.4s

    sqxtn v18.4h, v18.4s
    sqxtn2 v18.8h, v19.4s
    sqxtn v19.4h, v20.4s
    sqxtn2 v19.8h, v21.4s

    sqxtn v2.4h, v2.4s
    sqxtn2 v2.8h, v3.4s
    sqxtn v3.4h, v4.4s
    sqxtn2 v3.8h, v5.4s

    sqxtn v18.8b, v18.8h
    sqxtn2 v18.16b, v19.8h
    sqxtn v2.8b, v2.8h
    sqxtn2 v2.16b, v3.8h

    st1 {v18.16b}, [x0], #16
    st1 {v2.16b}, [x0], #16

    sub x6, x6, #8
    cmp x6, #8
    bge L8Loop
    cmp x6, #0
    ble End
    cmp x6, #4
    blt L1Loop

L4Loop:
    cmp x7, #0
    beq L4NeedBroadcast0
    cmp x7, #1
    beq L4NeedBroadcast1

    L4NotNeedBroadcast:
    ld1 {v27.16b}, [x1], #16
    ld1 {v28.16b}, [x2], #16
    b L4Compute

    L4NeedBroadcast0:
    ld1 {v27.b}[0], [x1]
    dup v27.16b, v27.b[0]
    ld1 {v28.16b}, [x2], #16
    b L4Compute

    L4NeedBroadcast1:
    ld1 {v28.b}[0], [x2]
    dup v28.16b, v28.b[0]
    ld1 {v27.16b}, [x1], #16
    b L4Compute

    L4Compute:
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

    fmin v0.4s, v0.4s, v4.4s
    fmin v1.4s, v1.4s, v5.4s
    fmin v2.4s, v2.4s, v6.4s
    fmin v3.4s, v3.4s, v7.4s

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

    st1 {v2.8b, v3.8b}, [x0], #16
    sub x6, x6, #4
    cmp x6, #4
    bge L4Loop

L1:
cmp x6, #0
beq End

L1Loop:
    cmp x7, #0
    beq L1NeedBroadcast0
    cmp x7, #1
    beq L1NeedBroadcast1

    L1NotNeedBroadcast:
    ld1 {v27.s}[0], [x1], #4
    ld1 {v28.s}[0], [x2], #4
    b L1Compute

    L1NeedBroadcast0:
    ld1 {v27.b}[0], [x1]
    dup v27.8b, v27.b[0]
    ld1 {v28.s}[0], [x2], #4
    b L1Compute

    L1NeedBroadcast1:
    ld1 {v28.b}[0], [x2]
    dup v28.8b, v28.b[0]
    ld1 {v27.s}[0], [x1], #4
    b L1Compute

    L1Compute:
    sxtl v16.8h, v27.8b
    sxtl v18.8h, v28.8b
    sxtl v17.4s, v16.4h
    sxtl v19.4s, v18.4h

    scvtf v0.4s, v17.4s
    scvtf v2.4s, v19.4s
    fmul v1.4s, v0.4s, v29.4s
    fmul v3.4s, v2.4s, v30.4s

    fmin v4.4s, v1.4s, v3.4s
    fmul v0.4s, v4.4s, v31.4s

    fcvtzs v5.4s, v0.4s
    sqxtn v6.4h, v5.4s
    sqxtn v7.8b, v6.8h
    st1 {v7.s}[0], [x0], #4

    subs x6, x6, #1
    bne L1Loop
End:

ret

#endif


