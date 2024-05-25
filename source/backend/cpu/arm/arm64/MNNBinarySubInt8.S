//
//  MNNBinarySubInt8.S
//
//  Created by MNN on 2019/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/*
struct QuanPrePostParameters{
    float* inputScale;
    float* outputScale;
    ssize_t* inputZeroPoint;
    ssize_t* outputZeroPoint;
    ssize_t minValue;
    ssize_t maxValue;
};
 */

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNBinarySubInt8
// MNNBinarySubInt8(int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, 
// float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast);
// Auto load:
// x0: dst, x1:src0, x2:src1, x3:inputScalesInt32, x4:inputScalesFp32, x5: params, x6: size, x7: needBroadcast
// w8: inputZeroPoint0, w9: inputZeroPoint1, w10: outputZeroPoint
stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

cmp x6, #0
beq End

ldr w3, [x4]
ldr w10, [x4, #8]
ldr w4, [x4, #4]
mov v0.s[0], w3
mov v0.s[1], w4
mov v0.s[2], w10

ldr x8, [x5, #16]
ldr x9, [x8, #8]   // input1 zeroPoint
ldr x8, [x8, #0]   // input0 zeroPoint
ldr x10, [x5, #24]
ldr x10, [x10]     // output zeroPoint
ldr x11, [x5, #32] // w11:minValue
ldr x12, [x5, #40] // w12:maxValue

cmp x6, #8
bge L8Loop
cmp x6, #4
bge L4
blt L1

L8Loop:
    cmp x7, #0
    beq L8NeedBroadcast0
    cmp x7, #1
    beq L8NeedBroadcast1

    L8NotNeedBroadcast:
    ld1 {v3.16b, v4.16b}, [x1], #32 // input00, input01
    ld1 {v5.16b, v6.16b}, [x2], #32 // input10, input11
    b L8Compute

    L8NeedBroadcast0:
    ld1r {v3.16b}, [x1]
    ld1r {v4.16b}, [x1]
    ld1 {v5.16b, v6.16b}, [x2], #32
    b L8Compute

    L8NeedBroadcast1:
    ld1 {v3.16b, v4.16b}, [x1], #32
    ld1r {v5.16b}, [x2]
    ld1r {v6.16b}, [x2]
    b L8Compute

    L8Compute:
    sxtl v7.8h, v3.8b
    sxtl2 v8.8h, v3.16b
    sxtl v9.8h, v4.8b     
    sxtl2 v10.8h, v4.16b   

    sxtl v11.8h, v5.8b
    sxtl2 v12.8h, v5.16b
    sxtl v13.8h, v6.8b   
    sxtl2 v14.8h, v6.16b
    
    INPUT0_SUB_ZERO:
    cmp w8, #0
    beq INPUT1_SUB_ZERO
    dup v2.8b, w8
    ssubw v7.8h, v7.8h, v2.8b
    ssubw v8.8h, v8.8h, v2.8b
    ssubw v9.8h, v9.8h, v2.8b
    ssubw v10.8h, v10.8h, v2.8b

    INPUT1_SUB_ZERO:
    cmp w9, #0
    beq L8SXTL_S32
    dup v1.8b, w9
    ssubw v11.8h, v11.8h, v1.8b
    ssubw v12.8h, v12.8h, v1.8b
    ssubw v13.8h, v13.8h, v1.8b
    ssubw v14.8h, v14.8h, v1.8b
    

    L8SXTL_S32:
    sxtl  v15.4s, v7.4h
    sxtl2 v16.4s, v7.8h
    sxtl  v17.4s, v8.4h
    sxtl2 v18.4s, v8.8h
    sxtl  v19.4s, v9.4h
    sxtl2 v20.4s, v9.8h
    sxtl  v21.4s, v10.4h
    sxtl2 v22.4s, v10.8h

    sxtl  v23.4s,v11.4h      
    sxtl2 v24.4s, v11.8h
    sxtl  v25.4s, v12.4h
    sxtl2 v26.4s, v12.8h
    sxtl  v27.4s, v13.4h
    sxtl2 v28.4s, v13.8h
    sxtl  v29.4s, v14.4h
    sxtl2 v30.4s, v14.8h

    scvtf v15.4s, v15.4s 
    scvtf v16.4s, v16.4s 
    scvtf v17.4s, v17.4s 
    scvtf v18.4s, v18.4s 
    scvtf v19.4s, v19.4s 
    scvtf v20.4s, v20.4s 
    scvtf v21.4s, v21.4s 
    scvtf v22.4s, v22.4s 

    scvtf v23.4s, v23.4s    
    scvtf v24.4s, v24.4s
    scvtf v25.4s, v25.4s
    scvtf v26.4s, v26.4s
    scvtf v27.4s, v27.4s
    scvtf v28.4s, v28.4s
    scvtf v29.4s, v29.4s
    scvtf v30.4s, v30.4s

    fmul v15.4s, v15.4s, v0.s[0]
    fmul v16.4s, v16.4s, v0.s[0]
    fmul v17.4s, v17.4s, v0.s[0]
    fmul v18.4s, v18.4s, v0.s[0]
    fmul v19.4s, v19.4s, v0.s[0]
    fmul v20.4s, v20.4s, v0.s[0]
    fmul v21.4s, v21.4s, v0.s[0]
    fmul v22.4s, v22.4s, v0.s[0]

    fmul v23.4s, v23.4s, v0.s[1]
    fmul v24.4s, v24.4s, v0.s[1]
    fmul v25.4s, v25.4s, v0.s[1]
    fmul v26.4s, v26.4s, v0.s[1]
    fmul v27.4s, v27.4s, v0.s[1]
    fmul v28.4s, v28.4s, v0.s[1]
    fmul v29.4s, v29.4s, v0.s[1]
    fmul v30.4s, v30.4s, v0.s[1]
    dup v11.16b, w11
    dup v12.16b, w12

    fsub v15.4s, v15.4s, v23.4s
    fsub v16.4s, v16.4s, v24.4s 
    fsub v17.4s, v17.4s, v25.4s 
    fsub v18.4s, v18.4s, v26.4s 
    fsub v19.4s, v19.4s, v27.4s 
    fsub v20.4s, v20.4s, v28.4s 
    fsub v21.4s, v21.4s, v29.4s 
    fsub v22.4s, v22.4s, v30.4s 

    fmul v15.4s, v15.4s, v0.s[2]
    fmul v16.4s, v16.4s, v0.s[2]
    fmul v17.4s, v17.4s, v0.s[2]
    fmul v18.4s, v18.4s, v0.s[2]
    fmul v19.4s, v19.4s, v0.s[2]
    fmul v20.4s, v20.4s, v0.s[2]
    fmul v21.4s, v21.4s, v0.s[2]
    fmul v22.4s, v22.4s, v0.s[2]

    fcvtas v15.4s, v15.4s
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s
    fcvtas v21.4s, v21.4s
    fcvtas v22.4s, v22.4s

    sqxtn v1.4h, v15.4s
    sqxtn2 v1.8h, v16.4s
    sqxtn v2.4h, v17.4s
    sqxtn2 v2.8h, v18.4s
    sqxtn v3.4h, v19.4s
    sqxtn2 v3.8h, v20.4s
    sqxtn v4.4h, v21.4s
    sqxtn2 v4.8h, v22.4s

    cmp w10, #0
    beq SQXTN_S8
    dup v14.8b, w10
    saddw v1.8h, v1.8h, v14.8b
    saddw v2.8h, v2.8h, v14.8b
    saddw v3.8h, v3.8h, v14.8b
    saddw v4.8h, v4.8h, v14.8b

    SQXTN_S8:
    sqxtn v5.8b, v1.8h
    sqxtn2 v5.16b, v2.8h
    sqxtn v6.8b, v3.8h
    sqxtn2 v6.16b, v4.8h

    smax v5.16b, v5.16b, v11.16b
    smax v6.16b, v6.16b, v11.16b
    smin v5.16b, v5.16b, v12.16b
    smin v6.16b, v6.16b, v12.16b

    st1 {v5.16b, v6.16b}, [x0], #32

    sub x6, x6, #8
    cmp x6, #8
    bge L8Loop
    cmp x6, #4
    blt L1

L4:
    dup v30.16b, w11
    dup v31.16b, w12
L4Loop:
    cmp x7, #0
    beq L4NeedBroadcast0
    cmp x7, #1
    beq L4NeedBroadcast1

    L4NotNeedBroadcast:
    ld1 {v3.16b}, [x1], #16 // input00, input01
    ld1 {v5.16b}, [x2], #16 // input10, input11
    b L4Compute

    L4NeedBroadcast0:
    ld1r {v3.16b}, [x1]
    ld1 {v5.16b}, [x2], #16
    b L4Compute

    L4NeedBroadcast1:
    ld1 {v3.16b}, [x1], #16
    ld1r {v5.16b}, [x2]
    b L4Compute

    L4Compute:
    sxtl v7.8h, v3.8b
    sxtl2 v8.8h, v3.16b  
    sxtl v11.8h, v5.8b
    sxtl2 v12.8h, v5.16b

    L4_INPUT0_SUB_ZERO:
    cmp w8, #0
    beq L4_INPUT1_SUB_ZERO
    dup v2.8b, w8
    ssubw v7.8h, v7.8h, v2.8b
    ssubw v8.8h, v8.8h, v2.8b

    L4_INPUT1_SUB_ZERO:
    cmp w9, #0
    beq L4SXTL_S32
    dup v1.8b, w9
    ssubw v11.8h, v11.8h, v1.8b
    ssubw v12.8h, v12.8h, v1.8b

    L4SXTL_S32:
    sxtl  v15.4s, v7.4h
    sxtl2 v16.4s, v7.8h
    sxtl  v17.4s, v8.4h
    sxtl2 v18.4s, v8.8h

    sxtl  v23.4s,v11.4h      
    sxtl2 v24.4s, v11.8h
    sxtl  v25.4s, v12.4h
    sxtl2 v26.4s, v12.8h

    scvtf v15.4s, v15.4s 
    scvtf v16.4s, v16.4s 
    scvtf v17.4s, v17.4s 
    scvtf v18.4s, v18.4s

    scvtf v23.4s, v23.4s    
    scvtf v24.4s, v24.4s
    scvtf v25.4s, v25.4s
    scvtf v26.4s, v26.4s

    fmul v15.4s, v15.4s, v0.s[0]
    fmul v16.4s, v16.4s, v0.s[0]
    fmul v17.4s, v17.4s, v0.s[0]
    fmul v18.4s, v18.4s, v0.s[0]

    fmul v23.4s, v23.4s, v0.s[1]
    fmul v24.4s, v24.4s, v0.s[1]
    fmul v25.4s, v25.4s, v0.s[1]
    fmul v26.4s, v26.4s, v0.s[1]

    fsub v15.4s, v15.4s, v23.4s
    fsub v16.4s, v16.4s, v24.4s 
    fsub v17.4s, v17.4s, v25.4s 
    fsub v18.4s, v18.4s, v26.4s

    fmul v15.4s, v15.4s, v0.s[2]
    fmul v16.4s, v16.4s, v0.s[2]
    fmul v17.4s, v17.4s, v0.s[2]
    fmul v18.4s, v18.4s, v0.s[2]

    fcvtas v15.4s, v15.4s
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s

    sqxtn v1.4h, v15.4s
    sqxtn2 v1.8h, v16.4s
    sqxtn v2.4h, v17.4s
    sqxtn2 v2.8h, v18.4s

    cmp w10, #0
    beq L4_SQXTN_S8
    dup v14.8b, w10
    saddw v1.8h, v1.8h, v14.8b
    saddw v2.8h, v2.8h, v14.8b

    L4_SQXTN_S8:
    sqxtn v5.8b, v1.8h
    sqxtn2 v5.16b, v2.8h
    smax v5.16b, v5.16b, v30.16b
    smin v5.16b, v5.16b, v31.16b
    st1 {v5.16b}, [x0], #16
    sub x6, x6, #4
    cmp x6, #4
    bge L4Loop

L1:
cmp x6, #0
beq End
    dup v30.16b, w11
    dup v31.16b, w12
L1Loop:
    cmp x7, #0
    beq L1NeedBroadcast0
    cmp x7, #1
    beq L1NeedBroadcast1

    L1NotNeedBroadcast:
    ld1 {v3.s}[0], [x1], #4 // input00, input01
    ld1 {v5.s}[0], [x2], #4 // input10, input11
    b L1Compute

    L1NeedBroadcast0:
    ld1 {v3.b}[0], [x1]
    dup v3.8b, v3.b[0]
    ld1 {v5.s}[0], [x2], #4
    b L1Compute

    L1NeedBroadcast1:
    ld1 {v3.s}[0], [x1], #4
    ld1r {v5.8b}, [x2]
    b L1Compute

    L1Compute:
    sxtl v7.8h, v3.8b
    sxtl v11.8h, v5.8b

    L1_INPUT0_SUB_ZERO:
    cmp w8, #0
    beq L1_INPUT1_SUB_ZERO
    dup v2.8b, w8
    ssubw v7.8h, v7.8h, v2.8b
    L1_INPUT1_SUB_ZERO:
    cmp w9, #0
    beq L1SXTL_S32
    dup v1.8b, w9
    ssubw v11.8h, v11.8h, v1.8b

    L1SXTL_S32:
    sxtl  v15.4s, v7.4h
    sxtl  v23.4s, v11.4h

    scvtf v15.4s, v15.4s
    scvtf v23.4s, v23.4s

    fmul v15.4s, v15.4s, v0.s[0]
    fmul v23.4s, v23.4s, v0.s[1]

    fsub v15.4s, v15.4s, v23.4s

    fmul v15.4s, v15.4s, v0.s[2]

    fcvtas v15.4s, v15.4s

    sqxtn v1.4h, v15.4s

    cmp w10, #0
    beq L1_SQXTN_S8
    dup v14.8b, w10
    saddw v1.8h, v1.8h, v14.8b

    L1_SQXTN_S8:
    sqxtn v5.8b, v1.8h
    smax v5.8b, v5.8b, v30.8b
    smin v5.8b, v5.8b, v31.8b
    st1 {v5.s}[0], [x0], #4

    subs x6, x6, #1
    bne L1Loop
End:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret

#endif
