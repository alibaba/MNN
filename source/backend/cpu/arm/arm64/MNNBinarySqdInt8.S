//
//  MNNBinarySqdInt8.S
//  MNN
//
//  Created by MNN on 2019/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNBinarySqdInt8
// MNNBinarySqdInt8(int8_t* dst, const int8_t* src0, const int8_t* src1, ssize_t* quantScalesInt32, 
// float* quantScalesFp32, const int8_t* inputOffset0, const int8_t* inputOffset1, const int8_t* outputOffset,,
// const size_t size, size_t needBroadcast)
// Auto load:
// x0: dst, x1:src0, x2:src1, x3:quantScalesInt32, x4:quantScalesFp32, x5: offset0, x6: offset1, x7: outputoffset
// Load from sp:
// x8: size, x9: needBroadcast

ldr x8, [sp, #0]
ldr x9, [sp, #8]

stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

cmp x8, #0
beq End

ldr w3, [x4]
ldr w10, [x4, #8]
ldr w4, [x4, #4]
mov v0.s[0], w3
mov v0.s[1], w4
mov v0.s[2], w10

cmp x8, #8
bge L8Loop
cmp x8, #4
bge L4Loop
blt L1

L8Loop:
    cmp x9, #0
    beq L8NeedBroadcast0
    cmp x9, #1
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
    ld1r {v2.8b}, [x5]
    
    ssubw v7.8h, v7.8h, v2.8b
    ssubw v8.8h, v8.8h, v2.8b
    ssubw v9.8h, v9.8h, v2.8b
    ssubw v10.8h, v10.8h, v2.8b

    INPUT1_SUB_ZERO:
    ld1r {v1.8b}, [x6]
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

    fsub v15.4s, v15.4s, v23.4s
    fsub v16.4s, v16.4s, v24.4s 
    fsub v17.4s, v17.4s, v25.4s 
    fsub v18.4s, v18.4s, v26.4s 
    fsub v19.4s, v19.4s, v27.4s 
    fsub v20.4s, v20.4s, v28.4s 
    fsub v21.4s, v21.4s, v29.4s 
    fsub v22.4s, v22.4s, v30.4s

    fmul v15.4s, v15.4s, v15.4s
    fmul v16.4s, v16.4s, v16.4s 
    fmul v17.4s, v17.4s, v17.4s 
    fmul v18.4s, v18.4s, v18.4s 
    fmul v19.4s, v19.4s, v19.4s 
    fmul v20.4s, v20.4s, v20.4s 
    fmul v21.4s, v21.4s, v21.4s 
    fmul v22.4s, v22.4s, v22.4s

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

    ld1r {v14.8b}, [x7]
    saddw v1.8h, v1.8h, v14.8b
    saddw v2.8h, v2.8h, v14.8b
    saddw v3.8h, v3.8h, v14.8b
    saddw v4.8h, v4.8h, v14.8b

    SQXTN_S8:
    sqxtn v5.8b, v1.8h
    sqxtn2 v5.16b, v2.8h
    sqxtn v6.8b, v3.8h
    sqxtn2 v6.16b, v4.8h

    st1 {v5.16b, v6.16b}, [x0], #32

    sub x8, x8, #8
    cmp x8, #8
    bge L8Loop
    cmp x8, #0
    ble End
    cmp x8, #4
    blt L1Loop

L4Loop:
    cmp x9, #0
    beq L4NeedBroadcast0
    cmp x9, #1
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
    ld1r {v2.8b}, [x5]
    ssubw v7.8h, v7.8h, v2.8b
    ssubw v8.8h, v8.8h, v2.8b

    L4_INPUT1_SUB_ZERO:
    ld1r {v1.8b}, [x6]
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

    fmul v15.4s, v15.4s, v15.4s
    fmul v16.4s, v16.4s, v16.4s 
    fmul v17.4s, v17.4s, v17.4s 
    fmul v18.4s, v18.4s, v18.4s

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

    ld1r {v14.8b}, [x7]
    saddw v1.8h, v1.8h, v14.8b
    saddw v2.8h, v2.8h, v14.8b

    L4_SQXTN_S8:
    sqxtn v5.8b, v1.8h
    sqxtn2 v5.16b, v2.8h
    st1 {v5.16b}, [x0], #16
    sub x8, x8, #4
    cmp x8, #4
    bge L4Loop

L1:
cmp x8, #0
beq End

L1Loop:
    cmp x9, #0
    beq L1NeedBroadcast0
    cmp x9, #1
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
    ld1r {v2.8b}, [x5]
    ssubw v7.8h, v7.8h, v2.8b
    L1_INPUT1_SUB_ZERO:
    ld1r {v1.8b}, [x6]
    ssubw v11.8h, v11.8h, v1.8b

    L1SXTL_S32:
    sxtl  v15.4s, v7.4h
    sxtl  v23.4s, v11.4h

    scvtf v15.4s, v15.4s
    scvtf v23.4s, v23.4s

    fmul v15.4s, v15.4s, v0.s[0]
    fmul v23.4s, v23.4s, v0.s[1]
    fsub v15.4s, v15.4s, v23.4s
    fmul v15.4s, v15.4s, v15.4s

    fmul v15.4s, v15.4s, v0.s[2]

    fcvtas v15.4s, v15.4s

    sqxtn v1.4h, v15.4s

    ld1r {v14.8b}, [x7]
    saddw v1.8h, v1.8h, v14.8b

    L1_SQXTN_S8:
    sqxtn v5.8b, v1.8h
    st1 {v5.s}[0], [x0], #4

    subs x8, x8, #1
    bne L1Loop
End:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret

#endif
