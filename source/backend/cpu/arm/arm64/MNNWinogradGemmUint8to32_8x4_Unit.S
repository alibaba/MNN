//
//  MNNWinogradGemmUint8to32_8x4_Unit.S
//  MNN
//
//  Created by MNN on 2018/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNWinogradGemmUint8to32_8x4_Unit
//void MNNGemmUint8to32_8x4_Unit(int32_t* dst, const uint8_t* src, const uint8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const int32_t* inputOffset);
//Auto: x0: dst, x1: src, x2:weight, x3: src_depth_quad
//x4: dst_step, x5: dst_depth_quad, x6: inputOffset

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

ldr w10, [x6, #0]
ldr w11, [x6, #4]
ldr w12, [x6, #8]
ldr w13, [x6, #12]

L6LoopDz:
    mov x6, x1
    subs x7, x3, #1

/*
v0  s00 s10  -  v4  w00 w20  -  v8  w20 w00
v1  s01 s11  -  v5  w01 w21  -  v9  w21 w01
v0  s20 s30  -  v6  w10 w30  - v10  w30 w10
v1  s21 s31  -  v7  w11 w31  - v11  w31 w11
 */
    ld4 {v4.2s, v5.2s, v6.2s, v7.2s}, [x2], #32

    rev64 v8.2s, v4.2s
    rev64 v9.2s, v5.2s
    ld2 {v0.2s, v1.2s}, [x1], #16
    rev64 v10.2s, v6.2s
    rev64 v11.2s, v7.2s
    

    uaddl v12.8h, v4.8b, v0.8b
    uaddl v14.8h, v5.8b, v1.8b
    uaddl v13.8h, v6.8b, v0.8b

    umull v16.4s, v12.4h, v14.4h
    uaddl v15.8h, v7.8b, v1.8b
    umull2 v22.4s, v12.8h, v14.8h
    umull v17.4s, v13.4h, v15.4h
    uaddl v12.8h, v8.8b, v0.8b
    umull2 v23.4s, v13.8h, v15.8h

    uaddl v14.8h, v9.8b, v1.8b
    uaddl v13.8h, v10.8b, v0.8b

    umull v18.4s, v12.4h, v14.4h
    uaddl v15.8h, v11.8b, v1.8b
    umull2 v20.4s, v12.8h, v14.8h
    ld2 {v0.2s, v1.2s}, [x1], #16
    umull v19.4s, v13.4h, v15.4h
    uaddl v12.8h, v4.8b, v0.8b
    umull2 v21.4s, v13.8h, v15.8h


    uaddl v14.8h, v5.8b, v1.8b
    uaddl v13.8h, v6.8b, v0.8b
    umull v24.4s, v12.4h, v14.4h
    uaddl v15.8h, v7.8b, v1.8b

    umull2 v30.4s, v12.8h, v14.8h
    umull v25.4s, v13.4h, v15.4h
    uaddl v12.8h, v8.8b, v0.8b
    umull2 v31.4s, v13.8h, v15.8h

    uaddl v14.8h, v9.8b, v1.8b
    uaddl v13.8h, v10.8b, v0.8b
    umull v26.4s, v12.4h, v14.4h
    uaddl v15.8h, v11.8b, v1.8b

    umull2 v28.4s, v12.8h, v14.8h
    umull v27.4s, v13.4h, v15.4h
    umull2 v29.4s, v13.8h, v15.8h

    beq L6LoopSzEnd

    subs x7, x7, #1
    ld4 {v4.2s, v5.2s, v6.2s, v7.2s}, [x2], #32

    rev64 v8.2s, v4.2s
    rev64 v9.2s, v5.2s
    ld2 {v0.2s, v1.2s}, [x1], #16
    rev64 v10.2s, v6.2s
    rev64 v11.2s, v7.2s
    
    uaddl v12.8h, v4.8b, v0.8b
    uaddl v14.8h, v5.8b, v1.8b
    uaddl v13.8h, v6.8b, v0.8b

    umlal v16.4s, v12.4h, v14.4h
    uaddl v15.8h, v7.8b, v1.8b
    umlal2 v22.4s, v12.8h, v14.8h
    umlal v17.4s, v13.4h, v15.4h
    uaddl v12.8h, v8.8b, v0.8b
    umlal2 v23.4s, v13.8h, v15.8h

    uaddl v14.8h, v9.8b, v1.8b
    uaddl v13.8h, v10.8b, v0.8b

    umlal v18.4s, v12.4h, v14.4h
    uaddl v15.8h, v11.8b, v1.8b
    umlal2 v20.4s, v12.8h, v14.8h
    ld2 {v0.2s, v1.2s}, [x1], #16

    beq L6LoopSzEndRemain

    L6LoopSz:
        umlal v19.4s, v13.4h, v15.4h
        uaddl v12.8h, v4.8b, v0.8b
        umlal2 v21.4s, v13.8h, v15.8h


        uaddl v14.8h, v5.8b, v1.8b
        uaddl v13.8h, v6.8b, v0.8b
        umlal v24.4s, v12.4h, v14.4h
        uaddl v15.8h, v7.8b, v1.8b

        umlal2 v30.4s, v12.8h, v14.8h
        umlal v25.4s, v13.4h, v15.4h
        uaddl v12.8h, v8.8b, v0.8b
        umlal2 v31.4s, v13.8h, v15.8h

        ld4 {v4.2s, v5.2s, v6.2s, v7.2s}, [x2], #32
        uaddl v14.8h, v9.8b, v1.8b
        rev64 v8.2s, v4.2s
        uaddl v13.8h, v10.8b, v0.8b
        rev64 v9.2s, v5.2s
        umlal v26.4s, v12.4h, v14.4h
        uaddl v15.8h, v11.8b, v1.8b

        rev64 v10.2s, v6.2s
        umlal2 v28.4s, v12.8h, v14.8h
        ld2 {v0.2s, v1.2s}, [x1], #16
        umlal v27.4s, v13.4h, v15.4h

        rev64 v11.2s, v7.2s
        
        uaddl v12.8h, v4.8b, v0.8b
        umlal2 v29.4s, v13.8h, v15.8h
        uaddl v14.8h, v5.8b, v1.8b
        uaddl v13.8h, v6.8b, v0.8b

        umlal v16.4s, v12.4h, v14.4h
        uaddl v15.8h, v7.8b, v1.8b
        umlal2 v22.4s, v12.8h, v14.8h
        umlal v17.4s, v13.4h, v15.4h
        uaddl v12.8h, v8.8b, v0.8b
        umlal2 v23.4s, v13.8h, v15.8h

        uaddl v14.8h, v9.8b, v1.8b
        uaddl v13.8h, v10.8b, v0.8b

        umlal v18.4s, v12.4h, v14.4h
        uaddl v15.8h, v11.8b, v1.8b
        umlal2 v20.4s, v12.8h, v14.8h
        ld2 {v0.2s, v1.2s}, [x1], #16

        subs x7, x7, #1

        bne L6LoopSz

    L6LoopSzEndRemain:
    umlal v19.4s, v13.4h, v15.4h
    uaddl v12.8h, v4.8b, v0.8b
    umlal2 v21.4s, v13.8h, v15.8h


    uaddl v14.8h, v5.8b, v1.8b
    uaddl v13.8h, v6.8b, v0.8b
    umlal v24.4s, v12.4h, v14.4h
    uaddl v15.8h, v7.8b, v1.8b

    umlal2 v30.4s, v12.8h, v14.8h
    umlal v25.4s, v13.4h, v15.4h
    uaddl v12.8h, v8.8b, v0.8b
    umlal2 v31.4s, v13.8h, v15.8h

    uaddl v14.8h, v9.8b, v1.8b
    uaddl v13.8h, v10.8b, v0.8b
    umlal v26.4s, v12.4h, v14.4h
    uaddl v15.8h, v11.8b, v1.8b

    umlal2 v28.4s, v12.8h, v14.8h
    umlal v27.4s, v13.4h, v15.4h
    umlal2 v29.4s, v13.8h, v15.8h

    L6LoopSzEnd:

    dup v2.4s, w10
    dup v3.4s, w11
    dup v4.4s, w12
    dup v5.4s, w13
    addp v16.4s, v16.4s, v17.4s
    addp v18.4s, v18.4s, v19.4s
    addp v20.4s, v20.4s, v21.4s
    addp v22.4s, v22.4s, v23.4s
    addp v24.4s, v24.4s, v25.4s
    addp v26.4s, v26.4s, v27.4s
    addp v28.4s, v28.4s, v29.4s
    addp v30.4s, v30.4s, v31.4s

    addp v10.4s, v16.4s, v18.4s
    addp v11.4s, v20.4s, v22.4s
    addp v12.4s, v24.4s, v26.4s
    addp v13.4s, v28.4s, v30.4s

    sqsub v10.4s, v10.4s, v2.4s
    sqsub v11.4s, v11.4s, v3.4s
    sqsub v12.4s, v12.4s, v4.4s
    st1 {v10.4s, v11.4s}, [x0], #32
    sqsub v13.4s, v13.4s, v5.4s

    subs x5, x5, #1
    st1 {v12.4s, v13.4s}, [x0], #32
    mov x1, x6
    bne L6LoopDz

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
