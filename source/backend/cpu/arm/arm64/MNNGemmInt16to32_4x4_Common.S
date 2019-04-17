//
//  MNNGemmInt16to32_4x4_Common.S
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNAsmGlobal.h"
#ifdef __aarch64__

.text
.align 5
asm_function MNNGemmInt16to32_4x4_Common
//void MNNGemmInt16to32_4x4_Common(int32_t* dst, const int16_t* src, const int16_t* weight, size_t src_depth_quad, size_t width, size_t dst_step, size_t dst_depth_quad)
//Auto:
//x0:dstOrigin, x1:src, x2: weight, x3:src_depth_quad
//x4: width, x5: dst_step, x6:dst_depth_quad

//step multi by sizeof(int32_t)
mov x12, #4
mul x5, x12, x5

//x11: weight_dz_step
mov x12, #32//16*sizeof(int16_t)
mul x11, x12, x3

//x12:src_step
mov x12, #8//4*sizeof(int16_t)
mul x12, x4, x12

LoopDz:
    mov x13, x0
    mov x14, x1
    mov x15, x2
    mov x7, x4

    cmp x4, #8
    blt L4
    
    L8Loop:
        mov x8, x1
        mov x9, x3
        mov x10, x2
    
        movi v16.4s, #0
        movi v17.4s, #0
        movi v18.4s, #0
        movi v19.4s, #0
        movi v20.4s, #0
        movi v21.4s, #0
        movi v22.4s, #0
        movi v23.4s, #0
    
        LoopZL8:
            ld1 {v2.4s, v3.4s}, [x2], #32
    
            //A
            ld1 {v0.4s}, [x1], #16
            smlal v16.4s, v2.4h, v0.h[0]
            smlal v17.4s, v2.4h, v0.h[4]
            smlal2 v16.4s, v2.8h, v0.h[1]
            smlal2 v17.4s, v2.8h, v0.h[5]
            smlal v16.4s, v3.4h, v0.h[2]
            smlal v17.4s, v3.4h, v0.h[6]
            smlal2 v16.4s, v3.8h, v0.h[3]
            smlal2 v17.4s, v3.8h, v0.h[7]
    
            //B
            ld1 {v1.4s}, [x1], #16
            smlal v18.4s, v2.4h, v1.h[0]
            smlal v19.4s, v2.4h, v1.h[4]
            smlal2 v18.4s, v2.8h, v1.h[1]
            smlal2 v19.4s, v2.8h, v1.h[5]
            smlal v18.4s, v3.4h, v1.h[2]
            smlal v19.4s, v3.4h, v1.h[6]
            smlal2 v18.4s, v3.8h, v1.h[3]
            smlal2 v19.4s, v3.8h, v1.h[7]
    
            //C
            ld1 {v0.4s}, [x1], #16
            smlal v20.4s, v2.4h, v0.h[0]
            smlal v21.4s, v2.4h, v0.h[4]
            smlal2 v20.4s, v2.8h, v0.h[1]
            smlal2 v21.4s, v2.8h, v0.h[5]
            smlal v20.4s, v3.4h, v0.h[2]
            smlal v21.4s, v3.4h, v0.h[6]
            smlal2 v20.4s, v3.8h, v0.h[3]
            smlal2 v21.4s, v3.8h, v0.h[7]
    
            //D
            ld1 {v1.4s}, [x1], #16
            smlal v22.4s, v2.4h, v1.h[0]
            smlal v23.4s, v2.4h, v1.h[4]
            smlal2 v22.4s, v2.8h, v1.h[1]
            smlal2 v23.4s, v2.8h, v1.h[5]
            smlal v22.4s, v3.4h, v1.h[2]
            smlal v23.4s, v3.4h, v1.h[6]
            smlal2 v22.4s, v3.8h, v1.h[3]
            smlal2 v23.4s, v3.8h, v1.h[7]

            sub x1, x1, #64
            add x1, x1, x12
    
            subs x3, x3, #1
            bne LoopZL8
        st1 {v16.4s, v17.4s}, [x0], #32
        st1 {v18.4s, v19.4s}, [x0], #32
        st1 {v20.4s, v21.4s}, [x0], #32
        st1 {v22.4s, v23.4s}, [x0], #32
        add x1, x8, #64 // 8*4*sizeof(int16_t)
        mov x3, x9
        mov x2, x10

        sub x4, x4, #8
        cmp x4, #8
        bge L8Loop
    L4:
        cmp x4, #4
        blt L1
        movi v20.4s, #0
        movi v21.4s, #0
        movi v22.4s, #0
        movi v23.4s, #0
        mov x8, x1
        mov x9, x3
        mov x10, x2


        LoopZL4:
            ld1 {v2.4s, v3.4s}, [x2], #32

            ld1 {v0.4s}, [x1], #16
            smlal v20.4s, v2.4h, v0.h[0]
            smlal v21.4s, v2.4h, v0.h[4]
            smlal2 v20.4s, v2.8h, v0.h[1]
            smlal2 v21.4s, v2.8h, v0.h[5]
            ld1 {v1.4s}, [x1], #16
            smlal v20.4s, v3.4h, v0.h[2]
            smlal v21.4s, v3.4h, v0.h[6]
            smlal2 v20.4s, v3.8h, v0.h[3]
            smlal2 v21.4s, v3.8h, v0.h[7]

            smlal v22.4s, v2.4h, v1.h[0]
            smlal v23.4s, v2.4h, v1.h[4]
            smlal2 v22.4s, v2.8h, v1.h[1]
            smlal2 v23.4s, v2.8h, v1.h[5]
            smlal v22.4s, v3.4h, v1.h[2]
            smlal v23.4s, v3.4h, v1.h[6]
            smlal2 v22.4s, v3.8h, v1.h[3]
            smlal2 v23.4s, v3.8h, v1.h[7]

            subs x3, x3, #1
            sub x1, x1, #32
            add x1, x1, x12
            bne LoopZL4

        sub x4, x4, #4
        add x1, x8, #32
        mov x3, x9
        mov x2, x10
        st1 {v20.4s, v21.4s}, [x0], #32
        st1 {v22.4s, v23.4s}, [x0], #32


    L1:
    cmp x4, #0
    beq EndL1

    LoopL1:
        mov x10, x2
        mov x8, x1
        mov x9, x3

        movi v23.4s, #0
        movi v22.4s, #0

        LoopZL1:
            ld1 {v2.4s, v3.4s}, [x2], #32
            ld1 {v0.4h}, [x1], x12

            smlal v22.4s, v2.4h, v0.h[0]
            smlal2 v23.4s, v2.8h, v0.h[1]
            smlal v22.4s, v3.4h, v0.h[2]
            smlal2 v23.4s, v3.8h, v0.h[3]

            subs x3, x3, #1
            bne LoopZL1

        subs x4, x4, #1
        add v23.4s, v23.4s, v22.4s
        mov x2, x10
        mov x3, x9
        st1 {v23.4s}, [x0], #16
        add x1, x8, #8
        bne LoopL1

    EndL1:

    mov x0, x13
    mov x1, x14
    mov x2, x15
    mov x4, x7
    subs x6, x6, #1

    add x2, x2, x11
    add x0, x0, x5
    bne LoopDz

ret

#endif
