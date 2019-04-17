//
//  MNNGemmInt16to32_4x4_Unit.S
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNAsmGlobal.h"
#ifdef __aarch64__

.text
.align 5
asm_function MNNGemmInt16to32_4x4_Unit
//void MNNGemmInt16to32_4x4_Unit(int32_t* dst, const int16_t* src, const int16_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad)
//Auto:
//x0:dstOrigin, x1:src, x2: weight, x3:src_depth_quad
//x4: dst_step, x5:dst_depth_quad

mov x12, #4
mul x4, x12, x4


L12Dz:
    mov x6, x1
    subs x7, x3, #1
    mov x12, x0

    ld1 {v4.4s, v5.4s}, [x2], #32

    //A
    ld1 {v0.4s}, [x1], #16
    smull v6.4s, v4.4h,  v0.h[0]
    smull v7.4s, v4.4h,  v0.h[4]
    smlal2 v6.4s, v4.8h,  v0.h[1]
    smlal2 v7.4s, v4.8h,  v0.h[5]
    smlal v6.4s, v5.4h, v0.h[2]
    smlal v7.4s, v5.4h, v0.h[6]
    smlal2 v6.4s, v5.8h, v0.h[3]
    smlal2 v7.4s, v5.8h, v0.h[7]
    ld1 {v1.4s}, [x1], #16
    smull v16.4s, v4.4h,  v1.h[0]
    smull v17.4s, v4.4h,  v1.h[4]
    smlal2 v16.4s, v4.8h,  v1.h[1]
    smlal2 v17.4s, v4.8h,  v1.h[5]
    smlal v16.4s, v5.4h, v1.h[2]
    smlal v17.4s, v5.4h, v1.h[6]
    smlal2 v16.4s, v5.8h, v1.h[3]
    smlal2 v17.4s, v5.8h, v1.h[7]

    //B
    ld1 {v0.4s}, [x1], #16
    smull v18.4s, v4.4h,  v0.h[0]
    smull v19.4s, v4.4h,  v0.h[4]
    smlal2 v18.4s, v4.8h,  v0.h[1]
    smlal2 v19.4s, v4.8h,  v0.h[5]
    smlal v18.4s, v5.4h, v0.h[2]
    smlal v19.4s, v5.4h, v0.h[6]
    smlal2 v18.4s, v5.8h, v0.h[3]
    smlal2 v19.4s, v5.8h, v0.h[7]
    ld1 {v1.4s}, [x1], #16
    smull v20.4s, v4.4h,  v1.h[0]
    smull v21.4s, v4.4h,  v1.h[4]
    smlal2 v20.4s, v4.8h,  v1.h[1]
    smlal2 v21.4s, v4.8h,  v1.h[5]
    smlal v20.4s, v5.4h, v1.h[2]
    smlal v21.4s, v5.4h, v1.h[6]
    smlal2 v20.4s, v5.8h, v1.h[3]
    smlal2 v21.4s, v5.8h, v1.h[7]

    //C
    ld1 {v0.4s}, [x1], #16
    smull v22.4s, v4.4h,  v0.h[0]
    smull v23.4s, v4.4h,  v0.h[4]
    smlal2 v22.4s, v4.8h,  v0.h[1]
    smlal2 v23.4s, v4.8h,  v0.h[5]
    smlal v22.4s, v5.4h, v0.h[2]
    smlal v23.4s, v5.4h, v0.h[6]
    smlal2 v22.4s, v5.8h, v0.h[3]
    smlal2 v23.4s, v5.8h, v0.h[7]
    ld1 {v1.4s}, [x1], #16
    smull v2.4s, v4.4h,  v1.h[0]
    smull v3.4s, v4.4h,  v1.h[4]

    beq LoopZEnd

    LoopZ:
        smlal2 v2.4s, v4.8h,  v1.h[1]
        smlal2 v3.4s, v4.8h,  v1.h[5]
        smlal v2.4s, v5.4h, v1.h[2]
        smlal v3.4s, v5.4h, v1.h[6]
        smlal2 v2.4s, v5.8h, v1.h[3]
        smlal2 v3.4s, v5.8h, v1.h[7]
        
        ld1 {v4.4s, v5.4s}, [x2], #32

        //A
        ld1 {v0.4s}, [x1], #16
        smlal v6.4s, v4.4h,  v0.h[0]
        smlal v7.4s, v4.4h,  v0.h[4]
        smlal2 v6.4s, v4.8h,  v0.h[1]
        smlal2 v7.4s, v4.8h,  v0.h[5]
        smlal v6.4s, v5.4h, v0.h[2]
        smlal v7.4s, v5.4h, v0.h[6]

        ld1 {v1.4s}, [x1], #16
        smlal2 v6.4s, v5.8h, v0.h[3]
        smlal2 v7.4s, v5.8h, v0.h[7]
        smlal v16.4s, v4.4h,  v1.h[0]
        smlal v17.4s, v4.4h,  v1.h[4]
        smlal2 v16.4s, v4.8h,  v1.h[1]
        smlal2 v17.4s, v4.8h,  v1.h[5]
        smlal v16.4s, v5.4h, v1.h[2]
        smlal v17.4s, v5.4h, v1.h[6]
        ld1 {v0.4s}, [x1], #16
        smlal2 v16.4s, v5.8h, v1.h[3]
        smlal2 v17.4s, v5.8h, v1.h[7]

        //B
        smlal v18.4s, v4.4h,  v0.h[0]
        smlal v19.4s, v4.4h,  v0.h[4]
        smlal2 v18.4s, v4.8h,  v0.h[1]
        smlal2 v19.4s, v4.8h,  v0.h[5]
        smlal v18.4s, v5.4h, v0.h[2]
        ld1 {v1.4s}, [x1], #16
        smlal v19.4s, v5.4h, v0.h[6]
        smlal2 v18.4s, v5.8h, v0.h[3]
        smlal2 v19.4s, v5.8h, v0.h[7]
        smlal v20.4s, v4.4h,  v1.h[0]
        smlal v21.4s, v4.4h,  v1.h[4]
        smlal2 v20.4s, v4.8h,  v1.h[1]
        smlal2 v21.4s, v4.8h,  v1.h[5]
        smlal v20.4s, v5.4h, v1.h[2]
        smlal v21.4s, v5.4h, v1.h[6]
        ld1 {v0.4s}, [x1], #16
        smlal2 v20.4s, v5.8h, v1.h[3]
        smlal2 v21.4s, v5.8h, v1.h[7]

        //C
        smlal v22.4s, v4.4h,  v0.h[0]
        smlal v23.4s, v4.4h,  v0.h[4]
        smlal2 v22.4s, v4.8h,  v0.h[1]
        smlal2 v23.4s, v4.8h,  v0.h[5]
        ld1 {v1.4s}, [x1], #16
        smlal v22.4s, v5.4h, v0.h[2]
        smlal v23.4s, v5.4h, v0.h[6]
        smlal2 v22.4s, v5.8h, v0.h[3]
        smlal2 v23.4s, v5.8h, v0.h[7]
        smlal v2.4s, v4.4h,  v1.h[0]
        smlal v3.4s, v4.4h,  v1.h[4]

        subs x7, x7, #1
        bne LoopZ

    LoopZEnd:

    smlal2 v2.4s, v4.8h,  v1.h[1]
    smlal2 v3.4s, v4.8h,  v1.h[5]
    smlal v2.4s, v5.4h, v1.h[2]
    smlal v3.4s, v5.4h, v1.h[6]
    smlal2 v2.4s, v5.8h, v1.h[3]
    smlal2 v3.4s, v5.8h, v1.h[7]

    st1 {v6.4s, v7.4s},   [x0], #32
    st1 {v16.4s, v17.4s}, [x0], #32
    st1 {v18.4s, v19.4s}, [x0], #32
    st1 {v20.4s, v21.4s}, [x0], #32
    st1 {v22.4s, v23.4s}, [x0], #32
    st1 {v2.4s, v3.4s}, [x0], #32
    mov x1, x6

    subs x5, x5, #1
    add x0, x12, x4
    bne L12Dz


ret


#endif
