//
//  MNNFloat16C8ToC4AddBiasRelu.S
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNFloat16C8ToC4AddBiasRelu
//void MNNFloat16C8ToC4AddBiasRelu(int16_t* dst, const int16_t* src, const int16_t* bias, size_t size, size_t ocUnit);
//Auto: x0: dst, x1:src, x2:bias, x3:size, x4:ocUnit

//Compute x5 as dst Z stride
mov x5, #8 // 4*sizeof(float16_t)
mul x5, x3, x5
movi v17.8h, #0
LoopOz:
    mov x6, x3
    ld1 {v16.8h}, [x2], #16

    //x7: dstZ0, x8:dstZ1
    mov x7, x0
    add x8, x0, x5

    LSize8:
    cmp x6, #8
    blt LSize1
    LoopSizeC8:
        sub x6, x6, #8
        ld1 {v0.8h, v1.8h}, [x1], #32
        cmp x6, #8
        fadd v0.8h, v0.8h, v16.8h
        fadd v1.8h, v1.8h, v16.8h
        ld1 {v2.8h, v3.8h}, [x1], #32
        fmax v0.8h, v0.8h, v17.8h
        fmax v1.8h, v1.8h, v17.8h
        fadd v2.8h, v2.8h, v16.8h
        st2 {v0.d, v1.d}[0], [x7], #16
        fmax v2.8h, v2.8h, v17.8h

        ld1 {v4.8h, v5.8h}, [x1], #32
        fadd v3.8h, v3.8h, v16.8h
        st2 {v0.d, v1.d}[1], [x8], #16
        fmax v3.8h, v3.8h, v17.8h
        fadd v4.8h, v4.8h, v16.8h
        st2 {v2.d, v3.d}[0], [x7], #16
        fadd v5.8h, v5.8h, v16.8h
        ld1 {v6.8h, v7.8h}, [x1], #32
        fmax v4.8h, v4.8h, v17.8h
        st2 {v2.d, v3.d}[1], [x8], #16
        fmax v5.8h, v5.8h, v17.8h

        fadd v6.8h, v6.8h, v16.8h
        st2 {v4.d, v5.d}[0], [x7], #16
        fadd v7.8h, v7.8h, v16.8h
        fmax v6.8h, v6.8h, v17.8h
        st2 {v4.d, v5.d}[1], [x8], #16
        fmax v7.8h, v7.8h, v17.8h
        st2 {v6.d, v7.d}[0], [x7], #16
        st2 {v6.d, v7.d}[1], [x8], #16

        bge LoopSizeC8

    LSize1:
    cmp x6, #0
    beq LSizeEnd

    LoopSize:
        ld1 {v0.8h}, [x1], #16

        fadd v0.8h, v0.8h, v16.8h
        fmax v0.8h, v0.8h, v17.8h

        st1 {v0.d}[0], [x7], #8
        st1 {v0.d}[1], [x8], #8

        subs x6, x6, #1
        bne LoopSize
    LSizeEnd:

    mov x0, x8


    subs x4, x4, #1
    bne LoopOz



EndOz:


ret

#endif
