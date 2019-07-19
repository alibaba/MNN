//
//  MNNQuanToDestUint8.S
//  MNN
//
//  Created by MNN on 2018/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNQuanToDestUint8

//void MNNQuanToDestUint8(uint8_t* outputInTile, const int32_t* gemmOutputAddr, const int32_t* bias_data,
//                          size_t ocUnit, size_t realDstCount,
//                          size_t dstZStep, size_t srcZstep,
//                          const MNN::CPUTFQuantizedConv2D::QuanParameter* parameter
//                          )

//struct QuanParameter
//{
//    int32_t output_multiplier;
//    int32_t output_shift_before;
//    int32_t output_shift_after;
//    int32_t output_activation_min;
//    int32_t output_activation_max;
//    int32_t output_offset;
//};


//Auto: x0: outputInTile, x1: gemmOutputAddr, x2: bias_data, x3: ocUnit

//x4: realDstCount, x5: dstZStep, x6: srcZstep, x7: parameter

//srcZstep -> srcZstep - realDstCount*4*sizeof(int32_t)
mov x12, #16//4*sizeof(int32_t)
mul x12, x4, x12
sub x6, x6, x12

//dstZStep -> dstZStep - realDstCount*4*sizeof(uint8_t)
mov x12, #4
mul x12, x4, x12
sub x5, x5, x12

//v23.4s: output_multiplier, v22.4s: output_shift_before, v21.4s: output_shift_after, v20.4s: output_activation_min, v19.4s: output_activation_max, v18.4s: output_offset
ld1 {v0.s}[0], [x7], #4
dup v23.4s, v0.s[0]
ld1 {v0.s}[0], [x7], #4
dup v22.4s, v0.s[0]
ld1 {v0.s}[0], [x7], #4
dup v21.4s, v0.s[0]
ld1 {v0.s}[0], [x7], #4
dup v20.4s, v0.s[0]
ld1 {v0.s}[0], [x7], #4
dup v19.4s, v0.s[0]
ld1 {v0.s}[0], [x7], #4
dup v18.4s, v0.s[0]

LoopDz:
    ld1 {v17.4s}, [x2], #16
    mov x8, x4
    L8:
    cmp x8, #4
    blt L1
    LoopW4:
        //A
        ld1 {v2.4s, v3.4s}, [x1], #32
        add v2.4s, v2.4s, v17.4s
        add v3.4s, v3.4s, v17.4s
        sqshl v2.4s, v2.4s, v22.4s
        sqshl v3.4s, v3.4s, v22.4s
        sqrdmulh v2.4s, v2.4s, v23.4s
        sqrdmulh v3.4s, v3.4s, v23.4s
        sqrshl v2.4s, v2.4s, v21.4s
        sqrshl v3.4s, v3.4s, v21.4s
        add v2.4s, v2.4s, v18.4s
        add v3.4s, v3.4s, v18.4s
        smax v2.4s, v2.4s, v20.4s
        smax v3.4s, v3.4s, v20.4s
        smin v2.4s, v2.4s, v19.4s
        smin v3.4s, v3.4s, v19.4s

        sqxtn v0.4h, v2.4s
        sqxtn2 v0.8h, v3.4s

        ld1 {v2.4s, v3.4s}, [x1], #32
        uqxtn v0.8b, v0.8h


        //B
        add v2.4s, v2.4s, v17.4s
        add v3.4s, v3.4s, v17.4s
        st1 {v0.8b}, [x0], #8
        sqshl v2.4s, v2.4s, v22.4s
        sqshl v3.4s, v3.4s, v22.4s
        sqrdmulh v2.4s, v2.4s, v23.4s
        sqrdmulh v3.4s, v3.4s, v23.4s
        sqrshl v2.4s, v2.4s, v21.4s
        sqrshl v3.4s, v3.4s, v21.4s
        add v2.4s, v2.4s, v18.4s
        add v3.4s, v3.4s, v18.4s
        smax v2.4s, v2.4s, v20.4s
        smax v3.4s, v3.4s, v20.4s
        smin v2.4s, v2.4s, v19.4s
        smin v3.4s, v3.4s, v19.4s

        sqxtn v0.4h, v2.4s
        sqxtn2 v0.8h, v3.4s

        uqxtn v0.8b, v0.8h

        st1 {v0.8b}, [x0], #8
        sub x8, x8, #4
        cmp x8, #4
        bge LoopW4
    L1:
    cmp x8, #0
    beq WEnd
    
    LoopW1:
        ld1 {v0.4s}, [x1], #16
        add v0.4s, v0.4s, v17.4s
        sqshl v0.4s, v0.4s, v22.4s
        sqrdmulh v0.4s, v23.4s, v0.4s
        sqrshl v0.4s, v0.4s, v21.4s
        add v0.4s, v0.4s, v18.4s
        smax v0.4s, v0.4s, v20.4s
        smin v0.4s, v0.4s, v19.4s

        sqxtn v0.4h, v0.4s
        uqxtn v0.8b, v0.8h
        st1 {v0.s}[0], [x0], #4

        subs x8, x8, #1
        bne LoopW1

    WEnd:
    subs x3, x3, #1
    add x0, x0, x5
    add x1, x1, x6
    bne LoopDz


ret

#endif
