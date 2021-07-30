//
//  MNNQuanToDestUint8.S
//  MNN
//
//  Created by MNN on 2018/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

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
//    int32_t filter_offset;
//    int32_t input_offset;
//    int32_t offset_add;
//};

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9

//Auto: r0: outputInTile, r1: gemmOutputAddr, r2: bias_data, r3: ocUnit

//Loap from sp:
//r4: realDstCount, r5: dstZStep, r6: srcZstep, r7: parameter

ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]
ldr r7, [sp, #44]

//srcZstep -> srcZstep - realDstCount*4*sizeof(int32_t)
mov r12, #16//4*sizeof(int32_t)
mul r12, r4, r12
sub r6, r6, r12

//dstZStep -> dstZStep - realDstCount*4*sizeof(uint8_t)
mov r12, #4
mul r12, r4, r12
sub r5, r5, r12

//q15: output_multiplier, q14: output_shift_before, q13: output_shift_after, d22: output_activation_min, d23: output_activation_max, q10: output_offset
ldr r8, [r7, #0]
vdup.32 q15, r8
ldr lr, [r7, #4]
vdup.32 q14, lr
ldr r8, [r7, #8]
vdup.32 q13, r8
ldr lr, [r7, #12]
vdup.32 q11, lr
ldr r8, [r7, #16]
vmovn.s32 d22, q11
vdup.32 q12, r8
vmovn.s32 d23, q12
ldr lr, [r7, #20]
vdup.32 q10, lr

LoopDz:
    vld1.32 {q9}, [r2]!
    mov r8, r4
    L2:
    cmp r8, #2
    blt L1
    LoopW2:
        //A
        vld1.32 {q2, q3}, [r1]!
        vadd.s32 q2, q2, q9
        vadd.s32 q3, q3, q9
        vshl.s32 q2, q2, q14
        vshl.s32 q3, q3, q14
        vqrdmulh.s32 q2, q2, q15
        vqrdmulh.s32 q3, q3, q15
        vrshl.s32 q2, q2, q13
        vrshl.s32 q3, q3, q13
        vadd.s32 q2, q2, q10
        vadd.s32 q3, q3, q10

        vmovn.s32 d0, q2
        vmovn.s32 d1, q3
        vmax.s16 d0, d0, d22
        vmax.s16 d1, d1, d22
        vmin.s16 d0, d0, d23
        vmin.s16 d1, d1, d23

        vmovn.s16 d0, q0

        vst1.32 {d0}, [r0]!

        sub r8, r8, #2
        cmp r8, #2
        bge LoopW2

    L1:
    cmp r8, #0
    beq WEnd

    LoopW1:
        vld1.32 {q0}, [r1]!
        vadd.s32 q0, q0, q9
        vshl.s32 q0, q0, q14
        vqrdmulh.s32 q0, q15, q0
        vrshl.s32 q0, q0, q13
        vadd.s32 q0, q0, q10

        vmovn.s32 d0, q0
        vmax.s16 d0, d0, d22
        vmin.s16 d0, d0, d23
        vmovn.s16 d0, q0
        vst1.32 {d0[0]}, [r0]!

        subs r8, r8, #1
        bne LoopW1

    WEnd:
    subs r3, r3, #1
    add r0, r0, r5
    add r1, r1, r6
    bne LoopDz



pop {r4-r8, r10, r11, pc}

#endif
#endif
