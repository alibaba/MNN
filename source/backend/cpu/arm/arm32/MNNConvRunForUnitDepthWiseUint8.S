//
//  MNNConvRunForUnitDepthWiseUint8.S
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
/*
struct MNN::ConstConvolutionParameter
{
    size_t kw;
    size_t kh;
    size_t weight_y_step;
    size_t dilate_x_step;
    size_t dilate_y_step;
    size_t stride_x_step;
    int32_t output_multiplier;
    int32_t output_shift_before;
    int32_t output_shift_after;
    int32_t output_offset;
    int32_t output_activation_min;
    int32_t output_activation_max;
};
*/

asm_function MNNConvRunForUnitDepthWiseUint8
//void MNNConvRunForUnitDepthWiseUint8(uint8_t* dst, const int16_t* src, const int16_t* weight,
//size_t fw, size_t fh,
//const MNN::ConstConvolutionParameter* parameter,
//const int32_t* bias_data)

//r0: dst, r1: src, r2: weight, r3:fw

//Load from sp:
//r4: fh, r5: parameter, r6:bias_data


push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9

ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]

/*Compute Convolution*/
vld1.32 {q8}, [r6]
//r7: weight_y_step
ldr r7, [r5, #8]

//r8: dilate_x_step
ldr r8, [r5, #12]

//lr:dilate_y_step
ldr lr, [r5, #16]

mul r12, r8, r3
sub lr, lr, r12

//sizeof(int16_t)*4
mov r12, #8
mul r12, r3, r12
sub r7, r7, r12

LoopFy:
    mov r10, r3
    LoopFx:
        vld1.32 {d0}, [r1], r8
        vld1.32 {d2}, [r2]!
        vmlal.s16 q8,  d0, d2
        subs r10, r10, #1
        bne LoopFx
    subs r4, r4, #1
    add r1, r1, lr
    add r2, r2, r7
    bne LoopFy

/*Compute Convolution End*/

/*Compute multi and relu*/

//r7: output_multiplier
ldr r7, [r5, #24]

//r8: output_shift_before
ldr r8, [r5, #28]
vdup.32 q13, r7
vdup.32 q14, r8

//lr: output_offset
ldr lr, [r5, #36]
vrshl.s32 q8, q8, q14

//r8: output_shift_after
ldr r8, [r5, #32]
vqrdmulh.s32 q8, q8, q13
vdup.32 q14, r8

//r10: output_activation_min
ldr r10, [r5, #40]
vrshl.s32 q8, q8, q14

//r11: output_activation_max
ldr r11, [r5, #44]

vdup.32 q12, lr
vdup.32 q14, r10
vadd.s32 q8, q8, q12
vdup.32 q13, r11
vmax.s32 q8, q14, q8
vmin.s32 q8, q13, q8

vmovn.s32 d0, q8
vmovn.s16 d16, q0

vst1.32 {d16[0]}, [r0]

pop {r4-r8, r10, r11, pc}

#endif
#endif
