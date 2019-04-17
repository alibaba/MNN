//
//  MNNConvRunForUnitDepthWiseUint8.S
//  MNN
//
//  Created by MNN on 2018/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

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

//x0: dst, x1: src, x2: weight, x3:fw
//x4: fh, x5: parameter, x6:bias_data


/*Compute Convolution*/
ld1 {v16.4s}, [x6]
//x7: weight_y_step
ldr x7, [x5, #16]

//x8: dilate_x_step
ldr x8, [x5, #24]

//x9:dilate_y_step
ldr x9, [x5, #32]

mul x12, x8, x3
sub x9, x9, x12

//sizeof(int16_t)*4
mov x12, #8
mul x12, x3, x12
sub x7, x7, x12

LoopFy:
    mov x10, x3
    LoopFx:
        ld1 {v0.4h}, [x1], x8
        ld1 {v1.4h}, [x2], #8
        smlal v16.4s, v0.4h, v1.4h
        subs x10, x10, #1
        bne LoopFx
    subs x4, x4, #1
    add x1, x1, x9
    add x2, x2, x7
    bne LoopFy

/*Compute Convolution End*/

/*Compute multi and relu*/

//x7: output_multiplier
add x5, x5, #48
ld1 {v0.s}[0], [x5], #4

//x8: output_shift_before
ld1 {v0.s}[1], [x5], #4
dup v21.4s, v0.s[0]
dup v22.4s, v0.s[1]


//x8: output_shift_after
ld1 {v1.s}[1], [x5], #4
sqrdmulh v16.4s, v16.4s, v21.4s

//x9: output_offset
ld1 {v0.s}[2], [x5], #4
sqshl v16.4s, v16.4s, v22.4s
dup v23.4s, v1.s[1]

//x10: output_activation_min
ld1 {v0.s}[3], [x5], #4
sqrshl v16.4s, v16.4s, v23.4s

//x11: output_activation_max
ld1 {v1.s}[0], [x5], #4

dup v20.4s, v0.s[2]
dup v22.4s, v0.s[3]
add v16.4s, v16.4s, v20.4s
dup v21.4s, v1.s[0]
smax v16.4s, v22.4s, v16.4s
smin v16.4s, v21.4s, v16.4s

sqxtn v16.4h, v16.4s
uqxtn v16.8b, v16.8h

st1 {v16.s}[0], [x0]


ret


#endif
