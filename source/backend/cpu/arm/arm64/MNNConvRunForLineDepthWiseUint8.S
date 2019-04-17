//
//  MNNConvRunForLineDepthWiseUint8.S
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

asm_function MNNConvRunForLineDepthWiseUint8
//void MNNConvRunForLineDepthWiseUint8(uint8_t* dst, const int16_t* src, const int16_t* weight, size_t width,
//                                       MNN::ConstConvolutionParameter* parameters,
//                                       const int32_t* bias_data)

//Auto: x0:dst, x1:src, x2:weight, x3:width
//x4: parameters, x5: bias_data

//Load All parameters
//v23.4s: output_multiplier, v22.4s: output_shift_before, v21.4s: output_shift_after
//v20.4s: output_offset, v19.4h: output_activation_min, v20.4h: output_activation_max, v17.4s: bias

ld1 {v17.4s}, [x5]
add x6, x4, #48
ld1 {v0.s}[0], [x6], #4
dup v23.4s, v0.s[0]
ld1 {v0.s}[0], [x6], #4
dup v22.4s, v0.s[0]
ld1 {v0.s}[0], [x6], #4
dup v21.4s, v0.s[0]
ld1 {v0.s}[0], [x6], #4
dup v20.4s, v0.s[0]
ld1 {v0.s}[0], [x6], #4
dup v19.4s, v0.s[0]
ld1 {v0.s}[0], [x6], #4
sqxtn v19.4h, v19.4s
dup v18.4s, v0.s[0]
sqxtn v18.4h, v18.4s
dup v19.2d, v19.d[0]
dup v18.2d, v18.d[0]


//Now x5 is no used, set x5 as stride_x_step
ldr x5, [x4, #40]
//x6:fw, x7:fh
ldr x6, [x4, #0]
ldr x7, [x4, #8]

//x8:dilate_x_step, x9:dilate_y_step
ldr x8, [x4, #24]
ldr x9, [x4, #32]

//dilate_y_step->dilate_y_step-fw*dilate_x_step
mul x12, x6, x8
sub x9, x9, x12

L8:
cmp x3, #8
blt L1

mov x12, #8
mul x10, x5, x12

LoopL8:
    mov x13, x1
    mov x14, x2
    mov v0.16b, v17.16b
    mov v1.16b,v17.16b 
    mov v2.16b,v17.16b
    mov v3.16b,v17.16b 
    mov v4.16b,v17.16b 
    mov v5.16b,v17.16b 
    mov v6.16b,v17.16b 
    mov v7.16b,v17.16b 

    mov x11, x7
    LoopL8FY:
        mov x12, x6
        LoopL8FX:
            ld1 {v16.4h}, [x2], #8
            ld1 {v24.4h}, [x1], x5
            smlal v0.4s, v16.4h, v24.4h
            ld1 {v25.4h}, [x1], x5
            smlal v1.4s, v16.4h, v25.4h
            ld1 {v24.4h}, [x1], x5
            smlal v2.4s, v16.4h, v24.4h
            ld1 {v25.4h}, [x1], x5
            smlal v3.4s, v16.4h, v25.4h
            ld1 {v24.4h}, [x1], x5
            smlal v4.4s, v16.4h, v24.4h
            ld1 {v25.4h}, [x1], x5
            smlal v5.4s, v16.4h, v25.4h
            ld1 {v24.4h}, [x1], x5
            smlal v6.4s, v16.4h, v24.4h
            ld1 {v25.4h}, [x1], x5
            smlal v7.4s, v16.4h, v25.4h
            sub x1, x1, x10
            add x1, x1, x8
            subs x12, x12, #1
            bne LoopL8FX
        subs x11, x11, #1
        add x1, x1, x9
        bne LoopL8FY
    mov x1, x13
    mov x2, x14
    sqshl v0.4s, v0.4s, v22.4s
    sqshl v1.4s, v1.4s, v22.4s
    sqshl v2.4s, v2.4s, v22.4s
    sqshl v3.4s, v3.4s, v22.4s
    sqshl v4.4s, v4.4s, v22.4s
    sqshl v5.4s, v5.4s, v22.4s
    sqshl v6.4s, v6.4s, v22.4s
    sqshl v7.4s, v7.4s, v22.4s
    
    sqrdmulh v0.4s, v23.4s, v0.4s
    sqrdmulh v1.4s, v23.4s, v1.4s
    sqrdmulh v2.4s, v23.4s, v2.4s
    sqrdmulh v3.4s, v23.4s, v3.4s
    sqrdmulh v4.4s, v23.4s, v4.4s
    sqrdmulh v5.4s, v23.4s, v5.4s
    sqrdmulh v6.4s, v23.4s, v6.4s
    sqrdmulh v7.4s, v23.4s, v7.4s

    sqrshl v0.4s, v0.4s, v21.4s
    sqrshl v1.4s, v1.4s, v21.4s
    sqrshl v2.4s, v2.4s, v21.4s
    sqrshl v3.4s, v3.4s, v21.4s
    sqrshl v4.4s, v4.4s, v21.4s
    sqrshl v5.4s, v5.4s, v21.4s
    sqrshl v6.4s, v6.4s, v21.4s
    sqrshl v7.4s, v7.4s, v21.4s

    add v0.4s, v0.4s, v20.4s
    add v1.4s, v1.4s, v20.4s
    add v2.4s, v2.4s, v20.4s
    add v3.4s, v3.4s, v20.4s
    add v4.4s, v4.4s, v20.4s
    add v5.4s, v5.4s, v20.4s
    add v6.4s, v6.4s, v20.4s
    add v7.4s, v7.4s, v20.4s

    add x1, x1, x10
    sqxtn v0.4h, v0.4s
    sqxtn2 v0.8h, v1.4s
    smax v0.8h, v0.8h, v19.8h
    sqxtn v1.4h, v2.4s
    sqxtn2 v1.8h, v3.4s
    smin v0.8h, v0.8h, v18.8h
    sqxtn v2.4h, v4.4s
    sqxtn2 v2.8h, v5.4s
    sqxtn v3.4h, v6.4s
    sqxtn2 v3.8h, v7.4s

    smax v1.8h, v1.8h, v19.8h
    smax v2.8h, v2.8h, v19.8h
    smax v3.8h, v3.8h, v19.8h

    smin v1.8h, v1.8h, v18.8h
    smin v2.8h, v2.8h, v18.8h
    smin v3.8h, v3.8h, v18.8h

    uqxtn v0.8b, v0.8h
    uqxtn2 v0.16b, v1.8h
    uqxtn v1.8b, v2.8h
    uqxtn2 v1.16b, v3.8h

    st1 {v0.4s, v1.4s}, [x0], #32
    sub x3, x3, #8
    cmp x3, #8
    bge LoopL8


L1:
cmp x3, #0
beq End

LoopL1:
    mov x13, x1
    mov x14, x2
    mov v0.16b,v17.16b
    mov x11, x7
    LoopL1FY:
        mov x12, x6
        LoopL1FX:
            ld1 {v1.4h}, [x1], x8
            ld1 {v2.4h}, [x2], #8
            smlal v0.4s, v1.4h, v2.4h
            subs x12, x12, #1
            bne LoopL1FX
        subs x11, x11, #1
        add x1, x1, x9
        bne LoopL1FY
    mov x1, x13
    mov x2, x14
    sqshl v0.4s, v0.4s, v22.4s
    sqrdmulh v0.4s, v23.4s, v0.4s
    sqrshl v0.4s, v0.4s, v21.4s
    add v0.4s, v0.4s, v20.4s

    add x1, x1, x5
    sqxtn v0.4h, v0.4s
    smax v0.4h, v0.4h, v19.4h
    smin v0.4h, v0.4h, v18.4h
    uqxtn v0.8b, v0.8h
    st1 {v0.s}[0], [x0], #4
    subs x3, x3, #1
    bne LoopL1

End:

ret

#endif
