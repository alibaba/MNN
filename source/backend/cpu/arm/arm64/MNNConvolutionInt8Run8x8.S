//
//  MNNConvolutionInt8Run8x8.S
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvolutionInt8Run8x8
//void MNNConvolutionInt8Run8x8(int16_t* dst_x, const int8_t* src_unit, const int8_t* weight_start,
//                                size_t icD8, size_t xCount, size_t yCount,
//                                size_t src_z_step, size_t dilate_y_step, size_t dilate_x_step,
//                                size_t weight_sz_step, size_t weight_sy_step
//                                );


//Auto
//x0: dst_x, x1:src_unit, x2: weight_start, x3: icD8


//Auto
//x4: xCount, x5:yCount,x6:dilate_y_step, x7:dilate_x_step

//Load
//x8:weight_sy_step

ldr x8, [sp, #0]

//Dst: v16.8h-v23.8h
movi v23.8h, #0
movi v22.8h, #0
movi v21.8h, #0
movi v20.8h, #0
movi v19.8h, #0
movi v18.8h, #0
movi v17.8h, #0
movi v16.8h, #0

LoopFY:
    mov x11, x4
    LoopFX:
        mov x12, x3
        LoopZ:
            ld1 {v0.8b}, [x1], #8

            ld1 {v1.8h}, [x2], #16

            dup v0.2d, v0.d[0]

            smlal v16.8h, v0.8b, v1.8b

            ld1 {v2.8h}, [x2], #16
            smlal2 v17.8h, v0.16b, v1.16b
            smlal v18.8h, v0.8b, v2.8b
            ld1 {v3.8h}, [x2], #16
            smlal2 v19.8h, v0.16b, v2.16b
            ld1 {v1.8h}, [x2], #16
            smlal v20.8h, v0.8b, v3.8b
            smlal2 v21.8h, v0.16b, v3.16b
            smlal v22.8h, v0.8b, v1.8b
            smlal2 v23.8h, v0.16b, v1.16b

            subs x12, x12, #1
            
            bne LoopZ

        subs x11, x11, #1
        add x1, x1, x7
        bne LoopFX
    add x2, x2, x8
    add x1, x1, x6
    subs x5, x5, #1
    bne LoopFY

//TODO Opt Transpose
//Transpose
//Swap Step 1
mov v0.16b, v16.16b
mov v1.16b, v18.16b
mov v2.16b, v20.16b
mov v3.16b, v22.16b
trn1 v16.8h, v16.8h, v17.8h
trn1 v18.8h, v18.8h, v19.8h
trn1 v20.8h, v20.8h, v21.8h
trn1 v22.8h, v22.8h, v23.8h
trn2 v17.8h, v0.8h, v17.8h
trn2 v19.8h, v1.8h, v19.8h
trn2 v21.8h, v2.8h, v21.8h
trn2 v23.8h, v3.8h, v23.8h

//Swap Step 2
mov v0.16b, v16.16b
mov v1.16b, v17.16b
mov v2.16b, v20.16b
mov v3.16b, v21.16b
trn1 v16.4s, v16.4s, v18.4s
trn1 v17.4s, v17.4s, v19.4s
trn1 v20.4s, v20.4s, v22.4s
trn1 v21.4s, v21.4s, v23.4s
trn2 v18.4s, v0.4s, v18.4s
trn2 v19.4s, v1.4s, v19.4s
trn2 v22.4s, v2.4s, v22.4s
trn2 v23.4s, v3.4s, v23.4s


//Swap Step 3
mov v0.16b, v16.16b
mov v1.16b, v17.16b
mov v2.16b, v18.16b
mov v3.16b, v19.16b
trn1 v16.2d, v16.2d, v20.2d
trn1 v17.2d, v17.2d, v21.2d
trn1 v18.2d, v18.2d, v22.2d
trn1 v19.2d, v19.2d, v23.2d
trn2 v20.2d, v0.2d, v20.2d
trn2 v21.2d, v1.2d, v21.2d
trn2 v22.2d, v2.2d, v22.2d
trn2 v23.2d, v3.2d, v23.2d

sqadd v0.8h, v16.8h, v17.8h
sqadd v1.8h, v18.8h, v19.8h
sqadd v2.8h, v20.8h, v21.8h
sqadd v3.8h, v22.8h, v23.8h

sqadd v0.8h, v0.8h, v1.8h
sqadd v2.8h, v2.8h, v3.8h

sqadd v0.8h, v0.8h, v2.8h


st1 {v0.8h}, [x0], #16



ret


#endif
