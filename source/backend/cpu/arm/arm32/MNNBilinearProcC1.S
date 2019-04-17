//
//  MNNBilinearProcC1.S
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNBilinearProcC1
//void MNNBilinearProcC1(const unsigned char *sample, unsigned char* dst, const int16_t* xFactor, const int16_t* yFactor, size_t w);

//Auto: r0:sample, r1:dst, r2:xFactor, r3:yFactor
//Load: r4:w
push {r4-r6, lr}
ldr r4, [sp, #16]

vld1.32 {d31[0]}, [r3]
//Now r3 is no used
vdup.s16 d30, d31[0]
vdup.s16 d31, d31[1]

L8:
cmp r4, #8
blt End

LoopL8:
    vld4.8 {d4, d5, d6, d7}, [r0]!//q2, q3
    vld2.16 {q0, q1}, [r2]!//q0, q1
    //(x00,x01) -> (y0)
    vmovl.u8 q8, d4
    vmovl.u8 q9, d5
    vmull.u16 q10, d16, d0
    vmull.u16 q11, d17, d1
    vmlal.u16 q10, d18, d2
    vmlal.u16 q11, d19, d3

    vshrn.u32 d4, q10, #4
    vshrn.u32 d5, q11, #4

    //(x10,x11) -> (y1)
    vmovl.u8 q8, d6
    vmovl.u8 q9, d7
    vmull.u16 q10, d16, d0
    vmull.u16 q11, d17, d1
    vmlal.u16 q10, d18, d2
    vmlal.u16 q11, d19, d3

    vshrn.u32 d6, q10, #4
    vshrn.u32 d7, q11, #4

    //(y0,y1) -> dst
    vmull.u16 q10, d4, d30
    vmull.u16 q11, d5, d30
    vmlal.u16 q10, d6, d31
    vmlal.u16 q11, d7, d31

    vshrn.u32 d4, q10, #16
    vshrn.u32 d5, q11, #16

    vrshrn.u16 d0, q2, #2

    vst1.8 {d0}, [r1]!

    sub r4, r4, #8
    cmp r4, #8
    bge LoopL8


End:



pop {r4-r6, pc}

#endif
#endif
