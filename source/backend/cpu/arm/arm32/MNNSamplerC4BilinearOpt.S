//
//  MNNSamplerC4BilinearOpt.S
//  MNN
//
//  Created by MNN on 2018/11/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
//void MNNSamplerC4BilinearOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t iw, size_t ih, size_t yStride);
asm_function MNNSamplerC4BilinearOpt

//Auto: r0:source, r1:dest, r2:points, r3:count
//Load: r4: xMax, r5: yMax, r6:yStride

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9
ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]
mov r12, #4

vpush {q4-q7}

vmov.i32 q15, #0
//q14: ih-1, q15:iw-1
vdup.i32 q14, r5
vdup.i32 q13, r4

vld1.32 {q12}, [r2]



L4:
cmp r3, #4
blt L1


LoopL4:
    vmov.32 d4, d24
    vadd.f32 d6, d24, d25
    vadd.f32 d5, d6, d25
    vadd.f32 d7, d5, d25
    vtrn.32 d4, d6
    vadd.f32 d24, d7, d25
    vtrn.32 d5, d7
    sub r3, r3, #4

    vcvt.s32.f32 q8, q2
    vcvt.s32.f32 q9, q3
    vcvt.f32.s32 q10, q8
    vcvt.f32.s32 q11, q9
    //q0: xF, q1:yF
    vsub.f32 q0, q2, q10
    vsub.f32 q1, q3, q11
    vmax.s32 q8, q8, q15
    vmax.s32 q9, q9, q15
    vmov.i32 q2, #1
    vadd.s32 q10, q8, q2
    vadd.s32 q11, q9, q2
    vmin.s32 q8, q8, q13
    vmin.s32 q9, q9, q14
    vmin.s32 q10, q10, q13
    vmin.s32 q11, q11, q14

    //q8, q9, q10, q11: x0, y0, x1, y1
    vmov.i32 d4[0], r6
    vmov.i32 d4[1], r12

    vmul.u32 q11, q11, d4[0]
    vmul.u32 q9, q9, d4[0]
    vmul.u32 q8, q8, d4[1]
    vmul.u32 q10, q10, d4[1]

    vadd.u32 q2, q8, q9
    vadd.u32 q3, q8, q11
    vadd.u32 q8, q10, q9
    vadd.u32 q9, q10, q11
    vdup.i32 q10, r0
    vadd.u32 q2, q2, q10
    vadd.u32 q3, q3, q10
    vadd.u32 q8, q8, q10
    vadd.u32 q9, q9, q10

    //q2, q3, q8, q9 : position of x00, x10, x01, x11
    //A
    vmov.i32 r7, d4[0]
    vld1.32 {d20[0]}, [r7]
    vmov.i32 r7, d6[0]
    vld1.32 {d22[0]}, [r7]
    vmov.i32 r7, d16[0]
    vld1.32 {d20[1]}, [r7]
    vmov.i32 r7, d18[0]
    vld1.32 {d22[1]}, [r7]

    vmovl.u8 q10, d20
    vmovl.u8 q11, d22
    vmovl.s16 q4, d20
    vmovl.s16 q5, d21
    vcvt.f32.s32 q4, q4
    vcvt.f32.s32 q5, q5

    vmovl.s16 q6, d22
    vmovl.s16 q7, d23
    vcvt.f32.s32 q6, q6
    vcvt.f32.s32 q7, q7

    vmov.i32 r7, d4[1]

    vsub.f32 q5, q5, q4
    vld1.32 {d20[0]}, [r7]
    vsub.f32 q7, q7, q6
    vmla.f32 q4, q5, d0[0]
    vmov.i32 r7, d6[1]
    vmla.f32 q6, q7, d0[0]
    vld1.32 {d22[0]}, [r7]
    vsub.f32 q5, q6, q4
    vmov.i32 r7, d16[1]

    vmla.f32 q4, q5, d2[0]

    vld1.32 {d20[1]}, [r7]
    vcvt.s32.f32 q4, q4
    vmov.i32 r7, d18[1]
    vmovn.u32 d8, q4
    vld1.32 {d22[1]}, [r7]
    vmovn.u16 d8, q4
    vst1.32 {d8[0]}, [r1]!

    //B

    vmovl.u8 q10, d20
    vmovl.u8 q11, d22
    vmovl.s16 q4, d20
    vmovl.s16 q5, d21
    vcvt.f32.s32 q4, q4
    vcvt.f32.s32 q5, q5

    vmovl.s16 q6, d22
    vmovl.s16 q7, d23
    vcvt.f32.s32 q6, q6
    vcvt.f32.s32 q7, q7

    vmov.i32 r7, d5[0]
    vsub.f32 q5, q5, q4
    vld1.32 {d20[0]}, [r7]
    vsub.f32 q7, q7, q6
    vmov.i32 r7, d7[0]
    vmla.f32 q4, q5, d0[1]
    vld1.32 {d22[0]}, [r7]
    vmla.f32 q6, q7, d0[1]
    vmov.i32 r7, d17[0]
    vsub.f32 q5, q6, q4
    vld1.32 {d20[1]}, [r7]
    vmla.f32 q4, q5, d2[1]
    vmov.i32 r7, d19[0]
    vcvt.s32.f32 q4, q4
    vld1.32 {d22[1]}, [r7]

    vmovn.u32 d8, q4
    vmovn.u16 d8, q4
    vst1.32 {d8[0]}, [r1]!

    //C

    vmovl.u8 q10, d20
    vmovl.u8 q11, d22
    vmovl.s16 q4, d20
    vmovl.s16 q5, d21
    vcvt.f32.s32 q4, q4
    vcvt.f32.s32 q5, q5

    vmovl.s16 q6, d22
    vmovl.s16 q7, d23
    vcvt.f32.s32 q6, q6
    vcvt.f32.s32 q7, q7

    vmov.i32 r7, d5[1]
    vsub.f32 q5, q5, q4
    vld1.32 {d20[0]}, [r7]
    vsub.f32 q7, q7, q6
    vmov.i32 r7, d7[1]
    vmla.f32 q4, q5, d1[0]
    vld1.32 {d22[0]}, [r7]
    vmla.f32 q6, q7, d1[0]
    vmov.i32 r7, d17[1]
    vsub.f32 q5, q6, q4
    vld1.32 {d20[1]}, [r7]
    vmla.f32 q4, q5, d3[0]
    vmov.i32 r7, d19[1]
    vcvt.s32.f32 q4, q4
    vld1.32 {d22[1]}, [r7]
    vmovn.u32 d8, q4
    vmovl.u8 q10, d20
    vmovn.u16 d8, q4
    vst1.32 {d8[0]}, [r1]!

    //D

    vmovl.u8 q11, d22
    vmovl.s16 q4, d20
    vmovl.s16 q5, d21
    vcvt.f32.s32 q4, q4
    vcvt.f32.s32 q5, q5

    vmovl.s16 q6, d22
    vmovl.s16 q7, d23
    vcvt.f32.s32 q6, q6
    vcvt.f32.s32 q7, q7

    vsub.f32 q5, q5, q4
    vsub.f32 q7, q7, q6
    vmla.f32 q4, q5, d1[1]
    vmla.f32 q6, q7, d1[1]

    vsub.f32 q5, q6, q4
    vmla.f32 q4, q5, d3[1]

    vcvt.s32.f32 q4, q4
    vmovn.u32 d8, q4
    vmovn.u16 d8, q4
    vst1.32 {d8[0]}, [r1]!
    cmp r3, #4
    bge LoopL4


L1:
cmp r3, #0
beq End

//int limit
vmov.i32 d26[0], r4
vmov.i32 d26[1], r5
vmov.i32 d28, #1

//float limit
vcvt.f32.s32 d27, d26

LoopL1:
    vmov.32 d0, d24
    vadd.f32 d24, d24, d25

    vmax.f32 d0, d0, d30

    //d1:x0y0, d2:x1y1
    vcvt.s32.f32 d1, d0
    vadd.s32 d2, d1, d28
    vcvt.f32.s32 d3, d1
    vmin.s32 d2, d2, d26
    vmin.s32 d1, d1, d26
    //d3:factor
    vsub.f32 d3, d0, d3

    //d16-d17 source pixels
    vmov.i32 r7, d1[0]
    mul r7, r7, r12
    vmov.i32 r8, d1[1]
    mul r8, r8, r6
    add r8, r8, r0
    add r7, r8, r7
    vld1.32 {d16[0]}, [r7]

    vmov.i32 r7, d2[0]
    mul r7, r7, r12
    add r7, r8, r7
    vld1.32 {d16[1]}, [r7]

    vmov.i32 r7, d1[0]
    mul r7, r7, r12
    vmov.i32 r8, d2[1]
    mul r8, r8, r6
    add r8, r8, r0
    add r7, r8, r7
    vld1.32 {d18[0]}, [r7]

    vmov.i32 r7, d2[0]
    mul r7, r7, r12
    add r7, r8, r7
    vld1.32 {d18[1]}, [r7]

    vmovl.u8 q8, d16
    vmovl.u8 q9, d18

    vmovl.s16 q0, d16
    vmovl.s16 q2, d17
    vcvt.f32.s32 q0, q0
    vcvt.f32.s32 q2, q2

    vsub.f32 q8, q2, q0
    vmla.f32 q0, q8, d3[0]

    vmovl.s16 q10, d18
    vmovl.s16 q11, d19
    vcvt.f32.s32 q10, q10
    vcvt.f32.s32 q11, q11

    vsub.f32 q8, q11, q10
    vmla.f32 q10, q8, d3[0]

    vsub.f32 q2, q10, q0
    vmla.f32 q0, q2, d3[1]

    vcvt.s32.f32 q0, q0
    vmovn.u32 d0, q0
    vmovn.u16 d0, q0

    vst1.32 {d0[0]}, [r1]!
    subs r3, r3, #1
    bne LoopL1

End:

vpop {q4-q7}
pop {r4-r8, r10, r11, pc}


#endif
#endif
