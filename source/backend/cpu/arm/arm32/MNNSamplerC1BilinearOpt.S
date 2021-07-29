//
//  MNNSamplerC1BilinearOpt.S
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
//void MNNSamplerC1BilinearOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t iw, size_t ih, size_t yStride);
asm_function MNNSamplerC1BilinearOpt

//Auto: r0:source, r1:dest, r2:points, r3:count
//Load: r4: xMax, r5: yMax, r6:yStride

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9
ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]

vpush {q4-q7}

vmov.i32 q15, #0
vmov.i32 q12, #1
//q14: ih-1, q15:iw-1
vdup.i32 q14, r5
vdup.i32 q13, r4

vcvt.f32.s32 q14, q14
vcvt.f32.s32 q13, q13

vld1.32 {q6}, [r2]


L4:
cmp r3, #4
blt L1

LoopL4:
    vmov.32 d4, d12
    vadd.f32 d6, d12, d13
    vadd.f32 d5, d6, d13
    vadd.f32 d7, d5, d13
    vtrn.32 d4, d6
    vadd.f32 d12, d7, d13
    vtrn.32 d5, d7
    sub r3, r3, #4
    vmin.f32 q2, q2, q13
    vmin.f32 q3, q3, q14
    vmax.f32 q2, q2, q15
    vmax.f32 q3, q3, q15

    vcvt.s32.f32 q8, q2
    vcvt.s32.f32 q9, q3
    vcvt.f32.s32 q10, q8
    vcvt.f32.s32 q11, q9
    //q0: xF, q1:yF
    vsub.f32 q0, q2, q10
    vsub.f32 q1, q3, q11
    vdup.i32 q2, r4
    vdup.i32 q3, r5
    vadd.s32 q10, q8, q12
    vadd.s32 q11, q9, q12
    vmin.s32 q10, q10, q2
    vmin.s32 q11, q11, q3

    //q8, q9, q10, q11: x0, y0, x1, y1

    vdup.i32 q2, r6
    vmul.s32 q11, q2, q11
    vmul.s32 q9, q2, q9

    vadd.s32 q2, q8, q9
    vadd.s32 q3, q8, q11
    vadd.s32 q8, q10, q9
    vdup.i32 q4, r0
    vadd.s32 q9, q10, q11
    vadd.u32 q2, q2, q4
    vadd.u32 q3, q3, q4
    vadd.u32 q8, q8, q4
    vadd.u32 q9, q9, q4

    //q2, q3, q8, q9 : position of x00, x10, x01, x11

    vmov.i32 r7, d4[0]
    vld1.8 {d8[0]}, [r7]
    vmov.i32 r7, d4[1]
    vld1.8 {d8[1]}, [r7]
    vmov.i32 r7, d5[0]
    vld1.8 {d8[2]}, [r7]
    vmov.i32 r7, d5[1]
    vld1.8 {d8[3]}, [r7]

    vmov.i32 r7, d6[0]
    vld1.8 {d8[4]}, [r7]
    vmov.i32 r7, d6[1]
    vld1.8 {d8[5]}, [r7]
    vmov.i32 r7, d7[0]
    vld1.8 {d8[6]}, [r7]
    vmov.i32 r7, d7[1]
    vld1.8 {d8[7]}, [r7]

    vmov.i32 r7, d16[0]
    vld1.8 {d10[0]}, [r7]
    vmov.i32 r7, d16[1]
    vld1.8 {d10[1]}, [r7]
    vmov.i32 r7, d17[0]
    vld1.8 {d10[2]}, [r7]
    vmov.i32 r7, d17[1]
    vld1.8 {d10[3]}, [r7]

    vmov.i32 r7, d18[0]
    vld1.8 {d10[4]}, [r7]
    vmov.i32 r7, d18[1]
    vld1.8 {d10[5]}, [r7]
    vmov.i32 r7, d19[0]
    vld1.8 {d10[6]}, [r7]
    vmov.i32 r7, d19[1]
    vld1.8 {d10[7]}, [r7]

    vmovl.u8 q4, d8
    vmovl.u8 q5, d10

    vmovl.s16 q2, d8
    vmovl.s16 q3, d9

    vcvt.f32.s32 q2, q2
    vcvt.f32.s32 q3, q3

    vmovl.s16 q4, d10
    vmovl.s16 q5, d11

    vcvt.f32.s32 q4, q4
    vcvt.f32.s32 q5, q5

    vsub.f32 q4, q4, q2
    vsub.f32 q5, q5, q3

    vmla.f32 q2, q4, q0
    vmla.f32 q3, q5, q0

    vsub.f32 q3, q3, q2
    vmla.f32 q2, q3, q1

    vcvt.s32.f32 q0, q2

    vmovn.u32 d0, q0
    vmovn.u16 d0, q0

    vst1.32 {d0[0]}, [r1]!

    cmp r3, #4
    bge LoopL4


L1:
cmp r3, #0
beq End

//int limit
vmov.i32 d26[0], r4
vmov.i32 d26[1], r5

//float limit
vcvt.f32.s32 d27, d26

LoopL1:
    vmov.32 d0, d12
    vadd.f32 d12, d12, d13
    vmin.f32 d0, d0, d27
    vmax.f32 d0, d0, d30

    //d1:x0y0, d2:x1y1
    vcvt.s32.f32 d1, d0
    vadd.s32 d2, d1, d24
    vcvt.f32.s32 d3, d1
    vmin.s32 d2, d2, d26
    //d3:factor
    vsub.f32 d3, d0, d3

    //d16-d17 source pixels
    vmov.i32 r7, d1[0]
    vmov.i32 r8, d1[1]
    mul r8, r8, r6
    add r8, r8, r0
    add r7, r8, r7
    vld1.8 {d16[0]}, [r7]

    vmov.i32 r7, d2[0]
    add r7, r8, r7
    vld1.8 {d16[1]}, [r7]

    vmov.i32 r7, d1[0]
    vmov.i32 r8, d2[1]
    mul r8, r8, r6
    add r8, r8, r0
    add r7, r8, r7
    vld1.8 {d16[2]}, [r7]

    vmov.i32 r7, d2[0]
    add r7, r8, r7
    vld1.8 {d16[3]}, [r7]

    vmovl.u8 q8, d16

    vmovl.s16 q0, d16
    vcvt.f32.s32 q0, q0

    vsub.f32 s8, s1, s0
    vsub.f32 s9, s3, s2
    vmla.f32 s0, s8, s6
    vmla.f32 s2, s9, s6

    vsub.f32 s2, s2, s0
    vmla.f32 s0, s2, s7

    vcvt.s32.f32 s0, s0
    vmovn.u32 d0, q0
    vmovn.u16 d0, q0

    vst1.8 {d0[0]}, [r1]!
    subs r3, r3, #1
    bne LoopL1

End:

vpop {q4-q7}
pop {r4-r8, r10, r11, pc}

#endif
#endif
