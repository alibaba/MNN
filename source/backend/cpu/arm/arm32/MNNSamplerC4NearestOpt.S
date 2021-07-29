//
//  MNNSamplerC4NearestOpt.S
//  MNN
//
//  Created by MNN on 2018/12/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
//void MNNSamplerC4NearestOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t iw, size_t ih, size_t yStride);
asm_function MNNSamplerC4NearestOpt

//Auto: r0:source, r1:dest, r2:points, r3:count
//Load: r4: xMax, r5: yMax, r6:yStride

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9
ldr r4, [sp, #32]
ldr r5, [sp, #36]
vdup.32 q12, r4
vdup.32 q13, r5
vmov.i32 q11, #0
vmov.i32 q10, #4
ldr r6, [sp, #40]

vld1.32 {q0}, [r2]
L8:
cmp r3, #8
blt L1
vdup.32 q15, r6
vdup.32 q14, r0

L8Loop:

vcvtr.s32.f32 s8, s0
vcvtr.s32.f32 s12, s1
vadd.f32 d3, d0, d1
vcvtr.s32.f32 s9, s6
vcvtr.s32.f32 s13, s7
vadd.f32 d0, d3, d1
vcvtr.s32.f32 s10, s0
vcvtr.s32.f32 s14, s1
vadd.f32 d3, d0, d1
vcvtr.s32.f32 s11, s6
vcvtr.s32.f32 s15, s7
vmax.s32 q2, q2, q11
vmax.s32 q3, q3, q11
vmin.s32 q2, q2, q12
vmin.s32 q3, q3, q13
vmul.s32 q2, q2, q10
vadd.f32 d0, d3, d1
vmla.s32 q2, q15, q3
vadd.s32 q2, q2, q14

vmov.i32 r4, d4[0]
vmov.i32 r5, d4[1]

vld1.32 {d16[0]}, [r4]
vcvtr.s32.f32 s8, s0
vld1.32 {d16[1]}, [r5]

vmov.i32 r4, d5[0]
vmov.i32 r5, d5[1]

vld1.32 {d17[0]}, [r4]
vcvtr.s32.f32 s12, s1
vld1.32 {d17[1]}, [r5]
vadd.f32 d3, d0, d1

vst1.32 {q8}, [r1]!
vcvtr.s32.f32 s9, s6

vcvtr.s32.f32 s13, s7
vadd.f32 d0, d3, d1
vcvtr.s32.f32 s10, s0
vcvtr.s32.f32 s14, s1
vadd.f32 d3, d0, d1
vcvtr.s32.f32 s11, s6
vcvtr.s32.f32 s15, s7
vadd.f32 d0, d3, d1
vmax.s32 q2, q2, q11
vmax.s32 q3, q3, q11
vmin.s32 q2, q2, q12
vmin.s32 q3, q3, q13
vmul.s32 q2, q2, q10
vmla.s32 q2, q15, q3
vadd.s32 q2, q2, q14

vmov.i32 r4, d4[0]
vmov.i32 r5, d4[1]

vld1.32 {d16[0]}, [r4]
vld1.32 {d16[1]}, [r5]

vmov.i32 r4, d5[0]
vmov.i32 r5, d5[1]

vld1.32 {d17[0]}, [r4]
vld1.32 {d17[1]}, [r5]

vst1.32 {q8}, [r1]!

sub r3, r3, #8
cmp r3, #8
bge L8Loop


L1:
cmp r3, #0

mov r12, #4
beq End

L1Loop:
vcvtr.s32.f32 s4, s0
vcvtr.s32.f32 s6, s1
vmax.s32 d2, d2, d22
vmax.s32 d3, d3, d22
vmin.s32 d2, d2, d24
vmin.s32 d3, d3, d26
vmov.i32 r4, d2[0]
vmov.i32 r5, d3[0]
mul r4, r4, r12
add r4, r4, r0
mul r5, r5, r6
add r4, r4, r5
vld1.32 {d3[0]}, [r4]
vadd.f32 d0, d0, d1
vst1.32 {d3[0]}, [r1]!

subs r3, r3, #1

bne L1Loop

End:

pop {r4-r8, r10, r11, pc}

#endif
#endif
