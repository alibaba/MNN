//
//  MNNBlitC3ToFloatRGBA.S
//  MNN
//
//  Created by MNN on 2018/09/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNBlitC3ToFloatRGBA
//        void MNNBlitC3ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count)

//Auto: r0: source, r1: dest, r2: mean, r3: normal

//Load from sp:
//r4: count

push {r4, lr}
ldr r4, [sp, #8]

//Mean
vld1.32 {q14}, [r2]

//Normal
vld1.32 {q15}, [r3]


L8:
cmp r4, #8
blt L1



LoopL8:

vld3.8 {d0, d1, d2}, [r0]!
vmovl.u8 q10, d0
vmovl.u8 q11, d1
vmovl.u8 q12, d2

//q0-q2, q8-q10 store origin rgb
vmovl.u16 q0, d20
vmovl.u16 q8, d21
vmovl.u16 q1, d22
vmovl.u16 q9, d23
vmovl.u16 q2, d24
vmovl.u16 q10, d25
vcvt.f32.u32 q0, q0
vcvt.f32.u32 q1, q1
vcvt.f32.u32 q2, q2

//Transpose: q0, q1, q2, 0 -> q0, q1, q2, q3
.macro COMPUTE
vmov.i32 q3, #0
vtrn.32 d0, d2
vtrn.32 d1, d3
vtrn.32 d4, d6
vtrn.32 d5, d7
vswp d1, d4
vswp d3, d6

vsub.f32 q1, q1, q14
vsub.f32 q0, q0, q14
vmul.f32 q1, q1, q15
vmul.f32 q0, q0, q15
vsub.f32 q2, q2, q14
vsub.f32 q3, q3, q14
vst1.32 {q0, q1}, [r1]!
vmul.f32 q3, q3, q15
vmul.f32 q2, q2, q15

vst1.32 {q2, q3}, [r1]!
.endm

COMPUTE

vmov.i32 q3, #0
vcvt.f32.u32 q0, q8
vcvt.f32.u32 q1, q9
vcvt.f32.u32 q2, q10


COMPUTE

sub r4, r4, #8
cmp r4, #8
bge LoopL8


L1:
cmp r4, #0
beq End

LoopL1:
vmov.i32 d0, #0
vld1.8 {d0[0]}, [r0]!
vld1.8 {d0[1]}, [r0]!
vld1.8 {d0[2]}, [r0]!

vmovl.u8 q0, d0
vmovl.u16 q0, d0
vcvt.f32.u32 q0, q0

vsub.f32 q0, q0, q14
vmul.f32 q0, q0, q15

vst1.32 {q0}, [r1]!

subs r4, r4, #1
bne LoopL1


End:

pop {r4, pc}




#endif
#endif
