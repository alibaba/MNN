//
//  MNNBlitC1ToFloatRGBA.S
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

asm_function MNNBlitC1ToFloatRGBA
//        void MNNBlitC1ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count)

//Auto: r0: source, r1: dest, r2: mean, r3: normal

//Load from sp:
//r4: count

push {r4, lr}
ldr r4, [sp, #8]

//Mean
vld1.32 {d0[0]}, [r2]
vdup.i32 q14, d0[0]

//Normal
vld1.32 {d1[0]}, [r3]
vdup.i32 q15, d1[0]


L8:
cmp r4, #8
blt L1


LoopL8:

vld1.8 {d0}, [r0]!
vmovl.u8 q0, d0
vmovl.u16 q1, d1
vmovl.u16 q0, d0
vcvt.f32.u32 q1, q1
vcvt.f32.u32 q0, q0
vsub.f32 q1, q1, q14
vsub.f32 q0, q0, q14
vmul.f32 q1, q1, q15
vmul.f32 q0, q0, q15
vdup.32 q8, d0[0]
vdup.32 q9, d0[1]
vdup.32 q10, d1[0]
vst1.32 {q8, q9}, [r1]!
vdup.32 q11, d1[1]

vdup.32 q8, d2[0]
vst1.32 {q10, q11}, [r1]!
vdup.32 q9, d2[1]
vdup.32 q10, d3[0]
vst1.32 {q8, q9}, [r1]!
vdup.32 q11, d3[1]
vst1.32 {q10, q11}, [r1]!

sub r4, r4, #8
cmp r4, #8
bge LoopL8


L1:
cmp r4, #0
beq End

LoopL1:
vld1.8 {d0[0]}, [r0]!

vmovl.u8 q0, d0
vmovl.u16 q0, d0
vdup.32 q0, d0[0]
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
