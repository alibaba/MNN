//
//  MNNBilinearLineC8.s
//  ALL_BUILD
//
//  Created by MNN on 2023/4/12.
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNBilinearLineC8
// void MNNBilinearLineC8(int8_t* dst, const int16_t* A, const int16_t* B, const float* t, int8_t* zeroPoint, size_t number)
// Auto load: r0: dst, r1: A, r2: B, r3: t
// r5: zeroPoint, r4: number
 
push {r4-r8, r10, lr} // avoid to touch platform-register r-9

ldr r5, [sp, #28]
ldr r4, [sp, #32]
ldr r3, [r3, #0]

vpush {q4-q7}
cmp r4, #0
beq END

vmov.s32 q0, #128
vcvt.f32.s32 q0, q0

vmov.f32 q15, #1.0
vdup.f32 q14, r3        // q14: df
vsub.f32 q15, q15, q14  // q15: sf

vmul.f32 q14, q14, d0[0]
vmul.f32 q15, q15, d0[0]
vcvt.s32.f32 q14, q14
vcvt.s32.f32 q15, q15

vqmovn.s32 d28, q14
vqmovn.s32 d29, q15

L1Loop:

vld1.16 {q0}, [r1]!  // A: q0: int16x8_t
vld1.16 {q1}, [r2]! //  B: q1

vmull.s16 q2, d0, d29
vmull.s16 q3, d1, d29
vmlal.s16 q2, d2, d28
vmlal.s16 q3, d3, d28

vshr.s32 q2, q2, #14
vshr.s32 q3, q3, #14

vqmovn.s32 d4, q2
vqmovn.s32 d5, q3

vqmovn.s16 d4, q2

vst1.8 {d4}, [r0]!

sub r4, r4, #1
cmp r4, #1
bge L1Loop

END:
vpop {q4-q7}
pop {r4-r8, r10, pc}

#endif
#endif
