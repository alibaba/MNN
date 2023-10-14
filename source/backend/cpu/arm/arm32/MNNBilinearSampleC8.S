//
//  MNNBilinearSampleC8.s
//  ALL_BUILD
//
//  Created by MNN on 2023/4/12.
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNBilinearSampleC8
// void MNNBilinearSampleC8(const int8_t* src, int16_t* dst, const int32_t* position, const float* factor, int8_t* zeroPoint, size_t number);
// Auto load: r0: src, r1: dst, r2: position, r3: factor
// r12: zeroPoint, r4:  number

push {r4-r8, r10, lr}
ldr r12, [sp, #28]
ldr r4, [sp, #32]
mov lr, #8
vpush {q4-q7}

vmov.s32 q0, #128
vcvt.f32.s32 q0, q0

cmp r4, #0
beq END

L1Loop:
ldr r5, [r2], #4
ldr r6, [r2], #4

mul r5, lr, r5
mul r6, lr, r6

add r7, r5, r0
add r8, r6, r0
vld1.8 {d2}, [r7]    // A: d2: int8x8_t
vld1.8 {d3}, [r8]    // B: d3

ldr r10, [r3], #4
vdup.f32 q14, r10         // q14: df
vmov.f32 q15, #1.0
vsub.f32 q15, q15, q14    // q15: sf

vmul.f32 q14, q14, d0[1]  // float->int8_t
vmul.f32 q15, q15, d0[1]
vcvt.s32.f32 q14, q14
vcvt.s32.f32 q15, q15

vqmovn.s32 d28, q14
vqmovn.s32 d30, q15
vqmovn.s16 d28, q14
vqmovn.s16 d29, q15

vdup.s8 d28, d28[0]
vdup.s8 d29, d29[0]

// A*sf+B*df
vmull.s8 q2, d2, d29     // q2: int16x8_t
vmlal.s8 q2, d3, d28

vst1.16 {q2}, [r1]!

sub r4, r4, #1
cmp r4, #1
bge L1Loop
cmp r4, #0
beq END

END:
vpop {q4-q7}
pop {r4-r8, r10, pc}

#endif
#endif
