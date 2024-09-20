//
//  MNNC3ToYUVFast.S
//  MNN
//
//  Created by MNN on 2024/08/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNC3ToYUVFast
// void MNNC3ToYUVFast(const unsigned char* source, unsigned char* dest, size_t count, int32_t* c);
// Auto Load: r0: source, r1: dest, r2: count, r3: c

push {lr}
vpush {q4-q7}

// q4-q6, const
vld1.32 {d8[0]}, [r3]! // C0
vld1.32 {d8[1]}, [r3]! // C1
vld1.32 {d9[0]}, [r3]! // C2
vld1.32 {d9[1]}, [r3]! // C3
vld1.32 {d10[0]}, [r3]! // C4
vld1.32 {d10[1]}, [r3]! // C5
vld1.32 {d11[0]}, [r3]! // C6
vld1.32 {d11[1]}, [r3]! // C7
vld1.32 {d12[0]}, [r3]! // C8

vmov.u16 q15, #128

L1:
vld3.8 {d0, d1, d2}, [r0]!
vmovl.u8 q2, d0 // r: uint8_t -> uint16_t
vmovl.u8 q3, d1
vmovl.u8 q13, d2

vmovl.u16 q7, d4 // r
vmovl.u16 q8, d5 // r
vmovl.u16 q9, d6 // g
vmovl.u16 q10, d7 // g
vmovl.u16 q11, d26 // b
vmovl.u16 q12, d27 // b

// r*C0, g*C1, b*C2
vmul.s32 q0, q7, d8[0]
vmul.s32 q1, q8, d8[0]
vmla.s32 q0, q9, d8[1]
vmla.s32 q1, q10, d8[1]
vmla.s32 q0, q11, d9[0]
vmla.s32 q1, q12, d9[0]

// r*C3, g*C4, b*C5
vmul.s32 q2, q7, d9[1]
vmul.s32 q3, q8, d9[1]
vmla.s32 q2, q9, d10[0]
vmla.s32 q3, q10, d10[0]
vmla.s32 q2, q11, d10[1]
vmla.s32 q3, q12, d10[1]

// r*C6, g*C7, b*C8
vmul.s32 q13, q7, d11[0]
vmul.s32 q14, q8, d11[0]
vmla.s32 q13, q9, d11[1]
vmla.s32 q14, q10, d11[1]
vmla.s32 q13, q11, d12[0]
vmla.s32 q14, q12, d12[0]

vrshrn.u32 d0, q0, #14
vrshrn.u32 d1, q1, #14
vrshrn.u32 d2, q2, #14
vrshrn.u32 d3, q3, #14
vrshrn.u32 d4, q13, #14
vrshrn.u32 d5, q14, #14

vadd.u16 q1, q1, q15
vadd.u16 q2, q2, q15

vqmovn.u16 d0, q0
vqmovn.u16 d1, q1
vqmovn.u16 d2, q2

vst3.8 {d0, d1, d2}, [r1]!

subs r2, r2, #1
bne L1

End:
vpop {q4-q7}
pop {pc}

#endif
#endif
