//
//  MNNExpC8.S
//  MNN
//
//  Created by MNN on 2019/01/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5
// void MNNPowC8(float* dest, const float* source, const float* powfParam, float powfConstant, size_t betaInt, size_t countC8)
asm_function MNNPowC8

//Auto: r0:dest, r1:source, r2: powfParam, r3: betaInt
//Load from sp: r4:countC8

push {r4, lr}
ldr r4, [sp, #8]
vpush {q4, q5, q6}

vld1.32 {q4, q5}, [r2]

vmov.f32 s23, #2.0
vmov.f32 s24, #3.0
vdiv.f32 s23, s23, s24  // s23 = 2.0/3.0
vmov.f32 s24, #1.25
vmov.f32 q13, #1.0

Loop:
// (q0, q1): src, (q2, q3): dst, (q9, q10): 1 / src, (r2 or r4): betaInt
vld1.32 {q0, q1}, [r1]!
vmov.f32 q2, #1.0
vmov.f32 q3, #1.0

cmp r3, #0
beq endSubLoop1

vrecpe.f32 q9, q0
vrecpe.f32 q10, q1
mov r2, r3

SubLoop1:
vmul.f32 q2, q2, q9
vmul.f32 q3, q3, q10
subs r2, r2, #1
bne SubLoop1

endSubLoop1:

.macro SUB_WORK loopDef z0 z1
// z0: x, z1: result
vcmp.f32 \z0, s24
vmrs APSR_nzcv, FPSCR
blt end\loopDef
\loopDef:
vmul.f32 \z0, \z0, s23 // x = x / 1.5
vmul.f32 \z1, \z1, s22 // result = result * powfConstant
vcmp.f32 \z0, s24
vmrs APSR_nzcv, FPSCR
bge \loopDef
end\loopDef:
.endm

SUB_WORK subLoop0, s0, s8
SUB_WORK subLoop1, s4, s12
SUB_WORK subLoop2, s1, s9
SUB_WORK subLoop3, s5, s13
SUB_WORK subLoop4, s2, s10
SUB_WORK subLoop5, s6, s14
SUB_WORK subLoop6, s3, s11
SUB_WORK subLoop7, s7, s15

vsub.f32 q0, q0, q13
vsub.f32 q1, q1, q13

.macro MLA_TWO z0 z1 z2 z3
vdup.32 \z1, \z0
vmla.f32 \z1, \z2, \z3
.endm

MLA_TWO d10[0], q9, q0, d10[1]
MLA_TWO d10[0], q10, q1, d10[1]
MLA_TWO d9[1], q11, q0, q9
MLA_TWO d9[1], q12, q1, q10
MLA_TWO d9[0], q9, q0, q11
MLA_TWO d9[0], q10, q1, q12
MLA_TWO d8[1], q11, q0, q9
MLA_TWO d8[1], q12, q1, q10
MLA_TWO d8[0], q9, q0, q11
MLA_TWO d8[0], q10, q1, q12

vmul.f32 q2, q2, q9
vmul.f32 q3, q3, q10
vst1.32 {q2, q3}, [r0]!

subs r4, r4, #1

bne Loop

vpop {q4, q5, q6}
pop {r4, pc}



#endif
#endif
