//
//  MNNNV21ToBGRAUnit.S
//  MNN
//
//  Created by MNN on 2018/12/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
//void MNNNV21ToBGRAUnit(const unsigned char* source, unsigned char* dest, size_t count, const unsigned char* uv);
//Auto: r0:source, r1:dest, r2:count, r3:uv
asm_function MNNNV21ToBGRAUnit
push {r4, lr}
vpush {q4}

mov r4, #73
vmov.i8 d31, #128
mov r12, #25
vmov.i16 d0[0], r4
vmov.i16 d0[1], r12
mov r4, #37
mov r12, #130
vmov.i16 d0[2], r4
vmov.i16 d0[3], r12
vmov.i16 q4, #0


LoopL1:
vld2.8 {d18, d19}, [r3]!

vsubl.u8 q11, d18, d31
vld2.8 {d16, d17}, [r0]!
vsubl.u8 q10, d19, d31

//q1-q3: BGR offset
vmul.s16 q1, q10, d0[3]// + B Offset
vmul.s16 q2, q10, d0[1]
vmla.s16 q2, q11, d0[2]// + G Offset
vmul.s16 q3, q11, d0[0]// - R Offset

vshll.u8 q10, d16, #6
vshll.u8 q11, d17, #6

vadd.s16 q12, q10, q1
vsub.s16 q13, q10, q2
vadd.s16 q14, q10, q3

vmax.s16 q12, q12, q4
vmax.s16 q13, q13, q4
vmax.s16 q14, q14, q4

vqshrn.u16 d16, q12, #6
vqshrn.u16 d17, q13, #6
vqshrn.u16 d18, q14, #6

vadd.s16 q12, q11, q1
vsub.s16 q13, q11, q2
vadd.s16 q14, q11, q3

vmax.s16 q12, q12, q4
vmax.s16 q13, q13, q4
vmax.s16 q14, q14, q4

vqshrn.u16 d20, q12, #6
vqshrn.u16 d21, q13, #6
vqshrn.u16 d22, q14, #6

vzip.8 d16, d20
vmov.i8 d19, #255
vzip.8 d17, d21
vzip.8 d18, d22
vmov.i8 d23, #255


vst4.u8 {d16, d17, d18, d19}, [r1]!

vst4.u8 {d20, d21, d22, d23}, [r1]!


subs r2, r2, #1
bne LoopL1

vpop {q4}
pop {r4, pc}

#endif
#endif
