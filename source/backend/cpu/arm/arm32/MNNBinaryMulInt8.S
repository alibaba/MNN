//
//  MNNBinaryMulInt8.S
//  MNN
//
//  Created by MNN on 2019/08/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNBinaryMulInt8
// MNNBinaryMulInt8(int8_t* dst, const int8_t* src0, const int8_t* src1, ssize_t* quantScalesInt32, 
// float* quantScalesFp32, const int8_t* inputOffset0, const int8_t* inputOffset1, const int8_t* outputOffset,,
//  const size_t size, size_t needBroadcast)
// Auto load:
// r0: dst, r1:src0, r2:src1, r3:quantScalesInt32
// Load from sp:
// r4:quantScalesFp32, r5: offset0, r6: offset1, r7: outputoffset
// r8: size, r9: needBroadcast
push {r4, r5, r6, r7, r8, r9, lr}

ldr r4, [sp, #28]
ldr r5, [sp, #32]
ldr r6, [sp, #36]
ldr r7, [sp, #40]
ldr r8, [sp, #44]
ldr r9, [sp, #48]

vpush {q4-q7}

ldr r12, [r4]
vdup.f32 q13, r12     // scale
ldr lr, [r4, #4]
vdup.f32 q14, lr
ldr lr, [r4, #8]
vdup.f32 q15, lr

vld1.8 {d0[0]}, [r5]
vld1.8 {d1[0]}, [r6]
vld1.8 {d4[0]}, [r7]
vdup.8 d0, d0[0]
vdup.8 d1, d1[0]
vdup.8 d4, d4[0]

L4:
cmp r8, #4
blt L1

L4Loop:
    cmp r9, #0
    beq L4NeedBroadcast0
    cmp r9, #1
    beq L4NeedBroadcast1

    L4NotNeedBroadcast:
    vld1.32 {q11}, [r1]!
    vld1.32 {q12}, [r2]!
    b L4Compute

    L4NeedBroadcast0:
    ldr r4, [r1]
    vdup.s8 q11, r4
    vld1.32 {q12}, [r2]!
    b L4Compute

    L4NeedBroadcast1:
    vld1.32 {q11}, [r1]!
    ldr r4, [r2]
    vdup.s8 q12, r4
    b L4Compute

    L4Compute:
    sub r8, r8, #4
    vmovl.s8 q4, d22
    vmovl.s8 q5, d23
    vmovl.s8 q6, d24
    vmovl.s8 q7, d25

    vsubw.s8 q4, q4, d0
    vsubw.s8 q5, q5, d0
    vsubw.s8 q6, q6, d1
    vsubw.s8 q7, q7, d1
    
    vmovl.s16 q8, d8
    vmovl.s16 q9, d9
    vmovl.s16 q10, d10
    vmovl.s16 q11, d11

    vmovl.s16 q3, d12
    vmovl.s16 q12, d13
    vmovl.s16 q1, d14
    vmovl.s16 q4, d15

    vcvt.f32.s32 q8, q8
    vcvt.f32.s32 q9, q9
    vcvt.f32.s32 q10, q10
    vcvt.f32.s32 q11, q11

    vcvt.f32.s32 q3, q3
    vcvt.f32.s32 q12, q12
    vcvt.f32.s32 q1, q1
    vcvt.f32.s32 q4, q4

    vmul.f32 q8, q8, q13
    vmul.f32 q9, q9, q13
    vmul.f32 q10, q10, q13
    vmul.f32 q11, q11, q13

    vmul.f32 q3, q3, q14
    vmul.f32 q12, q12, q14
    vmul.f32 q1, q1, q14
    vmul.f32 q4, q4, q14

    vmul.f32 q8, q8, q3
    vmul.f32 q9, q9, q12
    vmul.f32 q10, q10, q1
    vmul.f32 q11, q11, q4

    vmul.f32 q8, q8, q15
    vmul.f32 q9, q9, q15
    vmul.f32 q10, q10, q15
    vmul.f32 q11, q11, q15

    vcvt.s32.f32 q8, q8
    vcvt.s32.f32 q9, q9
    vcvt.s32.f32 q10, q10
    vcvt.s32.f32 q11, q11

    vqmovn.s32 d6, q8
    vqmovn.s32 d7, q9
    vqmovn.s32 d8, q10
    vqmovn.s32 d9, q11

    vaddw.s8 q3, q3, d4
    vaddw.s8 q4, q4, d4
    
    vqmovn.s16 d12, q3
    vqmovn.s16 d13, q4
    cmp r8, #4
    vst1.32 {q6}, [r0]!
    bge L4Loop

L1:
cmp r8, #0
beq End

L1Loop:
    cmp r9, #0
    beq L1NeedBroadcast0
    cmp r9, #1
    beq L1NeedBroadcast1

    L1NotNeedBroadcast:
    vld1.32 {d6[0]}, [r1]!
    vld1.32 {d8[0]}, [r2]!
    b L1Compute

    L1NeedBroadcast0:
    ldr r4, [r1]
    vdup.s8 d6, r4
    vld1.32 {d8[0]}, [r2]!
    b L1Compute

    L1NeedBroadcast1:
    vld1.32 {d6[0]}, [r1]!
    ldr r4, [r2]
    vdup.s8 d8, r4
    b L1Compute

    L1Compute:
    subs r8, r8, #1
    vmovl.s8 q3, d6
    vsubw.s8 q3, q3, d0
    vmovl.s16 q3, d6
    vcvt.f32.s32 q3, q3
    vmul.f32 q3, q3, q13

    vmovl.s8 q5, d8
    vsubw.s8 q5, q5, d1
    vmovl.s16 q6, d10
    vcvt.f32.s32 q6, q6
    vmul.f32 q6, q6, q14

    vmul.f32 q3, q3, q6
    vmul.f32 q3, q3, q15
    vcvt.s32.f32 q3, q3
    vqmovn.s32 d6, q3
    vaddw.s8 q3, q3, d4
    vqmovn.s16 d6, q3
    vst1.32 {d6[0]}, [r0]!
    bne L1Loop
End:
vpop {q4-q7}
pop {r4, r5, r6, r7, r8, r9, pc}

#endif
#endif

