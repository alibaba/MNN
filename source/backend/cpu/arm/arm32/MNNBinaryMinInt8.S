//
//  MNNBinaryMinInt8.S
//  MNN
//
//  Created by MNN on 2019/08/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/*
struct QuanPrePostParameters{
    float* inputScale;
    float* outputScale;
    ssize_t* inputZeroPoint;
    ssize_t* outputZeroPoint;
    ssize_t minValue;
    ssize_t maxValue;
};
 */

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNBinaryMinInt8
// MNNBinaryMinInt8(int8_t* outputRaw, const int8_t* inputRaw0, const int8_t* inputRaw1, ssize_t* inputScalesInt32, 
// float* inputScalesFp32, const QuanPrePostParameters* params, size_t elementSize, size_t needBroadcast);
// Auto load:
// r0: dst, r1:src0, r2:src1, r3:quantScalesInt32
// Load from sp:
// r4:quantScalesFp32, r5: params, r6: size, r7: needBroadcast
push {r4-r8, r10, r11, lr}

ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]
ldr r7, [sp, #44]

vpush {q4-q7}

ldr r12, [r3]
vdup.s32 q13, r12     // scale
ldr r4, [r3, #4]
vdup.s32 q14, r4

ldr lr, [r5, #8]
ldr r8, [lr, #0]
ldr r10,[lr, #4]
ldr r3, [r5, #16] // minValue
ldr r11, [r5, #20] // maxValue
ldr r5, [r5, #12]
ldr r5, [r5, #0]

vdup.s8 d0, r8
vdup.s8 d1, r10
vdup.s8 d4, r5

L4:
cmp r6, #4
blt L1

L4Loop:
    cmp r7, #0
    beq L4NeedBroadcast0
    cmp r7, #1
    beq L4NeedBroadcast1

    L4NotNeedBroadcast:
    vld1.32 {q11}, [r1]!
    vld1.32 {q12}, [r2]!
    b L4Compute

    L4NeedBroadcast0:
    vld1.8 {d22[0]}, [r1]
    vdup.s8 q11, d22[0]
    vld1.32 {q12}, [r2]!
    b L4Compute

    L4NeedBroadcast1:
    vld1.32 {q11}, [r1]!
    vld1.8 {d24[0]}, [r2]
    vdup.s8 q12, d24[0]
    b L4Compute

    L4Compute:
    sub r6, r6, #4
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
    vmovl.s16 q15, d14
    vmovl.s16 q4, d15

    vmulq.s32 q8, q8, q13
    vmulq.s32 q9, q9, q13
    vmulq.s32 q10, q10, q13
    vmulq.s32 q11, q11, q13

    vmulq.s32 q3, q3, q14
    vmulq.s32 q12, q12, q14
    vmulq.s32 q15, q15, q14
    vmulq.s32 q4, q4, q14

    vmin.s32 q8, q8, q3
    vmin.s32 q9, q9, q12
    vmin.s32 q10, q10, q15
    vmin.s32 q11, q11, q4
    vdup.8 q12, r3
    vdup.8 q15, r11
    vqshrn.s32 d6, q8, #16
    vqshrn.s32 d7, q9, #16
    vqshrn.s32 d8, q10, #16
    vqshrn.s32 d9, q11, #16

    vaddw.s8 q3, q3, d4
    vaddw.s8 q4, q4, d4
    
    vqmovn.s16 d12, q3
    vqmovn.s16 d13, q4
    vmax.s8 q6, q6, q12
    vmin.s8 q6, q6, q15
    cmp r6, #4
    vst1.32 {q6}, [r0]!
    bge L4Loop

L1:
cmp r6, #0
beq End
vdup.8 d20, r3
vdup.8 d22, r11
L1Loop:
    cmp r7, #0
    beq L1NeedBroadcast0
    cmp r7, #1
    beq L1NeedBroadcast1

    L1NotNeedBroadcast:
    vld1.32 {d6[0]}, [r1]!
    vld1.32 {d8[0]}, [r2]!
    b L1Compute

    L1NeedBroadcast0:
    vld1.8 {d6[0]}, [r1]
    vdup.s8 d6, d6[0]
    vld1.32 {d8[0]}, [r2]!
    b L1Compute

    L1NeedBroadcast1:
    vld1.32 {d6[0]}, [r1]!
    vld1.8 {d8[0]}, [r2]
    vdup.s8 d8, d8[0]
    b L1Compute

    L1Compute:
    subs r6, r6, #1
    vmovl.s8 q3, d6
    vsubw.s8 q3, q3, d0
    vmovl.s16 q3, d6
    vmulq.s32 q3, q3, q13

    vmovl.s8 q5, d8
    vsubw.s8 q5, q5, d1
    vmovl.s16 q6, d10
    vmulq.s32 q6, q6, q14

    vmin.s32 q3, q3, q6
    vqshrn.s32 d6, q3, #16
    vaddw.s8 q3, q3, d4
    vqmovn.s16 d6, q3
    vmax.s8 d6, d6, d20
    vmin.s8 d6, d6, d22
    vst1.32 {d6[0]}, [r0]!
    bne L1Loop
End:
vpop {q4-q7}
pop {r4-r8, r10, r11, pc}

#endif
#endif

