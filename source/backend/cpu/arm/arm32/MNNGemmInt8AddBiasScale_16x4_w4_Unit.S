//
//  MNNGemmInt8AddBiasScale_16x4_w4_Unit.S
//  MNN
//
//  Created by MNN on 2019/06/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmInt8AddBiasScale_16x4_w4_Unit
/* 
struct QuanPostTreatParameters {
    const float* scale;
    const float* biasFloat;
    int32_t maxValue;
    int32_t minValue;
    int32_t useInt8 = 1; // Save result as int8_t dataType; otherwise float32.
    float roundValuePos = 0.5f;
    float roundValueNeg = -0.5f;
    float* srcKernelSum;
    float* weightQuanBias;
    float* fp32minmax;
    ssize_t blockNum = 1;
    const int32_t* bias = nullptr;
    const float* extraScale = nullptr;
    const float* extraBias = nullptr;
    float* accumBuffer = nullptr;
};
*/

//void MNNGemmInt8AddBiasScale_16x4_w4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
//                                              size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t real) {

//Auto: r0: dst*, r1: src*, r2:weight*, r3: src_depth_quad
// Load from sp: r4: dst_step, r5: dst_depth_quad, r6: post, r10: real
// Load from post: r8: scale, lr: bias, r7: maxValue, r6: minValue

push {r4-r8, r10, lr} // avoid to touch platform-register r-9

ldr r4, [sp, #28]
ldr r5, [sp, #32]
ldr r6, [sp, #36]
ldr r10, [sp, #40]
ldr lr, [r6, #4]

vpush {q4-q7}
sub sp, sp, #52
/*
[sp, #8]          : bias
[sp, #12]         : fp32minmax
[sp, #16]         : int8 max
[sp, #20]         : int8 min
lr, [sp, #24]     : input scale
[sp, #32]         : accumBuffer
[sp, #36]         : blockNum
[sp, #40]         : input dequant bias
[sp, #28]         : input dequant bias
[sp, #44]         : block idx
r8, [sp, #48]     : input kernel sum
#


 */
str lr, [sp, #8]   // bias
ldr r7, [r6, #32]  // r7: weightKernelSum, if nullptr, dstBytes=1.

ldr r8, [r6, #28] // input kernel sum
ldr r12, [r6, #36]
str r12, [sp, #12] // f32minmax
ldr r12, [r6, #8]
str r12, [sp, #16] // int8 max
ldr r12, [r6, #12]
str r12, [sp, #20] // int8 min
ldr lr, [r6, #48]
str lr, [sp, #24] // input scale
ldr r12, [r6, #56]
str r12, [sp, #32] // accumBuffer
ldr r12, [r6, #40] 
str r12, [sp, #36] // blockNum
ldr r12, [r6, #52]
str r12, [sp, #40] // input dequant bias
str r12, [sp, #28] // input dequant bias
str r8, [sp, #48] // input kernel sum

Start:
cmp r10, #2
blt L1LoopDz

L2LoopDz:
    mov r10, r1
    mov r6, #0
    str r6, [sp, #44] // block idx
L2BLOCKNUM:
    subs r12, r3, #1
    // first four output
    vld1.8 {q2}, [r1]!
    vld1.8 {q4, q5}, [r2]! // weight, d8,d9,d10,d11
    // int4->int8
    vmov.i8 q6, #15
    vmov.i8 q7, #15
    vand.i8 q6, q6, q4
    vand.i8 q7, q7, q5
    vshr.u8 q4, q4, #4
    vshr.u8 q5, q5, #4

    vmull.s8 q0, d4, d8
    vmull.s8 q1, d4, d10
    vmlal.s8 q0, d5, d9
    vmlal.s8 q1, d5, d11
    vpaddl.s16 q8, q0
    vpaddl.s16 q9, q1

    vmull.s8 q0, d4, d12
    vmull.s8 q1, d4, d14
    vmlal.s8 q0, d5, d13
    vmlal.s8 q1, d5, d15
    vpaddl.s16 q10, q0
    vld1.8 {q3}, [r1]!
    vpaddl.s16 q11, q1
    // second four output
    vmull.s8 q0, d6, d8
    vmull.s8 q1, d6, d10
    vmlal.s8 q0, d7, d9
    vmlal.s8 q1, d7, d11
    vpaddl.s16 q12, q0
    vpaddl.s16 q13, q1

    vmull.s8 q0, d6, d12
    vmull.s8 q1, d6, d14
    vmlal.s8 q0, d7, d13
    vmlal.s8 q1, d7, d15
    vpaddl.s16 q14, q0
    vpaddl.s16 q15, q1

    beq L2LoopSzEnd

    L2LoopSz:
        // first four output
        vld1.8 {q2}, [r1]!
        vld1.8 {q4, q5}, [r2]! // weight, d8,d9,d10,d11
        // int4->int8
        vmov.i8 q6, #15
        vmov.i8 q7, #15
        vand.i8 q6, q6, q4
        vand.i8 q7, q7, q5
        vshr.u8 q4, q4, #4
        vshr.u8 q5, q5, #4
        vmull.s8 q0, d4, d8
        vmull.s8 q1, d4, d10
        vmlal.s8 q0, d5, d9
        vmlal.s8 q1, d5, d11
        vpadal.s16 q8, q0
        vpadal.s16 q9, q1

        vmull.s8 q0, d4, d12
        vmull.s8 q1, d4, d14
        vmlal.s8 q0, d5, d13
        vmlal.s8 q1, d5, d15
        vld1.8 {q3}, [r1]!
        vpadal.s16 q10, q0
        vpadal.s16 q11, q1
        // second four output
        vmull.s8 q0, d6, d8
        vmull.s8 q1, d6, d10
        vmlal.s8 q0, d7, d9
        vmlal.s8 q1, d7, d11
        vpadal.s16 q12, q0
        vpadal.s16 q13, q1

        vmull.s8 q0, d6, d12
        vmull.s8 q1, d6, d14
        vmlal.s8 q0, d7, d13
        vmlal.s8 q1, d7, d15
        vpadal.s16 q14, q0
        vpadal.s16 q15, q1

        subs r12, r12, #1
        bne L2LoopSz

    L2LoopSzEnd:

    L2Quan:
    vld1.f32 {q5}, [r2]! // scale

    vpadd.s32 d16, d16, d17
    vpadd.s32 d20, d20, d21
    vpadd.s32 d18, d18, d19
    vpadd.s32 d22, d22, d23

    vpadd.s32 d24, d24, d25
    vpadd.s32 d28, d28, d29
    vpadd.s32 d26, d26, d27
    vpadd.s32 d30, d30, d31
    
    vpadd.s32 d16, d16, d18
    vpadd.s32 d17, d20, d22
    vpadd.s32 d18, d24, d26
    vpadd.s32 d19, d28, d30

    vcvt.f32.s32 q0, q8
    vcvt.f32.s32 q1, q9

    vmulq.f32 q0, q0, q5 // mul scale
    vmulq.f32 q1, q1, q5

    // input scale if has
    cmp lr, #0
    beq L2_MLA
    vld1.f32 {d10[0]}, [lr]! // tile0
    vld1.f32 {d10[1]}, [lr]  // tile1
    vmulq.f32 q0, q0, d10[0]
    vmulq.f32 q1, q1, d10[1]
    sub lr, lr, #4

    L2_MLA:
    // input kernel sum
    vld1.f32 {d12[0]}, [r8]! // tile 0
    vld1.f32 {d12[1]}, [r8]! // tile 1
    vld1.f32 {q7}, [r2]! // weight bias

    vmla.f32 q0, q7, d12[0]
    vmla.f32 q1, q7, d12[1]

    ldr r6, [sp, #40]         // input bias
    cmp r6, #0
    beq L2_ADD_DSTV
    // input bias * weight kernel sum
    vld1.f32 {d12[0]}, [r6]!  // tile 0
    vld1.f32 {d12[1]}, [r6]!   // tile 1
    str r6, [sp, #40]
    vld1.f32 {q7}, [r7]!
    vmla.f32 q0, q7, d12[0]
    vmla.f32 q1, q7, d12[1]
    // update input scale
    add lr, lr, #8

    L2_ADD_DSTV:
    ldr r6, [sp, #44] // block idx
    cmp r6, #0
    beq L2_BUFFER
    ldr r6, [sp, #32] // accumBuffer
    vld1.f32 {q4, q5}, [r6]
    vadd.f32 q0, q0, q4
    vadd.f32 q1, q1, q5

    L2_BUFFER:
    ldr r6, [sp, #44] // block idx
    ldr r12, [sp, #36] // block num
    add r6, r6, #1
    cmp r6, r12
    beq L2_POST
    str r6, [sp, #44] // block idx
    ldr r6, [sp, #32] // accumBuffer
    vst1.f32 {q0, q1}, [r6]
    b L2BLOCKNUM

    L2_POST:
    ldr r12, [sp, #8] // load bias
    cmp r12, #0
    beq L2_RELU
    vld1.f32 {q4}, [r12]!
    vadd.f32 q0, q0, q4
    vadd.f32 q1, q1, q4
    str r12, [sp, #8] // save bias

    L2_RELU:
    ldr r6, [sp, #12] // fp32 minmax
    cmp r6, #0
    beq L2_STORE
    vld1.f32 {d20[0]}, [r6]!
    vld1.f32 {d22[0]}, [r6]
    vdup.f32 q10, d20[0]
    vdup.f32 q11, d22[0]
    vmax.f32 q0, q0, q10
    vmax.f32 q1, q1, q10
    vmin.f32 q0, q0, q11
    vmin.f32 q1, q1, q11

    L2_STORE:
    vst1.f32 {q0, q1}, [r0], r4

L2LoopCheck:
    subs r5, r5, #1
    mov r1, r10
    ldr lr, [sp, #24] // revert input scale
    ldr r8, [sp, #48] // revert input kernel sum
    ldr r12, [sp, #28] // load input bias
    str r12, [sp, #40] // revert input bias

    bne L2LoopDz

b End

L1LoopDz:
    mov r10, r1
    mov r6, #0
    str r6, [sp, #44] // block idx
L1BLOCKNUM:
    subs r12, r3, #1
    // first four output
    vld1.8 {q2}, [r1]!
    vld1.8 {q4, q5}, [r2]! // weight, d8,d9,d10,d11
    // int4->int8
    vmov.i8 q6, #15
    vmov.i8 q7, #15
    vand.i8 q6, q6, q4
    vand.i8 q7, q7, q5
    vshr.u8 q4, q4, #4
    vshr.u8 q5, q5, #4
    
    vmull.s8 q0, d4, d8
    vmull.s8 q1, d4, d10
    vmlal.s8 q0, d5, d9
    vmlal.s8 q1, d5, d11
    vpaddl.s16 q8, q0
    vpaddl.s16 q9, q1

    vmull.s8 q0, d4, d12
    vmull.s8 q1, d4, d14
    vmlal.s8 q0, d5, d13
    vmlal.s8 q1, d5, d15
    vpaddl.s16 q10, q0
    vpaddl.s16 q11, q1

    beq L1LoopSzEnd

    L1LoopSz:
        // first four output
        vld1.8 {q2}, [r1]!
        vld1.8 {q4, q5}, [r2]! // weight, d8,d9,d10,d11
        // int4->int8
        vmov.i8 q6, #15
        vmov.i8 q7, #15
        vand.i8 q6, q6, q4
        vand.i8 q7, q7, q5
        vshr.u8 q4, q4, #4
        vshr.u8 q5, q5, #4
        vmull.s8 q0, d4, d8
        vmull.s8 q1, d4, d10
        vmlal.s8 q0, d5, d9
        vmlal.s8 q1, d5, d11
        vpadal.s16 q8, q0
        vpadal.s16 q9, q1

        vmull.s8 q0, d4, d12
        vmull.s8 q1, d4, d14
        vmlal.s8 q0, d5, d13
        vmlal.s8 q1, d5, d15

        vpadal.s16 q10, q0
        vpadal.s16 q11, q1

        subs r12, r12, #1
        bne L1LoopSz

    L1LoopSzEnd:
    L1Quan:
    vld1.f32 {q5}, [r2]! // scale

    vpadd.s32 d16, d16, d17
    vpadd.s32 d20, d20, d21
    vpadd.s32 d18, d18, d19
    vpadd.s32 d22, d22, d23

    vpadd.s32 d16, d16, d18
    vpadd.s32 d17, d20, d22

    vcvt.f32.s32 q0, q8
    vmulq.f32 q0, q0, q5
    // extra scale if has
    cmp lr, #0
    beq L1_MLA
    vld1.f32 {d10[0]}, [lr] // tile0
    vmulq.f32 q0, q0, d10[0]

    L1_MLA:
    // input kernel sum
    vld1.f32 {d12[0]}, [r8]! // tile 0
    vld1.f32 {q7}, [r2]!
    vmla.f32 q0, q7, d12[0]

    ldr r6, [sp, #40]         // input bias
    cmp r6, #0
    beq L1_ADD_DSTV
    vld1.f32 {d12[0]}, [r6]!  // tile 0
    str r6, [sp, #40]
    vld1.f32 {q7}, [r7]!
    vmla.f32 q0, q7, d12[0]
    add lr, lr, #4

    L1_ADD_DSTV:
    ldr r6, [sp, #44] // block idx
    cmp r6, #0
    beq L1_BUFFER
    ldr r6, [sp, #32] // accumBuffer
    vld1.f32 {q4}, [r6]
    vadd.f32 q0, q0, q4

    L1_BUFFER:
    ldr r6, [sp, #44] // block idx
    ldr r12, [sp, #36] // block num
    add r6, r6, #1
    cmp r6, r12
    beq L1_POST
    str r6, [sp, #44]
    ldr r6, [sp, #32] // accumBuffer
    vst1.f32 {q0}, [r6]
    b L1BLOCKNUM

    L1_POST:
    ldr r12, [sp, #8] // load bias
    cmp r12, #0
    beq L1_RELU
    vld1.f32 {q4}, [r12]! // bias
    vadd.f32 q0, q0, q4
    str r12, [sp, #8] // save bias

    L1_RELU:
    ldr r6, [sp, #12] // fp32 minmax
    cmp r6, #0
    beq L1_STORE
    vld1.f32 {d20[0]}, [r6]!
    vld1.f32 {d22[0]}, [r6]
    vdup.f32 q10, d20[0]
    vdup.f32 q11, d22[0]
    vmax.f32 q0, q0, q10
    vmin.f32 q0, q0, q11
    L1_STORE:
    vst1.f32 {q0}, [r0], r4

L1LoopCheck:
    subs r5, r5, #1
    mov r1, r10
    ldr lr, [sp, #24] // revert input scale
    ldr r8, [sp, #48] // revert input kernel sum
    ldr r12, [sp, #28] // load input bias
    str r12, [sp, #40] // revert input bias
    bne L1LoopDz

End:
add sp, sp, #52
vpop {q4-q7}
pop {r4-r8, r10, pc}

#endif
#endif
