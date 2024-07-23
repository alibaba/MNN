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
    const int32_t* bias;
    const float* extraScale = nullptr;
    const float* extraBias = nullptr;
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
ldr r8, [r6, #0]
ldr lr, [r6, #4]

vpush {q4-q7}

// Branch1: input is int8_t, output is float32, DO NOT USE "scale".
// Branch2: input is int8_t, output is float32. USE "scale", DO NOT USE "minValue" and "maxValue".
// Branch3: input is int8_t, output is int8_t.  USE "scale", "minValue" and "maxValue".


ldr r7, [r6, #16]  // r7: useInt8

ldr r12, [r6, #28] // srcKernelSum
str r12, [sp, #4]
ldr r12, [r6, #32] // weightBias
str r12, [sp, #8]
ldr r12, [r6, #36] // f32minmax
str r12, [sp, #12]
ldr r12, [r6, #40] // blockNum
mul r12, r12, r3   // src_depth_quad=src_depth_quad*blockNum
lsl r12, r12, #6   // weight_stride = src_depth_quad*LP*HP
str r12, [sp, #16]
ldr r12, [r6, #48] // extraScale
str r12, [sp, #20]

Start:
cmp r10, #2
blt L1LoopDz

L2LoopDz:
    mov r10, r1
    str r2, [sp, #24] // store weight ptr
    subs r12, r3, #1
    // first four output
    vld1.8 {q2}, [r1]!
    vld1.8 {q4}, [r2]! // weight, d8,d9,d10,d11
    // int4->int8
    vmov.i8 q5, #15
    vand.i8 q5, q5, q4
    vshr.u8 q4, q4, #4
    vzip.8 q4, q5

    vmull.s8 q0, d4, d8
    vmull.s8 q1, d4, d10
    vmlal.s8 q0, d5, d9
    vmlal.s8 q1, d5, d11
    vpaddl.s16 q8, q0
    vpaddl.s16 q9, q1
    vld1.8 {q6}, [r2]! // weight,d12,d13,d14,d15
    // int4->int8
    vmov.i8 q7, #15
    vand.i8 q7, q7, q6
    vshr.u8 q6, q6, #4
    vzip.8 q6, q7

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
        vld1.8 {q4}, [r2]!
        // int4->int8
        vmov.i8 q5, #15
        vand.i8 q5, q5, q4
        vshr.u8 q4, q4, #4
        vzip.8 q4, q5
        vmull.s8 q0, d4, d8
        vmull.s8 q1, d4, d10
        vmlal.s8 q0, d5, d9
        vmlal.s8 q1, d5, d11
        vld1.8 {q6}, [r2]!
        // int4->int8
        vmov.i8 q7, #15
        vand.i8 q7, q7, q6
        vshr.u8 q6, q6, #4
        vzip.8 q6, q7
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
    vld1.f32 {q5}, [r8]! // scale

    vpadd.s32 d16, d16, d17
    vpadd.s32 d20, d20, d21
    vpadd.s32 d18, d18, d19
    vpadd.s32 d22, d22, d23

    vpadd.s32 d24, d24, d25
    vpadd.s32 d28, d28, d29
    vpadd.s32 d26, d26, d27
    vpadd.s32 d30, d30, d31

    // q8,q9
    
    vpadd.s32 d16, d16, d18
    vpadd.s32 d17, d20, d22
    vpadd.s32 d18, d24, d26
    vpadd.s32 d19, d28, d30

    // vaddq.s32 q0, q8, q4 // add bias
    // vaddq.s32 q1, q9, q4

    vcvt.f32.s32 q0, q0
    vcvt.f32.s32 q1, q1

    vmulq.f32 q0, q0, q5 // mul scale
    vmulq.f32 q1, q1, q5

    // extra scale if has
    ldr r6, [sp, #20]
    cmp r6, #0
    beq L2_MLA
    vld1.f32 {d10[0]}, [r6]! // tile0
    vld1.f32 {d10[1]}, [r6]  // tile1
    vmulq.f32 q0, q0, d10[0]
    vmulq.f32 q1, q1, d10[1]

    L2_MLA:
    ldr r6, [sp, #4] // srcKernelSum
    vld1.f32 {d12[0]}, [r6]! // tile 0
    vld1.f32 {d12[1]}, [r6] // tile 1
    ldr r6, [sp, #8] // weightBias
    vld1.f32 {q7}, [r6]!
    str r6, [sp, #8] // update next 4 weightBias

    vmla.f32 q0, q7, d12[0]
    vmla.f32 q1, q7, d12[1]

    L2_POST:
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
    ldr r2, [sp, #24] // origin weight ptr
    ldr r6, [sp, #16] // weight stride
    add r2, r2, r6    // next oc4 weight ptr
    bne L2LoopDz

b End

L1LoopDz:
    mov r10, r1
    str r2, [sp, #24] // store weight ptr
    subs r12, r3, #1
    // first four output
    vld1.8 {q2}, [r1]!
    vld1.8 {q4}, [r2]!
    // int4->int8
    vmov.i8 q5, #15
    vand.i8 q5, q5, q4
    vshr.u8 q4, q4, #4
    vzip.8 q4, q5
    
    vmull.s8 q0, d4, d8
    vmull.s8 q1, d4, d10
    vmlal.s8 q0, d5, d9
    vmlal.s8 q1, d5, d11
    vpaddl.s16 q8, q0
    vpaddl.s16 q9, q1
    vld1.8 {q6}, [r2]!
    // int4->int8
    vmov.i8 q7, #15
    vand.i8 q7, q7, q6
    vshr.u8 q6, q6, #4
    vzip.8 q6, q7

    vmull.s8 q0, d4, d12
    vmull.s8 q1, d4, d14
    vmlal.s8 q0, d5, d13
    vmlal.s8 q1, d5, d15
    vpaddl.s16 q10, q0
    add r1, r1, #16
    vpaddl.s16 q11, q1

    beq L1LoopSzEnd

    L1LoopSz:
        // first four output
        vld1.8 {q2}, [r1]!
        vld1.8 {q4}, [r2]!
        // int4->int8
        vmov.i8 q5, #15
        vand.i8 q5, q5, q4
        vshr.u8 q4, q4, #4
        vzip.8 q4, q5
        vmull.s8 q0, d4, d8
        vmull.s8 q1, d4, d10
        vmlal.s8 q0, d5, d9
        vmlal.s8 q1, d5, d11
        vld1.8 {q6}, [r2]!
        // int4->int8
        vmov.i8 q7, #15
        vand.i8 q7, q7, q6
        vshr.u8 q6, q6, #4
        vzip.8 q6, q7
        vpadal.s16 q8, q0
        vpadal.s16 q9, q1

        vmull.s8 q0, d4, d12
        vmull.s8 q1, d4, d14
        vmlal.s8 q0, d5, d13
        vmlal.s8 q1, d5, d15
        add r1, r1, #16
        vpadal.s16 q10, q0
        vpadal.s16 q11, q1

        subs r12, r12, #1
        bne L1LoopSz

    L1LoopSzEnd:
    L1Quan:
    //vld1.f32 {q4}, [lr]! // bias
    vld1.f32 {q5}, [r8]! // scale

    vpadd.s32 d16, d16, d17
    vpadd.s32 d20, d20, d21
    vpadd.s32 d18, d18, d19
    vpadd.s32 d22, d22, d23

    // q8
    vpadd.s32 d16, d16, d18
    vpadd.s32 d17, d20, d22

    // vaddq.s32 q0, q8, q4
    vcvt.f32.s32 q0, q0
    vmulq.f32 q0, q0, q5
    // extra scale if has
    ldr r6, [sp, #20]
    cmp r6, #0
    beq L1_MLA
    vld1.f32 {d10[0]}, [r6] // tile0
    vmulq.f32 q0, q0, d10[0]

    L1_MLA:
    ldr r6, [sp, #4] // srcKernelSum
    vld1.f32 {d12[0]}, [r6] // tile 0
    ldr r6, [sp, #8] // weightBias
    vld1.f32 {q7}, [r6]!
    str r6, [sp, #8] // update next 4 weightBias
    vmla.f32 q0, q7, d12[0]
    //vadd.f32 q0, q0, q4

    cmp lr, #0
    beq L1_ADD_DSTV
    vld1.f32 {q4}, [lr]! // bias
    vadd.f32 q0, q0, q4
    b L1_POST

    L1_ADD_DSTV:
    vld1.f32 {q4}, [r0]
    vadd.f32 q0, q0, q4

    L1_POST:
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
    ldr r2, [sp, #24] // origin weight ptr
    ldr r6, [sp, #16] // weight stride
    add r2, r2, r6    // next oc4 weight ptr
    bne L1LoopDz

End:
vpop {q4-q7}
pop {r4-r8, r10, pc}

#endif
#endif
