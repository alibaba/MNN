//
//  MNNTMacCompute.S
//  MNN
//
//  Created by MNN on 2025/02/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
/*
struct PlaneInfo {
    int planeSize;
    float offset;
    float dequantscale;
    float minValue;
    float maxValue;
    int ocDiv;
    const uint8_t* mWeightPtr;
    const uint8_t* mWeightScalePtr;
    const uint8_t* mBiasPtr;
};
struct TMacResource {
    int mBits;
    int mBlockSizeC4;
    int mBlockNumber;
    int mOutputCount;
    int mHp;
    std::shared_ptr<MNN::Tensor> mWeightInt8;
    std::shared_ptr<MNN::Tensor> mOriginScale;
    std::shared_ptr<MNN::Tensor> mOriginBias;
};
*/
asm_function MNNTMacCompute
// void MNNTMacCompute(float* dst, const int8_t* table, const float* inputSum, const TMacResource* res, const PlaneInfo* plane) {

//Auto Load:
//r0:dst, r1:table, r2:inputSum, r3:res

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9

//Load From Sp
//r4: plane
ldr r4, [sp, #32]

vpush {q4-q7}
sub sp, sp, #40

// q15: offset
/*
[sp, #16]: oc4
[sp, #20]: min
[sp, #24]: max
[sp, #28]: dequantscale
 */
ldr lr, [r4, #4]
vdup.f32 q15, lr
ldr lr, [r4, #20]
str lr, [sp, #16]
ldr lr, [r4, #12]
str lr, [sp, #20]
ldr lr, [r4, #16]
str lr, [sp, #24]
ldr lr, [r4, #8] // dequantscale
str lr, [sp, #28]

ldr r5, [r4, #24] // weight
ldr r6, [r4, #28] // scale
ldr r7, [r4, #32] // bias
ldr r4, [r4, #0]
mov lr, #16
mul r4, r4, lr    // dstStride=plansize*4*sizeof(float)

/*
[sp, #4]: bits
[sp, #8]: blc4
[sp, #12]: blocknumber
 */
ldr r8, [r3, #0]
str r8, [sp, #4]
ldr r11, [r3, #4]
str r11, [sp, #8]
ldr r10, [r3, #8]
str r10, [sp, #12]
str r2, [sp, #32]

vmov.i8 d18, #15

LoopDz:
    vld1.f32 {q0, q1}, [r7]!
    vld1.f32 {q2, q3}, [r7]!
    mov r12, r1

    ldr r10, [sp, #12]    // blocknumber
    ldr r2, [sp, #32]     // inputsum
    LoopBlock:
        ldr r8, [sp, #4]  // Load Bits
        ldr r11, [sp, #8] // Load blc4
        FirstBits:
            mov r3, r12
            vld1.f32 {q4}, [r3]!
            vld1.f32 {d10}, [r5]!
            vshr.u8 d11, d10, #4
            vand.u8 d10, d10, d18
            vtbl.8 d10, {d8, d9}, d10
            vtbl.8 d11, {d8, d9}, d11
            vmovl.s8 q7, d10
            vmovl.s8 q8, d11
            sub r11, r11, #1
            cmp r11, #0
            beq LoopInBlockFirstEnd

            LoopInBlockFirst:
                vld1.f32 {q4}, [r3]!
                vld1.f32 {d10}, [r5]!
                vshr.u8 d11, d10, #4
                vand.u8 d10, d10, d18
                vtbl.8 d10, {d8, d9}, d10
                vtbl.8 d11, {d8, d9}, d11
                vaddw.s8 q7, q7, d10
                vaddw.s8 q8, q8, d11
                sub r11, r11, #1
                cmp r11, #0
                bgt LoopInBlockFirst

        LoopInBlockFirstEnd:
        // int16_t -> int32_t
        vmovl.s16 q11, d14
        vmovl.s16 q12, d15
        vmovl.s16 q13, d16
        vmovl.s16 q14, d17
        sub r8, r8, #1
        cmp r8, #0
        beq EndBit

        LoopBits:
            vqshl.s32 q11, q11, #1
            vqshl.s32 q12, q12, #1
            vqshl.s32 q13, q13, #1
            vqshl.s32 q14, q14, #1
            
            mov r3, r12
            ldr r11, [sp, #8] // Load blc4
            sub r11, r11, #1
            vld1.f32 {q4}, [r3]!
            vld1.f32 {d10}, [r5]!
            vshr.u8 d11, d10, #4
            vand.u8 d10, d10, d18
            vtbl.8 d10, {d8, d9}, d10
            vtbl.8 d11, {d8, d9}, d11
            vmovl.s8 q7, d10
            vmovl.s8 q8, d11
            cmp r11, #0
            beq LoopInBlockEnd

            LoopInBlock:
                sub r11, r11, #1
                vld1.f32 {q4}, [r3]!
                vld1.f32 {d10}, [r5]!
                vshr.u8 d11, d10, #4
                vand.u8 d10, d10, d18
                vtbl.8 d10, {d8, d9}, d10
                vtbl.8 d11, {d8, d9}, d11
                vaddw.s8 q7, q7, d10
                vaddw.s8 q8, q8, d11
                cmp r11, #0
                bgt LoopInBlock

            LoopInBlockEnd:
                vaddw.s16 q11, q11, d14
                vaddw.s16 q12, q12, d15
                vaddw.s16 q13, q13, d16
                vaddw.s16 q14, q14, d17
            sub r8, r8, #1
            cmp r8, #0
            bgt LoopBits
        EndBit:
            mov r12, r3
        // Block scale
        vld1.f32 {q4, q5}, [r6]!
        vld1.f32 {q6, q7}, [r6]!
        vcvt.f32.s32 q11, q11
        vcvt.f32.s32 q12, q12
        vcvt.f32.s32 q13, q13
        vcvt.f32.s32 q14, q14

        vsub.f32 q11, q11, q15
        vsub.f32 q12, q12, q15
        vsub.f32 q13, q13, q15
        vsub.f32 q14, q14, q15

        vmul.f32 q11, q11, q4
        vmul.f32 q12, q12, q5
        vmul.f32 q13, q13, q6
        vmul.f32 q14, q14, q7

        ldr lr, [sp, #28]
        vdup.f32 q4, lr
        vmul.f32 q11, q11, q4
        vmul.f32 q12, q12, q4
        vmul.f32 q13, q13, q4
        vmul.f32 q14, q14, q4

        // input sum & block Bias
        vld1.f32 {d16[0]}, [r2]!
        vdup.f32 q8, d16[0]
        vld1.f32 {q4, q5}, [r6]!
        vld1.f32 {q6, q7}, [r6]!
        vmla.f32 q11, q4, q8
        vmla.f32 q12, q5, q8
        vmla.f32 q13, q6, q8
        vmla.f32 q14, q7, q8

        // Add bias
        vadd.f32 q0, q11, q0
        vadd.f32 q1, q12, q1
        vadd.f32 q2, q13, q2
        vadd.f32 q3, q14, q3

        sub r10, r10, #1 // block--
        cmp r10, #0

        bgt LoopBlock

    ldr lr, [sp, #20]
    vdup.f32 q4, lr
    ldr lr, [sp, #24]
    vdup.f32 q5, lr
    vmax.f32 q0, q0, q4
    vmax.f32 q1, q1, q4
    vmax.f32 q2, q2, q4
    vmax.f32 q3, q3, q4
    vmin.f32 q0, q0, q5
    vmin.f32 q1, q1, q5
    vmin.f32 q2, q2, q5
    vmin.f32 q3, q3, q5

    vst1.f32 {q0}, [r0], r4
    vst1.f32 {q1}, [r0], r4
    vst1.f32 {q2}, [r0], r4
    vst1.f32 {q3}, [r0], r4

    ldr lr, [sp, #16]
    sub lr, lr, #1
    str lr, [sp, #16]
    cmp lr, #0
    bgt LoopDz

End:
add sp, sp, #40
vpop {q4-q7}
pop {r4-r8, r10, r11, pc}



#endif
#endif
