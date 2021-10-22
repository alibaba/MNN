//
//  MNNConvWinoSourceTransformUnit6x6FP16.S
//  MNN
//
//  Created by MNN on 2021/10/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvWinoSourceTransformUnit6x6FP16
//    void MNNConvWinoSourceTransformUnit6x6FP16(const FLOAT16* srcBlock, FLOAT16* dstStart, size_t srcStep, size_t dstStep);

//Auto:
//r0: srcBlock, r1:dstStart, r2:srcStep, r3:dstStep

push {lr}

lsl r2, r2, #1
lsl r3, r3, #1

vld1.16 {q0}, [r0], r2        //s0
vld1.16 {q1}, [r0], r2        //s1
vld1.16 {q2}, [r0], r2        //s2
vld1.16 {q3}, [r0], r2        //s3
vld1.16 {q8}, [r0], r2        //s4
vld1.16 {q9}, [r0], r2        //s5

vmov.i16 q14, #0xc400

vsub.f16 q11, q3, q1
vsub.f16 q10, q8, q2           //b4
vadd.f16 q11, q11, q11         //b5

vadd.f16 q12, q10, q11         //m3
vsub.f16 q13, q10, q11         //m4

vmla.f16 q8, q14, q2           //b0
vmla.f16 q9, q14, q3           //b1
vmla.f16 q2, q14, q0           //b2
vmla.f16 q3, q14, q1           //b3

vsub.f16 q2, q8, q2            //m0
vsub.f16 q9, q9, q3            //m5
vadd.f16 q14, q8, q3           //m1
vsub.f16 q15, q8, q3           //m2

vst1.16 {q2}, [r1], r3
vst1.16 {q14}, [r1], r3
vst1.16 {q15}, [r1], r3
vst1.16 {q12}, [r1], r3
vst1.16 {q13}, [r1], r3
vst1.16 {q9}, [r1], r3

pop {pc}
#endif
#endif
