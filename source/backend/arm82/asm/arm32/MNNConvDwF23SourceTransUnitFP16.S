//
//  MNNConvDwF23SourceTransUnitFP16.S
//  MNN
//
//  Created by MNN on 2019/4/4.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvDwF23SourceTransUnitFP16
//    void MNNConvDwF23SourceTransUnitFP16(const FLOAT16 *source, FLOAT16 *dest, size_t unit);

//Auto:
//r0: source, r1:dest, r2:unit

push {lr}

L1:
cmp r2, #0
beq End

vld1.16 {q8, q9}, [r0]!
vld1.16 {q10, q11}, [r0]!
subs r2, r2, #1
vsub.f16 q0, q8, q10
vadd.f16 q1, q9, q10
beq L1LoopEnd

L1Loop:
    vsub.f16 q2, q10, q9
    vst1.16 {q0, q1}, [r1]!
    vsub.f16 q3, q11, q9
    vmov.i32 q8, q10
    vst1.16 {q2, q3}, [r1]!
    vmov.i32 q9, q11
    vld1.16 {q10, q11}, [r0]!
    vsub.f16 q0, q8, q10
    vadd.f16 q1, q9, q10

    subs r2, r2, #1
    bne L1Loop
L1LoopEnd:
vsub.f16 q2, q10, q9
vsub.f16 q3, q11, q9

vst1.16 {q0, q1}, [r1]!
vst1.16 {q2, q3}, [r1]!


End:

pop {pc}
#endif
#endif
