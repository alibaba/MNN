//
//  MNNQuantizeFP16_UNIT4.S
//  MNN
//
//  Created by MNN on 2020/02/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNQuantizeFP16_UNIT4
// void MNNQuantizeFP16_UNIT4(FLOAT16* dst, const float* src, int size);

// Auto:
//      r0:dst, r1:src, r2:size, r3: minMax

push {lr}
// Min max: q14, q15
ldr r12, [r3]
vdup.32 q14, r12
ldr r12, [r3, #4]
vdup.32 q15, r12

L4:
cmp r2, #4
blt L1

Loop4:
// {q0-q3} => {d16-d19 (q8-q9)}
vld1.32 {q0, q1}, [r1]!
vmax.f32 q0, q0, q14
vmax.f32 q1, q1, q14
vmin.f32 q0, q0, q15
vld1.32 {q2, q3}, [r1]!
vmin.f32 q1, q1, q15

vmax.f32 q2, q2, q14
vmax.f32 q3, q3, q14
vcvt.f16.f32 d16, q0

vmin.f32 q2, q2, q15
vmin.f32 q3, q3, q15
vcvt.f16.f32 d17, q1
vcvt.f16.f32 d18, q2
vst1.16 {d16, d17}, [r0]!
sub r2, r2, #4
vcvt.f16.f32 d19, q3
cmp r2, #4
vst1.16 {d18, d19}, [r0]!
bge Loop4

L1:
cmp r2, #0
beq End

Loop1:
vld1.32 {q0}, [r1]!
vmax.f32 q0, q0, q14
vmin.f32 q0, q0, q15
vcvt.f16.f32 d2, q0
vst1.16 {d2}, [r0]!
subs r2, r2, #1
bne Loop1

End:
pop {pc}

#endif
#endif
