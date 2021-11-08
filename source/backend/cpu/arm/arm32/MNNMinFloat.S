//
//  MNNMinFloat.S
//  MNN
//
//  Created by MNN on 2018/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNMinFloat
//void MNNMinFloat(float* input, float * minBuffer, int32_t inputCountUnit);

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9

//Auto: r0: input, r1:minBuffer, r2: inputCountUnit

//q15 : min
vld1.f32 {q15}, [r1]
cmp r2, #0
beq End
Loop:
    vld1.f32 {q0}, [r0]!
    vmin.f32 q15, q0, q15
    vld1.f32 {q1}, [r0]!
    vmin.f32 q15, q1, q15
    subs r2, r2, #1
    bne Loop

vst1.f32 {q15}, [r1]

End:

pop {r4-r8, r10, r11, pc}

#endif
#endif
