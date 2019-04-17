//
//  MNNUInt8ToInt16WithOffsetC4Common.S
//  MNN
//
//  Created by MNN on 2018/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNUInt8ToInt16WithOffsetC4Common

//void MNNUInt8ToInt16WithOffsetC4Common(int16_t* dst, const uint8_t* src, size_t zeroPoint, size_t sizeQuad, size_t dstStride, size_t srcStride)

//Auto: r0:dst, r1:src, r2: zeroPoint, r3: sizeQuad

//Load from sp: r4: dstStride, r5:srcStride

push {r4, r5, lr}
ldr r4, [sp, #12]
ldr r5, [sp, #16]

vdup.u8 d30, r2

L8:
cmp r3, #8
blt L1


LoopL8:

vld1.32 {d0[0]}, [r1], r5
vld1.32 {d0[1]}, [r1], r5
vld1.32 {d1[0]}, [r1], r5

vsubl.u8 q8, d0, d30

vld1.32 {d1[1]}, [r1], r5
vst1.32 {d16}, [r0], r4
vsubl.u8 q9, d1, d30
vld1.32 {d2[0]}, [r1], r5
vst1.32 {d17}, [r0], r4
vld1.32 {d2[1]}, [r1], r5
vst1.32 {d18}, [r0], r4
vld1.32 {d3[0]}, [r1], r5
vsubl.u8 q8, d2, d30
vst1.32 {d19}, [r0], r4
vld1.32 {d3[1]}, [r1], r5
vst1.32 {d16}, [r0], r4
vsubl.u8 q9, d3, d30
vst1.32 {d17}, [r0], r4
vst1.32 {d18}, [r0], r4
vst1.32 {d19}, [r0], r4

sub r3, r3, #8
cmp r3, #8
bge LoopL8


L1:
cmp r3, #0
beq End

LoopL1:
vld1.32 {d0[0]}, [r1], r5
vsubl.u8 q0, d0, d30

vst1.32 {d0}, [r0], r4

subs r3, r3, #1
bne LoopL1



End:

pop {r4, r5, pc}

#endif
#endif
