//
//  MNNRGBToBGR565Fast.S
//
//  Created by MNN on 2024/08/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNRGBToBGR565Fast
// void MNNRGBToBGR565Fast(const unsigned char* source, unsigned char* dest, size_t count);
// Auto Load: r0: source, r1: dest, r2: count

// b*7
// g*38
// r*19

push {lr}
vmov.s8 q15, #8
vneg.s8 q15, q15
vmov.s8 q14, #4
vneg.s8 q14, q14

L1:
vld3.8 {d0, d1, d2}, [r0]! // r,g,b

vand.u8 d0, d0, d30 // r & ~7
vand.u8 d1, d1, d28 // g & ~3
vshr.u8 d2, d2, #3  // b >> 3

vshll.u8 q2, d0, #7
vshl.u8 q2, q2, #1
vshll.u8 q3, d1, #3
vmovl.u8 q8, d2

vorr.u8 q2, q2, q3
vorr.u8 q2, q2, q8

vst1.16 {q2}, [r1]!

subs r2, r2, #1
bne L1

End:
pop {pc}

#endif
#endif
