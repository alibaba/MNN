//
//  MNNRGBToBGR555Fast.S
//  MNN
//
//  Created by MNN on 2024/08/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNRGBToBGR555Fast
// void MNNRGBToBGR555Fast(const unsigned char* source, unsigned char* dest, size_t count);
// Auto Load: r0: source, r1: dest, r2: count

push {lr}

vmov.s8 q15, #8
vneg.s8 q15, q15

L1:
vld3.8 {d0, d1, d2}, [r0]!
vand.u8 d0, d0, d30 // r & ~7
vand.u8 d1, d1, d30 // g & ~7
vshr.u8 d2, d2, #3  // b >> 3
vshll.u8 q2, d0, #7
vshll.u8 q3, d1, #2
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
