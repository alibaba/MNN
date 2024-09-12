//
//  MNNRGBToGRAYFast.S
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

asm_function MNNRGBToGRAYFast
// void MNNRGBToGRAYFast(const unsigned char* source, unsigned char* dest, size_t count);
// Auto Load: r0: source, r1: dest, r2: count

push {lr}

vmov.i8 d4, #7
vmov.i8 d5, #38
vmov.i8 d6, #19

L1:
vld3.8 {d0, d1, d2}, [r0]!
vmull.u8 q8, d2, d4 // b*7
vmlal.u8 q8, d1, d5 // g*38
vmlal.u8 q8, d0, d6 // r*19

vqshrn.u16 d16, q8, #6
vst1.8 {d16}, [r1]!

subs r2, r2, #1
bne L1

End:
pop {pc}

#endif
#endif
