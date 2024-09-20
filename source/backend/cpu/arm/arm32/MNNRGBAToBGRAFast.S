//
//  MNNRGBAToBGRAFast.S
//
//  Created by MNN on 2024/08/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNRGBAToBGRAFast
// void MNNRGBAToBGRAFast(const unsigned char* source, unsigned char* dest, size_t count);
// Auto Load: r0: source, r1: dest, r2: count

push {lr}

L1:
vld4.8 {d0, d1, d2, d3}, [r0]! // r,g,b,a

// swap d0,d2
vmov.32 d4, d2
vmov.32 d2, d0
vmov.32 d0, d4
vst4.8 {d0, d1, d2, d3}, [r1]!

subs r2, r2, #1
bne L1

End:
pop {pc}

#endif
#endif
