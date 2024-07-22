//
//  NEON_MNNAxByClampBroadcastC4_BF16.S
//  MNN
//
//  Created by MNN on 2021/03/09.
//  Copyright © 2018-2021 Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function NEON_MNNAxByClampBroadcastC4_BF16
//void NEON_MNNAxByClampBroadcastC4_BF16(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters)
//Auto: x0: C, x1:A, x2:B, x3:width
//x4:cStride, x5:aStride, x6:height, x7:parameters
ld4r {v28.4s, v29.4s, v30.4s, v31.4s}, [x7]
// ld1 {v7.4s}, [x7]
// dup v30.4s, v7.s[2]
// dup v31.4s, v7.s[3]
mov x12, #2 //sizeof(int16_t)
mul x4, x12, x4
mul x5, x12, x5

LoopY:
mov x8, x0
mov x9, x1
ld1 {v6.4h}, [x2], #8 // 4 * sizeof(int16_t)
shll v6.4s, v6.4h, #16

mov x11, x3

L8:
cmp x11, #8
blt L1

sub x11, x11, #8
cmp x11, #8
ldp d16, d17, [x1], #16 // 4 * 2 * sizeof(int16_t)
ldp d18, d19, [x1], #16 // 4 * 2 * sizeof(int16_t)
ldp d20, d21, [x1], #16
ldp d22, d23, [x1], #16

shll v16.4s, v16.4h, #16
shll v17.4s, v17.4h, #16
shll v18.4s, v18.4h, #16
shll v19.4s, v19.4h, #16
shll v20.4s, v20.4h, #16
shll v21.4s, v21.4h, #16
shll v22.4s, v22.4h, #16
shll v23.4s, v23.4h, #16

fmla v16.4s, v6.4s, v29.4s
fmla v17.4s, v6.4s, v29.4s
fmla v18.4s, v6.4s, v29.4s
fmla v19.4s, v6.4s, v29.4s
fmla v20.4s, v6.4s, v29.4s
fmla v21.4s, v6.4s, v29.4s
fmla v22.4s, v6.4s, v29.4s
fmla v23.4s, v6.4s, v29.4s

blt L8ComputeEnd

L8Loop:

fmax v16.4s, v16.4s, v30.4s
fmax v17.4s, v17.4s, v30.4s
fmax v18.4s, v18.4s, v30.4s
fmax v19.4s, v19.4s, v30.4s
fmax v20.4s, v20.4s, v30.4s
fmax v21.4s, v21.4s, v30.4s
fmax v22.4s, v22.4s, v30.4s
fmax v23.4s, v23.4s, v30.4s

add x0, x0, #(16 * 4)
add x1, x1, #(16 * 4)

fmin v16.4s, v16.4s, v31.4s
fmin v17.4s, v17.4s, v31.4s
fmin v18.4s, v18.4s, v31.4s
fmin v19.4s, v19.4s, v31.4s

fmin v20.4s, v20.4s, v31.4s
fmin v21.4s, v21.4s, v31.4s
fmin v22.4s, v22.4s, v31.4s
fmin v23.4s, v23.4s, v31.4s

shrn v16.4h, v16.4s, #16
shrn v17.4h, v17.4s, #16
shrn v18.4h, v18.4s, #16
shrn v19.4h, v19.4s, #16
shrn v20.4h, v20.4s, #16
shrn v21.4h, v21.4s, #16
shrn v22.4h, v22.4s, #16
shrn v23.4h, v23.4s, #16

stp d16, d17, [x0, #-(16 * 4)]
ldp d16, d17, [x1, #-(16 * 4)] // 4 * 2 * sizeof(int16_t)
stp d18, d19, [x0, #-(16 * 3)]
ldp d18, d19, [x1, #-(16 * 3)] // 4 * 2 * sizeof(int16_t)
stp d20, d21, [x0, #-(16 * 2)]
ldp d20, d21, [x1, #-(16 * 2)]
stp d22, d23, [x0, #-(16 * 1)]
ldp d22, d23, [x1, #-(16 * 1)]

shll v16.4s, v16.4h, #16
shll v17.4s, v17.4h, #16
shll v18.4s, v18.4h, #16
shll v19.4s, v19.4h, #16
shll v20.4s, v20.4h, #16
shll v21.4s, v21.4h, #16
shll v22.4s, v22.4h, #16
shll v23.4s, v23.4h, #16

fmla v16.4s, v6.4s, v29.4s
fmla v17.4s, v6.4s, v29.4s
fmla v18.4s, v6.4s, v29.4s
fmla v19.4s, v6.4s, v29.4s
fmla v20.4s, v6.4s, v29.4s
fmla v21.4s, v6.4s, v29.4s
fmla v22.4s, v6.4s, v29.4s
fmla v23.4s, v6.4s, v29.4s


sub x11, x11, #8
cmp x11, #8
bge L8Loop

L8ComputeEnd:

fmax v16.4s, v16.4s, v30.4s
fmax v17.4s, v17.4s, v30.4s
fmax v18.4s, v18.4s, v30.4s
fmax v19.4s, v19.4s, v30.4s
fmax v20.4s, v20.4s, v30.4s
fmax v21.4s, v21.4s, v30.4s
fmax v22.4s, v22.4s, v30.4s
fmax v23.4s, v23.4s, v30.4s
add x0, x0, #(16 * 4)
fmin v16.4s, v16.4s, v31.4s
fmin v17.4s, v17.4s, v31.4s
fmin v18.4s, v18.4s, v31.4s
fmin v19.4s, v19.4s, v31.4s
fmin v20.4s, v20.4s, v31.4s
fmin v21.4s, v21.4s, v31.4s
fmin v22.4s, v22.4s, v31.4s
fmin v23.4s, v23.4s, v31.4s

shrn v16.4h, v16.4s, #16
shrn v17.4h, v17.4s, #16
shrn v18.4h, v18.4s, #16
shrn v19.4h, v19.4s, #16
shrn v20.4h, v20.4s, #16
shrn v21.4h, v21.4s, #16
shrn v22.4h, v22.4s, #16
shrn v23.4h, v23.4s, #16

stp d16, d17, [x0, #-(16 * 4)]
stp d18, d19, [x0, #-(16 * 3)]
stp d20, d21, [x0, #-(16 * 2)]
stp d22, d23, [x0, #-(16 * 1)]

L1:
cmp x11, #0
beq EndLine

L1Loop:
ld1 {v0.4h}, [x1], #8
shll v0.4s, v0.4h, #16

fmla v0.4s, v6.4s, v29.4s
fmax v0.4s, v0.4s, v30.4s
fmin v0.4s, v0.4s, v31.4s

shrn v0.4h, v0.4s, #16
st1 {v0.4h}, [x0], #8
subs x11, x11, #1
bne L1Loop

EndLine:
add x0, x8, x4
add x1, x9, x5

subs x6, x6, #1
bne LoopY

ret

#endif
