//
//  MNNSamplerC1NearestOpt.S
//  MNN
//
//  Created by MNN on 2018/12/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
//void MNNSamplerC1NearestOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t iw, size_t ih, size_t yStride);
asm_function MNNSamplerC1NearestOpt

//Auto: x0:source, x1:dest, x2:points, x3:count
//x4: xMax, x5: yMax, x6:yStride

movi v19.4s, #0

ld1 {v0.2s, v1.2s}, [x2]
L4:
cmp x3, #4
blt L1
dup v16.4s, w4
dup v17.4s, w5
movi v3.2s, #4
scvtf v3.2s, v3.2s 
fmul v3.2s, v3.2s, v1.2s
dup v25.4s, v3.s[0]
dup v26.4s, v3.s[1]

fadd v2.2s, v0.2s, v1.2s
mov v4.s[0], v0.s[0]
fadd v3.2s, v2.2s, v1.2s
mov v5.s[0], v0.s[1]
mov v4.s[1], v2.s[0]
mov v5.s[1], v2.s[1]
mov v4.s[2], v3.s[0]
fadd v2.2s, v3.2s, v1.2s
mov v5.s[2], v3.s[1]
mov v4.s[3], v2.s[0]
mov v5.s[3], v2.s[1]

dup v23.4s, w6
dup v22.2d, x0

L4Loop:
fcvtns v6.4s, v4.4s
fcvtns v7.4s, v5.4s

smin v6.4s, v6.4s, v16.4s
smin v7.4s, v7.4s, v17.4s
smax v6.4s, v6.4s, v19.4s
smax v7.4s, v7.4s, v19.4s

mla v6.4s, v7.4s, v23.4s
uxtl2 v7.2d, v6.4s
uxtl v6.2d, v6.2s
add v6.2d, v6.2d, v22.2d
add v7.2d, v7.2d, v22.2d

mov x12, v6.d[0]
mov x13, v6.d[1]
ld1 {v3.b}[0], [x12]
mov x12, v7.d[0]
ld1 {v3.b}[1], [x13]
fadd v5.4s, v26.4s, v5.4s
mov x13, v7.d[1]
ld1 {v3.b}[2], [x12]
fadd v4.4s, v25.4s, v4.4s
ld1 {v3.b}[3], [x13]

st1 {v3.s}[0], [x1], #4


sub x3, x3, #4
cmp x3, #4
bge L4Loop

mov v0.s[0], v4.s[0]
mov v0.s[1], v5.s[0]


L1:
cmp x3, #0
beq End
mov v6.s[0], w4
mov v6.s[1], w5
mov w12, #1

L1Loop:

fcvtns v2.2s, v0.2s
smax v2.2s, v2.2s, v19.2s
smin v2.2s, v2.2s, v6.2s
mov w4, v2.s[0]
mov w5, v2.s[1]
umull x5, w5, w6
umull x4, w4, w12
add x7, x0, x5
add x4, x4, x7
ld1 {v2.b}[0], [x4]
fadd v0.2s, v0.2s, v1.2s
subs x3, x3, #1
st1 {v2.b}[0], [x1], #1


bne L1Loop

End:

ret
#endif
