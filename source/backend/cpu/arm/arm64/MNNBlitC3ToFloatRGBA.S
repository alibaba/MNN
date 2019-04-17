//
//  MNNBlitC3ToFloatRGBA.S
//  MNN
//
//  Created by MNN on 2018/09/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNBlitC3ToFloatRGBA
//        void MNNBlitC3ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count)

//Auto: x0: source, x1: dest, x2: mean, x3: normal

//Load from sp:
//x4: count

//Mean
ld1 {v22.4s}, [x2]

//Normal
ld1 {v23.4s}, [x3]


L8:
cmp x4, #8
blt L1

dup v24.4s, v22.s[0]
dup v25.4s, v22.s[1]
dup v26.4s, v22.s[2]

dup v27.4s, v23.s[0]
dup v28.4s, v23.s[1]
dup v29.4s, v23.s[2]



LoopL8:

ld3 {v0.8b, v1.8b, v2.8b}, [x0], #24
uxtl v18.8h, v0.8b
uxtl v19.8h, v1.8b
uxtl v20.8h, v2.8b

//v0.4s-v2.4s, v16.4s-v18.4s store origin rgb
uxtl v0.4s, v18.4h
uxtl2 v16.4s, v18.8h
uxtl v1.4s, v19.4h
uxtl2 v17.4s, v19.8h
uxtl v2.4s, v20.4h
uxtl2 v18.4s, v20.8h
ucvtf v0.4s, v0.4s
ucvtf v1.4s, v1.4s
ucvtf v2.4s, v2.4s
movi v3.4s, #0

//Transpose: v0.4s, v1.4s, v2.4s, 0 -> v0.4s, v1.4s, v2.4s, v3.4s
.macro COMPUTE
fsub v0.4s, v0.4s, v24.4s
fsub v1.4s, v1.4s, v25.4s
fsub v2.4s, v2.4s, v26.4s

fmul v0.4s, v0.4s, v27.4s
fmul v1.4s, v1.4s, v28.4s
fmul v2.4s, v2.4s, v29.4s

st4 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
.endm

COMPUTE

ucvtf v0.4s, v16.4s
ucvtf v1.4s, v17.4s
ucvtf v2.4s, v18.4s


COMPUTE

sub x4, x4, #8
cmp x4, #8
bge LoopL8


L1:
cmp x4, #0
beq End

LoopL1:
movi v0.8b, #0
ld1 {v0.b}[0], [x0], #1
ld1 {v0.b}[1], [x0], #1
ld1 {v0.b}[2], [x0], #1

uxtl v0.8h, v0.8b
uxtl v0.4s, v0.4h

ucvtf v0.4s, v0.4s

fsub v0.4s, v0.4s, v22.4s
fmul v0.4s, v0.4s, v23.4s

st1 {v0.4s}, [x1], #16

subs x4, x4, #1
bne LoopL1


End:

ret


#endif
