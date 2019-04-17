//
//  MNNBlitC1ToFloatRGBA.S
//  MNN
//
//  Created by MNN on 2018/09/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNBlitC1ToFloatRGBA
//        void MNNBlitC1ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count)

//Auto: x0: source, x1: dest, x2: mean, x3: normal

//Load from sp:
//x4: count

//Mean
ldr w5, [x2]
dup v22.4s, w5

//Normal
ldr w6, [x3]
dup v23.4s, w6


L8:
cmp x4, #8
blt L1


LoopL8:

ld1 {v0.8b}, [x0], #8
uxtl v0.8h, v0.8b
uxtl2 v1.4s, v0.8h
uxtl v0.4s, v0.4h
ucvtf v1.4s, v1.4s
ucvtf v0.4s, v0.4s
fsub v1.4s, v1.4s, v22.4s
fsub v0.4s, v0.4s, v22.4s
fmul v1.4s, v1.4s, v23.4s
fmul v0.4s, v0.4s, v23.4s
movi v16.4s, #0
movi v17.4s, #0
movi v18.4s, #0
movi v19.4s, #0

mov v16.s[0], v0.s[0]
mov v17.s[0], v0.s[1]
mov v18.s[0], v0.s[2]
st1 {v16.4s, v17.4s}, [x1],#32
mov v19.s[0], v0.s[3]
st1 {v18.4s, v19.4s}, [x1],#32
mov v16.s[0], v1.s[0]
mov v17.s[0], v1.s[1]
mov v18.s[0], v1.s[2]
st1 {v16.4s, v17.4s}, [x1],#32
mov v19.s[0], v1.s[3]
st1 {v18.4s, v19.4s}, [x1],#32

sub x4, x4, #8
cmp x4, #8
bge LoopL8


L1:
cmp x4, #0
beq End

LoopL1:
ld1 {v0.b}[0], [x0], #1

uxtl v0.8h, v0.8b
movi v1.4s, #0
uxtl v0.4s, v0.4h
mov v1.s[0], v0.s[0]
ucvtf v1.4s, v1.4s

fsub v1.4s, v1.4s, v22.4s
fmul v0.4s, v1.4s, v23.4s

st1 {v0.4s}, [x1], #16

subs x4, x4, #1
bne LoopL1


End:

ret


#endif
