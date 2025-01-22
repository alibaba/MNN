//
//  MNNGeneralIm2col_Fp32Arm82.S
//  MNN
//
//  Created by MNN on 2025/01/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

//void MNNGeneralIm2col_Fp32Arm82(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el, int32_t LP, int32_t pack)
asm_function MNNGeneralIm2col_Fp32Arm82

// x0:destOrigin, x1:sourceGroup, x2:info, x3:el, x4:LP, x5:pack
stp d14, d15, [sp, #(-16 * 5)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x19, x20, [sp, #(16 * 4)]

// load el info
ldr w6, [x2, #0]  // number
ldr w7, [x2, #4]  // eReal
ldr w15, [x2, #8]  // eDest (< EP)
ldr w9, [x2, #12] // offset (stride)
ldr x14, [x1, #0] // src start
lsl x9, x9, #4    // pack*offset*sizeof(float32_t)
// stride
lsl x19, x15, #4 // eDest*LP*sizeof(float32_t)
lsl x7, x7, #4  // eReal*pack*sizeof(float32_t)
mov x20, #3      // Arm82,LP=4

LoopNum:

ldr w10, [x3], #4 // e
ldr w11, [x3], #4 // l
ldr w12, [x3], #4 // eOffset
ldr w13, [x3], #4 // lOffset
// dst address: x2
and x2, x13, x20 // lR
sub x13, x13, x2 // lOffset-lR
mul x13, x13, x15 // (lOffset-lR)*(eDest)
add x13, x13, x2 // (lOffset-lR)*(eDest)+lR
add x13, x13, x12, LSL #2 // + eoffset*lp
add x2, x0, x13, LSL #2 // *sizeof(float32_t)

LoopL4:
mov x5, x2
mov x4, x14
mov x13, x10

cmp x13, #12
blt LoopL4E8

LoopL4E12:
sub x13, x13, #12
ld1 {v0.4s}, [x14], x9
ld1 {v1.4s}, [x14], x9
ld1 {v2.4s}, [x14], x9
ld1 {v3.4s}, [x14], x9
ld1 {v4.4s}, [x14], x9
ld1 {v5.4s}, [x14], x9
ld1 {v6.4s}, [x14], x9
ld1 {v7.4s}, [x14], x9
ld1 {v8.4s}, [x14], x9
ld1 {v9.4s}, [x14], x9
ld1 {v10.4s}, [x14], x9
ld1 {v11.4s}, [x14], x9

st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x2], #64
st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x2], #64
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x2], #64
cmp x13, #12
bge LoopL4E12

LoopL4E8:
cmp x13, #8
blt LoopL4E4
ld1 {v0.4s}, [x14], x9
ld1 {v1.4s}, [x14], x9
ld1 {v2.4s}, [x14], x9
ld1 {v3.4s}, [x14], x9
ld1 {v4.4s}, [x14], x9
ld1 {v5.4s}, [x14], x9
ld1 {v6.4s}, [x14], x9
ld1 {v7.4s}, [x14], x9
st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x2], #64
st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x2], #64
sub x13, x13, #8

LoopL4E4:
cmp x13, #4
blt LoopL4E2
ld1 {v0.4s}, [x14], x9
ld1 {v1.4s}, [x14], x9
ld1 {v2.4s}, [x14], x9
ld1 {v3.4s}, [x14], x9
st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x2], #64
sub x13, x13, #4

LoopL4E2:
cmp x13, #2
blt LoopL4E1
ld1 {v0.4s}, [x14], x9
ld1 {v1.4s}, [x14], x9
st1 {v0.4s, v1.4s}, [x2], #32
sub x13, x13, #2

LoopL4E1:
cmp x13, #1
blt EndL4LoopE
ld1 {v0.4s}, [x14], x9
st1 {v0.4s}, [x2], #16

EndL4LoopE:
add x2, x5, x19 // eDest*LP*sizeof(float32_t)
add x14, x4, x7
subs x11, x11, #4
bne LoopL4

EndLoopL:
subs x6, x6, #1
add x1, x1, #8
ldr x14, [x1, #0]
bne LoopNum


End:
ldp x19, x20, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 5)
ret

#endif