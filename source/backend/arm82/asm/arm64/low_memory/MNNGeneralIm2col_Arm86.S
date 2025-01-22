//
//  MNNGeneralIm2col_Arm86.S
//  MNN
//
//  Created by MNN on 2024/12/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

//void MNNGeneralIm2col_Arm86(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el, int32_t LP, int32_t pack)
asm_function MNNGeneralIm2col_Arm86

// x0:destOrigin, x1:sourceGroup, x2:info, x3:el, x4:LP, x5:pack
stp d14, d15, [sp, #(-16 * 5)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x19, x20, [sp, #(16 * 4)]

// load el info
ldr w6, [x2, #0]  // number
ldr w7, [x2, #4]  // eReal
ldr w8, [x2, #8]  // eDest (< EP)
ldr w9, [x2, #12] // offset (stride)
ldr x14, [x1, #0] // src start
lsl x9, x9, #4    // pack*offset*sizeof(float)
// stride
lsl x19, x8, #3 // eDest*LP
lsl x7, x7, #4  // eReal*pack*sizeof(float16_t)
mov x20, #7      // Arm86,LP=8

LoopNum:

ldr w10, [x3], #4 // e
ldr w11, [x3], #4 // l
ldr w12, [x3], #4 // eOffset
ldr w13, [x3], #4 // lOffset
// dst address: x2
and x2, x13, x20 // lR
sub x13, x13, x2 // lOffset-lR
mul x13, x13, x8 // (lOffset-lR)*(eDest)
add x13, x13, x2 // (lOffset-lR)*(eDest)+lR
add x13, x13, x12, LSL #3 // + eoffset*lp
add x2, x0, x13, LSL #1 // *sizeof(float16_t)

LoopL8:
mov x5, x2
mov x4, x14
mov x13, x10

cmp x13, #10
blt LoopL8E8

LoopL8E10:
sub x13, x13, #10
ld1 {v0.8h}, [x14], x9
ld1 {v1.8h}, [x14], x9
ld1 {v2.8h}, [x14], x9
ld1 {v3.8h}, [x14], x9
ld1 {v4.8h}, [x14], x9
ld1 {v5.8h}, [x14], x9
ld1 {v6.8h}, [x14], x9
ld1 {v7.8h}, [x14], x9
ld1 {v8.8h}, [x14], x9
ld1 {v9.8h}, [x14], x9
st1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x2], #64
st1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x2], #64
st1 {v8.8h, v9.8h}, [x2], #32
cmp x13, #10
bge LoopL8E10

LoopL8E8:
cmp x13, #8
blt LoopL8E4
ld1 {v0.8h}, [x14], x9
ld1 {v1.8h}, [x14], x9
ld1 {v2.8h}, [x14], x9
ld1 {v3.8h}, [x14], x9
ld1 {v4.8h}, [x14], x9
ld1 {v5.8h}, [x14], x9
ld1 {v6.8h}, [x14], x9
ld1 {v7.8h}, [x14], x9
st1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x2], #64
st1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x2], #64
sub x13, x13, #8

LoopL8E4:
cmp x13, #4
blt LoopL8E2
ld1 {v0.8h}, [x14], x9
ld1 {v1.8h}, [x14], x9
ld1 {v2.8h}, [x14], x9
ld1 {v3.8h}, [x14], x9
st1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x2], #64
sub x13, x13, #4

LoopL8E2:
cmp x13, #2
blt LoopL8E1
ld1 {v0.8h}, [x14], x9
ld1 {v1.8h}, [x14], x9
st1 {v0.8h, v1.8h}, [x2], #32
sub x13, x13, #2

LoopL8E1:
cmp x13, #1
blt EndL8LoopE
ld1 {v0.8h}, [x14], x9
st1 {v0.8h}, [x2], #16

EndL8LoopE:
sub x11, x11, #8
cmp x11, #8
add x2, x5, x19, LSL #1
add x14, x4, x7
bge LoopL8


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

