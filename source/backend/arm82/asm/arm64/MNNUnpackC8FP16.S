//
//  MNNUnpackC8FP16.S
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba GroDown Holding Limited
//


#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

// Ref from libjpeg-turbo-master/simd
.macro transpose_8x8 l0, l1, l2, l3, l4, l5, l6, l7, t0, t1, t2, t3
    trn1            \t0\().8h, \l0\().8h, \l1\().8h
    trn1            \t1\().8h, \l2\().8h, \l3\().8h
    trn1            \t2\().8h, \l4\().8h, \l5\().8h
    trn1            \t3\().8h, \l6\().8h, \l7\().8h
    trn2            \l1\().8h, \l0\().8h, \l1\().8h
    trn2            \l3\().8h, \l2\().8h, \l3\().8h
    trn2            \l5\().8h, \l4\().8h, \l5\().8h
    trn2            \l7\().8h, \l6\().8h, \l7\().8h

    trn1            \l4\().4s, \t2\().4s, \t3\().4s
    trn2            \t3\().4s, \t2\().4s, \t3\().4s
    trn1            \t2\().4s, \t0\().4s, \t1\().4s
    trn2            \l2\().4s, \t0\().4s, \t1\().4s
    trn1            \t0\().4s, \l1\().4s, \l3\().4s
    trn2            \l3\().4s, \l1\().4s, \l3\().4s
    trn2            \t1\().4s, \l5\().4s, \l7\().4s
    trn1            \l5\().4s, \l5\().4s, \l7\().4s

    trn2            \l6\().2d, \l2\().2d, \t3\().2d
    trn1            \l0\().2d, \t2\().2d, \l4\().2d
    trn1            \l1\().2d, \t0\().2d, \l5\().2d
    trn2            \l7\().2d, \l3\().2d, \t1\().2d
    trn1            \l2\().2d, \l2\().2d, \t3\().2d
    trn2            \l4\().2d, \t2\().2d, \l4\().2d
    trn1            \l3\().2d, \l3\().2d, \t1\().2d
    trn2            \l5\().2d, \t0\().2d, \l5\().2d
.endm

.macro transpose
transpose_8x8 v0, v1, v2, v3, v4, v5, v6, v7, v28,v29,v30,v31
.endm

asm_function MNNUnpackC8FP16_C8
//void MNNUnpackC8FP16_C8(float* dst, const float* src, size_t area, size_t depth, int32_t* areaOffset)
//Auto load:
//x0:dst, x1:src, x2:area, x3:depth, x4: areaOffset
//x9: srcArea, x10: dstArea

ldr w10, [x4, #4]
ldr w9, [x4, #0]
uxtw x10, w10
uxtw x9, w9

mul x4, x2, x3
cmp x4, #0
beq DownEnd

//Swap x0 and x1 for conviniense
mov x4, x0
mov x0, x1
mov x1, x4

//x4: dstDepthOffset:srcArea*sizeof(int16)
mov x4, #2
mul x4, x10, x4

//x9 -> 8 * (srcArea * sizeof(int16) - area * sizeof(int16))
mov x12, #16
sub x9, x9, x2
mul x9, x12, x9

//x10 -> (dstArea * sizeof(int16) - area * sizeof(int16))
mov x12, #2
sub x10, x10, x2
mul x10, x12, x10

DownL8:
//cmp x3, #8
//ble DownL7

DownL8Loop:
add x5, x4, x1
add x6, x4, x5
add x7, x4, x6
add x12, x4, x7
add x13, x4, x12
add x14, x4, x13
add x15, x4, x14

mov x8, x2
cmp x8, #8
ble DownL8AreaRemain
DownL8AreaLoop:

ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x0], #64

transpose

st1 {v0.8h}, [x1], #16
st1 {v1.8h}, [x5], #16
st1 {v2.8h}, [x6], #16
st1 {v3.8h}, [x7], #16
st1 {v4.8h}, [x12], #16
st1 {v5.8h}, [x13], #16
st1 {v6.8h}, [x14], #16
st1 {v7.8h}, [x15], #16

sub x8, x8, #8
cmp x8, #8
bge DownL8AreaLoop

DownL8AreaRemain:
cmp x8, #0
beq DownL8AreaRemainEnd
DownL8AreaRemainLoop:
ld1 {v0.8h}, [x0], #16

st1 {v0.h}[0], [x1], #2
st1 {v0.h}[1], [x5], #2
st1 {v0.h}[2], [x6], #2
st1 {v0.h}[3], [x7], #2
st1 {v0.h}[4], [x12], #2
st1 {v0.h}[5], [x13], #2
st1 {v0.h}[6], [x14], #2
st1 {v0.h}[7], [x15], #2

subs x8, x8, #1
bne DownL8AreaRemainLoop
DownL8AreaRemainEnd:
sub x3, x3, #8
add x1, x15, x10
cmp x3, #8
add x0, x9, x0
bge DownL8Loop

DownEnd:

ret
#endif
