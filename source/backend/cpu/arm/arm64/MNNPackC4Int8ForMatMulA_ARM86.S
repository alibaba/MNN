#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.macro SET_0 s0, s1, s2, s3
    movi \s0\().4s, #0
    movi \s1\().4s, #0
    movi \s2\().4s, #0
    movi \s3\().4s, #0
.endm

/*
struct SumByAxisParams {
    ssize_t kernelCountUnitDouble;
    ssize_t col_buffer_unit_size;
    ssize_t DST_XUNIT;
    ssize_t SRC_UNIT;
    ssize_t blockNum;
    ssize_t oneScale;
};
 */

asm_function MNNSumByAxisLForMatmul_A_ARM86
// MNNSumByAxisLForMatmul_A_ARM86(float* dest, int8_t* source, const float* dequantScale, ssize_t realDstCount, SumByAxisParams sumParams);
// x0: dest, x1: source, x2: dequantScale, x3: realDstCount, x4: sumParams
// Load from sp: x6: blockNum

ldr x6, [x4, #32] // blockNum
ldr x12, [x4, #40] // oneScale
ldr x11, [x4, #48] // valid
ldr x5, [x4, #56] // kx*ky
ldr x4, [x4, #64] // icDiv4

stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

movi v31.16b, #1
movi v29.16b, #1
ld1r {v30.4s}, [x2] // dequant scale
sdiv x4, x4, x6    // src_depth_quad_per_block

cbz x11, START
/* mask */
movi v31.16b, #1
mov x13, #0xFFFFFFFFFFFFFFFF
lsl x14, x11, #3
lsl x13, x13, x14
dup v27.2d, x13
bic v29.16b, v31.16b, v27.16b

START:
lsl x11, x3, #2

cmp x3, #1
mov x8, #8 // for LLM decode, otherwise update in Remain
beq TILE_1

TILE_10: // realDstCount >= EP(10)
cmp x3, #10
blt Remain
mov x9, x6 // blockNum

cbnz x12, TILE10_BLOCK_NUM
ld1 {v5.4s, v6.4s}, [x2], #32
ld1 {v7.d}[0], [x2], #8

TILE10_BLOCK_NUM:
mov x15, x5 // kx*ky
SET_0 v10, v11, v12, v13
movi v14.4s, #0

TILE10_BLOCK_INNER:
sub x14, x4, #1    // icDiv4
cbz x14, TILE10_LAST_QUAD
TILE10_PRE_QUAD:
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64 // E: 0,1,...,7
ld1 {v4.16b}, [x1], #16                         // E: 8,9
subs x14, x14, #1   // icDiv4--
.inst 0x4e80a7ea // smmla v10.4s, v31.16b, v0.16b // sum LP axis for E0 and E1
.inst 0x4e81a7eb // smmla v11.4s, v31.16b, v1.16b
.inst 0x4e82a7ec // smmla v12.4s, v31.16b, v2.16b
.inst 0x4e83a7ed // smmla v13.4s, v31.16b, v3.16b
.inst 0x4e84a7ee // smmla v14.4s, v31.16b, v4.16b
bne TILE10_PRE_QUAD

TILE10_LAST_QUAD:
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64 // E: 0,1,...,7
ld1 {v4.16b}, [x1], #16                         // E: 8,9
.inst 0x4e80a7aa // smmla v10.4s, v29.16b, v0.16b
.inst 0x4e81a7ab // smmla v11.4s, v29.16b, v1.16b
.inst 0x4e82a7ac // smmla v12.4s, v29.16b, v2.16b
.inst 0x4e83a7ad // smmla v13.4s, v29.16b, v3.16b
.inst 0x4e84a7ae // smmla v14.4s, v29.16b, v4.16b
subs x15, x15, #1
bne TILE10_BLOCK_INNER

TILE10_PER_BLOCK_END:
trn1 v20.2d, v10.2d, v11.2d
trn1 v21.2d, v12.2d, v13.2d

scvtf v20.4s, v20.4s
scvtf v21.4s, v21.4s
scvtf v14.4s, v14.4s

cbnz x12, TILE10_ONE_SCALE
fmul v20.4s, v20.4s, v5.4s
fmul v21.4s, v21.4s, v6.4s
fmul v14.4s, v14.4s, v7.4s
b TILE10_STORE

TILE10_ONE_SCALE:
fmul v20.4s, v20.4s, v30.4s
fmul v21.4s, v21.4s, v30.4s
fmul v14.4s, v14.4s, v30.4s

TILE10_STORE:
subs x9, x9, #1 // blockNum--
st1 {v20.4s, v21.4s}, [x0], #32
st1 {v14.d}[0], [x0], #8
bne TILE10_BLOCK_NUM // Finish one block

TILE10_END:
sub x3, x3, #10 // realDstCount-=10
b TILE_10


Remain: // remain realDstCount < EP
cbz x3, End

lsl x11, x3, #2
lsl x8, x3, #3 // x8: eDest*LP
/* For remain dstCount, each E's block step is x11. */
TILE_8: // realDstCount >= 8
cmp x3, #8
blt TILE_4

mov x7, x1  // tag begin src address for Remain8
mov x10, x0 // tag begin dst address for Remain8
mov x9, x6  // blockNum

cbnz x12, TILE8_BLOCK_NUM
ld1 {v5.4s, v6.4s}, [x2], #32

TILE8_BLOCK_NUM:
cbz x9, TILE8_END
mov x15, x5 // kx*ky
SET_0 v10, v11, v12, v13

TILE8_BLOCK_INNER:
sub x14, x4, #1
cbz x14, TILE8_LAST_QUAD
TILE8_PRE_QUAD:
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x7] // E: 0,1,...,7
add x7, x7, x8 // x7=x7+EP*LP
.inst 0x4e80a7ea // smmla v10.4s, v31.16b, v0.16b // sum LP axis for E0 and E1
.inst 0x4e81a7eb // smmla v11.4s, v31.16b, v1.16b
.inst 0x4e82a7ec // smmla v12.4s, v31.16b, v2.16b
.inst 0x4e83a7ed // smmla v13.4s, v31.16b, v3.16b
subs x14, x14, #1
bne TILE8_PRE_QUAD

TILE8_LAST_QUAD:
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x7] // E: 0,1,...,7
add x7, x7, x8 // x7=x7+EP*LP
.inst 0x4e80a7aa // smmla v10.4s, v29.16b, v0.16b // sum LP axis for E0 and E1
.inst 0x4e81a7ab // smmla v11.4s, v29.16b, v1.16b
.inst 0x4e82a7ac // smmla v12.4s, v29.16b, v2.16b
.inst 0x4e83a7ad // smmla v13.4s, v29.16b, v3.16b

subs x15, x15, #1 // kxky--
bne TILE8_BLOCK_INNER

TILE8_PER_BLOCK_END:

trn1 v20.2d, v10.2d, v11.2d
trn1 v21.2d, v12.2d, v13.2d

scvtf v20.4s, v20.4s
scvtf v21.4s, v21.4s

cbnz x12, TILE8_ONE_SCALE
fmul v20.4s, v20.4s, v5.4s
fmul v21.4s, v21.4s, v6.4s
b TILE8_STORE

TILE8_ONE_SCALE:
fmul v20.4s, v20.4s, v30.4s
fmul v21.4s, v21.4s, v30.4s

TILE8_STORE:
subs x9, x9, #1 // blockNum--
st1 {v20.4s, v21.4s}, [x10], x11 // Go to next block for this 8 remain.
bne TILE8_BLOCK_NUM

TILE8_END:
add x0, x0, #32  // finish 8 dstCount * sizeof(float)
sub x3, x3, #8 // realDstCount-=8
add x1, x1, #64 // LP*8


TILE_4: // realDstCount >= 4
cmp x3, #4
blt TILE_2

mov x7, x1  // tag begin src address for Remain4
mov x10, x0 // tag begin dst address for Remain4
mov x9, x6  // blockNum

cbnz x12, TILE4_BLOCK_NUM
ld1 {v5.4s}, [x2], #16

TILE4_BLOCK_NUM:
mov x15, x5 // kx*ky
movi v10.4s, #0
movi v11.4s, #0

TILE4_BLOCK_INNER:
sub x14, x4, #1
cbz x14, TILE4_LAST_QUAD
TILE4_PRE_QUAD:
ld1 {v0.16b, v1.16b}, [x7] // E: 0,1,2,3
add x7, x7, x8
subs x14, x14, #1
.inst 0x4e80a7ea // smmla v10.4s, v31.16b, v0.16b // sum LP axis for E0 and E1
.inst 0x4e81a7eb // smmla v11.4s, v31.16b, v1.16b
bne TILE4_PRE_QUAD

TILE4_LAST_QUAD:
ld1 {v0.16b, v1.16b}, [x7] // E: 0,1,2,3
add x7, x7, x8
.inst 0x4e80a7aa // smmla v10.4s, v29.16b, v0.16b // sum LP axis for E0 and E1
.inst 0x4e81a7ab // smmla v11.4s, v29.16b, v1.16b

subs x15, x15, #1
bne TILE4_BLOCK_INNER

TILE4_PER_BLOCK_END:
trn1 v20.2d, v10.2d, v11.2d
scvtf v20.4s, v20.4s

cbnz x12, TILE4_ONE_SCALE
fmul v20.4s, v20.4s, v5.4s
b TILE4_STORE
TILE4_ONE_SCALE:
fmul v20.4s, v20.4s, v30.4s
TILE4_STORE:
subs x9, x9, #1 // blockNum--
st1 {v20.4s}, [x10], x11
bne TILE4_BLOCK_NUM

TILE4_END:
add x0, x0, #16  // finish 4 dstCount * sizeof(float)
sub x3, x3, #4 // realDstCount-=4
add x1, x1, #32 // LP*4

TILE_2: // realDstCount >= 2
cmp x3, #2
blt TILE_1

mov x7, x1  // tag begin src address for Remain8
mov x10, x0 // tag begin dst address for Remain8
mov x9, x6  // blockNum

cbnz x12, TILE2_BLOCK_NUM
ld1 {v5.d}[0], [x2], #8

TILE2_BLOCK_NUM:
cbz x9, TILE2_END
mov x15, x5 // kx*ky
movi v10.4s, #0

TILE2_BLOCK_INNER:
sub x14, x4, #1
cbz x14, TILE2_LAST_QUAD

TILE2_PRE_QUAD:
ld1 {v0.16b}, [x7] // E: 0,1
add x7, x7, x8
.inst 0x4e80a7ea // smmla v10.4s, v31.16b, v0.16b // sum LP axis for E0 and E1
subs x14, x14, #1
bne TILE2_PRE_QUAD

TILE2_LAST_QUAD:
ld1 {v0.16b}, [x7] // E: 0,1
add x7, x7, x8
.inst 0x4e80a7aa // smmla v10.4s, v29.16b, v0.16b // sum LP axis for E0 and E1

subs x15, x15, #1
bne TILE2_BLOCK_INNER

TILE2_PER_BLOCK_END:
sub x9, x9, #1 // blockNum--

scvtf v10.4s, v10.4s
cbnz x12, TILE2_ONE_SCALE
fmul v10.4s, v10.4s, v5.4s
b TILE2_STORE
TILE2_ONE_SCALE:
fmul v10.4s, v10.4s, v30.4s
TILE2_STORE:
st1 {v10.d}[0], [x10], x11
b TILE2_BLOCK_NUM

TILE2_END:
add x0, x0, #8  // finish 2 dstCount: 2 * sizeof(float32)
sub x3, x3, #2 // realDstCount-=2
add x1, x1, #16 // LP * 2 * sizeof(int8_t)

TILE_1: // realDstCount >= 1
cmp x3, #1
blt End

mov x7, x1  // tag begin src address for Remain4
mov x10, x0 // tag begin dst address for Remain4
mov x9, x6  // blockNum

cbnz x12, TILE1_BLOCK_NUM
ld1 {v5.s}[0], [x2], #4

TILE1_BLOCK_NUM:
mov x15, x5 // kx*ky
movi v10.4s, #0

TILE1_BLOCK_INNER:
sub x14, x4, #1
cbz x14, TILE1_LAST_QUAD

TILE1_PRE_QUAD:
ld1 {v0.d}[0], [x7] // E: 0
add x7, x7, x8
.inst 0x4e80a7ea // smmla v10.4s, v31.16b, v0.16b // sum LP axis for E0
subs x14, x14, #1
bne TILE1_PRE_QUAD

TILE1_LAST_QUAD:
ld1 {v0.d}[0], [x7] // E: 0
add x7, x7, x8
.inst 0x4e80a7aa // smmla v10.4s, v29.16b, v0.16b // sum LP axis for E0

subs x15, x15, #1
bne TILE1_BLOCK_INNER

TILE1_PER_BLOCK_END:
scvtf v10.4s, v10.4s

cbnz x12, TILE1_ONE_SCALE
fmul v10.4s, v10.4s, v5.4s
b TILE1_STORE

TILE1_ONE_SCALE:
fmul v10.4s, v10.4s, v30.4s
TILE1_STORE:
subs x9, x9, #1 // blockNum--
st1 {v10.s}[0], [x10], x11
bne TILE1_BLOCK_NUM

TILE1_END:
sub x3, x3, #1 // realDstCount-=1
add x1, x1, #8 // LP * 1 * sizeof(int8_t)
add x0, x0, #4 // 1 * sizeof(float)

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret
#endif
