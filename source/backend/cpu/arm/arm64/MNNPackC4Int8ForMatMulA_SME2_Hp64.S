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

asm_function MNNSumByAxisLForMatmul_A_SME2_Hp64
// MNNSumByAxisLForMatmul_A_SME2_Hp64(float_t* dest, int8_t* source, float* dequantScale, ssize_t realDstCount, 
//                                ssize_t kernelCountUnitDouble, ssize_t col_buffer_unit_size, ssize_t EP, ssize_t LP, ssize_t blockNum, ssize_t oneScale);
// x0: dest, x1: source, x2: dequantScale, x3: realDstCount, x4: sumParams
// x5: oneScale
// Load from sp: x8: blockNum
// EP=16, LP=4, HP=16

ldr x12, [x4, #48] // Valid
ldr x8, [x4, #32]  // blockNum
ldr x5, [x4, #40]  // oneScale
ldr x14, [x4, #56] // kx*ky
ldr x15, [x4, #72] // input block quant, 0:no, 1:yes
ldr x4, [x4, #64]  // LU

stp d14, d15, [sp, #(-16 * 5)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x20, x21, [sp, #(16 * 4)]

movi v31.16b, #1
mov v29.16b, v31.16b
ld1r {v30.4s}, [x2] // Dequant scale
sdiv x4, x4, x8     // src_depth_quad per block
cbz x12, Start
mov x13, #0xFFFFFFFF
lsl x12, x12, #3
lsl x13, x13, x12
dup v28.4s, w13
bic v29.16b, v31.16b, v28.16b

Start:
mov x13, x15 // input block quant, 0:no, 1:yes

TILE_4:
cmp x3, #4
blt Remain

mov x9, x8 // blockNum
cbnz x13, TILE4_BLOCK_NUM
ld1 {v13.4s}, [x2], #16 // batch quant scale

TILE4_BLOCK_NUM:
mov x15, x14 // kx*ky
movi v9.4s, #0

/* for range(kx*ky)...for range(ic/pack) */
TILE4_BLOCK_INNER:
sub x12, x4, #1        // icDiv4
cbz x12, TILE4_LAST_QUAD

TILE4_PRE_QUAD:
ld1 {v0.16b}, [x1], #16 // E: 0,1,2,3
.inst 0x4e8097e9 // sdot v9.4s, v31.16b, v0.16b
subs x12, x12, #1      // icDiv4--
bne TILE4_PRE_QUAD

TILE4_LAST_QUAD:
ld1 {v0.16b}, [x1], #16 // E: 0,1,2,3
.inst 0x4e8097a9 // sdot v9.4s, v29.16b, v0.16b

subs x15, x15, #1
bne TILE4_BLOCK_INNER

TILE4_BLOCK_INNER_END:
subs x9, x9, #1    // blockNum--

scvtf v9.4s, v9.4s

cbnz x5, TILE4_MUL_ONE_SCALE
cbz x13, TILE4_MUL_BLOCK_SCALE
ld1 {v13.4s}, [x2], #16 // batch quant scale, input block quant
TILE4_MUL_BLOCK_SCALE:
fmul v9.4s, v9.4s, v13.4s
b TILE4_STORE

TILE4_MUL_ONE_SCALE:
fmul v9.4s, v9.4s, v30.4s

TILE4_STORE:
st1 {v9.4s}, [x0], #16
bne TILE4_BLOCK_NUM

TILE4_END:
subs x3, x3, #4 // realDstCount-=4
bne TILE_4


Remain: // remain realDstCount < EP
cbz x3, End
/* x11: Remain dstCount step for each block */
lsl x11, x3, #2
lsl x6, x3, #2 // x6=eDest * LP
mov x20, x2

TILE_2: // realDstCount >= 1
cmp x3, #2
blt TILE_1

mov x7, x1
mov x9, x8 // blockNum
mov x10, x0 // tag dst address

cbnz x13, TILE2_BLOCK_NUM
ld1 {v13.d}[0], [x2], #8 // batch quant scale

TILE2_BLOCK_NUM:
mov x15, x14 // kx*ky
movi v10.4s, #0

TILE2_BLOCK_INNER: // range(kxky)
sub x12, x4, #1    // icDiv4
cbz x12, TILE2_LAST_QUAD

TILE2_PRE_QUAD: // range(icDiv4)
ld1 {v0.d}[0], [x7], x6 // E: 0,1
subs x12, x12, #1
.inst 0x4e8097ea // sdot v10.4s, v31.16b, v0.16b // sum LP axis for E0
bne TILE2_PRE_QUAD

TILE2_LAST_QUAD:
ld1 {v0.d}[0], [x7], x6 // E: 0,1
.inst 0x4e8097aa // sdot v10.4s, v29.16b, v0.16b

subs x15, x15, #1   // kxky--
bne TILE2_BLOCK_INNER

TILE2_BLOCK_INNER_END:
scvtf v10.4s, v10.4s

cbnz x5, TILE2_MUL_ONE_SCALE
cbz x13, TILE2_MUL_BLOCK_SCALE
ld1 {v13.d}[0], [x2], x6 // batch quant scale
TILE2_MUL_BLOCK_SCALE:
fmul v10.4s, v10.4s, v13.4s
b TILE2_STORE

TILE2_MUL_ONE_SCALE:
fmul v10.4s, v10.4s, v30.4s

TILE2_STORE:
subs x9, x9, #1    // blockNum--
st1 {v10.d}[0], [x10], x11
bne TILE2_BLOCK_NUM

TILE2_END:
sub x3, x3, #2 // realDstCount-=2
add x1, x1, #8 // LP * 2
add x0, x0, #8 // finish remain 2
add x2, x20, #8 // x20 + 2 * sizeof(float)
mov x20, x2
b TILE_2


TILE_1: // realDstCount >= 1
cmp x3, #1
blt End

mov x7, x1
mov x9, x8  // blockNum
mov x10, x0

cbnz x13, TILE1_BLOCK_NUM
ld1 {v13.s}[0], [x2], #4 // batch quant scale

TILE1_BLOCK_NUM:
mov x15, x14 // kx*ky
movi v10.4s, #0

TILE1_BLOCK_INNER:
sub x12, x4, #1
cbz x12, TILE1_LAST_QUAD

TILE1_PRE_QUAD:
ld1 {v0.s}[0], [x7] // E: 0
add x7, x7, x6
.inst 0x4e8097ea // sdot v10.4s, v31.16b, v0.16b // sum LP axis for E0
subs x12, x12, #1 // icDiv4--
bne TILE1_PRE_QUAD

TILE1_LAST_QUAD:
ld1 {v0.s}[0], [x7], x6 // E: 0
.inst 0x4e8097aa // sdot v10.4s, v29.16b, v0.16b

subs x15, x15, #1 // kxky--
bne TILE1_BLOCK_INNER

TILE1_BLOCK_INNER_END:
scvtf v10.4s, v10.4s

cbnz x5, TILE1_MUL_ONE_SCALE
cbz x13, TILE1_MUL_BLOCK_SCALE
ld1 {v13.s}[0], [x2], x6 // batch quant scale
TILE1_MUL_BLOCK_SCALE:
fmul v10.4s, v10.4s, v13.4s
b TILE1_STORE

TILE1_MUL_ONE_SCALE:
fmul v10.4s, v10.4s, v30.4s

TILE1_STORE:
subs x9, x9, #1           // blockNum--
st1 {v10.s}[0], [x10], x11
bne TILE1_BLOCK_NUM

TILE1_END:
sub x3, x3, #1 // realDstCount-=1
add x1, x1, #4 // LP * 1
add x0, x0, #4 // finish remain 1
add x2, x20, #4 // x20 + 1 * sizeof(float)
mov x20, x2

b TILE_1

End:
ldp x20, x21, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 5)
ret
#endif
