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

asm_function MNNSumByAxisLForMatmul_A_ARM82
// MNNSumByAxisLForMatmul_A_ARM82(float_t* dest, int8_t* source, float* dequantScale, ssize_t realDstCount, 
//                                ssize_t kernelCountUnitDouble, ssize_t col_buffer_unit_size, ssize_t EP, ssize_t LP, ssize_t blockNum, ssize_t oneScale);
// x0: dest, x1: source, x2: dequantScale, x3: realDstCount, x4: sumParams
// x4: kernelCountUnitDouble, x5: col_buffer_unit_size
// Load from sp: x8: blockNum

ldr x8, [x4, #32] // blockNum
ldr x5, [x4, #40] // oneScale
ldr x4, [x4, #0]  // kernelCountUnitDouble

//ldr x8, [sp, #0] // blockNum

stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

movi v31.16b, #1
ld1r {v30.4s}, [x2] // Dequant scale
mov x6, #48 // EP*LP
sdiv x4, x4, x8     // src_depth_quad per block

TILE_12:
cmp x3, #12
blt Remain

mov x9, x8 // blockNum
cbnz x5, TILE12_BLOCK_NUM
ld1 {v13.4s, v14.4s, v15.4s}, [x2], #48 // batch quant scale

TILE12_BLOCK_NUM:
mov x15, x4 // kernelCountUnitDouble

movi v10.4s, #0
movi v11.4s, #0
movi v12.4s, #0

TILE12_BLOCK_INNER:

ld1 {v0.16b, v1.16b, v2.16b}, [x1], #48 // E: 0,1,2,3,...,11
subs x15, x15, #1

.inst 0x4e8097ea // sdot v10.4s, v31.16b, v0.16b // sum LP axis for E0, E1, E2, E3
.inst 0x4e8197eb // sdot v11.4s, v31.16b, v1.16b
.inst 0x4e8297ec // sdot v12.4s, v31.16b, v2.16b

bne TILE12_BLOCK_INNER

TILE12_BLOCK_INNER_END:
subs x9, x9, #1    // blockNum--

scvtf v10.4s, v10.4s
scvtf v11.4s, v11.4s
scvtf v12.4s, v12.4s

cbnz x5, TILE12_MUL_ONE_SCALE
fmul v10.4s, v10.4s, v13.4s
fmul v11.4s, v11.4s, v14.4s
fmul v12.4s, v12.4s, v15.4s
b TILE12_STORE

TILE12_MUL_ONE_SCALE:
fmul v10.4s, v10.4s, v30.4s
fmul v11.4s, v11.4s, v30.4s
fmul v12.4s, v12.4s, v30.4s

TILE12_STORE:
st1 {v10.4s, v11.4s, v12.4s}, [x0], #48
bne TILE12_BLOCK_NUM

TILE12_END:
subs x3, x3, #12 // realDstCount-=12
bne TILE_12


Remain: // remain realDstCount < EP
cbz x3, End
/* x11: Remain dstCount step for each block */
lsl x11, x3, #2

TILE_2: // realDstCount >= 1
cmp x3, #2
blt TILE_1

mov x7, x1
mov x9, x8 // blockNum
mov x10, x0 // tag dst address

cbnz x5, TILE2_BLOCK_NUM
ld1 {v13.d}[0], [x2], #8 // batch quant scale

TILE2_BLOCK_NUM:
mov x15, x4 // kernelCountUnitDouble
movi v10.4s, #0

TILE2_BLOCK_INNER:
ld1 {v0.d}[0], [x7] // E: 0,1
add x7, x7, x6
subs x15, x15, #1
.inst 0x4e8097ea // sdot v10.4s, v31.16b, v0.16b // sum LP axis for E0
bne TILE2_BLOCK_INNER

TILE2_BLOCK_INNER_ENd:
scvtf v10.4s, v10.4s

cbnz x5, TILE2_MUL_ONE_SCALE
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
b TILE_2


TILE_1: // realDstCount >= 1
cmp x3, #1
blt End

mov x7, x1
mov x9, x8  // blockNum
mov x10, x0

cbnz x5, TILE1_BLOCK_NUM
ld1 {v13.s}[0], [x2], #4 // batch quant scale

TILE1_BLOCK_NUM:
mov x15, x4 // kernelCountUnitDouble
movi v10.4s, #0

TILE1_BLOCK_INNER:
ld1 {v0.s}[0], [x7] // E: 0
subs x15, x15, #1
add x7, x7, x6
.inst 0x4e8097ea // sdot v10.4s, v31.16b, v0.16b // sum LP axis for E0

bne TILE1_BLOCK_INNER

TILE1_BLOCK_INNER_END:
scvtf v10.4s, v10.4s

cbnz x5, TILE1_MUL_ONE_SCALE
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

b TILE_1

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret
#endif