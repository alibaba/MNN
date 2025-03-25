#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

// void MNNRGBAToGRAYFast(const unsigned char* source, unsigned char* dest, size_t count);
asm_function MNNRGBAToGRAYFast
// x0: source, x1: dest, x2: count
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

movi v29.16b, #19
movi v30.16b, #38
movi v31.16b, #7

// b*7
// g*38
// r*19

L4:
cmp x2, #4
blt L2

sub x2, x2, #4
ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64
ld4 {v14.16b, v15.16b, v16.16b, v17.16b}, [x0], #64

umull v4.8h, v0.8b, v29.8b
umlal v4.8h, v1.8b, v30.8b
umlal v4.8h, v2.8b, v31.8b

umull2 v7.8h, v0.16b, v29.16b
umlal2 v7.8h, v1.16b, v30.16b
umlal2 v7.8h, v2.16b, v31.16b

umull v18.8h, v14.8b, v29.8b
umlal v18.8h, v15.8b, v30.8b
umlal v18.8h, v16.8b, v31.8b

umull2 v21.8h, v14.16b, v29.16b
umlal2 v21.8h, v15.16b, v30.16b
umlal2 v21.8h, v16.16b, v31.16b

uqshrn v4.8b, v4.8h, #6
uqshrn2 v4.16b, v7.8h, #6
uqshrn v5.8b, v18.8h, #6
uqshrn2 v5.16b, v21.8h, #6

st1 {v4.16b, v5.16b}, [x1], #32
b L4

L2:
cmp x2, #2
blt L1

sub x2, x2, #2
ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64

umull v4.8h, v0.8b, v29.8b
umlal v4.8h, v1.8b, v30.8b
umlal v4.8h, v2.8b, v31.8b

umull2 v7.8h, v0.16b, v29.16b
umlal2 v7.8h, v1.16b, v30.16b
umlal2 v7.8h, v2.16b, v31.16b

uqshrn v4.8b, v4.8h, #6
uqshrn2 v4.16b, v7.8h, #6

st1 {v4.16b}, [x1], #16
b L2

L1:
cmp x2, #1
blt End
ld4 {v0.8b, v1.8b, v2.8b, v3.8b}, [x0], #32

umull v4.8h, v0.8b, v29.8b
umlal v4.8h, v1.8b, v30.8b
umlal v4.8h, v2.8b, v31.8b

uqshrn v10.8b, v4.8h, #6

st1 {v10.8b}, [x1], #8

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret
#endif
