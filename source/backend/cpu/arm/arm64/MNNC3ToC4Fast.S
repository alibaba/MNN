#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
// void MNNC3ToC4Fast(const unsigned char* source, unsigned char* dest, size_t count);
asm_function MNNC3ToC4Fast
// x0: source, x1: dest, x2: count
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

movi v3.16b, #255
movi v7.16b, #255
movi v11.16b, #255
movi v15.16b, #255
movi v19.16b, #255
movi v23.16b, #255
movi v27.16b, #255
movi v31.16b, #255

L16:
cmp x2, #16
blt L12
ld3 {v0.16b, v1.16b, v2.16b}, [x0], #48
ld3 {v4.16b, v5.16b, v6.16b}, [x0], #48
ld3 {v8.16b, v9.16b, v10.16b}, [x0], #48
ld3 {v12.16b, v13.16b, v14.16b}, [x0], #48
ld3 {v16.16b, v17.16b, v18.16b}, [x0], #48
ld3 {v20.16b, v21.16b, v22.16b}, [x0], #48
ld3 {v24.16b, v25.16b, v26.16b}, [x0], #48
ld3 {v28.16b, v29.16b, v30.16b}, [x0], #48
sub x2, x2, #16

st4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
st4 {v4.16b, v5.16b, v6.16b, v7.16b}, [x1], #64
st4 {v8.16b, v9.16b, v10.16b, v11.16b}, [x1], #64
st4 {v12.16b, v13.16b, v14.16b, v15.16b}, [x1], #64
st4 {v16.16b, v17.16b, v18.16b, v19.16b}, [x1], #64
st4 {v20.16b, v21.16b, v22.16b, v23.16b}, [x1], #64
st4 {v24.16b, v25.16b, v26.16b, v27.16b}, [x1], #64
st4 {v28.16b, v29.16b, v30.16b, v31.16b}, [x1], #64
b L16

L12:
cmp x2, #12
blt L8
ld3 {v0.16b, v1.16b, v2.16b}, [x0], #48
ld3 {v4.16b, v5.16b, v6.16b}, [x0], #48
ld3 {v8.16b, v9.16b, v10.16b}, [x0], #48
ld3 {v12.16b, v13.16b, v14.16b}, [x0], #48
ld3 {v16.16b, v17.16b, v18.16b}, [x0], #48
ld3 {v20.16b, v21.16b, v22.16b}, [x0], #48
sub x2, x2, #12

st4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
st4 {v4.16b, v5.16b, v6.16b, v7.16b}, [x1], #64
st4 {v8.16b, v9.16b, v10.16b, v11.16b}, [x1], #64
st4 {v12.16b, v13.16b, v14.16b, v15.16b}, [x1], #64
st4 {v16.16b, v17.16b, v18.16b, v19.16b}, [x1], #64
st4 {v20.16b, v21.16b, v22.16b, v23.16b}, [x1], #64

b L12


L8:
cmp x2, #8
blt L4
ld3 {v0.16b, v1.16b, v2.16b}, [x0], #48
ld3 {v4.16b, v5.16b, v6.16b}, [x0], #48
ld3 {v8.16b, v9.16b, v10.16b}, [x0], #48
ld3 {v12.16b, v13.16b, v14.16b}, [x0], #48
sub x2, x2, #8

st4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
st4 {v4.16b, v5.16b, v6.16b, v7.16b}, [x1], #64
st4 {v8.16b, v9.16b, v10.16b, v11.16b}, [x1], #64
st4 {v12.16b, v13.16b, v14.16b, v15.16b}, [x1], #64
b L8

L4:
cmp x2, #4
blt L2
ld3 {v0.16b, v1.16b, v2.16b}, [x0], #48
ld3 {v4.16b, v5.16b, v6.16b}, [x0], #48
sub x2, x2, #4

st4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
st4 {v4.16b, v5.16b, v6.16b, v7.16b}, [x1], #64
b L4

L2:
cmp x2, #2
blt L1
ld3 {v0.16b, v1.16b, v2.16b}, [x0], #48
sub x2, x2, #2

st4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
b L2

L1:
cmp x2, #1
blt End
ld3 {v0.8b, v1.8b, v2.8b}, [x0], #24

st4 {v0.8b, v1.8b, v2.8b, v3.8b}, [x1], #32

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret
#endif
