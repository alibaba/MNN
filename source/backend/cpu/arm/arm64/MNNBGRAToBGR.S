#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
// void MNNBGRAToBGRC8(const unsigned char* source, unsigned char* dest, size_t count);
asm_function MNNBGRAToBGRC8
// x0: source, x1: dest, x2: count
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

L12:
cmp x2, #12
blt L8
ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64
ld4 {v4.16b, v5.16b, v6.16b, v7.16b}, [x0], #64
ld4 {v8.16b, v9.16b, v10.16b, v11.16b}, [x0], #64
ld4 {v12.16b, v13.16b, v14.16b, v15.16b}, [x0], #64
ld4 {v28.16b, v29.16b, v30.16b, v31.16b}, [x0], #64
sub x2, x2, #12
mov v16.16b, v0.16b
mov v17.16b, v1.16b
mov v18.16b, v2.16b
mov v19.16b, v4.16b
mov v20.16b, v5.16b
mov v21.16b, v6.16b

mov v22.16b, v8.16b
mov v23.16b, v9.16b
mov v24.16b, v10.16b
mov v25.16b, v12.16b
mov v26.16b, v13.16b
mov v27.16b, v14.16b

ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64

mov v4.16b, v28.16b
mov v5.16b, v29.16b
mov v6.16b, v30.16b
mov v8.16b, v0.16b
mov v9.16b, v1.16b
mov v10.16b, v2.16b


st3 {v16.16b, v17.16b, v18.16b}, [x1], #48
st3 {v19.16b, v20.16b, v21.16b}, [x1], #48
st3 {v22.16b, v23.16b, v24.16b}, [x1], #48
st3 {v25.16b, v26.16b, v27.16b}, [x1], #48
st3 {v4.16b, v5.16b, v6.16b}, [x1], #48
st3 {v8.16b, v9.16b, v10.16b}, [x1], #48

b L12


L8:
cmp x2, #8
blt L4
ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64
ld4 {v4.16b, v5.16b, v6.16b, v7.16b}, [x0], #64
ld4 {v8.16b, v9.16b, v10.16b, v11.16b}, [x0], #64
ld4 {v12.16b, v13.16b, v14.16b, v15.16b}, [x0], #64
sub x2, x2, #8
mov v16.16b, v0.16b
mov v17.16b, v1.16b
mov v18.16b, v2.16b
mov v19.16b, v4.16b
mov v20.16b, v5.16b
mov v21.16b, v6.16b

mov v22.16b, v8.16b
mov v23.16b, v9.16b
mov v24.16b, v10.16b
mov v25.16b, v12.16b
mov v26.16b, v13.16b
mov v27.16b, v14.16b

st3 {v16.16b, v17.16b, v18.16b}, [x1], #48
st3 {v19.16b, v20.16b, v21.16b}, [x1], #48
st3 {v22.16b, v23.16b, v24.16b}, [x1], #48
st3 {v25.16b, v26.16b, v27.16b}, [x1], #48
b L8

L4:
cmp x2, #4
blt L2
ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64
ld4 {v6.16b, v7.16b, v8.16b, v9.16b}, [x0], #64
sub x2, x2, #4
mov v10.16b, v0.16b
mov v11.16b, v1.16b
mov v12.16b, v2.16b
mov v13.16b, v6.16b
mov v14.16b, v7.16b
mov v15.16b, v8.16b

st3 {v10.16b, v11.16b, v12.16b}, [x1], #48
st3 {v13.16b, v14.16b, v15.16b}, [x1], #48
b L4

L2:
cmp x2, #2
blt L1
ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64
mov v4.16b, v0.16b
mov v5.16b, v1.16b
mov v6.16b, v2.16b
sub x2, x2, #2
st3 {v4.16b, v5.16b, v6.16b}, [x1], #48
b L2

L1:
cmp x2, #1
blt End
ld4 {v0.8b, v1.8b, v2.8b, v3.8b}, [x0], #32
mov v5.8b, v0.8b
mov v6.8b, v1.8b
mov v7.8b, v2.8b
st3 {v5.8b, v6.8b, v7.8b}, [x1], #24

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret
#endif
