#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

// void MNNC3ToYUVFast(const unsigned char* source, unsigned char* dest, size_t count);
asm_function MNNC3ToYUVFast
// x0: source, x1: dest, x2: count, x3: c
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

ld1r {v23.4s}, [x3], #4
ld1r {v24.4s}, [x3], #4
ld1r {v25.4s}, [x3], #4
ld1r {v26.4s}, [x3], #4
ld1r {v27.4s}, [x3], #4
ld1r {v28.4s}, [x3], #4
ld1r {v29.4s}, [x3], #4
ld1r {v30.4s}, [x3], #4
ld1r {v31.4s}, [x3], #4
movi v17.8h, #128

L1:
cmp x2, #1
blt End

ld3 {v0.8b, v1.8b, v2.8b}, [x0], #24
ushll v0.8h, v0.8b, #0 // r: uint8_t -> uint16_t
ushll v1.8h, v1.8b, #0
ushll v2.8h, v2.8b, #0

uxtl v3.4s, v0.4h  // r
uxtl2 v4.4s, v0.8h // r
uxtl v5.4s, v1.4h  // g
uxtl2 v6.4s, v1.8h  // g
uxtl v7.4s, v2.4h  // b
uxtl2 v8.4s, v2.8h // b

// r*C0, g*C1, b*C2
mul v9.4s, v3.4s, v23.4s
mul v10.4s, v4.4s, v23.4s
mla v9.4s, v5.4s, v24.4s
mla v10.4s, v6.4s, v24.4s
mla v9.4s, v7.4s, v25.4s
mla v10.4s, v8.4s, v25.4s

// r*C3, g*C4, b*C5
mul v15.4s, v3.4s, v26.4s
mul v16.4s, v4.4s, v26.4s
mla v15.4s, v5.4s, v27.4s
mla v16.4s, v6.4s, v27.4s
mla v15.4s, v7.4s, v28.4s
mla v16.4s, v8.4s, v28.4s

// r*C6, g*C7, b*C8
mul v21.4s, v3.4s, v29.4s
mul v22.4s, v4.4s, v29.4s
mla v21.4s, v5.4s, v30.4s
mla v22.4s, v6.4s, v30.4s
mla v21.4s, v7.4s, v31.4s
mla v22.4s, v8.4s, v31.4s

uqrshrn v11.4h, v9.4s, #14
uqrshrn2 v11.8h, v10.4s, #14
uqrshrn v12.4h, v15.4s, #14
uqrshrn2 v12.8h, v16.4s, #14
uqrshrn v13.4h, v21.4s, #14
uqrshrn2 v13.8h, v22.4s, #14

add v12.8h, v12.8h, v17.8h
add v13.8h, v13.8h, v17.8h

uqxtn v14.8b, v11.8h
uqxtn v15.8b, v12.8h
uqxtn v16.8b, v13.8h


st3 {v14.8b, v15.8b, v16.8b}, [x1], #24 
sub x2, x2, #1
b L1

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret
#endif
