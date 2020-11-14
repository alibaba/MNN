//
//  MNNNV21ToBGRAUnit.S
//  MNN
//
//  Created by MNN on 2018/12/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
//void MNNNV21ToBGRAUnit(const unsigned char* source, unsigned char* dest, size_t count, const unsigned char* uv);
//Auto: x0:source, x1:dest, x2:count, x3:uv
asm_function MNNNV21ToBGRAUnit

mov w4, #73
movi v31.8b, #128
mov w12, #25
mov v0.h[0], w4
mov v0.h[1], w12
mov w4, #37
mov w12, #130
mov v0.h[2], w4
mov v0.h[3], w12
movi v30.8h, #0
movi v7.8b, #255
movi v26.8b, #255


LoopL1:
ld2 {v19.8b, v20.8b}, [x3], #16
ld2 {v16.8b, v17.8b}, [x0], #16

usubl v18.8h, v20.8b, v31.8b
usubl v19.8h, v19.8b, v31.8b

//v1.4s-v3.4s: BGR offset
mul v1.8h, v18.8h, v0.h[3]// + B Offset
mul v2.8h, v18.8h, v0.h[1]
mla v2.8h, v19.8h, v0.h[2]// + G Offset
mul v3.8h, v19.8h, v0.h[0]// - R Offset

ushll v16.8h, v16.8b, #6
ushll v17.8h, v17.8b, #6

add v20.8h, v16.8h, v1.8h
sub v21.8h, v16.8h, v2.8h
add v22.8h, v16.8h, v3.8h

smax v20.8h, v20.8h, v30.8h
smax v21.8h, v21.8h, v30.8h
smax v22.8h, v22.8h, v30.8h

uqshrn v20.8b, v20.8h, #6
uqshrn v21.8b, v21.8h, #6
uqshrn v22.8b, v22.8h, #6

add v23.8h, v17.8h, v1.8h
sub v24.8h, v17.8h, v2.8h
add v25.8h, v17.8h, v3.8h

smax v23.8h, v23.8h, v30.8h
smax v24.8h, v24.8h, v30.8h
smax v25.8h, v25.8h, v30.8h

uqshrn v23.8b, v23.8h, #6
uqshrn v24.8b, v24.8h, #6
uqshrn v25.8b, v25.8h, #6

zip1 v4.8b, v20.8b, v23.8b
zip2 v23.8b, v20.8b, v23.8b
zip1 v5.8b, v21.8b, v24.8b
zip2 v24.8b, v21.8b, v24.8b
zip1 v6.8b, v22.8b, v25.8b
zip2 v25.8b, v22.8b, v25.8b

st4 {v4.8b, v5.8b, v6.8b, v7.8b}, [x1], #32
st4 {v23.8b, v24.8b, v25.8b, v26.8b}, [x1], #32

subs x2, x2, #1
bne LoopL1

ret
#endif
