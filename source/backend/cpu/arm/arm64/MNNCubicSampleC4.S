//
//  MNNCubicSampleC4.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNCubicSampleC4
//void MNNCubicSampleC4(const float* src, float* dst, int* position, const float* factor, size_t number)


//Auto: x0:src, x1:dst, x2:position, x3:factor
//x4:number
fmov s0, #5.000000000000000000e-01
dup v23.4s, v0.s[0]
mov w12, #16//sizeof(float)*4

CubicSampleLoop:
ld1 {v1.s}[0], [x3], #4

dup v22.4s, v1.s[0]
ld1 {v21.4s}, [x2], #16
dup v20.4s, w12
mul v21.4s, v20.4s, v21.4s

mov x5, #0
mov x6, #0

mov w5, v21.s[0]
mov w6, v21.s[1]

add x5, x0, x5
add x6, x0, x6

//A
ld1 {v0.4s}, [x5]
mov x7, #0
mov x5, #0
mov w7, v21.s[2]
mov w5, v21.s[3]
//B
ld1 {v1.4s}, [x6]
add x7, x0, x7
add x5, x0, x5
//C
ld1 {v2.4s}, [x7]
fsub v16.4s, v1.4s, v0.4s //B-A
//D
ld1 {v3.4s}, [x5]
fsub v17.4s, v1.4s, v2.4s//B-C
fsub v18.4s, v3.4s, v2.4s//D-C
fsub v19.4s, v2.4s, v0.4s//C-A

//Compute a
//d = v1.4s
fadd v20.4s, v16.4s, v18.4s //(D-C)+(B-A)
fadd v21.4s, v17.4s, v16.4s //(B-A)+(B-C), now B-A and B-C is no used
fmla v17.4s, v20.4s, v23.4s//a=v17.4s, now v20.4s is no used

fsub v2.4s, v2.4s, v21.4s //C-((B-A)+(B-C)), now v21.4s is no used
fadd v20.4s, v1.4s, v3.4s//B+D
fmul v19.4s, v19.4s, v23.4s//c=v19.4s

fmls v2.4s, v20.4s, v23.4s //b=v2.4s

fmla v2.4s, v17.4s, v22.4s
fmla v19.4s, v2.4s, v22.4s
fmla v1.4s, v19.4s, v22.4s

st1 {v1.4s}, [x1], #16

subs x4, x4, #1
bne CubicSampleLoop


ret

#endif
