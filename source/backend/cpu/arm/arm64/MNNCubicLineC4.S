//
//  MNNCubicLineC4.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNCubicLineC4
//void MNNCubicLineC4(float* dst, const float* A, const float* B, const float* C, const float* D, float* t, size_t number)

//Auto: x0:dst, x1:A, x2:B, x3:C
//x4: D, x5:t, x6:number

ld1 {v1.s}[0], [x5]

dup v22.4s, v1.s[0]

fmov s0, #5.000000000000000000e-01
dup v23.4s, v0.s[0]

subs x6, x6, #1
ld1 {v0.4s}, [x1], #16
ld1 {v1.4s}, [x2], #16
fsub v16.4s, v1.4s, v0.4s //B-A
ld1 {v2.4s}, [x3], #16
fsub v17.4s, v1.4s, v2.4s//B-C
ld1 {v3.4s}, [x4], #16

fsub v18.4s, v3.4s, v2.4s//D-C
fsub v19.4s, v2.4s, v0.4s//C-A
beq MNNCubicLineC4LoopEnd

//Compute a
//d = v1.4s
fadd v20.4s, v16.4s, v18.4s //(D-C)+(B-A)
fadd v21.4s, v17.4s, v16.4s //(B-A)+(B-C), now B-A and B-C is no used
fmla v17.4s, v20.4s, v23.4s//a=v17.4s, now v20.4s is no used

MNNCubicLineC4Loop:
    fsub v2.4s, v2.4s, v21.4s //C-((B-A)+(B-C)), now v21.4s is no used
    fadd v20.4s, v1.4s, v3.4s//B+D
    fmul v19.4s, v19.4s, v23.4s//c=v19.4s

    fmls v2.4s, v20.4s, v23.4s //b=v2.4s

    fmla v2.4s, v17.4s, v22.4s
    ld1 {v0.4s}, [x1], #16
    fmla v19.4s, v2.4s, v22.4s
    ld1 {v3.4s}, [x4], #16
    fmla v1.4s, v19.4s, v22.4s
    ld1 {v2.4s}, [x3], #16

    st1 {v1.4s}, [x0], #16

    ld1 {v1.4s}, [x2], #16
    
    fsub v16.4s, v1.4s, v0.4s //B-A
    fsub v17.4s, v1.4s, v2.4s//B-C
    fsub v18.4s, v3.4s, v2.4s//D-C
    fsub v19.4s, v2.4s, v0.4s//C-A

//Compute a
//d = v1.4s
fadd v20.4s, v16.4s, v18.4s //(D-C)+(B-A)
fadd v21.4s, v17.4s, v16.4s //(B-A)+(B-C), now B-A and B-C is no used
fmla v17.4s, v20.4s, v23.4s//a=v17.4s, now v20.4s is no used

subs x6, x6, #1
bne MNNCubicLineC4Loop

MNNCubicLineC4LoopEnd:
fsub v2.4s, v2.4s, v21.4s //C-((B-A)+(B-C)), now v21.4s is no used
fadd v20.4s, v1.4s, v3.4s//B+D
fmul v19.4s, v19.4s, v23.4s//c=v19.4s

fmls v2.4s, v20.4s, v23.4s //b=v2.4s

fmla v2.4s, v17.4s, v22.4s
fmla v19.4s, v2.4s, v22.4s
fmla v1.4s, v19.4s, v22.4s

st1 {v1.4s}, [x0]

ret
//MNNCubicLineC4 End
#endif
