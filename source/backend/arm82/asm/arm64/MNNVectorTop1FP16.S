//
//  MNNVectorTop1FP16.S
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.align 4
.LcNegInf: .float -1.0e30

.macro ARGMAX_LANE laneReg
    dup v3.4s, \laneReg
    fcmp s3, s4
    b.le 1f
    fmov s4, s3
    mov w5, w6
1:
    add w6, w6, #1
.endm

asm_function MNNVectorTop1FP16
//void MNNVectorTop1FP16(const FLOAT16* input, FLOAT16* maxValue, int32_t* maxIndex, size_t count)
//x0:input, x1:maxValue, x2:maxIndex, x3:count
cbz x3, .Lexit
ldr s4, .LcNegInf
mov w5, wzr
mov w6, wzr
mov x7, x0
.Lgroup:
cmp x3, #8
b.lt .Ltail
ld1 {v0.8h}, [x7], #16
fcvtl v1.4s, v0.4h
fcvtl2 v2.4s, v0.8h
ARGMAX_LANE v1.s[0]
ARGMAX_LANE v1.s[1]
ARGMAX_LANE v1.s[2]
ARGMAX_LANE v1.s[3]
ARGMAX_LANE v2.s[0]
ARGMAX_LANE v2.s[1]
ARGMAX_LANE v2.s[2]
ARGMAX_LANE v2.s[3]
sub x3, x3, #8
b .Lgroup
.Ltail:
cbz x3, .Lstore
.Ltail_loop:
ld1 {v0.h}[0], [x7], #2
fcvtl v1.4s, v0.4h
ARGMAX_LANE v1.s[0]
subs x3, x3, #1
b.ne .Ltail_loop
.Lstore:
fcvt h4, s4
str h4, [x1]
str w5, [x2]
.Lexit:
ret

#endif