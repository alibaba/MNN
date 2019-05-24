//
//  MNNMatrixMax.S
//  MNN
//
//  Created by MNN on 2019/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
// void MNNPowC8(float* dest, const float* source, const float* powfParam, size_t betaInt, size_t countC8)
asm_function MNNPowC8

// Auto: x0:dest, x1:source, x2:powfParam, x3:betaInt, x4:countC8

ld1 {v4.4s, v5.4s}, [x2], #32

fmov s16, #1.5
frecpe s17, s16
fmov s16, #1.25
mov s18, v5.s[2]
fmov v19.4s, #1.0

Loop:
// (v0, v1): src, (v2, v3): dst, (v4, v5): param, (v6, v7): 1/src
ld1 {v0.4s, v1.4s}, [x1], #32
fmov v2.4s, #1.0
fmov v3.4s, #1.0

cmp x3, #0
beq endSubLoop1

frecpe v6.4s, v0.4s
frecpe v7.4s, v1.4s
mov x2, x3

SubLoop1:
fmul v2.4s, v2.4s, v6.4s
fmul v3.4s, v3.4s, v7.4s
subs x2, x2, #1
bne SubLoop1

endSubLoop1:

.macro SUB_WORK loopDef z0 z1
// z0: x, z1: result
mov v6.s[0], \z0
fcmp s6, s16
blt end\loopDef
mov v7.s[0], \z1
\loopDef:
fmul s6, s6, s17
fmul s7, s7, s18
fcmp s6, s16
bge \loopDef
mov \z0, v6.s[0]
mov \z1, v7.s[0]
end\loopDef:
.endm

SUB_WORK subLoop0, v0.s[0], v2.s[0]
SUB_WORK subLoop1, v1.s[0], v3.s[0]
SUB_WORK subLoop2, v0.s[1], v2.s[1]
SUB_WORK subLoop3, v1.s[1], v3.s[1]
SUB_WORK subLoop4, v0.s[2], v2.s[2]
SUB_WORK subLoop5, v1.s[2], v3.s[2]
SUB_WORK subLoop6, v0.s[3], v2.s[3]
SUB_WORK subLoop7, v1.s[3], v3.s[3]

fsub v0.4s, v0.4s, v19.4s
fsub v1.4s, v1.4s, v19.4s

.macro MLA_TWO z0 z1 z2 z3
dup \z1, \z0
fmla \z1, \z2, \z3
.endm

MLA_TWO v5.s[0], v6.4s, v0.4s, v5.s[1]
MLA_TWO v5.s[0], v7.4s, v1.4s, v5.s[1]
MLA_TWO v4.s[3], v20.4s, v0.4s, v6.4s
MLA_TWO v4.s[3], v21.4s, v1.4s, v7.4s
MLA_TWO v4.s[2], v6.4s, v0.4s, v20.4s
MLA_TWO v4.s[2], v7.4s, v1.4s, v21.4s
MLA_TWO v4.s[1], v20.4s, v0.4s, v6.4s
MLA_TWO v4.s[1], v21.4s, v1.4s, v7.4s
MLA_TWO v4.s[0], v6.4s, v0.4s, v20.4s
MLA_TWO v4.s[0], v7.4s, v1.4s, v21.4s

fmul v2.4s, v2.4s, v6.4s
fmul v3.4s, v3.4s, v7.4s
st1 {v2.4s, v3.4s}, [x0], #32

subs x4, x4, #1
bne Loop

ret
#endif
