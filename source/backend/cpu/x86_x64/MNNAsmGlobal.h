//
//  MNNAsmGlobal.h
//  MNN
//
//  Created by MNN on 2019/01/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNAsmGlobal_h
#define MNNAsmGlobal_h
.macro asm_function fname
#ifdef __APPLE__
.globl _\fname
_\fname:
#else
.global \fname
#ifdef __ELF__
.hidden \fname
.type \fname, %function
#endif
\fname:
#endif
.endm


#endif /* MNNAsmGlobal_h */
