//
//  _AVX512_MNNGemmFloatUnitMainFMA.S
//  MNN
//
//  Created by MNN on 2020/12/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../MNNAsmGlobal.h"
.text
.align 4

asm_function _AVX512_MNNGemmFloatUnitMainFMA
//void _AVX512_MNNGemmFloatUnitMainFMA(float* C, const float* A, const float* B, const size_t* parameter, size_t hC4)

// SystemV Auto: rdi: C, rsi:A, rdx:B, rcx:parameter, r8: hC4
// Microsoft x64 Auto: rcx:C, rdx:A, r8:B, r9:parameter
pushq   %rbp
movq    %rsp, %rbp

#ifdef WIN32
movq 48(%rsp), %r10
pushq %rdi
pushq %rsi
pushq %r12
pushq %r13
movq %rcx, %rdi
movq %rdx, %rsi
movq %r8, %rdx
movq %r9, %rcx
movq %r10, %r9
#else
pushq   %r12
pushq   %r13
movq %r8, %r9
#endif

movq 40(%rcx), %r10 // bExtraStride
movq 24(%rcx), %r8 // cStride
movq 8(%rcx), %rcx // l

cmpq $0, %r9
je End

// zmm8-zmm31: Dst
// zmm0-zmm3: Src
// zmm4-zmm7: W

addq $3, %rcx
shrq $2, %rcx // l -> lC4
movq %rsi, %r13

LoopDz:
    movq %rcx, %r11
    movq %r13, %rsi

    subq $1, %r11

    vbroadcastf32x4 (%rdx), %zmm4
    vbroadcastf32x4 16(%rdx), %zmm5
    vbroadcastf32x4 32(%rdx), %zmm6
    vbroadcastf32x4 48(%rdx), %zmm7

    vmovups (%rsi), %zmm0
    vmovups 64(%rsi), %zmm1
    vmovups 128(%rsi), %zmm2
    vmovups 192(%rsi), %zmm3

    vmulps %zmm0, %zmm4, %zmm8
    vmulps %zmm0, %zmm5, %zmm9
    vmulps %zmm0, %zmm6, %zmm10
    vmulps %zmm0, %zmm7, %zmm11

    vmulps %zmm1, %zmm4, %zmm12
    vmulps %zmm1, %zmm5, %zmm13
    vmulps %zmm1, %zmm6, %zmm14
    vmulps %zmm1, %zmm7, %zmm15

    vmulps %zmm2, %zmm4, %zmm16
    vmulps %zmm2, %zmm5, %zmm17
    vmulps %zmm2, %zmm6, %zmm18
    vmulps %zmm2, %zmm7, %zmm19

    vmovups 256(%rsi), %zmm0

    vmulps %zmm3, %zmm4, %zmm20
    vmulps %zmm3, %zmm5, %zmm21
    vmulps %zmm3, %zmm6, %zmm22
    vmulps %zmm3, %zmm7, %zmm23

    vmovups 320(%rsi), %zmm1

    vmulps %zmm0, %zmm4, %zmm24
    vmulps %zmm0, %zmm5, %zmm25
    vmulps %zmm0, %zmm6, %zmm26
    vmulps %zmm0, %zmm7, %zmm27

    vmulps %zmm1, %zmm4, %zmm28
    vmulps %zmm1, %zmm5, %zmm29
    vmulps %zmm1, %zmm6, %zmm30
    vmulps %zmm1, %zmm7, %zmm31

    addq $64, %rdx
    addq $384, %rsi

    cmpq $0, %r11
    je LoopSzEnd

    LoopSz:
        vbroadcastf32x4 (%rdx), %zmm4
        vbroadcastf32x4 16(%rdx), %zmm5
        vbroadcastf32x4 32(%rdx), %zmm6
        vbroadcastf32x4 48(%rdx), %zmm7

        vmovups (%rsi), %zmm0
        vmovups 64(%rsi), %zmm1
        vmovups 128(%rsi), %zmm2
        vmovups 192(%rsi), %zmm3

        vfmadd231ps %zmm0, %zmm4, %zmm8
        vfmadd231ps %zmm0, %zmm5, %zmm9
        vfmadd231ps %zmm0, %zmm6, %zmm10
        vfmadd231ps %zmm0, %zmm7, %zmm11

        vfmadd231ps %zmm1, %zmm4, %zmm12
        vfmadd231ps %zmm1, %zmm5, %zmm13
        vfmadd231ps %zmm1, %zmm6, %zmm14
        vfmadd231ps %zmm1, %zmm7, %zmm15

        vfmadd231ps %zmm2, %zmm4, %zmm16
        vfmadd231ps %zmm2, %zmm5, %zmm17
        vfmadd231ps %zmm2, %zmm6, %zmm18
        vfmadd231ps %zmm2, %zmm7, %zmm19

        vmovups 256(%rsi), %zmm0

        vfmadd231ps %zmm3, %zmm4, %zmm20
        vfmadd231ps %zmm3, %zmm5, %zmm21
        vfmadd231ps %zmm3, %zmm6, %zmm22
        vfmadd231ps %zmm3, %zmm7, %zmm23

        vmovups 320(%rsi), %zmm1

        vfmadd231ps %zmm0, %zmm4, %zmm24
        vfmadd231ps %zmm0, %zmm5, %zmm25
        vfmadd231ps %zmm0, %zmm6, %zmm26
        vfmadd231ps %zmm0, %zmm7, %zmm27

        vfmadd231ps %zmm1, %zmm4, %zmm28
        vfmadd231ps %zmm1, %zmm5, %zmm29
        vfmadd231ps %zmm1, %zmm6, %zmm30
        vfmadd231ps %zmm1, %zmm7, %zmm31

        addq $64, %rdx
        addq $384, %rsi

        subq $1, %r11
        cmpq $0, %r11

        jne LoopSz
    LoopSzEnd:

.macro HADD_SAVE x0, x1, x2, x3
    vextractf64x4 $0, \x0, %ymm0
    vextractf64x4 $1, \x0, %ymm1

    vextractf64x4 $0, \x1, %ymm2
    vextractf64x4 $1, \x1, %ymm3

    vextractf64x4 $0, \x2, %ymm4
    vextractf64x4 $1, \x2, %ymm5

    vextractf64x4 $0, \x3, %ymm6
    vextractf64x4 $1, \x3, %ymm7

    vhaddps %ymm2, %ymm0, %ymm0
    vhaddps %ymm6, %ymm4, %ymm4
    vhaddps %ymm3, %ymm1, %ymm1
    vhaddps %ymm7, %ymm5, %ymm5

    vhaddps %ymm4, %ymm0, %ymm0
    vhaddps %ymm5, %ymm1, %ymm1

    vmovups %ymm0, (%r11)
    vmovups %ymm1, 32(%r11)
.endm
    movq %rdi, %r11

    HADD_SAVE %zmm8, %zmm9, %zmm10, %zmm11

    addq $64, %r11
    HADD_SAVE %zmm12, %zmm13, %zmm14, %zmm15

    addq $64, %r11
    HADD_SAVE %zmm16, %zmm17, %zmm18, %zmm19

    addq $64, %r11
    HADD_SAVE %zmm20, %zmm21, %zmm22, %zmm23

    addq $64, %r11
    HADD_SAVE %zmm24, %zmm25, %zmm26, %zmm27

    addq $64, %r11
    HADD_SAVE %zmm28, %zmm29, %zmm30, %zmm31

    addq %r10, %rdx
    addq %r8, %rdi
    subq $1, %r9
    testq %r9, %r9
    jne LoopDz


End:

#ifdef WIN32
popq    %r13
popq    %r12
popq    %rsi
popq    %rdi
popq    %rbp
#else
popq    %r13
popq    %r12
popq    %rbp
#endif

retq

