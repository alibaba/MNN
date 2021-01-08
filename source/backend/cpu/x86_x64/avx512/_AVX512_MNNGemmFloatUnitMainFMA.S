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

// zmm8-zmm31: Dst
// zmm0-zmm3: Src
// zmm4-zmm7: W

movq %rsi, %r13
cmpq $2, %r9
jl LD1

LoopDz2:
    movq %rcx, %r11
    movq %r13, %rsi

    subq $1, %r11

    vmovups (%rsi), %zmm0
    vmovups 64(%rsi), %zmm1
    vmovups 128(%rsi), %zmm2

    vbroadcastss (%rdx), %zmm4
    vmulps %zmm0, %zmm4, %zmm8
    vmulps %zmm1, %zmm4, %zmm9
    vmulps %zmm2, %zmm4, %zmm10

    vbroadcastss 4(%rdx), %zmm5
    vmulps %zmm0, %zmm5, %zmm11
    vmulps %zmm1, %zmm5, %zmm12
    vmulps %zmm2, %zmm5, %zmm13

    vbroadcastss 8(%rdx), %zmm6
    vmulps %zmm0, %zmm6, %zmm14
    vmulps %zmm1, %zmm6, %zmm15
    vmulps %zmm2, %zmm6, %zmm16

    vbroadcastss 12(%rdx), %zmm7
    vmulps %zmm0, %zmm7, %zmm17
    vmulps %zmm1, %zmm7, %zmm18
    vmulps %zmm2, %zmm7, %zmm19

    vbroadcastss 16(%rdx), %zmm4
    vmulps %zmm0, %zmm4, %zmm20
    vmulps %zmm1, %zmm4, %zmm21
    vmulps %zmm2, %zmm4, %zmm22

    vbroadcastss 20(%rdx), %zmm5
    vmulps %zmm0, %zmm5, %zmm23
    vmulps %zmm1, %zmm5, %zmm24
    vmulps %zmm2, %zmm5, %zmm25

    vbroadcastss 24(%rdx), %zmm6
    vmulps %zmm0, %zmm6, %zmm26
    vmulps %zmm1, %zmm6, %zmm27
    vmulps %zmm2, %zmm6, %zmm28

    vbroadcastss 28(%rdx), %zmm7
    vmulps %zmm0, %zmm7, %zmm29
    vmulps %zmm1, %zmm7, %zmm30
    vmulps %zmm2, %zmm7, %zmm31

    addq $32, %rdx
    addq $192, %rsi

    cmpq $2, %r11
    jl LastS1
    
    LoopSz:
        vmovups (%rsi), %zmm0
        vmovups 64(%rsi), %zmm1
        vmovups 128(%rsi), %zmm2

        vbroadcastss (%rdx), %zmm4
        vfmadd231ps %zmm0, %zmm4, %zmm8
        vfmadd231ps %zmm1, %zmm4, %zmm9
        vfmadd231ps %zmm2, %zmm4, %zmm10

        vbroadcastss 4(%rdx), %zmm5
        vfmadd231ps %zmm0, %zmm5, %zmm11
        vfmadd231ps %zmm1, %zmm5, %zmm12
        vfmadd231ps %zmm2, %zmm5, %zmm13

        vbroadcastss 8(%rdx), %zmm6
        vfmadd231ps %zmm0, %zmm6, %zmm14
        vfmadd231ps %zmm1, %zmm6, %zmm15
        vfmadd231ps %zmm2, %zmm6, %zmm16

        vbroadcastss 12(%rdx), %zmm7
        vfmadd231ps %zmm0, %zmm7, %zmm17
        vfmadd231ps %zmm1, %zmm7, %zmm18
        vfmadd231ps %zmm2, %zmm7, %zmm19

        vbroadcastss 16(%rdx), %zmm4
        vfmadd231ps %zmm0, %zmm4, %zmm20
        vfmadd231ps %zmm1, %zmm4, %zmm21
        vfmadd231ps %zmm2, %zmm4, %zmm22

        vbroadcastss 20(%rdx), %zmm5
        vfmadd231ps %zmm0, %zmm5, %zmm23
        vfmadd231ps %zmm1, %zmm5, %zmm24
        vfmadd231ps %zmm2, %zmm5, %zmm25

        vbroadcastss 24(%rdx), %zmm6
        vfmadd231ps %zmm0, %zmm6, %zmm26
        vfmadd231ps %zmm1, %zmm6, %zmm27
        vfmadd231ps %zmm2, %zmm6, %zmm28

        vbroadcastss 28(%rdx), %zmm7
        vfmadd231ps %zmm0, %zmm7, %zmm29
        vfmadd231ps %zmm1, %zmm7, %zmm30
        vfmadd231ps %zmm2, %zmm7, %zmm31

        vmovups 192(%rsi), %zmm0
        vmovups 256(%rsi), %zmm1
        vmovups 320(%rsi), %zmm2

        vbroadcastss 32(%rdx), %zmm4
        vfmadd231ps %zmm0, %zmm4, %zmm8
        vfmadd231ps %zmm1, %zmm4, %zmm9
        vfmadd231ps %zmm2, %zmm4, %zmm10

        vbroadcastss 36(%rdx), %zmm5
        vfmadd231ps %zmm0, %zmm5, %zmm11
        vfmadd231ps %zmm1, %zmm5, %zmm12
        vfmadd231ps %zmm2, %zmm5, %zmm13

        vbroadcastss 40(%rdx), %zmm6
        vfmadd231ps %zmm0, %zmm6, %zmm14
        vfmadd231ps %zmm1, %zmm6, %zmm15
        vfmadd231ps %zmm2, %zmm6, %zmm16

        vbroadcastss 44(%rdx), %zmm7
        vfmadd231ps %zmm0, %zmm7, %zmm17
        vfmadd231ps %zmm1, %zmm7, %zmm18
        vfmadd231ps %zmm2, %zmm7, %zmm19

        vbroadcastss 48(%rdx), %zmm4
        vfmadd231ps %zmm0, %zmm4, %zmm20
        vfmadd231ps %zmm1, %zmm4, %zmm21
        vfmadd231ps %zmm2, %zmm4, %zmm22

        vbroadcastss 52(%rdx), %zmm5
        vfmadd231ps %zmm0, %zmm5, %zmm23
        vfmadd231ps %zmm1, %zmm5, %zmm24
        vfmadd231ps %zmm2, %zmm5, %zmm25

        vbroadcastss 56(%rdx), %zmm6
        vfmadd231ps %zmm0, %zmm6, %zmm26
        vfmadd231ps %zmm1, %zmm6, %zmm27
        vfmadd231ps %zmm2, %zmm6, %zmm28

        vbroadcastss 60(%rdx), %zmm7
        vfmadd231ps %zmm0, %zmm7, %zmm29
        vfmadd231ps %zmm1, %zmm7, %zmm30
        vfmadd231ps %zmm2, %zmm7, %zmm31

        addq $64, %rdx
        addq $384, %rsi
        subq $2, %r11
        cmpq $2, %r11
        jge LoopSz

    LastS1:
    cmpq $1, %r11
    jl Last
    vmovups (%rsi), %zmm0
    vmovups 64(%rsi), %zmm1
    vmovups 128(%rsi), %zmm2

    vbroadcastss (%rdx), %zmm4
    vbroadcastss 4(%rdx), %zmm5
    vbroadcastss 8(%rdx), %zmm6
    vbroadcastss 12(%rdx), %zmm7

    vfmadd231ps %zmm0, %zmm4, %zmm8
    vfmadd231ps %zmm1, %zmm4, %zmm9
    vfmadd231ps %zmm2, %zmm4, %zmm10

    vfmadd231ps %zmm0, %zmm5, %zmm11
    vfmadd231ps %zmm1, %zmm5, %zmm12
    vfmadd231ps %zmm2, %zmm5, %zmm13

    vfmadd231ps %zmm0, %zmm6, %zmm14
    vfmadd231ps %zmm1, %zmm6, %zmm15
    vfmadd231ps %zmm2, %zmm6, %zmm16

    vfmadd231ps %zmm0, %zmm7, %zmm17
    vfmadd231ps %zmm1, %zmm7, %zmm18
    vfmadd231ps %zmm2, %zmm7, %zmm19

    vbroadcastss 16(%rdx), %zmm4
    vbroadcastss 20(%rdx), %zmm5
    vbroadcastss 24(%rdx), %zmm6
    vbroadcastss 28(%rdx), %zmm7

    vfmadd231ps %zmm0, %zmm4, %zmm20
    vfmadd231ps %zmm1, %zmm4, %zmm21
    vfmadd231ps %zmm2, %zmm4, %zmm22

    vfmadd231ps %zmm0, %zmm5, %zmm23
    vfmadd231ps %zmm1, %zmm5, %zmm24
    vfmadd231ps %zmm2, %zmm5, %zmm25

    vfmadd231ps %zmm0, %zmm6, %zmm26
    vfmadd231ps %zmm1, %zmm6, %zmm27
    vfmadd231ps %zmm2, %zmm6, %zmm28

    vfmadd231ps %zmm0, %zmm7, %zmm29
    vfmadd231ps %zmm1, %zmm7, %zmm30
    vfmadd231ps %zmm2, %zmm7, %zmm31

    addq $32, %rdx

    Last:

.macro TRANSPOSE_SAVE x0, x1, x2, x3
    vpunpckldq \x1, \x0, %zmm0
    vpunpckldq \x3, \x2, %zmm2
    vpunpckhdq \x1, \x0, %zmm1
    vpunpckhdq \x3, \x2, %zmm3

    vpunpcklqdq %zmm2, %zmm0, \x0
    vpunpckhqdq %zmm2, %zmm0, \x1
    vpunpcklqdq %zmm3, %zmm1, \x2
    vpunpckhqdq %zmm3, %zmm1, \x3

    vextractf32x8 $0, \x0, %ymm0
    vextractf32x8 $0, \x1, %ymm1
    vperm2f128 $32, %ymm1, %ymm0, %ymm4
    vperm2f128 $49, %ymm1, %ymm0, %ymm5
    vextractf32x8 $0, \x2, %ymm2
    vextractf32x8 $0, \x3, %ymm3
    vmovups %ymm4, (%r11)
    vmovups %ymm5, 64(%r11)
    vperm2f128 $32, %ymm3, %ymm2, %ymm6
    vperm2f128 $49, %ymm3, %ymm2, %ymm7
    vmovups %ymm6, 32(%r11)
    vmovups %ymm7, 96(%r11)

    vextractf32x8 $1, \x0, %ymm0
    vextractf32x8 $1, \x1, %ymm1
    vperm2f128 $32, %ymm1, %ymm0, %ymm4
    vperm2f128 $49, %ymm1, %ymm0, %ymm5
    vextractf32x8 $1, \x2, %ymm2
    vextractf32x8 $1, \x3, %ymm3
    vmovups %ymm4, 128(%r11)
    vmovups %ymm5, 192(%r11)
    vperm2f128 $32, %ymm3, %ymm2, %ymm6
    vperm2f128 $49, %ymm3, %ymm2, %ymm7
    vmovups %ymm6, 160(%r11)
    vmovups %ymm7, 224(%r11)

.endm
    movq %rdi, %r11
    TRANSPOSE_SAVE %zmm8, %zmm11, %zmm14, %zmm17
    addq $256, %r11
    TRANSPOSE_SAVE %zmm9, %zmm12, %zmm15, %zmm18
    addq $256, %r11
    TRANSPOSE_SAVE %zmm10, %zmm13, %zmm16, %zmm19

    addq %r8, %rdi

    movq %rdi, %r11
    TRANSPOSE_SAVE %zmm20, %zmm23, %zmm26, %zmm29
    addq $256, %r11
    TRANSPOSE_SAVE %zmm21, %zmm24, %zmm27, %zmm30
    addq $256, %r11
    TRANSPOSE_SAVE %zmm22, %zmm25, %zmm28, %zmm31

    addq %r8, %rdi

    addq %r10, %rdx

    subq $2, %r9
    cmpq $2, %r9
    jge LoopDz2

LD1:
cmpq $1, %r9
jl End

movq %rcx, %r11
movq %r13, %rsi

subq $1, %r11

vmovups (%rsi), %zmm0
vmovups 64(%rsi), %zmm1
vmovups 128(%rsi), %zmm2

vbroadcastss (%rdx), %zmm4
vbroadcastss 4(%rdx), %zmm5
vbroadcastss 8(%rdx), %zmm6
vbroadcastss 12(%rdx), %zmm7

vmulps %zmm0, %zmm4, %zmm8
vmulps %zmm1, %zmm4, %zmm9
vmulps %zmm2, %zmm4, %zmm10

vmulps %zmm0, %zmm5, %zmm11
vmulps %zmm1, %zmm5, %zmm12
vmulps %zmm2, %zmm5, %zmm13

vmulps %zmm0, %zmm6, %zmm14
vmulps %zmm1, %zmm6, %zmm15
vmulps %zmm2, %zmm6, %zmm16

vmulps %zmm0, %zmm7, %zmm17
vmulps %zmm1, %zmm7, %zmm18
vmulps %zmm2, %zmm7, %zmm19

addq $32, %rdx
addq $192, %rsi

cmpq $1, %r11
jl LastLD1

LoopSzLD1:
    vmovups (%rsi), %zmm0
    vmovups 64(%rsi), %zmm1
    vmovups 128(%rsi), %zmm2

    vbroadcastss (%rdx), %zmm4
    vbroadcastss 4(%rdx), %zmm5
    vbroadcastss 8(%rdx), %zmm6
    vbroadcastss 12(%rdx), %zmm7

    vfmadd231ps %zmm0, %zmm4, %zmm8
    vfmadd231ps %zmm1, %zmm4, %zmm9
    vfmadd231ps %zmm2, %zmm4, %zmm10

    vfmadd231ps %zmm0, %zmm5, %zmm11
    vfmadd231ps %zmm1, %zmm5, %zmm12
    vfmadd231ps %zmm2, %zmm5, %zmm13

    vfmadd231ps %zmm0, %zmm6, %zmm14
    vfmadd231ps %zmm1, %zmm6, %zmm15
    vfmadd231ps %zmm2, %zmm6, %zmm16

    vfmadd231ps %zmm0, %zmm7, %zmm17
    vfmadd231ps %zmm1, %zmm7, %zmm18
    vfmadd231ps %zmm2, %zmm7, %zmm19

    addq $32, %rdx
    addq $192, %rsi
    subq $1, %r11
    cmpq $1, %r11
    jge LoopSzLD1

LastLD1:

movq %rdi, %r11
TRANSPOSE_SAVE %zmm8, %zmm11, %zmm14, %zmm17
addq $256, %r11
TRANSPOSE_SAVE %zmm9, %zmm12, %zmm15, %zmm18
addq $256, %r11
TRANSPOSE_SAVE %zmm10, %zmm13, %zmm16, %zmm19

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

