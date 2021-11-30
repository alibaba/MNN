//
//  _AVX512_MNNGemmFloatUnit1x8.S
//  MNN
//
//  Created by MNN on 2020/05/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../MNNAsmGlobal.h"
.text
.align 4

asm_function _AVX512_MNNGemmFloatUnit1x8
//void _AVX512_MNNGemmFloatUnit1x8(float* C, const float* A, const float* B, const size_t* parameter, size_t hC4)

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
pushq %r14
pushq   %r15
movq %rcx, %rdi
movq %rdx, %rsi
movq %r8, %rdx
movq %r9, %rcx
movq %r10, %r9
#else
pushq   %r12
pushq   %r13
pushq   %r14
pushq   %r15
movq %r8, %r9
#endif

movq 40(%rcx), %r10 // bExtraStride
movq 24(%rcx), %r8 // cStride
movq 16(%rcx), %r9 // h
movq (%rcx), %r14 // aStride
movq 8(%rcx), %rcx // l

// h -> UP_DIV(h, 8)
addq $7, %r9
shrq $3, %r9

// zmm8-zmm31: Dst
// zmm0-zmm2: Src
// zmm3-zmm7: W
cmpq $4, %r9
jl L1
L4:

// r10->bStride = bExtraStride + 8 * sizeof(float) * l
movq %rcx, %r15
imulq $32, %r15
addq %r15, %r10

LoopDz4:
    movq %rcx, %r11
    movq %r13, %rsi
    movq %rdx, %r15
    cmpq $2, %r11
    jge LoopDz4L2Init
    vzeroall
    jmp RemainDz4

    LoopDz4L2Init:
        subq $2, %r11
        vbroadcastss (%rsi), %ymm0
        vbroadcastss (%rsi, %r14), %ymm1
        VSHUFI32x4 $68, %zmm1, %zmm0, %zmm0

        vmovups (%r15), %zmm4
        vmovups (%r15, %r10), %zmm5
        vmovups (%r15, %r10, 2), %zmm6
        addq %r10, %r15
        vmovups (%r15, %r10, 2), %zmm7

        vmulps %zmm0, %zmm4, %zmm16
        vmulps %zmm0, %zmm5, %zmm17
        vmulps %zmm0, %zmm6, %zmm18
        vmulps %zmm0, %zmm7, %zmm19

        addq %r14, %rsi
        subq %r10, %r15
        addq %r14, %rsi
        addq $64, %r15
        cmpq $2, %r11
        jl RemainDz4

    LoopDz4L2:
        subq $2, %r11
        vbroadcastss (%rsi), %ymm0
        vbroadcastss (%rsi, %r14), %ymm1
        VSHUFI32x4 $68, %zmm1, %zmm0, %zmm0

        vmovups (%r15), %zmm4
        vmovups (%r15, %r10), %zmm5
        vmovups (%r15, %r10, 2), %zmm6
        addq %r10, %r15
        vmovups (%r15, %r10, 2), %zmm7

        vfmadd231ps %zmm0, %zmm4, %zmm16
        vfmadd231ps %zmm0, %zmm5, %zmm17
        vfmadd231ps %zmm0, %zmm6, %zmm18
        vfmadd231ps %zmm0, %zmm7, %zmm19

        addq %r14, %rsi
        subq %r10, %r15
        addq %r14, %rsi
        addq $64, %r15
        cmpq $2, %r11
        jge LoopDz4L2

    VEXTRACTF64x4 $0, %zmm16, %ymm0
    VEXTRACTF64x4 $1, %zmm16, %ymm1
    VEXTRACTF64x4 $0, %zmm17, %ymm2
    VEXTRACTF64x4 $1, %zmm17, %ymm3
    VEXTRACTF64x4 $0, %zmm18, %ymm4
    VEXTRACTF64x4 $1, %zmm18, %ymm5
    VEXTRACTF64x4 $0, %zmm19, %ymm6
    VEXTRACTF64x4 $1, %zmm19, %ymm7

    vaddps %ymm1, %ymm0, %ymm0
    vaddps %ymm3, %ymm2, %ymm2
    vaddps %ymm5, %ymm4, %ymm4
    vaddps %ymm7, %ymm6, %ymm6

    RemainDz4:
        cmpq $0, %r11
        je EndDz4
        vbroadcastss (%rsi), %ymm16
        vmovups (%r15), %ymm1
        vmovups (%r15, %r10), %ymm3
        vmovups (%r15, %r10, 2), %ymm5
        addq %r10, %r15
        vmovups (%r15, %r10, 2), %ymm7
    
        vfmadd231ps %ymm1, %ymm16, %ymm0
        vfmadd231ps %ymm3, %ymm16, %ymm2
        vfmadd231ps %ymm5, %ymm16, %ymm4
        vfmadd231ps %ymm7, %ymm16, %ymm6
    
    EndDz4:
        // Store
        vmovups %ymm0, (%rdi)
        vmovups %ymm2, 32(%rdi)
        vmovups %ymm4, (%rdi, %r8)
        vmovups %ymm6, 32(%rdi, %r8)
        addq %r8, %rdi
        addq %r8, %rdi

    subq $4, %r9
    movq %r10, %r15
    imulq $4, %r15
    addq %r15, %rdx
    cmpq $4, %r9
    jge LoopDz4

// r10 -> bExtraStride = bStride - 8 * sizeof(float) * l
movq %rcx, %r15
imulq $32, %r15
subq %r15, %r10

L1:
cmpq $0, %r9
je End

movq $0, %r12
movq %rsi, %r13
LoopDz:
    movq %rcx, %r11
    movq %r13, %rsi
    subq $1, %r11
    Init:
        vbroadcastss (%rsi), %ymm0
        vmovups (%rdx), %ymm4
        vmulps %ymm4, %ymm0, %ymm8
        addq %r14, %rsi
        addq $32, %rdx
 
    cmpq $0, %r11
    je Last

    LoopSz:
        vbroadcastss (%rsi), %ymm0
        vmovups (%rdx), %ymm4
        vfmadd231ps %ymm4, %ymm0, %ymm8
        addq %r14, %rsi
        addq $32, %rdx
        subq $1, %r11
        cmpq $0, %r11
        jne LoopSz
    Last:

    vmovups %ymm8, (%rdi)

    cmpq $0, %r12
    je EndAdd8
    subq $32, %rdi
    addq %r8, %rdi
    jmp EndLoop
    EndAdd8:
    addq $32, %rdi

    EndLoop:

    addq %r10, %rdx

    addq $1, %r12
    andq $1, %r12

    subq $1, %r9
    cmpq $0, %r9
    jne LoopDz

End:

#ifdef WIN32
popq    %r15
popq    %r14
popq    %r13
popq    %r12
popq    %rsi
popq    %rdi
popq    %rbp
#else
popq    %r15
popq    %r14
popq    %r13
popq    %r12
popq    %rbp
#endif

retq

