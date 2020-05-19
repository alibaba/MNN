#include "../MNNAsmGlobal.h"

.text
.align 4

asm_function _AVX_MNNMatrixAdd
//void MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride, size_t bStride, size_t height)

//Auto: rdi: C, rsi:A, rdx:B, rcx:widthC4 , r8:cStride, r9:aStride
// r10:bStride, r11:height
pushq   %rbp
movq    %rsp, %rbp
cmpq $0, %rcx
je End

movq 16(%rbp), %r10
movq 24(%rbp), %r11
cmpq $0, %r11
je End
imulq $4, %r8
imulq $4, %r9
imulq $4, %r10

LoopY:
    movq %rcx, %rax
    cmpq $8, %rax
    jl LoopXRemain
    vmovups (%rsi), %ymm0
    vmovups 32(%rsi), %ymm2
    vmovups 64(%rsi), %ymm4
    vmovups 96(%rsi), %ymm6
    vaddps (%rdx), %ymm0, %ymm1
    vaddps 32(%rdx), %ymm2, %ymm3
    vaddps 64(%rdx), %ymm4, %ymm5
    vaddps 96(%rdx), %ymm6, %ymm7

    subq $8, %rax
    cmpq $8, %rax
    jl LoopX8End
    LoopX:
        vmovups %ymm1, (%rdi)
        vmovups %ymm3, 32(%rdi)
        addq $128, %rsi
        vmovups %ymm5, 64(%rdi)
        addq $128, %rdx
        vmovups (%rsi), %ymm0
        vmovups %ymm7, 96(%rdi)
        vmovups 32(%rsi), %ymm2
        addq $128, %rdi

        vmovups 64(%rsi), %ymm4
        vmovups 96(%rsi), %ymm6
        vaddps (%rdx), %ymm0, %ymm1
        vaddps 32(%rdx), %ymm2, %ymm3
        vaddps 64(%rdx), %ymm4, %ymm5
        vaddps 96(%rdx), %ymm6, %ymm7

        subq $8, %rax
        cmpq $8, %rax
        jge LoopX
    
    LoopX8End:
    vmovups %ymm1, (%rdi)
    vmovups %ymm3, 32(%rdi)
    vmovups %ymm5, 64(%rdi)
    vmovups %ymm7, 96(%rdi)
    addq $128, %rsi
    addq $128, %rdx
    addq $128, %rdi    
    testq %rax, %rax
    je LoopXEnd
    LoopXRemain:
        vmovups (%rsi), %xmm0
        vaddps (%rdx), %xmm0, %xmm1
        vmovups %xmm1, (%rdi)
        addq $16, %rsi
        addq $16, %rdx
        addq $16, %rdi
        subq $1, %rax
        testq %rax, %rax
        jne LoopXRemain
    LoopXEnd:
    addq %r8, %rdi
    addq %r9, %rsi
    addq %r10, %rdx
    subq $1, %r11
    testq %r11, %r11
    jne LoopY

End:
popq    %rbp

retq
