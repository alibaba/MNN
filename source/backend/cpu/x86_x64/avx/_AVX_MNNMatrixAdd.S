#include "../MNNAsmGlobal.h"

.text
.align 5

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
    LoopX:
        vmovups (%rsi), %xmm0
        vaddps (%rdx), %xmm0, %xmm1
        vmovups %xmm1, (%rdi)
        addq $16, %rsi
        addq $16, %rdx
        addq $16, %rdi
        subq $1, %rax
        testq %rax, %rax
        jne LoopX
    addq %r8, %rdi
    addq %r9, %rsi
    addq %r10, %rdx
    subq $1, %r11
    testq %r11, %r11
    jne LoopY

End:
popq    %rbp

retq
