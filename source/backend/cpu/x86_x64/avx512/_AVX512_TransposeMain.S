//
//  _AVX512_TransposeMain.S
//  MNN
//
//  Created by MNN on 2020/05/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../MNNAsmGlobal.h"
.text
.align 4

asm_function _AVX512_TransposeMain
//void _AVX512_TransposeMain(const float* source, float* dest, const size_t* info, size_t lC4)
//    int srcStride = info[0] / sizeof(float);
//    int dstStride = info[1] / sizeof(float);
//    int pOffset = info[2] / sizeof(float);

// SystemV Auto: rdi: source, rsi:dest, rdx:info, rcx:lC4
pushq   %rbp
movq    %rsp, %rbp

#ifdef WIN32
pushq %rdi
pushq %rsi
pushq %r12
pushq %r13
pushq %r14
movq %rcx, %rdi
movq %rdx, %rsi
movq %r8, %rdx
movq %r9, %rcx
leaq (-1280)(%rsp), %rsp
vmovdqu %xmm6,  (128*0)(%rsp)
vmovdqu %xmm7,  (128*1)(%rsp)
vmovdqu %xmm8,  (128*2)(%rsp)
vmovdqu %xmm9,  (128*3)(%rsp)
vmovdqu %xmm10, (128*4)(%rsp)
vmovdqu %xmm11, (128*5)(%rsp)
vmovdqu %xmm12, (128*6)(%rsp)
vmovdqu %xmm13, (128*7)(%rsp)
vmovdqu %xmm14, (128*8)(%rsp)
vmovdqu %xmm15, (128*9)(%rsp)
#else
pushq   %r12
pushq   %r13
pushq   %r14
#endif

cmpq $0, %rcx
je End
movq (%rdx), %r9
movq 8(%rdx), %r10
movq 16(%rdx), %r11
movq %r11, %r13
imulq $3, %r13

.macro TRANPOSE p, q
    vmovups (%r12), %zmm0
    vmovups (%r12, %r11), %zmm1
    vmovups (%r12, %r11, 2), %zmm2
    addq %r13, %r12
    vmovups (%r12), %zmm3
    vmovups (%r12, %r11), %zmm4
    vmovups (%r12, %r11, 2), %zmm5
    addq %r13, %r12
    vmovups (%r12), %zmm6
    vmovups (%r12, %r11), %zmm7
    vmovups (%r12, %r11, 2), %zmm8
    addq %r13, %r12
    vmovups (%r12), %zmm9
    vmovups (%r12, %r11), %zmm10
    vmovups (%r12, %r11, 2), %zmm11
    addq %r13, %r12
    vmovups (%r12), %zmm12
    vmovups (%r12, %r11), %zmm13
    vmovups (%r12, %r11, 2), %zmm14
    addq %r13, %r12
    vmovups (%r12), %zmm15
    addq %r11, %r12

    vpunpckldq %zmm1, %zmm0, %zmm16
    vpunpckhdq %zmm1, %zmm0, %zmm17
    vpunpckldq %zmm3, %zmm2, %zmm18
    vpunpckhdq %zmm3, %zmm2, %zmm19
    vpunpckldq %zmm5, %zmm4, %zmm20
    vpunpckhdq %zmm5, %zmm4, %zmm21
    vpunpckldq %zmm7, %zmm6, %zmm22
    vpunpckhdq %zmm7, %zmm6, %zmm23
    vpunpckldq %zmm9, %zmm8, %zmm24
    vpunpckhdq %zmm9, %zmm8, %zmm25
    vpunpckldq %zmm11, %zmm10, %zmm26
    vpunpckhdq %zmm11, %zmm10, %zmm27
    vpunpckldq %zmm13, %zmm12, %zmm28
    vpunpckhdq %zmm13, %zmm12, %zmm29
    vpunpckldq %zmm15, %zmm14, %zmm30
    vpunpckhdq %zmm15, %zmm14, %zmm31

    vpunpcklqdq %zmm18, %zmm16, %zmm0
    vpunpckhqdq %zmm18, %zmm16, %zmm1
    vpunpcklqdq %zmm19, %zmm17, %zmm2
    vpunpckhqdq %zmm19, %zmm17, %zmm3
    vpunpcklqdq %zmm22, %zmm20, %zmm4
    vpunpckhqdq %zmm22, %zmm20, %zmm5
    vpunpcklqdq %zmm23, %zmm21, %zmm6
    vpunpckhqdq %zmm23, %zmm21, %zmm7
    vpunpcklqdq %zmm26, %zmm24, %zmm8
    vpunpckhqdq %zmm26, %zmm24, %zmm9
    vpunpcklqdq %zmm27, %zmm25, %zmm10
    vpunpckhqdq %zmm27, %zmm25, %zmm11
    vpunpcklqdq %zmm30, %zmm28, %zmm12
    vpunpckhqdq %zmm30, %zmm28, %zmm13
    vpunpcklqdq %zmm31, %zmm29, %zmm14
    vpunpckhqdq %zmm31, %zmm29, %zmm15

    VSHUFI32x4 \p, %zmm4, %zmm0, %zmm16
    VSHUFI32x4 \p, %zmm5, %zmm1, %zmm17
    VSHUFI32x4 \p, %zmm6, %zmm2, %zmm18
    VSHUFI32x4 \p, %zmm7, %zmm3, %zmm19
    VSHUFI32x4 \q, %zmm4, %zmm0, %zmm20
    VSHUFI32x4 \q, %zmm5, %zmm1, %zmm21
    VSHUFI32x4 \q, %zmm6, %zmm2, %zmm22
    VSHUFI32x4 \q, %zmm7, %zmm3, %zmm23
    VSHUFI32x4 \p, %zmm12, %zmm8, %zmm24
    VSHUFI32x4 \p, %zmm13, %zmm9, %zmm25
    VSHUFI32x4 \p, %zmm14, %zmm10, %zmm26
    VSHUFI32x4 \p, %zmm15, %zmm11, %zmm27
    VSHUFI32x4 \q, %zmm12, %zmm8, %zmm28
    VSHUFI32x4 \q, %zmm13, %zmm9, %zmm29
    VSHUFI32x4 \q, %zmm14, %zmm10, %zmm30
    VSHUFI32x4 \q, %zmm15, %zmm11, %zmm31

    VSHUFI32x4 \p, %zmm24, %zmm16, %zmm0
    VSHUFI32x4 \p, %zmm25, %zmm17, %zmm1
    VSHUFI32x4 \p, %zmm26, %zmm18, %zmm2
    VSHUFI32x4 \p, %zmm27, %zmm19, %zmm3
    VSHUFI32x4 \p, %zmm28, %zmm20, %zmm4
    VSHUFI32x4 \p, %zmm29, %zmm21, %zmm5
    VSHUFI32x4 \p, %zmm30, %zmm22, %zmm6
    VSHUFI32x4 \p, %zmm31, %zmm23, %zmm7
    VSHUFI32x4 \q, %zmm24, %zmm16, %zmm8
    VSHUFI32x4 \q, %zmm25, %zmm17, %zmm9
    VSHUFI32x4 \q, %zmm26, %zmm18, %zmm10
    VSHUFI32x4 \q, %zmm27, %zmm19, %zmm11
    VSHUFI32x4 \q, %zmm28, %zmm20, %zmm12
    VSHUFI32x4 \q, %zmm29, %zmm21, %zmm13
    VSHUFI32x4 \q, %zmm30, %zmm22, %zmm14
    VSHUFI32x4 \q, %zmm31, %zmm23, %zmm15

    vmovups %zmm0, (%r14)
    vmovups %zmm1, 192(%r14)
    vmovups %zmm2, 384(%r14)
    vmovups %zmm3, 576(%r14)
    vmovups %zmm4, 768(%r14)
    vmovups %zmm5, 960(%r14)
    vmovups %zmm6, 1152(%r14)
    vmovups %zmm7, 1344(%r14)
    vmovups %zmm8, 1536(%r14)
    vmovups %zmm9, 1728(%r14)
    vmovups %zmm10, 1920(%r14)
    vmovups %zmm11, 2112(%r14)
    vmovups %zmm12, 2304(%r14)
    vmovups %zmm13, 2496(%r14)
    vmovups %zmm14, 2688(%r14)
    vmovups %zmm15, 2880(%r14)


.endm

Loop:
    movq %rdi, %r12
    movq %rsi, %r14

    // Transpose 16x16
    TRANPOSE $136, $221
    addq $64, %r14
    TRANPOSE $136, $221
    addq $64, %r14
    TRANPOSE $136, $221

    addq %r9, %rdi
    addq %r10, %rsi

    subq $1, %rcx
    cmpq $0, %rcx
    jne Loop

End:

#ifdef WIN32
vmovdqu (128*0)(%rsp), %xmm6
vmovdqu (128*1)(%rsp), %xmm7
vmovdqu (128*2)(%rsp), %xmm8
vmovdqu (128*3)(%rsp), %xmm9
vmovdqu (128*4)(%rsp), %xmm10
vmovdqu (128*5)(%rsp), %xmm11
vmovdqu (128*6)(%rsp), %xmm12
vmovdqu (128*7)(%rsp), %xmm13
vmovdqu (128*8)(%rsp), %xmm14
vmovdqu (128*9)(%rsp), %xmm15
leaq (1280)(%rsp), %rsp
popq    %r14
popq    %r13
popq    %r12
popq    %rsi
popq    %rdi
popq    %rbp
#else
popq    %r14
popq    %r13
popq    %r12
popq    %rbp
#endif

retq

