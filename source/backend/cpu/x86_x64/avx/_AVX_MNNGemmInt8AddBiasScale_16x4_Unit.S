//
//  _AVX_MNNGemmInt8AddBiasScale_16x4_Unit.S
//  MNN
//
//  Created by MNN on 2020/11/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../MNNAsmGlobal.h"
.text
.align 4

//struct QuanPostTreatParameters {
//    const float* scale;
//    const int32_t* bias;
//    int32_t maxValue;
//    int32_t minValue;
//    float roundValuePos = 0.5f;
//    float roundValueNeg = -0.5f;
//};

asm_function _AVX_MNNGemmInt8AddBiasScale_16x4_UnitMain
//void _AVX_MNNGemmInt8AddBiasScale_16x4_UnitMain(int8_t* dst, const int8_t* src, const int8_t* weight, const size_t* strides, const QuanPostTreatParameters* post);


// SystemV Auto: rdi: dst, rsi:src, rdx:weight, rcx:strides, r8: post
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
pushq   %r14
pushq   %r15
#else
pushq   %r12
pushq   %r13
pushq   %r14
pushq   %r15
movq %r8, %r9
#endif

movq 8(%rcx), %r10 // dst_step
movq 16(%rcx), %r8 // dst_depth_quad
movq (%rcx), %rcx // src_depth_quad
movq (%r9), %r12 // scale
movq 8(%r9), %r15 // bias


// ymm0-ymm1: Src
// ymm2-ymm3: Weight
// ymm4-ymm7: TmpDst
// ymm8-ymm15: Dst Sum

// Last dst save to ymm8-ymm11

cmpq $0, %r8
je End

movq %rsi, %r13
subq $64, %rsp
LoopDz:
    movq %rcx, %r11
    movq %r13, %rsi
    movq %rdx, %r14
    subq $1, %r11
    vpmovzxbw (%rsi), %ymm0
    vpmovzxbw 16(%rsi), %ymm1
    vpmovsxbw (%rdx), %ymm2
    vpmovsxbw 16(%rdx), %ymm3

    vpmaddwd %ymm0, %ymm2, %ymm8
    vpmaddwd %ymm0, %ymm3, %ymm9
    vpmaddwd %ymm1, %ymm2, %ymm12
    vpmaddwd %ymm1, %ymm3, %ymm13
    vpmovsxbw 32(%rdx), %ymm2
    vpmovsxbw 48(%rdx), %ymm3

    vpmaddwd %ymm0, %ymm2, %ymm10
    vpmaddwd %ymm0, %ymm3, %ymm11
    vpmaddwd %ymm1, %ymm2, %ymm14
    vpmaddwd %ymm1, %ymm3, %ymm15
    addq $64, %rdx
    addq $64, %rsi

    testq %r11, %r11
    je FirstLoopSzEnd

    FirstLoopSz:
        vpmovzxbw (%rsi), %ymm0
        vpmovzxbw 16(%rsi), %ymm1
        vpmovsxbw (%rdx), %ymm2
        vpmovsxbw 16(%rdx), %ymm3

        vpmaddwd %ymm0, %ymm2, %ymm4
        vpmaddwd %ymm0, %ymm3, %ymm5
        vpmaddwd %ymm1, %ymm2, %ymm6
        vpmaddwd %ymm1, %ymm3, %ymm7
        vpaddd %ymm4, %ymm8, %ymm8
        vpaddd %ymm5, %ymm9, %ymm9
        vpmovsxbw 32(%rdx), %ymm2
        vpmovsxbw 48(%rdx), %ymm3
        vpaddd %ymm6, %ymm12, %ymm12
        vpaddd %ymm7, %ymm13, %ymm13


        vpmaddwd %ymm0, %ymm2, %ymm4
        vpmaddwd %ymm0, %ymm3, %ymm5
        vpmaddwd %ymm1, %ymm2, %ymm6
        vpmaddwd %ymm1, %ymm3, %ymm7
        vpaddd %ymm4, %ymm10, %ymm10
        vpaddd %ymm5, %ymm11, %ymm11
        vpaddd %ymm6, %ymm14, %ymm14
        vpaddd %ymm7, %ymm15, %ymm15

        addq $64, %rdx
        addq $64, %rsi

        subq $1, %r11
        testq %r11, %r11
        jne FirstLoopSz

    FirstLoopSzEnd:
    
    vphaddd %ymm9, %ymm8, %ymm8
    vphaddd %ymm11, %ymm10, %ymm10
    vphaddd %ymm13, %ymm12, %ymm12
    vphaddd %ymm15, %ymm14, %ymm14

    vphaddd %ymm10, %ymm8, %ymm8
    vphaddd %ymm14, %ymm12, %ymm9

    vmovups %ymm8, (%rsp)
    vmovups %ymm9, 32(%rsp)

    movq %rcx, %r11
    movq %r13, %rsi
    movq %r14, %rdx
    vpmovzxbw 32(%rsi), %ymm0
    vpmovzxbw 48(%rsi), %ymm1
    vpmovsxbw (%rdx), %ymm2
    vpmovsxbw 16(%rdx), %ymm3

    vpmaddwd %ymm0, %ymm2, %ymm8
    vpmaddwd %ymm0, %ymm3, %ymm9
    vpmaddwd %ymm1, %ymm2, %ymm12
    vpmaddwd %ymm1, %ymm3, %ymm13

    vpmovsxbw 32(%rdx), %ymm2
    vpmovsxbw 48(%rdx), %ymm3

    vpmaddwd %ymm0, %ymm2, %ymm10
    vpmaddwd %ymm0, %ymm3, %ymm11
    vpmaddwd %ymm1, %ymm2, %ymm14
    vpmaddwd %ymm1, %ymm3, %ymm15

    addq $64, %rdx
    addq $64, %rsi

    subq $1, %r11
    testq %r11, %r11
    je SecondLoopSzEnd

    SecondLoopSz:
        vpmovzxbw 32(%rsi), %ymm0
        vpmovzxbw 48(%rsi), %ymm1
        vpmovsxbw (%rdx), %ymm2
        vpmovsxbw 16(%rdx), %ymm3

        vpmaddwd %ymm0, %ymm2, %ymm4
        vpmaddwd %ymm0, %ymm3, %ymm5
        vpmaddwd %ymm1, %ymm2, %ymm6
        vpmaddwd %ymm1, %ymm3, %ymm7
        vpaddd %ymm4, %ymm8, %ymm8
        vpaddd %ymm5, %ymm9, %ymm9
        vpmovsxbw 32(%rdx), %ymm2
        vpmovsxbw 48(%rdx), %ymm3
        vpaddd %ymm6, %ymm12, %ymm12
        vpaddd %ymm7, %ymm13, %ymm13


        vpmaddwd %ymm0, %ymm2, %ymm4
        vpmaddwd %ymm0, %ymm3, %ymm5
        vpmaddwd %ymm1, %ymm2, %ymm6
        vpmaddwd %ymm1, %ymm3, %ymm7
        vpaddd %ymm4, %ymm10, %ymm10
        vpaddd %ymm5, %ymm11, %ymm11
        vpaddd %ymm6, %ymm14, %ymm14
        vpaddd %ymm7, %ymm15, %ymm15

        addq $64, %rdx
        addq $64, %rsi

        subq $1, %r11
        testq %r11, %r11
        jne SecondLoopSz
    SecondLoopSzEnd:

    vphaddd %ymm9, %ymm8, %ymm8
    vphaddd %ymm11, %ymm10, %ymm10
    vphaddd %ymm13, %ymm12, %ymm12
    vphaddd %ymm15, %ymm14, %ymm14

    vphaddd %ymm10, %ymm8, %ymm10
    vphaddd %ymm14, %ymm12, %ymm11

    vmovups (%rsp), %ymm8
    vmovups 32(%rsp), %ymm9

    Last:
.macro TRANSPOSE x0, x1, x2, x3
    // 32 = 0 + 16 * 2: frist 128 x0_lo, second 128 x1_lo
    // 49 = 1 + 16 * 3: frist 128 x0_hi, second 128 x1_hi
    vperm2f128 $32, \x1, \x0, \x2
    vperm2f128 $49, \x1, \x0, \x3
.endm
    cmpq $0, %r12
    jne LoopDzQuan
    TRANSPOSE %ymm8, %ymm9, %ymm0, %ymm1
    TRANSPOSE %ymm10, %ymm11, %ymm2, %ymm3
    vbroadcastf128 (%r15), %ymm9
    vpaddd %ymm0, %ymm1, %ymm0
    vpaddd %ymm2, %ymm3, %ymm2
    vpaddd %ymm9, %ymm0, %ymm0
    vpaddd %ymm9, %ymm2, %ymm2
    vcvtdq2ps %ymm0, %ymm0
    vcvtdq2ps %ymm2, %ymm2
    vmovups %ymm0, (%rdi)
    vmovups %ymm2, 32(%rdi)
    addq $16, %r15
    addq %r10, %rdi
    jmp LoopDzCheck
LoopDzQuan:
    TRANSPOSE %ymm8, %ymm10, %ymm0, %ymm1
    TRANSPOSE %ymm9, %ymm11, %ymm2, %ymm3
    vpaddd %ymm0, %ymm1, %ymm0
    vpaddd %ymm2, %ymm3, %ymm2

    vbroadcastf128 (%r12), %ymm8
    vbroadcastf128 (%r15), %ymm9

    vpaddd %ymm9, %ymm0, %ymm0
    vpaddd %ymm9, %ymm2, %ymm2

    vcvtdq2ps %ymm0, %ymm0
    vcvtdq2ps %ymm2, %ymm2

    vmulps %ymm8, %ymm0, %ymm0
    vmulps %ymm8, %ymm2, %ymm2
    // zero
    vxorps %ymm13, %ymm13, %ymm13

    vbroadcastss 24(%r9), %ymm14
    vbroadcastss 28(%r9), %ymm15
    vbroadcastss 16(%r9), %ymm10
    vbroadcastss 20(%r9), %ymm11

    // Round
    vcmpltps %ymm13, %ymm0, %ymm4
    vcmpltps %ymm13, %ymm2, %ymm5

    vblendvps %ymm4, %ymm15, %ymm14, %ymm4
    vblendvps %ymm5, %ymm15, %ymm14, %ymm5

    vaddps %ymm0, %ymm4, %ymm0
    vaddps %ymm2, %ymm5, %ymm2

    // 3: ROUND to Zero
    vroundps $3, %ymm0, %ymm0
    vroundps $3, %ymm2, %ymm2
    vcvtps2dq %ymm0, %ymm0
    vcvtps2dq %ymm2, %ymm2

    vpminsd %ymm10, %ymm0, %ymm0
    vpminsd %ymm10, %ymm2, %ymm2

    vpmaxsd %ymm11, %ymm0, %ymm0
    vpmaxsd %ymm11, %ymm2, %ymm2

    vpackssdw %ymm2, %ymm0, %ymm0
    vperm2f128 $1, %ymm0, %ymm0, %ymm1
    vpacksswb %ymm1, %ymm0, %ymm0

    addq $16, %r12
    addq $16, %r15

    vmovups %xmm0, (%rdi)
    addq %r10, %rdi
LoopDzCheck:
    subq $1, %r8
    testq %r8, %r8
    jne LoopDz
addq $64, %rsp

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

// FIXME: if don't vzeroall, it will cause other op slow
vzeroall
retq

