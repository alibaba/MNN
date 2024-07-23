//
//  MNNPackedSparseQuantMatMulEpx4.S
//  MNN
//
//  Created by MNN on 2021/06/23.
//  Copyright © 2018-2021 Alibaba Group Holding Limited
//
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"
#define sizeof_value 4
#define sizeof_value_lg2 2
#define sparse_blockoc 4

.macro TYPE_CVT op, z0, z1, z2, z3
    \op \z0, \z0
    \op \z1, \z1
    \op \z2, \z2
    \op \z3, \z3
.endm

.macro CLAMP op, z0, z1, z2, z3, m0
    \op \z0, \z0, \m0
    \op \z1, \z1, \m0
    \op \z2, \z2, \m0
    \op \z3, \z3, \m0
.endm

.macro SCALE z0, z1, z2, z3, scale
    vmul.f32 \z0, \z0, \scale
    vmul.f32 \z1, \z1, \scale
    vmul.f32 \z2, \z2, \scale
    vmul.f32 \z3, \z3, \scale
.endm

.macro ROUND_MODE z0, z1, z2, z3
    vcgt.f32 q0, \z0, #0
    vcgt.f32 q1, \z1, #0
    vcgt.f32 q2, \z2, #0
    vcgt.f32 q3, \z3, #0
    vbsl.f32 q0, q4, q5
    vbsl.f32 q1, q4, q5
    vbsl.f32 q2, q4, q5
    vbsl.f32 q3, q4, q5
    vadd.f32 \z0, \z0, q0
    vadd.f32 \z1, \z1, q1
    vadd.f32 \z2, \z2, q2
    vadd.f32 \z3, \z3, q3
.endm

.text
.align 5
// caution!!! this is 8 * 4 Sparse MatMul
asm_function MNNPackedSparseQuantMatMulEpx4

// void MNNPackedSparseQuantMatMulEpx4(int8_t* C, const int8_t* A, const int8_t* B, const size_t* sparseQuantParam,
// const QuanPostTreatParameters* post, unsigned int* NNZMap, int* dataOffsetMap) {
//Auto load: r0: C, r1:A, r2:B, r3:sparseQuantParam,
//load from stack r4:QuanPostTreatParameters, r5:NNZMap, r6:dataOffsetMap

// var not defined: bias,

push {r4-r8, r10, r11, lr}
vpush {q4-q7}
#define push_registers_bytes (8 * 4 + 4 * 16)
ldr r4, [sp, #push_registers_bytes]
ldr r7, [r4, #8]
ldr r8, [r4, #12]
vmov.f32 q4, #0.5
vmov.f32 q5, #-0.5
vdup.32 q6, r7 // max
vdup.32 q7, r8 // min

// r0: C
// r1: A
// r2: B
// r3: sparseQuantParam mem(6*4byte) [eSize, eP, aStride, l, h, cStride]
// r4: QuanPostTreatParameters mem(4*4byte) [scale, bias, max, min]
// r5: NNZMap
// r6: dataOffsetMap
// r7: scale
// r8: bias
// r10: loop_counter (loop_e8 / loop_e4 / loop_e2 / loop_e1), cStride
// r11: loop_counter (loop_e8h4 / loop_e4h4 / loop_e2h4 / loop_e1h4)
// r12: loop_counter (loop_e8h4l1 / loop_e4h4l1 / loop_e2h4l1 / loop_e1h4l1)
// lr: temp var

ldr r10, [r3]
loop_e8:
    cmp r10, #8
    blt loop_e4
    sub r10, r10, #8
    ldr r5, [sp, #(push_registers_bytes + 4)]
    ldr r6, [sp, #(push_registers_bytes + 8)]
    ldr r7, [r4]
    ldr r8, [r4, #44]
    push {r0-r2, r10}
    ldr r10, [r3, #20] // cStride
    ldr lr, [r6], #4 // dataOffset
    add r1, r1, lr
    ldr r11, [r3, #16] // h
    lsr r11, r11, #2 // hDiv4 (C4)
    loop_e8h4:
        vld1.32 {q8}, [r8]!
        vmov q9, q8
        vmov q10, q8
        vmov q11, q8
        vmov q12, q8
        vmov q13, q8
        vmov q14, q8
        vmov q15, q8
        ldr r12, [r5], #4
        cmp r12, #0
        beq loop_e8h4_end
        loop_e8h4l1:
            vld1.32 {d0[0]}, [r2]!
            vld1.8 {d2}, [r1]
            vmovl.s8 q0, d0
            vmovl.s8 q1, d2
            ldr lr, [r6], #4
            add r1, r1, lr
            subs r12, r12, #1

            vmlal.s16 q8, d0, d2[0]
            vmlal.s16 q9, d0, d2[1]
            vmlal.s16 q10, d0, d2[2]
            vmlal.s16 q11, d0, d2[3]
            vmlal.s16 q12, d0, d3[0]
            vmlal.s16 q13, d0, d3[1]
            vmlal.s16 q14, d0, d3[2]
            vmlal.s16 q15, d0, d3[3]

            bne loop_e8h4l1
        loop_e8h4_end:
            vld1.32 {q0}, [r7]!
            TYPE_CVT vcvt.f32.s32, q8, q9, q10, q11
            TYPE_CVT vcvt.f32.s32, q12, q13, q14, q15
            SCALE q8, q9, q10, q11, q0
            SCALE q12, q13, q14, q15, q0
            ROUND_MODE q8, q9, q10, q11
            ROUND_MODE q12, q13, q14, q15
            TYPE_CVT vcvt.s32.f32, q8, q9, q10, q11
            TYPE_CVT vcvt.s32.f32, q12, q13, q14, q15
            CLAMP vmin.s32, q8, q9, q10, q11, q6
            CLAMP vmin.s32, q12, q13, q14, q15, q6
            CLAMP vmax.s32, q8, q9, q10, q11, q7
            CLAMP vmax.s32, q12, q13, q14, q15, q7
            vqmovn.s32 d0, q8
            vqmovn.s32 d1, q9
            vqmovn.s32 d2, q10
            vqmovn.s32 d3, q11
            vqmovn.s32 d4, q12
            vqmovn.s32 d5, q13
            vqmovn.s32 d6, q14
            vqmovn.s32 d7, q15
            vqmovn.s16 d0, q0
            vqmovn.s16 d1, q1
            vqmovn.s16 d2, q2
            vqmovn.s16 d3, q3
            vst1.8 {q0, q1}, [r0], r10
            subs r11, r11, #1
        bne loop_e8h4
        pop {r0-r2, r10}
        add r0, r0, #32
        add r1, r1, #8
    b loop_e8

loop_e4:
    cmp r10, #4
    blt loop_e2
    sub r10, r10, #4
    ldr r5, [sp, #(push_registers_bytes + 4)]
    ldr r6, [sp, #(push_registers_bytes + 8)]
    ldr r7, [r4]
    ldr r8, [r4, #44]
    push {r0-r2, r10}
    ldr r10, [r3, #20] // cStride
    ldr lr, [r6], #4 // dataOffset
    add r1, r1, lr
    ldr r11, [r3, #16] // h
    lsr r11, r11, #2 // hDiv4 (C4)
    loop_e4h4:
        vld1.32 {q8}, [r8]!
        vmov q9, q8
        vmov q10, q8
        vmov q11, q8
        ldr r12, [r5], #4
        cmp r12, #0
        beq loop_e4h4_end
        loop_e4h4l1:
            vld1.32 {d0[0]}, [r2]!
            vld1.32 {d2[0]}, [r1]
            vmovl.s8 q0, d0
            vmovl.s8 q1, d2
            ldr lr, [r6], #4
            add r1, r1, lr
            subs r12, r12, #1

            vmlal.s16 q8, d0, d2[0]
            vmlal.s16 q9, d0, d2[1]
            vmlal.s16 q10, d0, d2[2]
            vmlal.s16 q11, d0, d2[3]
            bne loop_e4h4l1
        loop_e4h4_end:
            vld1.32 {q0}, [r7]!
            TYPE_CVT vcvt.f32.s32, q8, q9, q10, q11
            SCALE q8, q9, q10, q11, q0
            ROUND_MODE q8, q9, q10, q11
            TYPE_CVT vcvt.s32.f32, q8, q9, q10, q11
            CLAMP vmin.s32, q8, q9, q10, q11, q6
            CLAMP vmax.s32, q8, q9, q10, q11, q7
            vqmovn.s32 d0, q8
            vqmovn.s32 d1, q9
            vqmovn.s32 d2, q10
            vqmovn.s32 d3, q11
            vqmovn.s16 d0, q0
            vqmovn.s16 d1, q1
            vst1.8 {q0}, [r0], r10
            subs r11, r11, #1
        bne loop_e4h4
        pop {r0-r2, r10}
        add r0, r0, #16
        add r1, r1, #4
    b loop_e4

loop_e2:
    cmp r10, #2
    blt loop_e1
    sub r10, r10, #2
    ldr r5, [sp, #(push_registers_bytes + 4)]
    ldr r6, [sp, #(push_registers_bytes + 8)]
    ldr r7, [r4]
    ldr r8, [r4, #44]
    push {r0-r2, r10}
    ldr r10, [r3, #20] // cStride
    ldr lr, [r6], #4 // dataOffset
    add r1, r1, lr
    ldr r11, [r3, #16] // h
    lsr r11, r11, #2 // hDiv4 (C4)
    loop_e2h4:
        vld1.32 {q8}, [r8]!
        vmov q9, q8
        ldr r12, [r5], #4
        cmp r12, #0
        beq loop_e2h4_end
        loop_e2h4l1:
            vld1.32 {d0[0]}, [r2]!
            vld1.16 {d2[0]}, [r1]
            vmovl.s8 q0, d0
            vmovl.s8 q1, d2
            ldr lr, [r6], #4
            add r1, r1, lr
            subs r12, r12, #1

            vmlal.s16 q8, d0, d2[0]
            vmlal.s16 q9, d0, d2[1]
            bne loop_e2h4l1
        loop_e2h4_end:
            vld1.32 {q0}, [r7]!
            vcvt.f32.s32 q8, q8
            vcvt.f32.s32 q9, q9
            vmul.f32 q8, q8, q0
            vmul.f32 q9, q9, q0
            vcgt.f32 q1, q8, #0
            vcgt.f32 q2, q9, #0
            vbsl.f32 q1, q4, q5
            vbsl.f32 q2, q4, q5
            vadd.f32 q8, q8, q1
            vadd.f32 q9, q9, q2
            vcvt.s32.f32 q8, q8
            vcvt.s32.f32 q9, q9
            vmin.s32 q8, q8, q6
            vmin.s32 q9, q9, q6
            vmax.s32 q8, q8, q7
            vmax.s32 q9, q9, q7
            vqmovn.s32 d0, q8
            vqmovn.s32 d1, q9
            vqmovn.s16 d0, q0
            vst1.8 {d0}, [r0], r10
            subs r11, r11, #1
        bne loop_e2h4
        pop {r0-r2, r10}
        add r0, r0, #8
        add r1, r1, #2
    b loop_e2

loop_e1:
    cmp r10, #1
    blt End
    sub r10, r10, #1
    ldr r5, [sp, #(push_registers_bytes + 4)]
    ldr r6, [sp, #(push_registers_bytes + 8)]
    ldr r7, [r4]
    ldr r8, [r4, #44]
    push {r0-r2, r10}
    ldr r10, [r3, #20] // cStride
    ldr lr, [r6], #4 // dataOffset
    add r1, r1, lr
    ldr r11, [r3, #16] // h
    lsr r11, r11, #2 // hDiv4 (C4)
    loop_e1h4:
        vld1.32 {q8}, [r8]!
        ldr r12, [r5], #4
        cmp r12, #0
        beq loop_e1h4_end
        loop_e1h4l1:
            vld1.32 {d0[0]}, [r2]!
            vld1.8 {d2[0]}, [r1]
            vmovl.s8 q0, d0
            vmovl.s8 q1, d2
            ldr lr, [r6], #4
            add r1, r1, lr
            subs r12, r12, #1

            vmlal.s16 q8, d0, d2[0]
            bne loop_e1h4l1
        loop_e1h4_end:
            vld1.32 {q0}, [r7]!
            vcvt.f32.s32 q8, q8
            vmul.f32 q8, q8, q0
            vcgt.f32 q1, q8, #0
            vbsl.f32 q1, q4, q5
            vadd.f32 q8, q8, q1
            vcvt.s32.f32 q8, q8
            vmin.s32 q8, q8, q6
            vmax.s32 q8, q8, q7
            vqmovn.s32 d0, q8
            vqmovn.s16 d0, q0
            vst1.32 {d0[0]}, [r0], r10
            subs r11, r11, #1
        bne loop_e1h4
        pop {r0-r2, r10}
        add r0, r0, #4
        add r1, r1, #1
    b loop_e1

End:
vpop {q4-q7}
pop {r4-r8, r10, r11, pc}

#undef push_registers_bytes
#undef sizeof_value
#undef sizeof_value_lg2
#undef sparse_blockoc

#endif
#endif


