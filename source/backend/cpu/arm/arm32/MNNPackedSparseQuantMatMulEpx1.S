//
//  MNNPackedSparseQuantMatMulEpx1.S
//  MNN
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018-2021 Alibaba Group Holding Limited
//
//

/*
struct QuanPostTreatParameters {
    const float* scale;
    const float* biasFloat;
    int32_t maxValue;
    int32_t minValue;
    int32_t useInt8 = 1; // Save result as int8_t dataType; otherwise float32.
    float roundValuePos = 0.5f;
    float roundValueNeg = -0.5f;
    float* srcKernelSum;
    float* weightQuanBias;
    float* fp32minmax;
    ssize_t blockNum = 1;
    const int32_t* bias;

};
 */

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"
#define sizeof_value 4
#define sizeof_value_lg2 2
#define sparse_blockoc 4

.text
.align 5
// caution!!! this is 8 * 1 Sparse MatMul
asm_function MNNPackedSparseQuantMatMulEpx1

// void MNNPackedSparseQuantMatMulEpx1(int8_t* C, const int8_t* A, const int8_t* B, const size_t* sparseQuantParam,
// const QuanPostTreatParameters* post, unsigned int* NNZMap, int* dataOffsetMap) {
//Auto load: r0: C, r1:A, r2:B, r3:sparseQuantParam,
//load from stack r4:QuanPostTreatParameters, r5:NNZMap, r6:dataOffsetMap

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
// r10: loop_counter (loop_e8 / loop_e4 / loop_e2 / loop_e1), h
// r11: loop_counter (loop_e8h1 / loop_e4h1 / loop_e2h1 / loop_e1h1)
// r12: loop_counter (loop_e8h1l1 / loop_e4h1l1 / loop_e2h1l1 / loop_e1h1l1)
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
    ldr lr, [r6], #4 // dataOffset
    add r1, r1, lr
    ldr r10, [r3, #16] // h
    mov r11, #0
    loop_e8h1:
        vld1.32 {d16[0]}, [r8]!
        vdup.32 q8, d16[0]
        vdup.32 q9, d16[0]
        ldr r12, [r5], #4
        cmp r12, #0
        beq loop_e8h1_end
        loop_e8h1l1:
            vld1.8 {d0[0]}, [r2]!
            vld1.8 {d2}, [r1]
            vmovl.s8 q0, d0
            vmovl.s8 q1, d2
            ldr lr, [r6], #4
            add r1, r1, lr
            subs r12, r12, #1

            vmlal.s16 q8, d2, d0[0]
            vmlal.s16 q9, d3, d0[0]

            bne loop_e8h1l1
        loop_e8h1_end:
            vld1.32 {d0[0]}, [r7]!
            vcvt.f32.s32 q8, q8
            vcvt.f32.s32 q9, q9
            vmul.f32 q8, q8, d0[0]
            vmul.f32 q9, q9, d0[0]
            vcgt.f32 q0, q8, #0
            vcgt.f32 q1, q9, #0
            vbsl.f32 q0, q4, q5
            vbsl.f32 q1, q4, q5
            vadd.f32 q8, q8, q0
            vadd.f32 q9, q9, q1
            vcvt.s32.f32 q8, q8
            vcvt.s32.f32 q9, q9
            vmin.s32 q8, q8, q6
            vmin.s32 q9, q9, q6
            vmax.s32 q8, q8, q7
            vmax.s32 q9, q9, q7
            vqmovn.s32 d0, q8
            vqmovn.s32 d1, q9
            vqmovn.s16 d0, q0
            mov lr, #4
            vst1.8 {d0[0]}, [r0], lr
            vst1.8 {d0[1]}, [r0], lr
            vst1.8 {d0[2]}, [r0], lr
            vst1.8 {d0[3]}, [r0], lr
            vst1.8 {d0[4]}, [r0], lr
            vst1.8 {d0[5]}, [r0], lr
            vst1.8 {d0[6]}, [r0], lr
            vst1.8 {d0[7]}, [r0], lr
            sub r0, r0, lr, lsl #3
            add r11, r11, #1
            ands lr, r11, #0x03
            addne r0, r0, #1
            ldr lr, [r3, #20] // cStride
            subeq lr, lr, #3
            addeq r0, r0, lr
            cmp r11, r10
        blt loop_e8h1
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
    ldr lr, [r6], #4 // dataOffset
    add r1, r1, lr
    ldr r10, [r3, #16] // h
    mov r11, #0
    loop_e4h1:
        vld1.32 {d16[0]}, [r8]!
        vdup.32 q8, d16[0]
        ldr r12, [r5], #4
        cmp r12, #0
        beq loop_e4h1_end
        loop_e4h1l1:
            vld1.8 {d0[0]}, [r2]!
            vld1.32 {d2[0]}, [r1]
            vmovl.s8 q0, d0
            vmovl.s8 q1, d2
            ldr lr, [r6], #4
            add r1, r1, lr
            subs r12, r12, #1

            vmlal.s16 q8, d2, d0[0]
            bne loop_e4h1l1
        loop_e4h1_end:
            vld1.32 {d0[0]}, [r7]!
            vcvt.f32.s32 q8, q8
            vmul.f32 q8, q8, d0[0]
            vcgt.f32 q0, q8, #0
            vbsl.f32 q0, q4, q5
            vadd.f32 q8, q8, q0
            vcvt.s32.f32 q8, q8
            vmin.s32 q8, q8, q6
            vmax.s32 q8, q8, q7
            vqmovn.s32 d0, q8
            vqmovn.s16 d0, q0
            mov lr, #4
            vst1.8 {d0[0]}, [r0], lr
            vst1.8 {d0[1]}, [r0], lr
            vst1.8 {d0[2]}, [r0], lr
            vst1.8 {d0[3]}, [r0], lr
            sub r0, r0, lr, lsl #2
            add r11, r11, #1
            ands lr, r11, #0x03
            addne r0, r0, #1
            ldr lr, [r3, #20] // cStride
            subeq lr, lr, #3
            addeq r0, r0, lr
            cmp r11, r10
        blt loop_e4h1
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
    ldr lr, [r6], #4 // dataOffset
    add r1, r1, lr
    ldr r10, [r3, #16] // h
    mov r11, #0
    loop_e2h1:
        vld1.32 {d16[0]}, [r8]!
        vdup.32 d16, d16[0]
        ldr r12, [r5], #4
        cmp r12, #0
        beq loop_e2h1_end
        loop_e2h1l1:
            vld1.8 {d0[0]}, [r2]!
            vld1.16 {d2[0]}, [r1]
            vmovl.s8 q0, d0
            vmovl.s8 q1, d2
            ldr lr, [r6], #4
            add r1, r1, lr
            subs r12, r12, #1

            vmlal.s16 q8, d2, d0[0]
            bne loop_e2h1l1
        loop_e2h1_end:
            vld1.32 {d0[0]}, [r7]!
            vcvt.f32.s32 d16, d16
            vmul.f32 d16, d16, d0[0]
            vcgt.f32 d0, d16, #0
            vbsl.f32 d0, d8, d10
            vadd.f32 d16, d16, d0
            vcvt.s32.f32 d16, d16
            vmin.s32 d16, d16, d12
            vmax.s32 d16, d16, d14
            vqmovn.s32 d0, q8
            vqmovn.s16 d0, q0
            mov lr, #4
            vst1.8 {d0[0]}, [r0], lr
            vst1.8 {d0[1]}, [r0], lr
            sub r0, r0, lr, lsl #1
            add r11, r11, #1
            ands lr, r11, #0x03
            addne r0, r0, #1
            ldr lr, [r3, #20] // cStride
            subeq lr, lr, #3
            addeq r0, r0, lr
            cmp r11, r10
        blt loop_e2h1
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
    ldr lr, [r6], #4 // dataOffset
    add r1, r1, lr
    ldr r10, [r3, #16] // h
    mov r11, #0
    loop_e1h1:
        vld1.32 {d16[0]}, [r8]!
        ldr r12, [r5], #4
        cmp r12, #0
        beq loop_e1h1_end
        loop_e1h1l1:
            vld1.8 {d0[0]}, [r2]!
            vld1.8 {d2[0]}, [r1]
            vmovl.s8 q0, d0
            vmovl.s8 q1, d2
            ldr lr, [r6], #4
            add r1, r1, lr
            subs r12, r12, #1

            vmlal.s16 q8, d2, d0[0]
            bne loop_e1h1l1
        loop_e1h1_end:
            vld1.32 {d0[0]}, [r7]!
            vcvt.f32.s32 d16, d16
            vmul.f32 d16, d16, d0[0]
            vcgt.f32 d0, d16, #0
            vbsl.f32 d0, d8, d10
            vadd.f32 d16, d16, d0
            vcvt.s32.f32 d16, d16
            vmin.s32 d16, d16, d12
            vmax.s32 d16, d16, d14
            vqmovn.s32 d0, q8
            vqmovn.s16 d0, q0
            mov lr, #4
            vst1.8 {d0[0]}, [r0]
            add r11, r11, #1
            ands lr, r11, #0x03
            addne r0, r0, #1
            ldr lr, [r3, #20] // cStride
            subeq lr, lr, #3
            addeq r0, r0, lr
            cmp r11, r10
        blt loop_e1h1
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

