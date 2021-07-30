//
//  MNNPackedSparseMatMulEpx1.S
//  MNN
//
//  Created by MNN on 2021/05/10.
//  Copyright © 2018-2021 Alibaba Group Holding Limited
//
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"
#define sizeof_value 4
#define sizeof_value_lg2 2
#define sparse_blockoc 4

#define push_registers_bytes (8 * 4 + 4 * 16)

.text
.align 5
// caution!!! this is 8 * 1 Sparse MatMul
asm_function MNNPackedSparseMatMulEpx1
// void MNNPackedSparseMatMulEpx1(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap) {
//Auto r0: C, r1:A, r2:B, r3:eSize,
//load from stack r4:parameter, r5:postParameters, r6:bias, r7:NNZMap, r8:dataOffsetMap

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9
vpush {q4-q7}


ldr r4, [sp, #push_registers_bytes]
ldr r5, [sp, #(push_registers_bytes + 4)]
vmov d13, r0, r2

ldr r7, [sp, #(push_registers_bytes + 12)]
ldr r8, [sp, #(push_registers_bytes + 16)]
vmov d14, r7, r8

ldr lr, [r4, #0]                                   // x9: aStride, x10: l
ldr r10, [r4, #4]
ldr r6, [sp, #(push_registers_bytes + 8)]
vmov d10, r3, lr                                  // eSize, eP

mul r10, lr, r10                                  // x13: aStride with sizeof()
lsr lr, lr, #2                                    // x9: eP

ldr r11, [r4, #8]                                 // x11: h, x12: cStride
ldr r12, [r4, #12]
lsr r0, r11, #2
add r5, r5, #(2 * 4)                              // move to float element [2], [3]
lsl r0, r0, #2                                    // x14:  (h / 4) * 4

vmov d12, r12, r10                                // cStride, aStride
vmov d15, r6, r6                                  // compile error when 'vmov d15[0], r6'
vmov d11, r0, r11                                 // h_even_4, h

vld1.32 {d6[], d7[]}, [r5:32]!
vld1.32 {d8[], d9[]}, [r5:32]

mov r4, #0
cmp lr, r3
bgt loop_e4

loop_e8:

    vmov r7, r8, d14
    vmov r0, r2, d13

    ldr lr, [r8], #4
    add r3, r0, r4, lsl #(sizeof_value_lg2 + 2) // float* blockC = C + (ie << 2);
    add r1, r1, lr, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

    vmov lr, r11, d11
    vmov r12, r10, d12 // cStride
    vmov r6, r10, d15
    cmp r11, #0
    mov r5, #0
    beq loop_e8h_end

    loop_e8h1:
        lsr r10, r5, #2
        and r0, r5, #0x03 // NC4HW4
        mul r10, r10, r12
        add r0, r3, r0, lsl #sizeof_value_lg2 // r0: c = blockC + ihpack * cStride + isubIndex
        cmp r6, #0
        add r0, r0, r10

        beq load_e8h1_zero
            vld1.32 {d16[], d17[]}, [r6:32]!
            b load_e8h1_end
        load_e8h1_zero:
            vmov.i32 q8, #0

        load_e8h1_end:
        ldr r10, [r7], #4
        vmov q9, q8
        cmp r10, #0
        beq loop_e8h1l1_end

        loop_e8h1l1:

          vld1.32 {q0, q1}, [r1]
          vld1.32 {d4[], d5[]}, [r2:32]!
          ldr lr, [r8], #4
          subs r10, r10, #1
          add r1, r1, lr, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

          vmla.f32 q8, q2, q0
          vmla.f32 q9, q2, q1

          bne loop_e8h1l1

    loop_e8h1l1_end:

    // layout3:
    vmin.f32 q8, q8, q4
    vmin.f32 q9, q9, q4
    vmov r10, r11, d11
    add r5, r5, #1
    vmax.f32 q8, q8, q3
    vmax.f32 q9, q9, q3
    add lr, r0, #(4 * sizeof_value)
    mov r10, #(2 * 4 * sizeof_value)

    cmp r5, r11
    vst1.32 {d16[0]}, [r0], r10 // st1 donot support immediate increasement other than sizeof stored element
    vst1.32 {d16[1]}, [lr], r10
    vst1.32 {d17[0]}, [r0], r10
    vst1.32 {d17[1]}, [lr], r10
    vst1.32 {d18[0]}, [r0], r10
    vst1.32 {d18[1]}, [lr], r10
    vst1.32 {d19[0]}, [r0]
    vst1.32 {d19[1]}, [lr]

    blt loop_e8h1

    loop_e8h_end:

    vmov r3, lr, d10
    vmov r10, r6, d12

    add r4, r4, lr
    add r1, r1, r6

    add r5, r4, lr
    cmp r5, r3
    ble loop_e8

loop_e4:
ands r5, r3, #0x04
beq loop_e2

    vmov r7, r8, d14
    vmov r0, r2, d13

    ldr lr, [r8], #4
    add r3, r0, r4, lsl #(sizeof_value_lg2 + 2) // float* blockC = C + (ie << 2);
    add r1, r1, lr, lsl #sizeof_value_lg2 // a += diff * sizeof(float)
    vmov lr, r11, d11
    vmov r12, r10, d12 // cStride
    vmov r6, r10, d15
    cmp r11, #0
    mov r5, #0
    beq loop_e4h_end

    loop_e4h1:
        lsr r10, r5, #2
        and r0, r5, #0x03 // NC4HW4
        mul r10, r10, r12
        add r0, r3, r0, lsl #sizeof_value_lg2 // r0: c = blockC + ihpack * cStride + isubIndex
        cmp r6, #0
        add r0, r0, r10
        beq load_e4h1_zero
            vld1.32 {d16[], d17[]}, [r6:32]!
            b load_e4h1_end
        load_e4h1_zero:
            vmov.i32 q8, #0

        load_e4h1_end:
        ldr r10, [r7], #4
        cmp r10, #0
        beq loop_e4h1l1_end

        loop_e4h1l1:

          vld1.32 {q0}, [r1]
          vld1.32 {d4[], d5[]}, [r2:32]!
          ldr lr, [r8], #4
          subs r10, r10, #1
          add r1, r1, lr, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

          vmla.f32 q8, q2, q0

          bne loop_e4h1l1

    loop_e4h1l1_end:
    // layout3:
    vmin.f32 q8, q8, q4
    vmov r10, r11, d11
    add r5, r5, #1
    vmax.f32 q8, q8, q3
    add lr, r0, #(4 * sizeof_value)
    mov r10, #(2 * 4 * sizeof_value)

    vst1.32 {d16[0]}, [r0], r10 // st1 donot support immediate increasement other than sizeof stored element
    vst1.32 {d16[1]}, [lr], r10
    cmp r5, r11
    vst1.32 {d17[0]}, [r0]
    vst1.32 {d17[1]}, [lr]

    blt loop_e4h1

    loop_e4h_end:
    vmov r3, lr, d10 // caution: r3=eSize is used in next loop.
    add r4, r4, #4
    add r1, r1, #(4 * sizeof_value) // Has not exceed one aStride, just 4

loop_e2:
ands r5, r3, #0x02
beq loop_e1


    vmov r7, r8, d14
    vmov r0, r2, d13
    ldr lr, [r8], #4
    add r3, r0, r4, lsl #(sizeof_value_lg2 + 2) // float* blockC = C + (ie << 2);
    add r1, r1, lr, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

    vmov lr, r11, d11
    vmov r12, r10, d12 // cStride
    vmov r6, r10, d15
    cmp r11, #0
    mov r5, #0
    beq loop_e2h_end
    loop_e2h1:
        lsr r10, r5, #2
        and r0, r5, #0x03 // NC4HW4
        mul r10, r10, r12
        add r0, r3, r0, lsl #sizeof_value_lg2 // r0: c = blockC + ihpack * cStride + isubIndex
        cmp r6, #0
        add r0, r0, r10

        beq load_e2h1_zero
            vld1.32 {d16[]}, [r6:32]!
            b load_e2h1_end
        load_e2h1_zero:
            vmov.i32 q8, #0

        load_e2h1_end:
        ldr r10, [r7], #4
        cmp r10, #0
        beq loop_e2h1l1_end

        loop_e2h1l1:

          vld1.32 {d0}, [r1]
          vld1.32 {d4[]}, [r2:32]!
          ldr lr, [r8], #4
          subs r10, r10, #1
          add r1, r1, lr, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

          vmla.f32 d16, d4, d0

          bne loop_e2h1l1

    loop_e2h1l1_end:
    // layout3:
    vmin.f32 d16, d16, d8
    add r5, r5, #1
    vmax.f32 d16, d16, d6
    add lr, r0, #(4 * sizeof_value)

    vst1.32 {d16[0]}, [r0] // st1 donot support immediate increasement other than sizeof stored element
    vst1.32 {d16[1]}, [lr]
    cmp r5, r11

    blt loop_e2h1

    loop_e2h_end:
    vmov r3, lr, d10 // caution: r3=eSize is used in next loop.
    add r4, r4, #2
    add r1, r1, #(2 * sizeof_value) // Has not exceed one aStride, just 2

loop_e1:
ands r5, r3, #0x01
beq loop_end

    vmov r7, r8, d14
    vmov r0, r2, d13
    ldr lr, [r8], #4
    add r3, r0, r4, lsl #(sizeof_value_lg2 + 2) // float* blockC = C + (ie << 2);
    add r1, r1, lr, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

    vmov lr, r11, d11
    vmov r12, r10, d12 // cStride
    vmov r6, r10, d15
    cmp r11, #0
    mov r5, #0
    beq loop_e1h_end

    loop_e1h1:
        lsr r10, r5, #2
        and r0, r5, #0x03 // NC4HW4
        mul r10, r10, r12
        add r0, r3, r0, lsl #sizeof_value_lg2 // r0: c = blockC + ihpack * cStride + isubIndex
        cmp r6, #0
        add r0, r0, r10

        beq load_e1h1_zero
            vld1.32 {d16[0]}, [r6]!
            b load_e1h1_end
        load_e1h1_zero:
            vmov.i32 d16, #0

        load_e1h1_end:
        ldr r10, [r7], #4
        cmp r10, #0
        beq loop_e1h1l1_end

        loop_e1h1l1:

          vld1.32 {d0[0]}, [r1]
          vld1.32 {d4[0]}, [r2]!
          ldr lr, [r8], #4
          subs r10, r10, #1
          add r1, r1, lr, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

          vmla.f32 d16, d4, d0[0]

          bne loop_e1h1l1

    loop_e1h1l1_end:
    // layout3:
    vmin.f32 d16, d16, d8
    add r5, r5, #1
    vmax.f32 d16, d16, d6

    vst1.32 {d16[0]}, [r0:32] // st1 donot support immediate increasement other than sizeof stored element
    cmp r5, r11
    blt loop_e1h1

    loop_e1h_end:

loop_end:

vpop {q4-q7}
pop {r4-r8, r10, r11, pc}

#undef push_registers_bytes
#undef sizeof_value
#undef sizeof_value_lg2
#undef sparse_blockoc

#endif
#endif

