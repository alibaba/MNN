//
//  MNNPackedSparseMatMulEpx4.S
//  MNN
//
//  Created by MNN on 2021/04/28.
//  Copyright © 2018-2021 Alibaba Group Holding Limited
//
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"
#define sizeof_value 4
#define sizeof_value_lg2 2
#define sparse_blockoc 4

.text
.align 5
// 16 * 4 MatMul
asm_function MNNPackedSparseMatMulEpx4
// void MNNPackedSparseMatMulEpx4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap) {
//Auto x0: C, x1:A, x2:B, x3:eSize, x4:parameter, x5:postParameters, x6:bias, x7:NNZMap
// avoid to touch platform-register x-18
ldr x8, [sp, #0] // x7: unsigned int* NNZMap, x8: int* dataOffsetMap
ldp x9, x10, [x4, #0]   // x9: aStride, x10: l
ldp x11, x12, [x4, #16] // x11: h, x12: cStride

stp x25, x26, [sp, #(-16 * 4)]!
stp x23, x24, [sp, #(16 * 1)]
stp x21, x22, [sp, #(16 * 2)]
stp x19, x20, [sp, #(16 * 3)]

mul x13, x9, x10 // x13: aStride with sizeof()
lsr x9, x9, #2 // x9: eP = parameter[0] / sizeof(float);

lsr x14, x11, #2
lsl x14, x14, #2 // x14:  (h / 4) * 4

add x5, x5, #(2 * 4) // move to float element [2], [3]
ld2r {v5.4s, v6.4s}, [x5]

//x0:C,
//x1:A,
//x2:B,
//x3:eSize,
//x4:parameter,      // free
//x5:postParameters, // free
//x6:bias
// x7, x15: unsigned int* NNZMap,
// x8, x25: int* dataOffsetMap
// x9: eP,
// x10: l             // free
// x11: h,
// x12: cStride with sizeof
// x13: aStride with sizeof
// x14: (h / 4) * 4

// v0-v3: A
// v4:  B
// v5: minValue
// v6: maxValue
// v16-v31: C
// sparse_blockoc = 4


// x4 as ie
// x5 as ih
// w20 as il

mov x10, x2
mov x4, xzr
cmp x9, x3
bgt loop_e8

loop_e16:

    mov x25, x8
    ldrsw x26, [x25], #4
    add x1, x1, x26, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

    mov x2, x10
    mov x15, x7
    add x26, x0, x4, lsl #(sizeof_value_lg2 + 2) // float* blockC = C + (ie << 2);

    mov x5, xzr
    mov x24, x6 // bias
    cbz x14, loop_e16h1

    loop_e16h4:

        lsr x20, x5, #2 // NC4HW4
        mul x20, x20, x12
        add x19, x26, x20 // x19: c = blockC + ihpack * cStride
        cbz x6, load_e16h4_zero
            ldr q16, [x24], #(4 * sparse_blockoc)
            b load_e16h4_end
        load_e16h4_zero:
            movi v16.4s, #0000000000000000

        load_e16h4_end:
        ldr w20, [x15], #4
        mov v17.16b, v16.16b
        mov v18.16b, v16.16b
        mov v19.16b, v16.16b
        mov v20.16b, v16.16b
        mov v21.16b, v16.16b
        mov v22.16b, v16.16b
        mov v23.16b, v16.16b
        mov v24.16b, v16.16b
        mov v25.16b, v16.16b
        mov v26.16b, v16.16b
        mov v27.16b, v16.16b
        mov v28.16b, v16.16b
        mov v29.16b, v16.16b
        mov v30.16b, v16.16b
        mov v31.16b, v16.16b
        cbz w20, loop_e16h4l1_end

        loop_e16h4l1:

            ldp q0, q1, [x1]
            ldp q2, q3, [x1, #(8 * sizeof_value)]
            ldr q4, [x2], #(4 * sizeof_value)
            ldrsw x21, [x25], #4
            subs w20, w20, #1
            add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

            fmla v16.4s, v4.4s, v0.s[0]
            fmla v17.4s, v4.4s, v0.s[1]
            fmla v18.4s, v4.4s, v0.s[2]
            fmla v19.4s, v4.4s, v0.s[3]

            fmla v20.4s, v4.4s, v1.s[0]
            fmla v21.4s, v4.4s, v1.s[1]
            fmla v22.4s, v4.4s, v1.s[2]
            fmla v23.4s, v4.4s, v1.s[3]

            fmla v24.4s, v4.4s, v2.s[0]
            fmla v25.4s, v4.4s, v2.s[1]
            fmla v26.4s, v4.4s, v2.s[2]
            fmla v27.4s, v4.4s, v2.s[3]

            fmla v28.4s, v4.4s, v3.s[0]
            fmla v29.4s, v4.4s, v3.s[1]
            fmla v30.4s, v4.4s, v3.s[2]
            fmla v31.4s, v4.4s, v3.s[3]
            bne loop_e16h4l1

        loop_e16h4l1_end:
        fmin v16.4s, v16.4s, v6.4s
        fmin v17.4s, v17.4s, v6.4s
        fmin v18.4s, v18.4s, v6.4s
        fmin v19.4s, v19.4s, v6.4s
        fmin v20.4s, v20.4s, v6.4s
        fmin v21.4s, v21.4s, v6.4s
        fmin v22.4s, v22.4s, v6.4s
        fmin v23.4s, v23.4s, v6.4s
        fmin v24.4s, v24.4s, v6.4s
        fmin v25.4s, v25.4s, v6.4s
        fmin v26.4s, v26.4s, v6.4s
        fmin v27.4s, v27.4s, v6.4s
        fmin v28.4s, v28.4s, v6.4s
        fmin v29.4s, v29.4s, v6.4s
        fmin v30.4s, v30.4s, v6.4s
        fmin v31.4s, v31.4s, v6.4s
       add x5, x5, #sparse_blockoc
        fmax v16.4s, v16.4s, v5.4s
        fmax v17.4s, v17.4s, v5.4s
        fmax v18.4s, v18.4s, v5.4s
        fmax v19.4s, v19.4s, v5.4s
        fmax v20.4s, v20.4s, v5.4s
        fmax v21.4s, v21.4s, v5.4s
        fmax v22.4s, v22.4s, v5.4s
        fmax v23.4s, v23.4s, v5.4s
        fmax v24.4s, v24.4s, v5.4s
        fmax v25.4s, v25.4s, v5.4s
        fmax v26.4s, v26.4s, v5.4s
        fmax v27.4s, v27.4s, v5.4s
        fmax v28.4s, v28.4s, v5.4s
        fmax v29.4s, v29.4s, v5.4s
        fmax v30.4s, v30.4s, v5.4s
        fmax v31.4s, v31.4s, v5.4s

        cmp x5, x14
        stp q16, q17, [x19]
        stp q18, q19, [x19, #(32 * 1)]
        stp q20, q21, [x19, #(32 * 2)]
        stp q22, q23, [x19, #(32 * 3)]
        stp q24, q25, [x19, #(32 * 4)]
        stp q26, q27, [x19, #(32 * 5)]
        stp q28, q29, [x19, #(32 * 6)]
        stp q30, q31, [x19, #(32 * 7)]
        blt loop_e16h4

        cmp x5, x11
        bge loop_e16h_end

        lsr x19, x5, #2 // NC4HW4
        mul x19, x19, x12
        // mov x24, x6 // bias
        add x26, x26, x19 // x19: c = blockC + ihpack * cStride

    loop_e16h1:
        and x20, x5, #0x03 // NC4HW4
        add x19, x26, x20, lsl #sizeof_value_lg2 // x19: c = blockC + isubIndex

        cbz x6, load_e16h1_zero
            ld1r {v16.4s}, [x24], #(4)
            b load_e16h1_end
        load_e16h1_zero:
            movi v16.4s, #0000000000000000

        load_e16h1_end:
        ldr w20, [x15], #4
        mov v17.16b, v16.16b
        mov v18.16b, v16.16b
        mov v19.16b, v16.16b
        cbz w20, loop_e16h1l1_end

        loop_e16h1l1:

          ldp q0, q1, [x1]
          ldp q2, q3, [x1, #(8 * sizeof_value)]
          ld1r {v4.4s}, [x2], #(sizeof_value)
          ldrsw x21, [x25], #4
          subs w20, w20, #1
          add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

          fmla v16.4s, v4.4s, v0.4s
          fmla v17.4s, v4.4s, v1.4s
          fmla v18.4s, v4.4s, v2.4s
          fmla v19.4s, v4.4s, v3.4s

          bne loop_e16h1l1

    loop_e16h1l1_end:

     // layout1:
     // fmin v16.4s, v16.4s, v6.4s
     // add x20, x19, #(4 * sizeof_value)
     // add x5, x5, #1
     // fmax v16.4s, v16.4s, v5.4s
     // add x21, x19, #(8 * sizeof_value)
     // add x22, x20, #(8 * sizeof_value)
     // str s16, [x19]
     // st1 {v16.s}[1], [x20]
//
     // fmin v17.4s, v17.4s, v6.4s
     // st1 {v16.s}[2], [x21]
     // st1 {v16.s}[3], [x22]
     // add x20, x20, #(4 * 4 * sizeof_value)
     // fmax v17.4s, v17.4s, v5.4s
     // add x21, x21, #(4 * 4 * sizeof_value)
     // add x22, x22, #(4 * 4 * sizeof_value)
     // str s17, [x19, #(4 * 4 * sizeof_value)]
     // st1 {v17.s}[1], [x20]
//
     // fmin v18.4s, v18.4s, v6.4s
     // st1 {v17.s}[2], [x21]
     // st1 {v17.s}[3], [x22]
     // add x20, x20, #(4 * 4 * sizeof_value)
     // fmax v18.4s, v18.4s, v5.4s
     // add x21, x21, #(4 * 4 * sizeof_value)
     // add x22, x22, #(4 * 4 * sizeof_value)
     // str s18, [x19, #(8 * 4 * sizeof_value)]
     // st1 {v18.s}[1], [x20]
     // fmin v19.4s, v19.4s, v6.4s
     // st1 {v18.s}[2], [x21]
     // st1 {v18.s}[3], [x22]
     // add x20, x20, #(4 * 4 * sizeof_value)
//
     // fmax v19.4s, v19.4s, v5.4s
     // add x21, x21, #(4 * 4 * sizeof_value)
     // add x22, x22, #(4 * 4 * sizeof_value)
     // str s19, [x19, #(12 * 4 * sizeof_value)]
     // st1 {v19.s}[1], [x20]
     // st1 {v19.s}[2], [x21]
     // st1 {v19.s}[3], [x22]
     // cmp x5, x11


    // // layout2:
     // fmin v16.4s, v16.4s, v6.4s
     // fmin v17.4s, v17.4s, v6.4s
     // //add x20, x19, #(4 * sizeof_value)
     // //mov x21, #(8 * sizeof_value)
     // mov x20, #(4 * sizeof_value)
     // fmin v18.4s, v18.4s, v6.4s
     // fmin v19.4s, v19.4s, v6.4s
     // add x5, x5, #1
     // fmax v16.4s, v16.4s, v5.4s
     // fmax v17.4s, v17.4s, v5.4s
     // cmp x5, x11
     // fmax v18.4s, v18.4s, v5.4s
     // fmax v19.4s, v19.4s, v5.4s
//
     // st1 {v16.s}[0], [x19], x20 // st1 donot support immediate increasement other than sizeof stored element
     // st1 {v16.s}[1], [x19], x20
     // st1 {v16.s}[2], [x19], x20
     // st1 {v16.s}[3], [x19], x20
     // st1 {v17.s}[0], [x19], x20
     // st1 {v17.s}[1], [x19], x20
     // st1 {v17.s}[2], [x19], x20
     // st1 {v17.s}[3], [x19], x20
     // st1 {v18.s}[0], [x19], x20
     // st1 {v18.s}[1], [x19], x20
     // st1 {v18.s}[2], [x19], x20
     // st1 {v18.s}[3], [x19], x20
     // st1 {v19.s}[0], [x19], x20
     // st1 {v19.s}[1], [x19], x20
     // st1 {v19.s}[2], [x19], x20
     // st1 {v19.s}[3], [x19]

    // layout3:
    fmin v16.4s, v16.4s, v6.4s
    fmin v17.4s, v17.4s, v6.4s
    mov x23, #(4 * 4 * sizeof_value)
    add x20, x19, #(4 * sizeof_value)
    fmin v18.4s, v18.4s, v6.4s
    fmin v19.4s, v19.4s, v6.4s
    add x5, x5, #1
    add x21, x19, #(8 * sizeof_value)
    fmax v16.4s, v16.4s, v5.4s
    fmax v17.4s, v17.4s, v5.4s
    add x22, x20, #(8 * sizeof_value)
    cmp x5, x11
    fmax v18.4s, v18.4s, v5.4s
    fmax v19.4s, v19.4s, v5.4s

    st1 {v16.s}[0], [x19], x23 // st1 donot support immediate increasement other than sizeof stored element
    st1 {v16.s}[1], [x20], x23
    st1 {v16.s}[2], [x21], x23
    st1 {v16.s}[3], [x22], x23
    st1 {v17.s}[0], [x19], x23
    st1 {v17.s}[1], [x20], x23
    st1 {v17.s}[2], [x21], x23
    st1 {v17.s}[3], [x22], x23
    st1 {v18.s}[0], [x19], x23
    st1 {v18.s}[1], [x20], x23
    st1 {v18.s}[2], [x21], x23
    st1 {v18.s}[3], [x22], x23
    st1 {v19.s}[0], [x19]
    st1 {v19.s}[1], [x20]
    st1 {v19.s}[2], [x21]
    st1 {v19.s}[3], [x22]

    blt loop_e16h1

    loop_e16h_end:

    add x4, x4, x9
    add x1, x1, x13

    add x5, x4, x9
    cmp x5, x3
    ble loop_e16

loop_e8:
ands x5, x3, #0x08
beq loop_e4

    mov x25, x8
    ldrsw x26, [x25], #4
    add x1, x1, x26, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

    mov x2, x10
    mov x15, x7
    add x26, x0, x4, lsl #(sizeof_value_lg2 + 2) // float* blockC = C + (ie << 2);

    mov x5, xzr
    mov x24, x6 // bias
    cbz x14, loop_e8h1

    loop_e8h4:

        lsr x20, x5, #2 // NC4HW4
        mul x20, x20, x12
        add x19, x26, x20 // x19: c = blockC + ihpack * cStride
        cbz x6, load_e8h4_zero
            ldr q16, [x24], #(4 * sparse_blockoc)
            b load_e8h4_end
        load_e8h4_zero:
            movi v16.4s, #0000000000000000

        load_e8h4_end:
        ldr w20, [x15], #4
        mov v17.16b, v16.16b
        mov v18.16b, v16.16b
        mov v19.16b, v16.16b
        mov v20.16b, v16.16b
        mov v21.16b, v16.16b
        mov v22.16b, v16.16b
        mov v23.16b, v16.16b
        cbz w20, loop_e8h4l1_end

        loop_e8h4l1:

            ldp q0, q1, [x1]
            ldr q4, [x2], #(4 * sizeof_value)
            ldrsw x21, [x25], #4
            subs w20, w20, #1
            add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

            fmla v16.4s, v4.4s, v0.s[0]
            fmla v17.4s, v4.4s, v0.s[1]
            fmla v18.4s, v4.4s, v0.s[2]
            fmla v19.4s, v4.4s, v0.s[3]

            fmla v20.4s, v4.4s, v1.s[0]
            fmla v21.4s, v4.4s, v1.s[1]
            fmla v22.4s, v4.4s, v1.s[2]
            fmla v23.4s, v4.4s, v1.s[3]
            bne loop_e8h4l1

        loop_e8h4l1_end:
        fmin v16.4s, v16.4s, v6.4s
        fmin v17.4s, v17.4s, v6.4s
        fmin v18.4s, v18.4s, v6.4s
        fmin v19.4s, v19.4s, v6.4s
        fmin v20.4s, v20.4s, v6.4s
        fmin v21.4s, v21.4s, v6.4s
        fmin v22.4s, v22.4s, v6.4s
        fmin v23.4s, v23.4s, v6.4s
       add x5, x5, #sparse_blockoc
        fmax v16.4s, v16.4s, v5.4s
        fmax v17.4s, v17.4s, v5.4s
        fmax v18.4s, v18.4s, v5.4s
        fmax v19.4s, v19.4s, v5.4s
        fmax v20.4s, v20.4s, v5.4s
        fmax v21.4s, v21.4s, v5.4s
        fmax v22.4s, v22.4s, v5.4s
        fmax v23.4s, v23.4s, v5.4s

        cmp x5, x14
        stp q16, q17, [x19]
        stp q18, q19, [x19, #(32 * 1)]
        stp q20, q21, [x19, #(32 * 2)]
        stp q22, q23, [x19, #(32 * 3)]
        blt loop_e8h4

        cmp x5, x11
        bge loop_e8h_end

        lsr x19, x5, #2 // NC4HW4
        mul x19, x19, x12
        // mov x24, x6 // bias
        add x26, x26, x19 // x19: c = blockC + ihpack * cStride

    loop_e8h1:
        and x20, x5, #0x03 // NC4HW4
        add x19, x26, x20, lsl #sizeof_value_lg2 // x19: c = blockC + isubIndex

        cbz x6, load_e8h1_zero
            ld1r {v16.4s}, [x24], #(4)
            b load_e8h1_end
        load_e8h1_zero:
            movi v16.4s, #0000000000000000

        load_e8h1_end:
        ldr w20, [x15], #4
        mov v17.16b, v16.16b
        cbz w20, loop_e8h1l1_end

        loop_e8h1l1:

          ldp q0, q1, [x1]
          ld1r {v4.4s}, [x2], #(sizeof_value)
          ldrsw x21, [x25], #4
          subs w20, w20, #1
          add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

          fmla v16.4s, v4.4s, v0.4s
          fmla v17.4s, v4.4s, v1.4s

          bne loop_e8h1l1

    loop_e8h1l1_end:

    fmin v16.4s, v16.4s, v6.4s
    fmin v17.4s, v17.4s, v6.4s
    mov x23, #(4 * 4 * sizeof_value)
    add x20, x19, #(4 * sizeof_value)
    add x5, x5, #1
    fmax v16.4s, v16.4s, v5.4s
    fmax v17.4s, v17.4s, v5.4s
    add x21, x19, #(8 * sizeof_value)
    add x22, x20, #(8 * sizeof_value)

    cmp x5, x11
    st1 {v16.s}[0], [x19], X23 // st1 donot support immediate increasement other than sizeof stored element
    st1 {v16.s}[1], [x20], X23
    st1 {v16.s}[2], [x21], X23
    st1 {v16.s}[3], [x22], X23

    st1 {v17.s}[0], [x19]
    st1 {v17.s}[1], [x20]
    st1 {v17.s}[2], [x21]
    st1 {v17.s}[3], [x22]
    blt loop_e8h1

    loop_e8h_end:

    add x4, x4, #8 // e8
    add x1, x1, #(8 * sizeof_value) // Has not exceed one aStride, just 8


loop_e4:
ands x5, x3, #0x04
beq loop_e2

    mov x25, x8
    ldrsw x26, [x25], #4
    add x1, x1, x26, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

    mov x2, x10
    mov x15, x7
    add x26, x0, x4, lsl #(sizeof_value_lg2 + 2) // float* blockC = C + (ie << 2);
    mov x5, xzr
    mov x24, x6 // bias
    cbz x14, loop_e4h1

    loop_e4h4:

        lsr x20, x5, #2 // NC4HW4
        mul x20, x20, x12
        add x19, x26, x20 // x19: c = blockC + ihpack * cStride
        cbz x6, load_e4h4_zero
            ldr q16, [x24], #(4 * sparse_blockoc)
            b load_e4h4_end
        load_e4h4_zero:
            movi v16.4s, #0000000000000000

        load_e4h4_end:
        ldr w20, [x15], #4
        mov v17.16b, v16.16b
        mov v18.16b, v16.16b
        mov v19.16b, v16.16b
        cbz w20, loop_e4h4l1_end

        loop_e4h4l1:

            ldr q0, [x1]
            ldr q4, [x2], #(4 * sizeof_value)
            ldrsw x21, [x25], #4
            subs w20, w20, #1
            add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

            fmla v16.4s, v4.4s, v0.s[0]
            fmla v17.4s, v4.4s, v0.s[1]
            fmla v18.4s, v4.4s, v0.s[2]
            fmla v19.4s, v4.4s, v0.s[3]
            bne loop_e4h4l1

        loop_e4h4l1_end:
        fmin v16.4s, v16.4s, v6.4s
        fmin v17.4s, v17.4s, v6.4s
        fmin v18.4s, v18.4s, v6.4s
        fmin v19.4s, v19.4s, v6.4s
        add x5, x5, #sparse_blockoc
        fmax v16.4s, v16.4s, v5.4s
        fmax v17.4s, v17.4s, v5.4s
        fmax v18.4s, v18.4s, v5.4s
        fmax v19.4s, v19.4s, v5.4s
        cmp x5, x14
        stp q16, q17, [x19]
        stp q18, q19, [x19, #(32 * 1)]
        blt loop_e4h4

        cmp x5, x11
        bge loop_e4h_end

        lsr x19, x5, #2 // NC4HW4
        mul x19, x19, x12
        // mov x24, x6 // bias
        add x26, x26, x19 // x19: c = blockC + ihpack * cStride

    loop_e4h1:
        and x20, x5, #0x03 // NC4HW4
        add x19, x26, x20, lsl #sizeof_value_lg2 // x20: c = blockC + isubIndex

        cbz x6, load_e4h1_zero
            ld1r {v16.4s}, [x24], #(4)
            b load_e4h1_end
        load_e4h1_zero:
            movi v16.4s, #0000000000000000

        load_e4h1_end:
        ldr w20, [x15], #4
        cbz w20, loop_e4h1l1_end

        loop_e4h1l1:

          ldr q0, [x1]
          ld1r {v4.4s}, [x2], #(sizeof_value)
          ldrsw x21, [x25], #4
          subs w20, w20, #1
          add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

          fmla v16.4s, v4.4s, v0.4s
          bne loop_e4h1l1

    loop_e4h1l1_end:

    fmin v16.4s, v16.4s, v6.4s
    add x20, x19, #(4 * sizeof_value)
    add x5, x5, #1
    fmax v16.4s, v16.4s, v5.4s
    add x21, x19, #(8 * sizeof_value)
    add x22, x20, #(8 * sizeof_value)
    cmp x5, x11
    st1 {v16.s}[0], [x19] // st1 donot support immediate increasement other than sizeof stored element
    st1 {v16.s}[1], [x20]
    st1 {v16.s}[2], [x21]
    st1 {v16.s}[3], [x22]
    blt loop_e4h1

    loop_e4h_end:

    add x4, x4, #4 // e4
    add x1, x1, #(4 * sizeof_value) // Has not exceed one aStride, just 4

loop_e2:
ands x5, x3, #0x02
beq loop_e1

    mov x25, x8
    ldrsw x26, [x25], #4
    add x1, x1, x26, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

    mov x2, x10
    mov x15, x7
    add x26, x0, x4, lsl #(sizeof_value_lg2 + 2) // float* blockC = C + (ie << 2);
    mov x5, xzr
    mov x24, x6 // bias
    cbz x14, loop_e2h1

    loop_e2h4:
        lsr x20, x5, #2 // NC4HW4
        mul x20, x20, x12
        add x19, x26, x20 // x19: c = blockC + ihpack * cStride
        cbz x6, load_e2h4_zero
            ldr q16, [x24], #(4 * sparse_blockoc)
            b load_e2h4_end
        load_e2h4_zero:
            movi v16.4s, #0000000000000000

        load_e2h4_end:
        ldr w20, [x15], #4
        mov v17.16b, v16.16b
        cbz w20, loop_e2h4l1_end

        loop_e2h4l1:

            ldr d0, [x1]
            ldr q4, [x2], #(4 * sizeof_value)
            ldrsw x21, [x25], #4
            subs w20, w20, #1
            add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

            fmla v16.4s, v4.4s, v0.s[0]
            fmla v17.4s, v4.4s, v0.s[1]

            bne loop_e2h4l1

        loop_e2h4l1_end:
        fmin v16.4s, v16.4s, v6.4s
        fmin v17.4s, v17.4s, v6.4s
        add x5, x5, #sparse_blockoc
        fmax v16.4s, v16.4s, v5.4s
        fmax v17.4s, v17.4s, v5.4s
        cmp x5, x14
        stp q16, q17, [x19]
        blt loop_e2h4

        cmp x5, x11
        bge loop_e2h_end

        lsr x19, x5, #2 // NC4HW4
        mul x19, x19, x12
        // mov x24, x6 // bias
        add x26, x26, x19 // x19: c = blockC + ihpack * cStride

    loop_e2h1:
        and x20, x5, #0x03 // NC4HW4
        add x19, x26, x20, lsl #sizeof_value_lg2 // x20: c = blockC + isubIndex

        cbz x6, load_e2h1_zero
            ld1r {v16.2s}, [x24], #(sparse_blockoc)
            b load_e2h1_end
        load_e2h1_zero:
            movi v16.4s, #0000000000000000

        load_e2h1_end:
        ldr w20, [x15], #4

        cbz w20, loop_e2h1l1_end

        loop_e2h1l1:

          ldr d0, [x1]
          ld1r {v4.2s}, [x2], #(sizeof_value)
          ldrsw x21, [x25], #4
          subs w20, w20, #1
          add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

          fmla v16.2s, v4.2s, v0.2s
          bne loop_e2h1l1

    loop_e2h1l1_end:

    fmin v16.2s, v16.2s, v6.2s
    add x20, x19, #(4 * sizeof_value)
    add x5, x5, #1
    fmax v16.2s, v16.2s, v5.2s
    cmp x5, x11
    st1 {v16.s}[0], [x19] // st1 donot support immediate increasement other than sizeof stored element
    st1 {v16.s}[1], [x20]

    blt loop_e2h1

    loop_e2h_end:
    add x4, x4, #2 // e2
    add x1, x1, #(2 * sizeof_value) // Has not exceed one aStride, just 2


loop_e1:
ands x5, x3, #0x01
beq loop_end

    mov x25, x8
    ldrsw x26, [x25], #4
    add x1, x1, x26, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

    mov x2, x10
    mov x15, x7
    add x26, x0, x4, lsl #(sizeof_value_lg2 + 2) // float* blockC = C + (ie << 2);

    mov x5, xzr
    mov x24, x6 // bias
    cbz x14, loop_e1h1

    loop_e1h4:
        lsr x20, x5, #2 // NC4HW4
        mul x20, x20, x12
        add x19, x26, x20 // x19: c = blockC + ihpack * cStride
        cbz x6, load_e1h4_zero
            ldr q16, [x24], #(4 * sparse_blockoc)
            b load_e1h4_end
        load_e1h4_zero:
            movi v16.4s, #0000000000000000

        load_e1h4_end:
        ldr w20, [x15], #4
        cbz w20, loop_e1h4l1_end

        loop_e1h4l1:

            ldr s0, [x1]
            ldr q4, [x2], #(4 * sizeof_value)
            ldrsw x21, [x25], #4
            subs w20, w20, #1
            add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

            fmla v16.4s, v4.4s, v0.s[0]

            bne loop_e1h4l1

        loop_e1h4l1_end:
        fmin v16.4s, v16.4s, v6.4s
        add x5, x5, #sparse_blockoc
        fmax v16.4s, v16.4s, v5.4s
        cmp x5, x14
        str q16, [x19]
        blt loop_e1h4

        cmp x5, x11
        bge loop_e1h_end

        lsr x19, x5, #2 // NC4HW4
        mul x19, x19, x12
        // mov x24, x6 // bias
        add x26, x26, x19 // x19: c = blockC + ihpack * cStride

    loop_e1h1:
        and x20, x5, #0x03 // NC4HW4
        add x19, x26, x20, lsl #sizeof_value_lg2 // x20: c = blockC + isubIndex

        cbz x6, load_e1h1_zero
            ld1 {v16.s}[0], [x24], #(4)
            b load_e1h1_end
        load_e1h1_zero:
            movi v16.4s, #0000000000000000

        load_e1h1_end:
        ldr w20, [x15], #4

        cbz w20, loop_e1h1l1_end

        loop_e1h1l1:

          ldr s0, [x1]
          ldr s4, [x2], #(sizeof_value)
          ldrsw x21, [x25], #4
          subs w20, w20, #1
          add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

          fmla s16, s4, v0.s[0]
          bne loop_e1h1l1

    loop_e1h1l1_end:

    fmin s16, s16, s6
    add x5, x5, #1
    fmax s16, s16, s5

    cmp x5, x11
    str s16, [x19]
    blt loop_e1h1

    loop_e1h_end:
    add x4, x4, #1 // e1
    // add x1, x1, #(1 * sizeof_value) // Has not exceed one aStride, just 1


loop_end:
ldp x19, x20, [sp, #(16 * 3)]
ldp x21, x22, [sp, #(16 * 2)]
ldp x23, x24, [sp, #(16 * 1)]
ldp x25, x26, [sp], #(16 * 4)
ret

#undef sizeof_value
#undef sizeof_value_lg2
#undef sparse_blockoc

#endif


