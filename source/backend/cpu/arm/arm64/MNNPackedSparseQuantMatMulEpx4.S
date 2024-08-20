//
//  MNNPackedSparseQuantMatMulEpx4.S
//  MNN
//
//  Created by MNN on 2021/06/20.
//  Copyright © 2018-2021 Alibaba Group Holding Limited
//
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"
#define sizeof_value 1
#define sizeof_value_lg2 0
#define sparse_blockoc 4


.text
.align 5
// 16 * 4 MatMul
asm_function MNNPackedSparseQuantMatMulEpx4
// void MNNPackedSparseQuantMatMulEpx4(int8_t* C, const int8_t* A, const int8_t* B, const size_t* sparseQuantParam,
// const QuanPostTreatParameters* post, unsigned int* NNZMap, int* dataOffsetMap) {
// x0: C, x1:A, x2:B, x3:sparseQuantParam, x4:QuanPostTreatParameters, x5:NNZMap, x6:dataOffsetMap

str d14, [sp, #(-16 * 9)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x27, x28, [sp, #(16 * 4)]
stp x25, x26, [sp, #(16 * 5)]
stp x23, x24, [sp, #(16 * 6)]
stp x21, x22, [sp, #(16 * 7)]
stp x19, x20, [sp, #(16 * 8)]

ldp x13, x10, [x3, #16]     // x13: aStride, x10: l
ldp x11, x12, [x3, #32]     // x11: h, x12: cStride
ldp x3, x9, [x3]            // x3: eSize, x9: eP

mov x8, x6                  // x8: dataOffsetMap
mov x7, x5                  // x7: NNZMap
ldr x24, [x4]
ldr x6, [x4, #72]
add x4, x4, #16
//ldp x24, x6, [x4], #16      // x5: scale , x6: bias
lsr x14, x11, #2
lsl x14, x14, #2            // x14:  (h / 4) * 4
ld2r {v13.4s, v14.4s}, [x4] // first two elements of x4 are pointers, 'max, min ' locate at [2], [3]


//x0:C,
//x1:A,
//x2:B,
//x3:eSize,
//x4:parameter,      // free
//x5:postParameters, // free
//x6:bias
// x7, x15: unsigned int* NNZMap,
// x8, x26: int* dataOffsetMap
// x9: eP,
// x10: l             // free
// x11: h,
// x12: cStride with sizeof
// x13: aStride with sizeof
// x14: (h / 4) * 4
// x24: scale

// v0-v3: A
// v4:  B
// v13: maxValue
// v14: minValue
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

    mov x26, x8
    ldrsw x27, [x26], #4
    add x1, x1, x27, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

    mov x2, x10
    mov x15, x7
    add x27, x0, x4, lsl #(sizeof_value_lg2 + 2) // float* blockC = C + (ie << 2);

    mov x5, xzr
    mov x28, x6 // bias
    mov x25, x24 // scale
    cbz x14, loop_e16h1
    loop_e16h4:

        lsr x20, x5, #2 // NC4HW4
        mul x20, x20, x12
        add x19, x27, x20 // x19: c = blockC + ihpack * cStride
        cbz x6, load_e16h4_zero
            ldr q16, [x28], #(4 * sparse_blockoc)
            b load_e16h4_end
        load_e16h4_zero:
            movi v16.4s, #0000000000000000

        load_e16h4_end:
        ldr w20, [x15], #4
        dup v20.4s, v16.s[1]
        dup v24.4s, v16.s[2]
        dup v28.4s, v16.s[3]
        dup v16.4s, v16.s[0]

        mov v21.16b, v20.16b
        mov v22.16b, v20.16b
        mov v23.16b, v20.16b

        mov v25.16b, v24.16b
        mov v26.16b, v24.16b
        mov v27.16b, v24.16b

        mov v29.16b, v28.16b
        mov v30.16b, v28.16b
        mov v31.16b, v28.16b

        mov v17.16b, v16.16b
        mov v18.16b, v16.16b
        mov v19.16b, v16.16b
        cbz w20, loop_e16h4l1_end

        loop_e16h4l1:
/*
            ld4r {v1.16b, v2.16b, v3.16b, v4.16b}, [x2], #(4 * sizeof_value)
            ldr q0, [x1]
            ldrsw x21, [x26], #4
            subs w20, w20, #1
            add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(int8)

            smull v5.8h, v0.8b, v1.8b
            smull v6.8h, v0.8b, v2.8b
            smull v7.8h, v0.8b, v3.8b
            smull v8.8h, v0.8b, v4.8b
            smull2 v9.8h, v0.16b, v1.16b
            smull2 v10.8h, v0.16b, v2.16b
            smull2 v11.8h, v0.16b, v3.16b
            smull2 v12.8h, v0.16b, v4.16b

            saddw v16.4s, v16.4s, v5.4h
            saddw v18.4s, v18.4s, v9.4h
            saddw v20.4s, v20.4s, v6.4h
            saddw v22.4s, v22.4s, v10.4h
            saddw v24.4s, v24.4s, v7.4h
            saddw v26.4s, v26.4s, v11.4h
            saddw v28.4s, v28.4s, v8.4h
            saddw v30.4s, v30.4s, v12.4h

            saddw2 v17.4s, v17.4s, v5.8h
            saddw2 v19.4s, v19.4s, v9.8h
            saddw2 v21.4s, v21.4s, v6.8h
            saddw2 v23.4s, v23.4s, v10.8h
            saddw2 v25.4s, v25.4s, v7.8h
            saddw2 v27.4s, v27.4s, v11.8h
            saddw2 v29.4s, v29.4s, v8.8h
            saddw2 v31.4s, v31.4s, v12.8h
*/

            ldr s0, [x2], #(4 * sizeof_value)
            ldr q1, [x1]
            sxtl v0.8h, v0.8b
            ldrsw x21, [x26], #4
            sxtl v2.8h, v1.8b
            sxtl2 v3.8h, v1.16b
            subs w20, w20, #1
            add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(int8)

            smlal v16.4s, v2.4h, v0.h[0]
            smlal v18.4s, v3.4h, v0.h[0]
            smlal v20.4s, v2.4h, v0.h[1]
            smlal v22.4s, v3.4h, v0.h[1]

            smlal v24.4s, v2.4h, v0.h[2]
            smlal v26.4s, v3.4h, v0.h[2]
            smlal v28.4s, v2.4h, v0.h[3]
            smlal v30.4s, v3.4h, v0.h[3]

            smlal2 v17.4s, v2.8h, v0.h[0]
            smlal2 v19.4s, v3.8h, v0.h[0]
            smlal2 v21.4s, v2.8h, v0.h[1]
            smlal2 v23.4s, v3.8h, v0.h[1]

            smlal2 v25.4s, v2.8h, v0.h[2]
            smlal2 v27.4s, v3.8h, v0.h[2]
            smlal2 v29.4s, v2.8h, v0.h[3]
            smlal2 v31.4s, v3.8h, v0.h[3]

            bne loop_e16h4l1

        loop_e16h4l1_end:

        cbz x24, clamp_noscale_e16h4
         // deal with scale
          ldr q0, [x25], #(4 * sparse_blockoc)
          scvtf v16.4s, v16.4s
          scvtf v17.4s, v17.4s
          scvtf v18.4s, v18.4s
          scvtf v19.4s, v19.4s
          scvtf v20.4s, v20.4s
          scvtf v21.4s, v21.4s
          scvtf v22.4s, v22.4s
          scvtf v23.4s, v23.4s
          scvtf v24.4s, v24.4s
          scvtf v25.4s, v25.4s
          scvtf v26.4s, v26.4s
          scvtf v27.4s, v27.4s
          scvtf v28.4s, v28.4s
          scvtf v29.4s, v29.4s
          scvtf v30.4s, v30.4s
          scvtf v31.4s, v31.4s
          fmul v16.4s, v16.4s, v0.s[0]
          fmul v17.4s, v17.4s, v0.s[0]
          fmul v18.4s, v18.4s, v0.s[0]
          fmul v19.4s, v19.4s, v0.s[0]
          fmul v20.4s, v20.4s, v0.s[1]
          fmul v21.4s, v21.4s, v0.s[1]
          fmul v22.4s, v22.4s, v0.s[1]
          fmul v23.4s, v23.4s, v0.s[1]
          fmul v24.4s, v24.4s, v0.s[2]
          fmul v25.4s, v25.4s, v0.s[2]
          fmul v26.4s, v26.4s, v0.s[2]
          fmul v27.4s, v27.4s, v0.s[2]
          fmul v28.4s, v28.4s, v0.s[3]
          fmul v29.4s, v29.4s, v0.s[3]
          fmul v30.4s, v30.4s, v0.s[3]
          fmul v31.4s, v31.4s, v0.s[3]
          fcvtas v16.4s, v16.4s
          fcvtas v17.4s, v17.4s
          fcvtas v18.4s, v18.4s
          fcvtas v19.4s, v19.4s
          fcvtas v20.4s, v20.4s
          fcvtas v21.4s, v21.4s
          fcvtas v22.4s, v22.4s
          fcvtas v23.4s, v23.4s
          fcvtas v24.4s, v24.4s
          fcvtas v25.4s, v25.4s
          fcvtas v26.4s, v26.4s
          fcvtas v27.4s, v27.4s
          fcvtas v28.4s, v28.4s
          fcvtas v29.4s, v29.4s
          fcvtas v30.4s, v30.4s
          fcvtas v31.4s, v31.4s

        clamp_noscale_e16h4:

        smin v16.4s, v16.4s, v13.4s
        smin v17.4s, v17.4s, v13.4s
        smin v18.4s, v18.4s, v13.4s
        smin v19.4s, v19.4s, v13.4s
        smin v20.4s, v20.4s, v13.4s
        smin v21.4s, v21.4s, v13.4s
        smin v22.4s, v22.4s, v13.4s
        smin v23.4s, v23.4s, v13.4s
        smin v24.4s, v24.4s, v13.4s
        smin v25.4s, v25.4s, v13.4s
        smin v26.4s, v26.4s, v13.4s
        smin v27.4s, v27.4s, v13.4s
        smin v28.4s, v28.4s, v13.4s
        smin v29.4s, v29.4s, v13.4s
        smin v30.4s, v30.4s, v13.4s
        smin v31.4s, v31.4s, v13.4s
        add x5, x5, #sparse_blockoc
        smax v16.4s, v16.4s, v14.4s
        smax v17.4s, v17.4s, v14.4s
        smax v18.4s, v18.4s, v14.4s
        smax v19.4s, v19.4s, v14.4s
        smax v20.4s, v20.4s, v14.4s
        smax v21.4s, v21.4s, v14.4s
        smax v22.4s, v22.4s, v14.4s
        smax v23.4s, v23.4s, v14.4s
        smax v24.4s, v24.4s, v14.4s
        smax v25.4s, v25.4s, v14.4s
        smax v26.4s, v26.4s, v14.4s
        smax v27.4s, v27.4s, v14.4s
        smax v28.4s, v28.4s, v14.4s
        smax v29.4s, v29.4s, v14.4s
        smax v30.4s, v30.4s, v14.4s
        smax v31.4s, v31.4s, v14.4s

        sqxtn v0.4h, v16.4s
        sqxtn2 v0.8h, v17.4s
        sqxtn v1.4h, v18.4s
        sqxtn2 v1.8h, v19.4s
        sqxtn v2.4h, v20.4s
        sqxtn2 v2.8h, v21.4s
        sqxtn v3.4h, v22.4s
        sqxtn2 v3.8h, v23.4s
        sqxtn v4.4h, v24.4s
        sqxtn2 v4.8h, v25.4s
        sqxtn v5.4h, v26.4s
        sqxtn2 v5.8h, v27.4s
        sqxtn v6.4h, v28.4s
        sqxtn2 v6.8h, v29.4s
        sqxtn v7.4h, v30.4s
        sqxtn2 v7.8h, v31.4s

        sqxtn v16.8b, v0.8h
        sqxtn2 v16.16b, v1.8h
        sqxtn v17.8b, v2.8h
        sqxtn2 v17.16b, v3.8h
        sqxtn v18.8b, v4.8h
        sqxtn2 v18.16b, v5.8h
        sqxtn v19.8b, v6.8h
        sqxtn2 v19.16b, v7.8h

        cmp x5, x14
        st4 {v16.16b, v17.16b, v18.16b, v19.16b}, [x19] // if want to use 'stp', have to transpose v16-v19
        blt loop_e16h4

        cmp x5, x11
        bge loop_e16h_end

        lsr x19, x5, #2 // NC4HW4
        mul x19, x19, x12
        add x27, x27, x19 // x19: c = blockC + ihpack * cStride

    loop_e16h1:
        and x20, x5, #0x03 // NC4HW4
        add x19, x27, x20, lsl #sizeof_value_lg2 // x19: c = blockC + isubIndex

        cbz x6, load_e16h1_zero
            ld1r {v16.4s}, [x28], #(4)
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

          ldr q0, [x1]
          ld1r {v1.16b}, [x2], #(sizeof_value)
          ldrsw x21, [x26], #4
          subs w20, w20, #1
          add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)


            smull v5.8h, v0.8b, v1.8b
            smull2 v9.8h, v0.16b, v1.16b

            saddw v16.4s, v16.4s, v5.4h
            saddw v18.4s, v18.4s, v9.4h
            saddw2 v17.4s, v17.4s, v5.8h
            saddw2 v19.4s, v19.4s, v9.8h

          bne loop_e16h1l1

    loop_e16h1l1_end:

    cbz x24, clamp_noscale_e16h1
        // deal with scale
        ldr s0, [x25], #(4)
        scvtf v16.4s, v16.4s
        scvtf v17.4s, v17.4s
        scvtf v18.4s, v18.4s
        scvtf v19.4s, v19.4s
        fmul v16.4s, v16.4s, v0.s[0]
        fmul v17.4s, v17.4s, v0.s[0]
        fmul v18.4s, v18.4s, v0.s[0]
        fmul v19.4s, v19.4s, v0.s[0]
        fcvtas v16.4s, v16.4s
        fcvtas v17.4s, v17.4s
        fcvtas v18.4s, v18.4s
        fcvtas v19.4s, v19.4s

    clamp_noscale_e16h1:
    smin v16.4s, v16.4s, v13.4s
    smin v17.4s, v17.4s, v13.4s
    smin v18.4s, v18.4s, v13.4s
    smin v19.4s, v19.4s, v13.4s
    add x5, x5, #1
    smax v16.4s, v16.4s, v14.4s
    smax v17.4s, v17.4s, v14.4s
    smax v18.4s, v18.4s, v14.4s
    smax v19.4s, v19.4s, v14.4s

    sqxtn v0.4h, v16.4s
    sqxtn2 v0.8h, v17.4s
    sqxtn v1.4h, v18.4s
    sqxtn2 v1.8h, v19.4s

    sqxtn v16.8b, v0.8h
    sqxtn2 v16.16b, v1.8h

    mov x23, #(4 * 4 * sizeof_value)
    add x20, x19, #(4 * sizeof_value)
    add x21, x19, #(8 * sizeof_value)
    add x22, x20, #(8 * sizeof_value)
    cmp x5, x11

    st1 {v16.b}[0], [x19], x23 // st1 donot support immediate increasement other than sizeof stored element
    st1 {v16.b}[1], [x20], x23
    st1 {v16.b}[2], [x21], x23
    st1 {v16.b}[3], [x22], x23
    st1 {v16.b}[4], [x19], x23
    st1 {v16.b}[5], [x20], x23
    st1 {v16.b}[6], [x21], x23
    st1 {v16.b}[7], [x22], x23
    st1 {v16.b}[8], [x19], x23
    st1 {v16.b}[9], [x20], x23
    st1 {v16.b}[10], [x21], x23
    st1 {v16.b}[11], [x22], x23
    st1 {v16.b}[12], [x19]
    st1 {v16.b}[13], [x20]
    st1 {v16.b}[14], [x21]
    st1 {v16.b}[15], [x22]

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

    mov x26, x8
    ldrsw x27, [x26], #4
    add x1, x1, x27, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

    mov x2, x10
    mov x15, x7
    add x27, x0, x4, lsl #(sizeof_value_lg2 + 2) // float* blockC = C + (ie << 2);

    mov x5, xzr
    mov x28, x6 // bias
    mov x25, x24 // scale
    cbz x14, loop_e8h1

    loop_e8h4:

        lsr x20, x5, #2 // NC4HW4
        mul x20, x20, x12
        add x19, x27, x20 // x19: c = blockC + ihpack * cStride
        cbz x6, load_e8h4_zero
            ldr q16, [x28], #(4 * sparse_blockoc)
            b load_e8h4_end
        load_e8h4_zero:
            movi v16.4s, #0000000000000000

        load_e8h4_end:
        ldr w20, [x15], #4
        dup v20.4s, v16.s[1]
        dup v24.4s, v16.s[2]
        dup v28.4s, v16.s[3]
        dup v16.4s, v16.s[0]
        mov v25.16b, v24.16b
        mov v29.16b, v28.16b
        mov v21.16b, v20.16b
        mov v17.16b, v16.16b
        cbz w20, loop_e8h4l1_end

        loop_e8h4l1:

            ldr s0, [x2], #(4 * sizeof_value)
            ldr d1, [x1]
            sxtl v0.8h, v0.8b
            ldrsw x21, [x26], #4
            sxtl v2.8h, v1.8b
            subs w20, w20, #1
            add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

            smlal v16.4s, v2.4h, v0.h[0]
            smlal v20.4s, v2.4h, v0.h[1]
            smlal v24.4s, v2.4h, v0.h[2]
            smlal v28.4s, v2.4h, v0.h[3]
            smlal2 v17.4s, v2.8h, v0.h[0]
            smlal2 v21.4s, v2.8h, v0.h[1]
            smlal2 v25.4s, v2.8h, v0.h[2]
            smlal2 v29.4s, v2.8h, v0.h[3]

            bne loop_e8h4l1

        loop_e8h4l1_end:
        cbz x24, clamp_noscale_e8h4
         // deal with scale
          ldr q0, [x25], #(4 * sparse_blockoc)
          scvtf v16.4s, v16.4s
          scvtf v17.4s, v17.4s
          scvtf v20.4s, v20.4s
          scvtf v21.4s, v21.4s
          scvtf v24.4s, v24.4s
          scvtf v25.4s, v25.4s
          scvtf v28.4s, v28.4s
          scvtf v29.4s, v29.4s
          fmul v16.4s, v16.4s, v0.s[0]
          fmul v17.4s, v17.4s, v0.s[0]
          fmul v20.4s, v20.4s, v0.s[1]
          fmul v21.4s, v21.4s, v0.s[1]
          fmul v24.4s, v24.4s, v0.s[2]
          fmul v25.4s, v25.4s, v0.s[2]
          fmul v28.4s, v28.4s, v0.s[3]
          fmul v29.4s, v29.4s, v0.s[3]
          fcvtas v16.4s, v16.4s
          fcvtas v17.4s, v17.4s
          fcvtas v20.4s, v20.4s
          fcvtas v21.4s, v21.4s
          fcvtas v24.4s, v24.4s
          fcvtas v25.4s, v25.4s
          fcvtas v28.4s, v28.4s
          fcvtas v29.4s, v29.4s

        clamp_noscale_e8h4:

        smin v16.4s, v16.4s, v13.4s
        smin v17.4s, v17.4s, v13.4s
        smin v20.4s, v20.4s, v13.4s
        smin v21.4s, v21.4s, v13.4s
        smin v24.4s, v24.4s, v13.4s
        smin v25.4s, v25.4s, v13.4s
        smin v28.4s, v28.4s, v13.4s
        smin v29.4s, v29.4s, v13.4s
        add x5, x5, #sparse_blockoc
        smax v16.4s, v16.4s, v14.4s
        smax v17.4s, v17.4s, v14.4s
        smax v20.4s, v20.4s, v14.4s
        smax v21.4s, v21.4s, v14.4s
        smax v24.4s, v24.4s, v14.4s
        smax v25.4s, v25.4s, v14.4s
        smax v28.4s, v28.4s, v14.4s
        smax v29.4s, v29.4s, v14.4s

        sqxtn v0.4h, v16.4s
        sqxtn2 v0.8h, v17.4s
        sqxtn v2.4h, v20.4s
        sqxtn2 v2.8h, v21.4s
        sqxtn v4.4h, v24.4s
        sqxtn2 v4.8h, v25.4s
        sqxtn v6.4h, v28.4s
        sqxtn2 v6.8h, v29.4s

        sqxtn v16.8b, v0.8h
        sqxtn v17.8b, v2.8h
        sqxtn v18.8b, v4.8h
        sqxtn v19.8b, v6.8h

        cmp x5, x14
        st4 {v16.8b, v17.8b, v18.8b, v19.8b}, [x19] // if want to use 'stp', have to transpose v16-v19
        blt loop_e8h4

        cmp x5, x11
        bge loop_e8h_end

        lsr x19, x5, #2 // NC4HW4
        mul x19, x19, x12
        add x27, x27, x19 // x19: c = blockC + ihpack * cStride

    loop_e8h1:
        and x20, x5, #0x03 // NC4HW4
        add x19, x27, x20, lsl #sizeof_value_lg2 // x19: c = blockC + isubIndex

        cbz x6, load_e8h1_zero
            ld1r {v16.4s}, [x28], #(4)
            b load_e8h1_end
        load_e8h1_zero:
            movi v16.4s, #0000000000000000

        load_e8h1_end:
        ldr w20, [x15], #4
        mov v17.16b, v16.16b
        cbz w20, loop_e8h1l1_end

        loop_e8h1l1:
          ldr d0, [x1]
          ld1r {v1.8b}, [x2], #(sizeof_value)
          ldrsw x21, [x26], #4
          subs w20, w20, #1
          add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)
          smull v5.8h, v0.8b, v1.8b
          saddw v16.4s, v16.4s, v5.4h
          saddw2 v17.4s, v17.4s, v5.8h
          bne loop_e8h1l1

    loop_e8h1l1_end:
    cbz x24, clamp_noscale_e8h1
        // deal with scale
        ldr s0, [x25], #(4)
        scvtf v16.4s, v16.4s
        scvtf v17.4s, v17.4s
        fmul v16.4s, v16.4s, v0.s[0]
        fmul v17.4s, v17.4s, v0.s[0]
        fcvtas v16.4s, v16.4s
        fcvtas v17.4s, v17.4s
    clamp_noscale_e8h1:
    smin v16.4s, v16.4s, v13.4s
    smin v17.4s, v17.4s, v13.4s
    add x5, x5, #1
    smax v16.4s, v16.4s, v14.4s
    smax v17.4s, v17.4s, v14.4s

    sqxtn v0.4h, v16.4s
    sqxtn2 v0.8h, v17.4s
    sqxtn v16.8b, v0.8h

    mov x23, #(4 * 4 * sizeof_value)
    add x20, x19, #(4 * sizeof_value)
    add x21, x19, #(8 * sizeof_value)
    add x22, x20, #(8 * sizeof_value)

    cmp x5, x11
    st1 {v16.b}[0], [x19], X23 // st1 donot support immediate increasement other than sizeof stored element
    st1 {v16.b}[1], [x20], X23
    st1 {v16.b}[2], [x21], X23
    st1 {v16.b}[3], [x22], X23
    st1 {v16.b}[4], [x19]
    st1 {v16.b}[5], [x20]
    st1 {v16.b}[6], [x21]
    st1 {v16.b}[7], [x22]
    blt loop_e8h1

    loop_e8h_end:

    add x4, x4, #8 // e8
    add x1, x1, #(8 * sizeof_value) // Has not exceed one aStride, just 8


loop_e4:
ands x5, x3, #0x04
beq loop_e2

    mov x26, x8
    ldrsw x27, [x26], #4
    add x1, x1, x27, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

    mov x2, x10
    mov x15, x7
    add x27, x0, x4, lsl #(sizeof_value_lg2 + 2) // float* blockC = C + (ie << 2);
    mov x5, xzr
    mov x28, x6 // bias
    mov x25, x24 // scale
    cbz x14, loop_e4h1

    loop_e4h4:

        lsr x20, x5, #2 // NC4HW4
        mul x20, x20, x12
        add x19, x27, x20 // x19: c = blockC + ihpack * cStride
        cbz x6, load_e4h4_zero
            ldr q16, [x28], #(4 * sparse_blockoc)
            b load_e4h4_end
        load_e4h4_zero:
            movi v16.4s, #0000000000000000

        load_e4h4_end:
        ldr w20, [x15], #4
        dup v20.4s, v16.s[1]
        dup v24.4s, v16.s[2]
        dup v28.4s, v16.s[3]
        dup v16.4s, v16.s[0]
        cbz w20, loop_e4h4l1_end

        loop_e4h4l1:

            ldr s0, [x2], #(4 * sizeof_value)
            ldr s1, [x1]
            sxtl v0.8h, v0.8b
            ldrsw x21, [x26], #4
            sxtl v2.8h, v1.8b
            subs w20, w20, #1
            add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

            smlal v16.4s, v2.4h, v0.h[0]
            smlal v20.4s, v2.4h, v0.h[1]
            smlal v24.4s, v2.4h, v0.h[2]
            smlal v28.4s, v2.4h, v0.h[3]

            bne loop_e4h4l1

        loop_e4h4l1_end:
        cbz x24, clamp_noscale_e4h4
         // deal with scale
          ldr q0, [x25], #(4 * sparse_blockoc)
          scvtf v16.4s, v16.4s
          scvtf v20.4s, v20.4s
          scvtf v24.4s, v24.4s
          scvtf v28.4s, v28.4s
          fmul v16.4s, v16.4s, v0.s[0]
          fmul v20.4s, v20.4s, v0.s[1]
          fmul v24.4s, v24.4s, v0.s[2]
          fmul v28.4s, v28.4s, v0.s[3]
          fcvtas v16.4s, v16.4s
          fcvtas v20.4s, v20.4s
          fcvtas v24.4s, v24.4s
          fcvtas v28.4s, v28.4s

        clamp_noscale_e4h4:

        smin v16.4s, v16.4s, v13.4s
        smin v20.4s, v20.4s, v13.4s
        smin v24.4s, v24.4s, v13.4s
        smin v28.4s, v28.4s, v13.4s
        add x5, x5, #sparse_blockoc
        smax v16.4s, v16.4s, v14.4s
        smax v20.4s, v20.4s, v14.4s
        smax v24.4s, v24.4s, v14.4s
        smax v28.4s, v28.4s, v14.4s

        sqxtn v0.4h, v16.4s
        sqxtn v2.4h, v20.4s
        sqxtn v4.4h, v24.4s
        sqxtn v6.4h, v28.4s

        sqxtn v16.8b, v0.8h // only 4b is valid
        sqxtn v17.8b, v2.8h
        sqxtn v18.8b, v4.8h
        sqxtn v19.8b, v6.8h

        cmp x5, x14
        st4 {v16.b, v17.b, v18.b, v19.b}[0], [x19], #(sizeof_value * sparse_blockoc)
        st4 {v16.b, v17.b, v18.b, v19.b}[1], [x19], #(sizeof_value * sparse_blockoc)
        st4 {v16.b, v17.b, v18.b, v19.b}[2], [x19], #(sizeof_value * sparse_blockoc)
        st4 {v16.b, v17.b, v18.b, v19.b}[3], [x19]
        blt loop_e4h4

        cmp x5, x11
        bge loop_e4h_end

        lsr x19, x5, #2 // NC4HW4
        mul x19, x19, x12
        add x27, x27, x19 // x19: c = blockC + ihpack * cStride

    loop_e4h1:
        and x20, x5, #0x03 // NC4HW4
        add x19, x27, x20, lsl #sizeof_value_lg2 // x20: c = blockC + isubIndex

        cbz x6, load_e4h1_zero
            ld1r {v16.4s}, [x28], #(4)
            b load_e4h1_end
        load_e4h1_zero:
            movi v16.4s, #0000000000000000

        load_e4h1_end:
        ldr w20, [x15], #4
        cbz w20, loop_e4h1l1_end

        loop_e4h1l1:

          ldr s0, [x1]
          ld1r {v1.8b}, [x2], #(sizeof_value) // try 4b
          ldrsw x21, [x26], #4
          subs w20, w20, #1
          add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

          smull v5.8h, v0.8b, v1.8b
          saddw v16.4s, v16.4s, v5.4h
          bne loop_e4h1l1

    loop_e4h1l1_end:
    cbz x24, clamp_noscale_e4h1
        // deal with scale
        ldr s0, [x25], #(4)
        scvtf v16.4s, v16.4s
        fmul v16.4s, v16.4s, v0.s[0]
        fcvtas v16.4s, v16.4s
    clamp_noscale_e4h1:
    smin v16.4s, v16.4s, v13.4s
    add x5, x5, #1
    smax v16.4s, v16.4s, v14.4s

    sqxtn v0.4h, v16.4s
    sqxtn v16.8b, v0.8h // 4b is valid

    add x20, x19, #(4 * sizeof_value)
    add x21, x19, #(8 * sizeof_value)
    add x22, x20, #(8 * sizeof_value)

    cmp x5, x11
    st1 {v16.b}[0], [x19] // st1 donot support immediate increasement other than sizeof stored element
    st1 {v16.b}[1], [x20]
    st1 {v16.b}[2], [x21]
    st1 {v16.b}[3], [x22]
    blt loop_e4h1

    loop_e4h_end:

    add x4, x4, #4 // e4
    add x1, x1, #(4 * sizeof_value) // Has not exceed one aStride, just 4


loop_e2:
ands x5, x3, #0x02
beq loop_e1

    mov x26, x8
    ldrsw x27, [x26], #4
    add x1, x1, x27, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

    mov x2, x10
    mov x15, x7
    add x27, x0, x4, lsl #(sizeof_value_lg2 + 2) // float* blockC = C + (ie << 2);
    mov x5, xzr
    mov x28, x6 // bias
    mov x25, x24 // scale
    cbz x14, loop_e2h1

    loop_e2h4:
        lsr x20, x5, #2 // NC4HW4
        mul x20, x20, x12
        add x19, x27, x20 // x19: c = blockC + ihpack * cStride
        cbz x6, load_e2h4_zero
            ldr q16, [x28], #(4 * sparse_blockoc)
            b load_e2h4_end
        load_e2h4_zero:
            movi v16.4s, #0000000000000000

        load_e2h4_end:
        ldr w20, [x15], #4
        dup v20.2s, v16.s[1]
        dup v24.2s, v16.s[2]
        dup v28.2s, v16.s[3]
        dup v16.2s, v16.s[0]
        cbz w20, loop_e2h4l1_end

        loop_e2h4l1:

            ldr s0, [x2], #(4 * sizeof_value)
            ld1 {v1.h}[0], [x1]
            sxtl v0.8h, v0.8b
            ldrsw x21, [x26], #4
            sxtl v2.8h, v1.8b
            subs w20, w20, #1
            add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

            smlal v16.4s, v2.4h, v0.h[0] // only 2s valid
            smlal v20.4s, v2.4h, v0.h[1]
            smlal v24.4s, v2.4h, v0.h[2]
            smlal v28.4s, v2.4h, v0.h[3]

            bne loop_e2h4l1

        loop_e2h4l1_end:
        cbz x24, clamp_noscale_e2h4
         // deal with scale
          ldr q0, [x25], #(4 * sparse_blockoc)
          scvtf v16.2s, v16.2s
          scvtf v20.2s, v20.2s
          scvtf v24.2s, v24.2s
          scvtf v28.2s, v28.2s
          fmul v16.2s, v16.2s, v0.s[0]
          fmul v20.2s, v20.2s, v0.s[1]
          fmul v24.2s, v24.2s, v0.s[2]
          fmul v28.2s, v28.2s, v0.s[3]
          fcvtas v16.2s, v16.2s
          fcvtas v20.2s, v20.2s
          fcvtas v24.2s, v24.2s
          fcvtas v28.2s, v28.2s

        clamp_noscale_e2h4:

        smin v16.2s, v16.2s, v13.2s
        smin v20.2s, v20.2s, v13.2s
        smin v24.2s, v24.2s, v13.2s
        smin v28.2s, v28.2s, v13.2s
        add x5, x5, #sparse_blockoc
        smax v16.2s, v16.2s, v14.2s
        smax v20.2s, v20.2s, v14.2s
        smax v24.2s, v24.2s, v14.2s
        smax v28.2s, v28.2s, v14.2s

        sqxtn v0.4h, v16.4s // only 2s -> 2h is valid
        sqxtn v2.4h, v20.4s
        sqxtn v4.4h, v24.4s
        sqxtn v6.4h, v28.4s

        sqxtn v16.8b, v0.8h // only 2h -> 2b is valid
        sqxtn v17.8b, v2.8h
        sqxtn v18.8b, v4.8h
        sqxtn v19.8b, v6.8h

        cmp x5, x14
        st4 {v16.b, v17.b, v18.b, v19.b}[0], [x19], #(sizeof_value * sparse_blockoc)
        st4 {v16.b, v17.b, v18.b, v19.b}[1], [x19]
        blt loop_e2h4

        cmp x5, x11
        bge loop_e2h_end

        lsr x19, x5, #2 // NC4HW4
        mul x19, x19, x12
        add x27, x27, x19 // x19: c = blockC + ihpack * cStride

    loop_e2h1:
        and x20, x5, #0x03 // NC4HW4
        add x19, x27, x20, lsl #sizeof_value_lg2 // x20: c = blockC + isubIndex

        cbz x6, load_e2h1_zero
            ld1r {v16.2s}, [x28], #(4)
            b load_e2h1_end
        load_e2h1_zero:
            movi v16.4s, #0000000000000000
        load_e2h1_end:
        ldr w20, [x15], #4
        cbz w20, loop_e2h1l1_end
        loop_e2h1l1:

          ld1 {v0.h}[0], [x1]
          ld1r {v1.8b}, [x2], #(sizeof_value) // try 2b
          ldrsw x21, [x26], #4
          subs w20, w20, #1
          add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)
          smull v5.8h, v0.8b, v1.8b // only 2b valid
          saddw v16.4s, v16.4s, v5.4h
          bne loop_e2h1l1

    loop_e2h1l1_end:

        cbz x24, clamp_noscale_e2h1
        // deal with scale
        ldr s0, [x25], #(4)
        scvtf v16.2s, v16.2s
        fmul v16.2s, v16.2s, v0.s[0]
        fcvtas v16.2s, v16.2s
    clamp_noscale_e2h1:
    smin v16.2s, v16.2s, v13.2s
    add x5, x5, #1
    smax v16.2s, v16.2s, v14.2s
    add x20, x19, #(4 * sizeof_value)
    sqxtn v0.4h, v16.4s
    sqxtn v16.8b, v0.8h // 2h -> 2b is valid
    cmp x5, x11
    st1 {v16.b}[0], [x19] // st1 donot support immediate increasement other than sizeof stored element
    st1 {v16.b}[1], [x20]
    blt loop_e2h1

    loop_e2h_end:
    add x4, x4, #2 // e2
    add x1, x1, #(2 * sizeof_value) // Has not exceed one aStride, just 2


loop_e1:
ands x5, x3, #0x01
beq loop_end

    mov x26, x8
    ldrsw x27, [x26], #4
    add x1, x1, x27, lsl #sizeof_value_lg2 // a += diff * sizeof(float)

    mov x2, x10
    mov x15, x7
    add x27, x0, x4, lsl #(sizeof_value_lg2 + 2) // float* blockC = C + (ie << 2);

    mov x5, xzr
    mov x28, x6 // bias
    mov x25, x24 // scale
    cbz x14, loop_e1h1

    loop_e1h4:
        lsr x20, x5, #2 // NC4HW4
        mul x20, x20, x12
        add x19, x27, x20 // x19: c = blockC + ihpack * cStride
        cbz x6, load_e1h4_zero
            ldr q16, [x28], #(4 * sparse_blockoc)
            b load_e1h4_end
        load_e1h4_zero:
            movi v16.4s, #0000000000000000
        load_e1h4_end:
        ldr w20, [x15], #4
        cbz w20, loop_e1h4l1_end

        loop_e1h4l1:

            ld1r {v0.8b}, [x1] // only 1b valid
            ldr s1, [x2], #(4 * sizeof_value)
            ldrsw x21, [x26], #4
            subs w20, w20, #1
            add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)
            smull v5.8h, v0.8b, v1.8b // only 4h valid
            saddw v16.4s, v16.4s, v5.4h // 4s is valid
            bne loop_e1h4l1

        loop_e1h4l1_end:

        cbz x24, clamp_noscale_e1h4
         // deal with scale
          ldr q0, [x25], #(4 * sparse_blockoc)
          scvtf v16.4s, v16.4s
          fmul v16.4s, v16.4s, v0.4s
          fcvtas v16.4s, v16.4s
        clamp_noscale_e1h4:

        smin v16.4s, v16.4s, v13.4s
        add x5, x5, #sparse_blockoc
        smax v16.4s, v16.4s, v14.4s

        sqxtn v0.4h, v16.4s
        sqxtn v16.8b, v0.8h // 4b is valid

        cmp x5, x14
        str s16, [x19]
        blt loop_e1h4

        cmp x5, x11
        bge loop_e1h_end

        lsr x19, x5, #2 // NC4HW4
        mul x19, x19, x12
        add x27, x27, x19 // x19: c = blockC + ihpack * cStride

    loop_e1h1:
        and x20, x5, #0x03 // NC4HW4
        add x19, x27, x20, lsl #sizeof_value_lg2 // x20: c = blockC + isubIndex

        cbz x6, load_e1h1_zero
            ld1 {v16.s}[0], [x28], #(4)
            b load_e1h1_end
        load_e1h1_zero:
            movi v16.4s, #0000000000000000

        load_e1h1_end:
        ldr w20, [x15], #4

        cbz w20, loop_e1h1l1_end

        loop_e1h1l1:

          ld1 {v0.b}[0], [x1]
          ld1 {v1.b}[0], [x2], #(sizeof_value)
          ldrsw x21, [x26], #4
          subs w20, w20, #1
          add x1, x1, x21, lsl #sizeof_value_lg2 // a += diff * sizeof(float)
          smull v5.8h, v0.8b, v1.8b // only 1h valid
          saddw v16.4s, v16.4s, v5.4h // only 1s is valid
          bne loop_e1h1l1

    loop_e1h1l1_end:

    cbz x24, clamp_noscale_e1h1
     // deal with scale
      ldr s0, [x25], #(4)
      scvtf s16, s16
      fmul s16, s16, v0.s[0]
      fcvtas s16, s16
    clamp_noscale_e1h1:

    smin v16.2s, v16.2s, v13.2s
    add x5, x5, #1
    smax v16.2s, v16.2s, v14.2s
    sqxtn v0.4h, v16.4s
    sqxtn v16.8b, v0.8h // 1b is valid
    cmp x5, x11
    st1 {v16.b}[0], [x19]
    blt loop_e1h1

    loop_e1h_end:
    add x4, x4, #1 // e1

loop_end:

ldp x19, x20, [sp, #(16 * 8)]
ldp x21, x22, [sp, #(16 * 7)]
ldp x23, x24, [sp, #(16 * 6)]
ldp x25, x26, [sp, #(16 * 5)]
ldp x27, x28, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldr d14, [sp], #(16 * 9)

ret

#undef sizeof_value
#undef sizeof_value_lg2
#undef sparse_blockoc

#endif


