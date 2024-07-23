//
//  MNNPackedSparseQuantMatMulEpx1.S
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
asm_function MNNPackedSparseQuantMatMulEpx1
// void MNNPackedSparseQuantMatMulEpx1(int8_t* C, const int8_t* A, const int8_t* B, const size_t* sparseQuantParam,
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
ldr x24, [x4, #0]
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
    loop_e16h1:

        lsr x21, x5, #2
        and x20, x5, #0x03 // NC4HW4
        mul x21, x21, x12
        add x19, x27, x20, lsl #sizeof_value_lg2
        add x19, x19, x21
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

    loop_e8h1:
        lsr x21, x5, #2
        and x20, x5, #0x03 // NC4HW4
        mul x21, x21, x12
        add x19, x27, x20, lsl #sizeof_value_lg2
        add x19, x19, x21

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

    loop_e4h1:
        lsr x21, x5, #2
        and x20, x5, #0x03 // NC4HW4
        mul x21, x21, x12
        add x19, x27, x20, lsl #sizeof_value_lg2
        add x19, x19, x21

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

    loop_e2h1:
        lsr x21, x5, #2
        and x20, x5, #0x03 // NC4HW4
        mul x21, x21, x12
        add x19, x27, x20, lsl #sizeof_value_lg2
        add x19, x19, x21

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
    loop_e1h1:
        lsr x21, x5, #2
        and x20, x5, #0x03 // NC4HW4
        mul x21, x21, x12
        add x19, x27, x20, lsl #sizeof_value_lg2
        add x19, x19, x21

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
