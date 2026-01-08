//
//  MNNTMacCompute.S
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5
/*
struct PlaneInfo {
    int planeSize;
    float offset;
    float dequantscale;
    float minValue;
    float maxValue;
    int ocDivSize;
    const uint8_t* mWeightPtr;
    const uint8_t* mWeightScalePtr;
    const uint8_t* mBiasPtr;
};
struct TMacResource {
    int mBits;
    int mBlockSizeC4;
    int mBlockNumber;
    int mOutputCount;
    int mHp;
};
*/
asm_function MNNTMacCompute
//void MNNTMacCompute(float* dst, const int8_t* table, const float* inputSum, const TMacResource* res, const PlaneInfo* plane) {
//Auto load:
//x0:dst, x1:table, x2:inputSum, x3:resource, x4: plane

stp d14, d15, [sp, #-128]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]
stp x19, x20, [sp, #64]

// x5: weight, x6: scale, x7: bias, x8: bits
// x9: blc4, x10: blocknumber, x3: oc4
ldr w8, [x3, #0]
ldr w9, [x3, #4]
ldr w10, [x3, #8]

// v28: offset, v29: dequantscale, v30: min, v31: max
ldr w11, [x4, #4]
ldr w12, [x4, #8]
dup v28.4s, w11
dup v29.4s, w12
ldr w11, [x4, #12]
ldr w12, [x4, #16]
dup v30.4s, w11
dup v31.4s, w12
ldr w3, [x4, #20]
ldr x5, [x4, #24]
ldr x6, [x4, #32]
ldr x7, [x4, #40]
ldr w4, [x4]
uxtw x4, w4
// x4: dstStride (planesize * 4 * sizeof(float))
lsl x4, x4, #4
movi v7.16b, #15

LoopDz:
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x7], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x7], #64
    mov x12, x1
    mov x13, x2
    mov w11, w10
    LoopBlock:
        mov w15, w8
        FirstBits:
            mov x19, x12
            mov w14, w9
            ld1 {v1.16b}, [x19], #16
            ld1 {v2.16b}, [x5], #16
            ushr v3.16b, v2.16b, #4
            and v2.16b, v2.16b, v7.16b
            tbl v2.16b, {v1.16b}, v2.16b
            tbl v3.16b, {v1.16b}, v3.16b
            sxtl v24.8h, v2.8b
            sxtl2 v25.8h, v2.16b
            sxtl v26.8h, v3.8b
            sxtl2 v27.8h, v3.16b
            subs w14, w14, #1
            beq LoopInBlockFirstEnd

            LoopInBlockFirst:
                ld1 {v1.16b}, [x19], #16
                ld1 {v2.16b}, [x5], #16
                ushr v3.16b, v2.16b, #4
                and v2.16b, v2.16b, v7.16b
                tbl v2.16b, {v1.16b}, v2.16b
                tbl v3.16b, {v1.16b}, v3.16b
                subs w14, w14, #1
                saddw v24.8h, v24.8h, v2.8b
                saddw2 v25.8h, v25.8h, v2.16b
                saddw v26.8h, v26.8h, v3.8b
                saddw2 v27.8h, v27.8h, v3.16b

                bne LoopInBlockFirst
            LoopInBlockFirstEnd:
            sxtl v16.4s, v24.4h
            sxtl2 v17.4s, v24.8h
            sxtl v18.4s, v25.4h
            sxtl2 v19.4s, v25.8h
            subs w15, w15, #1
            sxtl v20.4s, v26.4h
            sxtl2 v21.4s, v26.8h
            sxtl v22.4s, v27.4h
            sxtl2 v23.4s, v27.8h
            beq EndBit        
        LoopBits:
            sqshl v16.4s, v16.4s, #1
            sqshl v17.4s, v17.4s, #1
            sqshl v18.4s, v18.4s, #1
            sqshl v19.4s, v19.4s, #1
            sqshl v20.4s, v20.4s, #1
            sqshl v21.4s, v21.4s, #1
            sqshl v22.4s, v22.4s, #1
            sqshl v23.4s, v23.4s, #1

            mov x19, x12
            mov w14, w9

            ld1 {v1.16b}, [x19], #16
            ld1 {v2.16b}, [x5], #16
            ushr v3.16b, v2.16b, #4
            and v2.16b, v2.16b, v7.16b
            tbl v2.16b, {v1.16b}, v2.16b
            tbl v3.16b, {v1.16b}, v3.16b
            subs w14, w14, #1
            sxtl v24.8h, v2.8b
            sxtl2 v25.8h, v2.16b
            sxtl v26.8h, v3.8b
            sxtl2 v27.8h, v3.16b

            beq LoopInBlockEnd
            LoopInBlock:
                ld1 {v1.16b}, [x19], #16
                ld1 {v2.16b}, [x5], #16
                ushr v3.16b, v2.16b, #4
                and v2.16b, v2.16b, v7.16b
                tbl v2.16b, {v1.16b}, v2.16b
                tbl v3.16b, {v1.16b}, v3.16b
                subs w14, w14, #1
                saddw v24.8h, v24.8h, v2.8b
                saddw2 v25.8h, v25.8h, v2.16b
                saddw v26.8h, v26.8h, v3.8b
                saddw2 v27.8h, v27.8h, v3.16b

                bne LoopInBlock
            LoopInBlockEnd:
            saddw v16.4s, v16.4s, v24.4h
            saddw2 v17.4s, v17.4s, v24.8h
            saddw v18.4s, v18.4s, v25.4h
            saddw2 v19.4s, v19.4s, v25.8h
            saddw v20.4s, v20.4s, v26.4h
            saddw2 v21.4s, v21.4s, v26.8h
            saddw v22.4s, v22.4s, v27.4h
            saddw2 v23.4s, v23.4s, v27.8h

            subs w15, w15, #1
            bne LoopBits
        EndBit:
        mov x12, x19 // Revert table

        // Block Scale
        ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x6], #64

        scvtf v16.4s, v16.4s
        scvtf v17.4s, v17.4s
        scvtf v18.4s, v18.4s
        scvtf v19.4s, v19.4s

        scvtf v20.4s, v20.4s
        scvtf v21.4s, v21.4s
        scvtf v22.4s, v22.4s
        scvtf v23.4s, v23.4s

        fsub v16.4s, v16.4s, v28.4s
        fsub v17.4s, v17.4s, v28.4s
        fsub v18.4s, v18.4s, v28.4s
        fsub v19.4s, v19.4s, v28.4s

        fmul v16.4s, v16.4s, v24.4s
        fmul v17.4s, v17.4s, v25.4s
        fmul v18.4s, v18.4s, v26.4s
        fmul v19.4s, v19.4s, v27.4s

        fmul v16.4s, v16.4s, v29.4s
        fmul v17.4s, v17.4s, v29.4s
        fmul v18.4s, v18.4s, v29.4s
        fmul v19.4s, v19.4s, v29.4s

        ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x6], #64

        fsub v20.4s, v20.4s, v28.4s
        fsub v21.4s, v21.4s, v28.4s
        fsub v22.4s, v22.4s, v28.4s
        fsub v23.4s, v23.4s, v28.4s

        fmul v20.4s, v20.4s, v29.4s
        fmul v21.4s, v21.4s, v29.4s
        fmul v22.4s, v22.4s, v29.4s
        fmul v23.4s, v23.4s, v29.4s

        ld1 {v0.s}[0], [x13], #4

        fmul v20.4s, v20.4s, v24.4s
        fmul v21.4s, v21.4s, v25.4s
        fmul v22.4s, v22.4s, v26.4s
        fmul v23.4s, v23.4s, v27.4s


        // Block Bias
        ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x6], #64
        fmla v16.4s, v24.4s, v0.s[0]
        fmla v17.4s, v25.4s, v0.s[0]
        fmla v18.4s, v26.4s, v0.s[0]
        fmla v19.4s, v27.4s, v0.s[0]

        ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x6], #64

        fadd v8.4s, v8.4s, v16.4s
        fadd v9.4s, v9.4s, v17.4s
        fadd v10.4s, v10.4s, v18.4s
        fadd v11.4s, v11.4s, v19.4s

        fmla v20.4s, v24.4s, v0.s[0]
        fmla v21.4s, v25.4s, v0.s[0]
        fmla v22.4s, v26.4s, v0.s[0]
        fmla v23.4s, v27.4s, v0.s[0]

        fadd v12.4s, v12.4s, v20.4s
        fadd v13.4s, v13.4s, v21.4s
        fadd v14.4s, v14.4s, v22.4s
        fadd v15.4s, v15.4s, v23.4s

        subs w11, w11, #1
        bne LoopBlock
    fmax v8.4s, v8.4s, v30.4s
    fmax v9.4s, v9.4s, v30.4s
    fmax v10.4s, v10.4s, v30.4s
    fmax v11.4s, v11.4s, v30.4s

    fmin v8.4s, v8.4s, v31.4s
    fmin v9.4s, v9.4s, v31.4s
    fmin v10.4s, v10.4s, v31.4s
    fmin v11.4s, v11.4s, v31.4s

    fmax v12.4s, v12.4s, v30.4s
    fmax v13.4s, v13.4s, v30.4s
    fmax v14.4s, v14.4s, v30.4s
    fmax v15.4s, v15.4s, v30.4s

    fmin v12.4s, v12.4s, v31.4s
    fmin v13.4s, v13.4s, v31.4s
    fmin v14.4s, v14.4s, v31.4s
    fmin v15.4s, v15.4s, v31.4s


    st1 {v8.4s}, [x0], x4
    st1 {v9.4s}, [x0], x4
    st1 {v10.4s}, [x0], x4
    st1 {v11.4s}, [x0], x4

    st1 {v12.4s}, [x0], x4
    st1 {v13.4s}, [x0], x4
    st1 {v14.4s}, [x0], x4
    st1 {v15.4s}, [x0], x4


    subs w3, w3, #1
    bne LoopDz


ldp x19, x20, [sp, #64]
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #128


ret


#endif

