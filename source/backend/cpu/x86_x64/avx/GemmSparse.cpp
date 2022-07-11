//
//  GemmCommon.cpp
//  MNN
//
//  Created by MNN on 2021/07/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//


#include "GemmCommon.hpp"
#include "FunctionSummary.hpp"
#include "Vec8.hpp"
#include "core/Macro.h"

#ifdef MNN_X86_USE_ASM
extern "C" {
void _AVX_MNNPackedSparseMatMulEpx4EFMA_ASM(SparseMatMulParas* temp, const float* bias, const size_t* parameter, const float* postParameters);
void _AVX_MNNPackedSparseMatMulEpx1EFMA_ASM(SparseMatMulParas* temp, const float* bias, const size_t* parameter, const float* postParameters);
}
#endif

void _AVX_MNNGetSparseMatMulPackMode(int* eP, int *lP, int* hP){
    *eP = 24;
    *lP = 1;
    *hP = 4;
    // hp is corresponding to sparse block along right matrix colum dimension. in ramdom sparse, it is 1.
    return;
}

#define EMULATED_AVX2_FMA(dst, src0, src1) dst = _mm256_add_ps(dst, _mm256_mul_ps(src0, src1));

#define MIN_MAX_VEC(cVec)               \
    cVec = _mm256_max_ps(cVec, minVec); \
    cVec = _mm256_min_ps(cVec, maxVec);

#define ONE_H_STORE_E24(cTilePtr)   \
    cTilePtr[8 * 0] = c0VecPtr[0];  \
    cTilePtr[8 * 1] = c0VecPtr[1];  \
    cTilePtr[8 * 2] = c0VecPtr[2];  \
    cTilePtr[8 * 3] = c0VecPtr[3];  \
    cTilePtr[8 * 4] = c0VecPtr[4];  \
    cTilePtr[8 * 5] = c0VecPtr[5];  \
    cTilePtr[8 * 6] = c0VecPtr[6];  \
    cTilePtr[8 * 7] = c0VecPtr[7];  \
                                    \
    cTilePtr[8 * 8]  = c1VecPtr[0]; \
    cTilePtr[8 * 9]  = c1VecPtr[1]; \
    cTilePtr[8 * 10] = c1VecPtr[2]; \
    cTilePtr[8 * 11] = c1VecPtr[3]; \
    cTilePtr[8 * 12] = c1VecPtr[4]; \
    cTilePtr[8 * 13] = c1VecPtr[5]; \
    cTilePtr[8 * 14] = c1VecPtr[6]; \
    cTilePtr[8 * 15] = c1VecPtr[7]; \
                                    \
    cTilePtr[8 * 16] = c2VecPtr[0]; \
    cTilePtr[8 * 17] = c2VecPtr[1]; \
    cTilePtr[8 * 18] = c2VecPtr[2]; \
    cTilePtr[8 * 19] = c2VecPtr[3]; \
    cTilePtr[8 * 20] = c2VecPtr[4]; \
    cTilePtr[8 * 21] = c2VecPtr[5]; \
    cTilePtr[8 * 22] = c2VecPtr[6]; \
    cTilePtr[8 * 23] = c2VecPtr[7];

#define TRANSPOSE_4x4_WITH_STORE(rowIdx, offset, cVec0, cVec1, cVec2, cVec3, cTilePtr)     \
    {                                                                                      \
        transposeTemp0 = _mm256_extractf128_ps(cVec0, offset);                             \
        transposeTemp1 = _mm256_extractf128_ps(cVec1, offset);                             \
        transposeTemp2 = _mm256_extractf128_ps(cVec2, offset);                             \
        transposeTemp3 = _mm256_extractf128_ps(cVec3, offset);                             \
        _MM_TRANSPOSE4_PS(transposeTemp0, transposeTemp1, transposeTemp2, transposeTemp3); \
        _mm_storeu_ps(cTilePtr + (rowIdx + 0) * unit, transposeTemp0);                      \
        _mm_storeu_ps(cTilePtr + (rowIdx + 1) * unit, transposeTemp1);                      \
        _mm_storeu_ps(cTilePtr + (rowIdx + 2) * unit, transposeTemp2);                      \
        _mm_storeu_ps(cTilePtr + (rowIdx + 3) * unit, transposeTemp3);                      \
    }

#define TRANSPOSE_4x24_WITH_STORE(cTilePtr, unit)                               \
    {                                                                           \
        __m128 transposeTemp0;                                                  \
        __m128 transposeTemp1;                                                  \
        __m128 transposeTemp2;                                                  \
        __m128 transposeTemp3;                                                  \
        TRANSPOSE_4x4_WITH_STORE(0, 0, c0Vec, c3Vec, c6Vec, c9Vec, cTilePtr);   \
        TRANSPOSE_4x4_WITH_STORE(4, 1, c0Vec, c3Vec, c6Vec, c9Vec, cTilePtr);   \
        TRANSPOSE_4x4_WITH_STORE(8, 0, c1Vec, c4Vec, c7Vec, c10Vec, cTilePtr);  \
        TRANSPOSE_4x4_WITH_STORE(12, 1, c1Vec, c4Vec, c7Vec, c10Vec, cTilePtr); \
        TRANSPOSE_4x4_WITH_STORE(16, 0, c2Vec, c5Vec, c8Vec, c11Vec, cTilePtr); \
        TRANSPOSE_4x4_WITH_STORE(20, 1, c2Vec, c5Vec, c8Vec, c11Vec, cTilePtr); \
    }

#define REMAIN_TRANSPOSE_4x24_WITH_STORE(cTilePtr, unit)                                       \
    {                                                                                          \
        __m128 transposeTemp0;                                                                 \
        __m128 transposeTemp1;                                                                 \
        __m128 transposeTemp2;                                                                 \
        __m128 transposeTemp3;                                                                 \
        int tailE  = eSize % 4;                                                                \
        int eFull4 = eSize / 4;                                                                \
        switch (eFull4) {                                                                      \
            case 5:                                                                            \
                TRANSPOSE_4x4_WITH_STORE(16, 0, c2Vec, c5Vec, c8Vec, c11Vec, cTilePtr);        \
            case 4:                                                                            \
                TRANSPOSE_4x4_WITH_STORE(12, 1, c1Vec, c4Vec, c7Vec, c10Vec, cTilePtr);        \
            case 3:                                                                            \
                TRANSPOSE_4x4_WITH_STORE(8, 0, c1Vec, c4Vec, c7Vec, c10Vec, cTilePtr);         \
            case 2:                                                                            \
                TRANSPOSE_4x4_WITH_STORE(4, 1, c0Vec, c3Vec, c6Vec, c9Vec, cTilePtr);          \
            case 1:                                                                            \
                TRANSPOSE_4x4_WITH_STORE(0, 0, c0Vec, c3Vec, c6Vec, c9Vec, cTilePtr);          \
            default:                                                                           \
                break;                                                                         \
        }                                                                                      \
        if (tailE) {                                                                           \
            if (eFull4 == 5) {                                                                 \
                transposeTemp0 = _mm256_extractf128_ps(c2Vec, 1);                              \
                transposeTemp1 = _mm256_extractf128_ps(c5Vec, 1);                              \
                transposeTemp2 = _mm256_extractf128_ps(c8Vec, 1);                              \
                transposeTemp3 = _mm256_extractf128_ps(c11Vec, 1);                             \
            } else if (eFull4 == 4) {                                                          \
                transposeTemp0 = _mm256_extractf128_ps(c2Vec, 0);                              \
                transposeTemp1 = _mm256_extractf128_ps(c5Vec, 0);                              \
                transposeTemp2 = _mm256_extractf128_ps(c8Vec, 0);                              \
                transposeTemp3 = _mm256_extractf128_ps(c11Vec, 0);                             \
            } else if (eFull4 == 3) {                                                          \
                transposeTemp0 = _mm256_extractf128_ps(c1Vec, 1);                              \
                transposeTemp1 = _mm256_extractf128_ps(c4Vec, 1);                              \
                transposeTemp2 = _mm256_extractf128_ps(c7Vec, 1);                              \
                transposeTemp3 = _mm256_extractf128_ps(c10Vec, 1);                             \
            } else if (eFull4 == 2) {                                                          \
                transposeTemp0 = _mm256_extractf128_ps(c1Vec, 0);                              \
                transposeTemp1 = _mm256_extractf128_ps(c4Vec, 0);                              \
                transposeTemp2 = _mm256_extractf128_ps(c7Vec, 0);                              \
                transposeTemp3 = _mm256_extractf128_ps(c10Vec, 0);                             \
            } else if (eFull4 == 1) {                                                          \
                transposeTemp0 = _mm256_extractf128_ps(c0Vec, 1);                              \
                transposeTemp1 = _mm256_extractf128_ps(c3Vec, 1);                              \
                transposeTemp2 = _mm256_extractf128_ps(c6Vec, 1);                              \
                transposeTemp3 = _mm256_extractf128_ps(c9Vec, 1);                              \
            }                                                                                  \
            else{\
                transposeTemp0 = _mm256_extractf128_ps(c0Vec, 0);                              \
                transposeTemp1 = _mm256_extractf128_ps(c3Vec, 0);                              \
                transposeTemp2 = _mm256_extractf128_ps(c6Vec, 0);                              \
                transposeTemp3 = _mm256_extractf128_ps(c9Vec, 0);                              \
            }\
            _MM_TRANSPOSE4_PS(transposeTemp0, transposeTemp1, transposeTemp2, transposeTemp3); \
            int offset = 4 * eFull4;                                                           \
            switch (tailE) {                                                                   \
                case 3:                                                                        \
                    _mm_storeu_ps(cTilePtr + (offset + 2) * unit, transposeTemp2);             \
                case 2:                                                                        \
                    _mm_storeu_ps(cTilePtr + (offset + 1) * unit, transposeTemp1);             \
                case 1:                                                                        \
                    _mm_storeu_ps(cTilePtr + (offset + 0) * unit, transposeTemp0);             \
                default:                                                                       \
                    break;                                                                     \
            }                                                                                  \
        }                                                                                      \
    }

#define FP32_BYTES      4
#define AVX2_SPARSE_EP  24
#define AVX2_SP_BLOCK4  4

void _AVX_MNNPackedSparseMatMulEpx1EFMA(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                    const float* postParameters, const float* bias, unsigned int* NNZMap,
                                    int* dataOffsetMap) {
    /*
    mat_a: [eSize/eP, l, eP]
    mat_c: [h/unit, e, unit]
    bias: [h, ]
    parameter[0]: eP * bytes
    parameter[1]: l
    parameter[2]: h
    parameter[3]: h/unit stride, equals to e * unit * sizeof(dataType)
    parameter[4]: unit
    eSize: this tile`s real e size, which can be greater or less than eP!
    postParameters[2]: min_val of output
    postParameters[3]: max_val of output
    */

    /*
    This func performs the sparse matmul with bias add and post process of min/max threshold.
    The basic process of the dense version of func is:
    batch_matmul([l, eP], [h/hP, l, hP]) --> [h/hP, eP, hP].
    However, when mat_b is sparsed encoded, this func changes accordingly.
    First, divide the whole process into two part, the full hP part and the remain part.
    The full hP part means, in each iteration, mat_b`s col (or row actually) is processed in hP count,
    and the non-zero value is hP continous encoded.
    The remain part means, in each iteration, mat_b`s col (or row actually) is processed in 1 count,
    and the non-zero value is encoded one by one.
    (Although this func is specialized for hP = 1)

    ***********************************************
    Specialization description:
    1. eP = 24, hP = 1, lP = 1;
    2. mat_a stores in [eSize/eP, l, eP] format;
    3. mat_c stores in [h/unit, e, unit] format;
    4. data type is fixed as float32, which means the bytes = 4;
    5. unit is fixed as 8;
    ***********************************************

    Note that, the function reserves the aStride, which is for mat_a that contains more than one l * eP tile.
    But for now, limit the eSize <= eP!
    */
#ifdef MNN_X86_USE_ASM
   if (eSize == AVX2_SPARSE_EP && parameter[2] % 4 == 0){
        // use the asm function when eSize == 24 and h == 4x
        SparseMatMulParas temp = {C, A, B, NNZMap, dataOffsetMap};
        SparseMatMulParas* tempPtr = &temp;
        _AVX_MNNPackedSparseMatMulEpx1EFMA_ASM(tempPtr, bias, parameter, postParameters);
        return;
    }
#endif
    const size_t aStride = parameter[0] / FP32_BYTES;
    const size_t l       = parameter[1];
    const size_t h       = parameter[2];
    const size_t cStride = parameter[3] / FP32_BYTES; // intrinsic do not need the byte stride.
    const size_t unit    = 8;

    MNN_ASSERT(eSize <= aStride);

    auto minVec = _mm256_broadcast_ss(postParameters + 2);
    auto maxVec = _mm256_broadcast_ss(postParameters + 3);

    // full [l, eP] X [h/unit, e, unit]
    for (int matALoopIdx = 0; matALoopIdx < eSize / aStride; matALoopIdx++) {
        const float* aTilePtrSt  = A + l * aStride * matALoopIdx;
        const int* aRowOffsetPtr = dataOffsetMap;
        const float* weightPtr   = B;

        // as this func is specialized for hP = 1,
        // iteration in h axis is all full hP method.
        __m256 c0Vec;
        __m256 c1Vec;
        __m256 c2Vec;
        auto c0VecPtr = (float*)&c0Vec;
        auto c1VecPtr = (float*)&c1Vec;
        auto c2VecPtr = (float*)&c2Vec;

        for (int hLoopIdx = 0; hLoopIdx < h; hLoopIdx++) {
            float* cTilePtrSt = C + (unit * aStride * matALoopIdx) + (hLoopIdx / unit * cStride) + (hLoopIdx % unit);
            size_t nonZeroCnt = *NNZMap;
            NNZMap++;

            // inittialize mat_c tile with bias if existed.
            // [eP, hP] bias initialize.

            if (bias != nullptr) {
                c0Vec = _mm256_broadcast_ss(bias + hLoopIdx);
                c1Vec = c0Vec;
                c2Vec = c0Vec;
            } else {
                c0Vec = _mm256_setzero_ps();
                c1Vec = _mm256_setzero_ps();
                c2Vec = _mm256_setzero_ps();
            }

            for (int lLoopIdx = 0; lLoopIdx < nonZeroCnt; lLoopIdx++) {
                aTilePtrSt += aRowOffsetPtr[0];
                aRowOffsetPtr++;
                auto a0Vec = _mm256_loadu_ps(aTilePtrSt + 0);
                auto a1Vec = _mm256_loadu_ps(aTilePtrSt + 8);
                auto a2Vec = _mm256_loadu_ps(aTilePtrSt + 16);
                auto b0Vec = _mm256_broadcast_ss(weightPtr);
                weightPtr++;
                c0Vec = EMULATED_AVX2_FMA(c0Vec, a0Vec, b0Vec);
                c1Vec = EMULATED_AVX2_FMA(c1Vec, a1Vec, b0Vec);
                c2Vec = EMULATED_AVX2_FMA(c2Vec, a2Vec, b0Vec);
            }

            // min-max post process and store process.
            MIN_MAX_VEC(c0Vec);
            MIN_MAX_VEC(c1Vec);
            MIN_MAX_VEC(c2Vec);

            ONE_H_STORE_E24(cTilePtrSt);
        }
        NNZMap -= h;
    }

    // remained [l, eSize%eP] X [h/unit, e, unit]
    A += (eSize / aStride) * aStride * l;
    C += (eSize / aStride) * aStride * unit;
    eSize = eSize % aStride; // eSize % 24

    // remained eSize part
    if (eSize) {
        // as this func is specialized for hP = 1,
        // iteration in h axis is all full hP method.
        __m256 c0Vec;
        __m256 c1Vec;
        __m256 c2Vec;
        auto c0VecPtr  = (float*)&c0Vec;
        auto c1VecPtr  = (float*)&c1Vec;
        auto c2VecPtr  = (float*)&c2Vec;
        for (int hLoopIdx = 0; hLoopIdx < h; hLoopIdx++) {
            float* cTilePtrSt = C + (hLoopIdx / unit * cStride) + (hLoopIdx % unit);
            size_t nonZeroCnt = *NNZMap;
            NNZMap++;

            // inittialize mat_c tile with bias if existed.
            // [eP, hP] bias initialize.

            if (bias != nullptr) {
                c0Vec = _mm256_broadcast_ss(bias + hLoopIdx);
                c1Vec = c0Vec;
                c2Vec = c0Vec;
            } else {
                c0Vec = _mm256_setzero_ps();
                c1Vec = _mm256_setzero_ps();
                c2Vec = _mm256_setzero_ps();
            }

            for (int lLoopIdx = 0; lLoopIdx < nonZeroCnt; lLoopIdx++) {
                A += dataOffsetMap[0];
                dataOffsetMap++;
                auto a0Vec      = _mm256_loadu_ps(A + 0);
                auto a1Vec      = _mm256_loadu_ps(A + 8);
                auto a2Vec      = _mm256_loadu_ps(A + 16);
                auto b0Vec = _mm256_broadcast_ss(B);
                B++;
                c0Vec = EMULATED_AVX2_FMA(c0Vec, a0Vec, b0Vec);
                c1Vec = EMULATED_AVX2_FMA(c1Vec, a1Vec, b0Vec);
                c2Vec = EMULATED_AVX2_FMA(c2Vec, a2Vec, b0Vec);
            }

            // min-max post process and store process.
            MIN_MAX_VEC(c0Vec);
            MIN_MAX_VEC(c1Vec);
            MIN_MAX_VEC(c2Vec);

            auto CStorePtr = cTilePtrSt;
            auto cxVecPtr  = c0VecPtr;
            if (eSize >= 8) {
                CStorePtr[8 * 0] = cxVecPtr[0];
                CStorePtr[8 * 1] = cxVecPtr[1];
                CStorePtr[8 * 2] = cxVecPtr[2];
                CStorePtr[8 * 3] = cxVecPtr[3];
                CStorePtr[8 * 4] = cxVecPtr[4];
                CStorePtr[8 * 5] = cxVecPtr[5];
                CStorePtr[8 * 6] = cxVecPtr[6];
                CStorePtr[8 * 7] = cxVecPtr[7];
                CStorePtr += 8 * unit;
                cxVecPtr = c1VecPtr;
            }
            if (eSize >= 16){
                CStorePtr[8 * 0] = cxVecPtr[0];
                CStorePtr[8 * 1] = cxVecPtr[1];
                CStorePtr[8 * 2] = cxVecPtr[2];
                CStorePtr[8 * 3] = cxVecPtr[3];
                CStorePtr[8 * 4] = cxVecPtr[4];
                CStorePtr[8 * 5] = cxVecPtr[5];
                CStorePtr[8 * 6] = cxVecPtr[6];
                CStorePtr[8 * 7] = cxVecPtr[7];
                CStorePtr += 8 * unit;
                cxVecPtr = c2VecPtr;
            }
            for (int i = 0; i < eSize % 8; i++) {
                CStorePtr[8 * i] = cxVecPtr[i];
            }
        }
        NNZMap -= h;
    }
    return;
}

void _AVX_MNNPackedSparseMatMulEpx4EFMA(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                    const float* postParameters, const float* bias, unsigned int* NNZMap,
                                    int* dataOffsetMap) {
    /*
    mat_a: [eSize/eP, l, eP]
    mat_c: [h/unit, e, unit]
    bias: [h, ]
    parameter[0]: eP * bytes
    parameter[1]: l
    parameter[2]: h
    parameter[3]: h/unit stride, equals to e * unit * sizeof(dataType)
    parameter[4]: unit
    eSize: this tile`s real e size, which can be greater or less than eP!
    postParameters[2]: min_val of output
    postParameters[3]: max_val of output
    */

    /*
    This func performs the sparse matmul with bias add and post process of min/max threshold.
    The basic process of the dense version of func is:
    batch_matmul([l, eP], [h/hP, l, hP]) --> [h/hP, eP, hP].
    However, when mat_b is sparsed encoded, this func changes accordingly.
    First, divide the whole process into two part, the full hP part and the remain part.
    The full hP part means, in each iteration, mat_b`s col (or row actually) is processed in hP count,
    and the non-zero value is hP continous encoded.
    The remain part means, in each iteration, mat_b`s col (or row actually) is processed in 1 count,
    and the non-zero value is encoded one by one.

    ***********************************************
    Specialization description:
    1. eP = 24, hP = 4, lP = 1;
    2. mat_a stores in [eSize/eP, l, eP] format;
    3. mat_c stores in [h/unit, e, unit] format;
    4. data type is fixed as float32, which means the bytes = 4;
    5. unit is fixed as 8;
    ***********************************************

    Note that, the function reserves the aStride, which is for mat_a that contains more than one l * eP tile.
    But for now, limit the eSize <= eP!
    */
#define ONE_LP_ACT_E24(cVecFirst, cVecSecond, cVecThird)       \
    b0Vec = _mm256_broadcast_ss(weightPtr);                    \
    weightPtr++;                                               \
    cVecFirst  = EMULATED_AVX2_FMA(cVecFirst, a0Vec, b0Vec);  \
    cVecSecond = EMULATED_AVX2_FMA(cVecSecond, a1Vec, b0Vec); \
    cVecThird  = EMULATED_AVX2_FMA(cVecThird, a2Vec, b0Vec);

#define REMAIN_E_ONE_LP_ACT_E24(cVecFirst, cVecSecond, cVecThird) \
    b0Vec = _mm256_broadcast_ss(B);                               \
    B++;                                                          \
    cVecFirst  = EMULATED_AVX2_FMA(cVecFirst, a0Vec, b0Vec);     \
    cVecSecond = EMULATED_AVX2_FMA(cVecSecond, a1Vec, b0Vec);    \
    cVecThird  = EMULATED_AVX2_FMA(cVecThird, a2Vec, b0Vec);

#ifdef MNN_X86_USE_ASM
   if (eSize == AVX2_SPARSE_EP && parameter[2] % 4 == 0){
        // use the asm function when eSize == eP(24) and h == 4x
        SparseMatMulParas temp = {C, A, B, NNZMap, dataOffsetMap};
        SparseMatMulParas* tempPtr = &temp;
        _AVX_MNNPackedSparseMatMulEpx4EFMA_ASM(tempPtr, bias, parameter, postParameters);
        return;
    }
#endif
    const size_t aStride = parameter[0] / FP32_BYTES; // intrinsic do not need the byte stride.
    const size_t l       = parameter[1];
    const size_t h       = parameter[2];
    const size_t cStride = parameter[3] / FP32_BYTES; // intrinsic do not need the byte stride.
    const size_t unit    = 8;

    MNN_ASSERT(eSize <= aStride);

    const float minVal = postParameters[2];
    const float maxVal = postParameters[3];
    const int fullHCnt = h / AVX2_SP_BLOCK4 * AVX2_SP_BLOCK4;

    // full [l, eP] X [h/unit, e, unit]
    for (int matALoopIdx = 0; matALoopIdx < eSize / aStride; matALoopIdx++) {
        const float* aTilePtrSt  = A + l * aStride * matALoopIdx;
        const int* aRowOffsetPtr = dataOffsetMap;
        const float* weightPtr   = B;
        int hLoopIdx             = 0;

        // full hP method!
        for (; hLoopIdx < fullHCnt; hLoopIdx += AVX2_SP_BLOCK4) {
            float* cTilePtrSt = C + (unit * aStride * matALoopIdx) + (hLoopIdx / unit * cStride) + (hLoopIdx % unit);
            size_t nonZeroCnt = *NNZMap;
            NNZMap++;

            __m256 c0Vec;
            __m256 c1Vec;
            __m256 c2Vec;
            __m256 c3Vec;
            __m256 c4Vec;
            __m256 c5Vec;
            __m256 c6Vec;
            __m256 c7Vec;
            __m256 c8Vec;
            __m256 c9Vec;
            __m256 c10Vec;
            __m256 c11Vec;
            if (bias != nullptr) {
                c0Vec = _mm256_broadcast_ss(bias + hLoopIdx);
                c3Vec = _mm256_broadcast_ss(bias + hLoopIdx + 1);
                c6Vec = _mm256_broadcast_ss(bias + hLoopIdx + 2);
                c9Vec = _mm256_broadcast_ss(bias + hLoopIdx + 3);
                c1Vec = c0Vec;
                c2Vec = c0Vec;
                c4Vec = c3Vec;
                c5Vec = c3Vec;
                c7Vec = c6Vec;
                c8Vec = c6Vec;
                c10Vec = c9Vec;
                c11Vec = c9Vec;

            } else {
                // [intrinsic bug] zeroall will not work after the first iteration!
                c0Vec = _mm256_setzero_ps();
                c3Vec = _mm256_setzero_ps();
                c6Vec = _mm256_setzero_ps();
                c9Vec = _mm256_setzero_ps();
                c1Vec = _mm256_setzero_ps();
                c2Vec = _mm256_setzero_ps();
                c4Vec = _mm256_setzero_ps();
                c5Vec = _mm256_setzero_ps();
                c7Vec = _mm256_setzero_ps();
                c8Vec = _mm256_setzero_ps();
                c10Vec = _mm256_setzero_ps();
                c11Vec = _mm256_setzero_ps();
            }

            {
                __m256 a0Vec;
                __m256 a1Vec;
                __m256 a2Vec;
                __m256 b0Vec;

                for (int lLoopIdx = 0; lLoopIdx < nonZeroCnt; lLoopIdx++) {
                    //printf("aRowOffset: %d\t", *aRowOffsetPtr);
                    aTilePtrSt += *aRowOffsetPtr;
                    aRowOffsetPtr++;
                    a0Vec = _mm256_loadu_ps(aTilePtrSt + 0);
                    a1Vec = _mm256_loadu_ps(aTilePtrSt + 8);
                    a2Vec = _mm256_loadu_ps(aTilePtrSt + 16);
                    ONE_LP_ACT_E24(c0Vec, c1Vec, c2Vec);
                    ONE_LP_ACT_E24(c3Vec, c4Vec, c5Vec);
                    ONE_LP_ACT_E24(c6Vec, c7Vec, c8Vec);
                    ONE_LP_ACT_E24(c9Vec, c10Vec, c11Vec);
                }
            }
            {
                auto minVec = _mm256_set1_ps(minVal);
                auto maxVec = _mm256_set1_ps(maxVal);

                MIN_MAX_VEC(c0Vec);
                MIN_MAX_VEC(c1Vec);
                MIN_MAX_VEC(c2Vec);
                MIN_MAX_VEC(c3Vec);
                MIN_MAX_VEC(c4Vec);
                MIN_MAX_VEC(c5Vec);
                MIN_MAX_VEC(c6Vec);
                MIN_MAX_VEC(c7Vec);
                MIN_MAX_VEC(c8Vec);
                MIN_MAX_VEC(c9Vec);
                MIN_MAX_VEC(c10Vec);
                MIN_MAX_VEC(c11Vec);
            }
            TRANSPOSE_4x24_WITH_STORE(cTilePtrSt, unit);
        }

        // remain hP method!
        __m256 c0Vec;
        __m256 c1Vec;
        __m256 c2Vec;
        auto minVec   = _mm256_set1_ps(minVal);
        auto maxVec   = _mm256_set1_ps(maxVal);
        auto c0VecPtr = (float*)&c0Vec;
        auto c1VecPtr = (float*)&c1Vec;
        auto c2VecPtr = (float*)&c2Vec;

        for (; hLoopIdx < h; hLoopIdx++) {
            float* cTilePtrSt = C + (unit * aStride * matALoopIdx) + (hLoopIdx / unit * cStride) + (hLoopIdx % unit);
            size_t nonZeroCnt = *NNZMap;
            NNZMap++;

            // inittialize mat_c tile with bias if existed.
            // [eP, hP] bias initialize.

            if (bias != nullptr) {
                c0Vec = _mm256_broadcast_ss(bias + hLoopIdx);
                c1Vec = c0Vec;
                c2Vec = c0Vec;
            } else {
                c0Vec = _mm256_setzero_ps();
                c1Vec = _mm256_setzero_ps();
                c2Vec = _mm256_setzero_ps();
            }

            for (int lLoopIdx = 0; lLoopIdx < nonZeroCnt; lLoopIdx++) {
                aTilePtrSt += aRowOffsetPtr[0];
                aRowOffsetPtr++;
                auto a0Vec = _mm256_loadu_ps(aTilePtrSt + 0);
                auto a1Vec = _mm256_loadu_ps(aTilePtrSt + 8);
                auto a2Vec = _mm256_loadu_ps(aTilePtrSt + 16);
                auto b0Vec = _mm256_broadcast_ss(weightPtr);
                weightPtr++;
                c0Vec = EMULATED_AVX2_FMA(c0Vec, a0Vec, b0Vec);
                c1Vec = EMULATED_AVX2_FMA(c1Vec, a1Vec, b0Vec);
                c2Vec = EMULATED_AVX2_FMA(c2Vec, a2Vec, b0Vec);
            }

            // min-max post process and store process.
            MIN_MAX_VEC(c0Vec);
            MIN_MAX_VEC(c1Vec);
            MIN_MAX_VEC(c2Vec);

            ONE_H_STORE_E24(cTilePtrSt);
        }
        NNZMap -= fullHCnt / AVX2_SP_BLOCK4 + h - fullHCnt;
    }

    // remained [l, eSize%eP] X [h/unit, e, unit]
    A += (eSize / aStride) * aStride * l;
    C += (eSize / aStride) * aStride * unit;
    eSize = eSize % aStride; // eSize % 24

    // remained eSize part
    if (eSize) {
        int hLoopIdx   = 0;
        for (; hLoopIdx < fullHCnt; hLoopIdx += AVX2_SP_BLOCK4) {
            float* cTilePtrSt = C + (hLoopIdx / unit * cStride) + (hLoopIdx % unit);
            size_t nonZeroCnt = *NNZMap;
            NNZMap++;

            __m256 c0Vec;
            __m256 c1Vec;
            __m256 c2Vec;
            __m256 c3Vec;
            __m256 c4Vec;
            __m256 c5Vec;
            __m256 c6Vec;
            __m256 c7Vec;
            __m256 c8Vec;
            __m256 c9Vec;
            __m256 c10Vec;
            __m256 c11Vec;
            if (bias != nullptr) {
                c0Vec = _mm256_broadcast_ss(bias + hLoopIdx);
                c3Vec = _mm256_broadcast_ss(bias + hLoopIdx + 1);
                c6Vec = _mm256_broadcast_ss(bias + hLoopIdx + 2);
                c9Vec = _mm256_broadcast_ss(bias + hLoopIdx + 3);
                c1Vec = c0Vec;
                c2Vec = c0Vec;
                c4Vec = c3Vec;
                c5Vec = c3Vec;
                c7Vec = c6Vec;
                c8Vec = c6Vec;
                c10Vec = c9Vec;
                c11Vec = c9Vec;

            } else {
                // [intrinsic bug] zeroall will not work after the first iteration!
                c0Vec = _mm256_setzero_ps();
                c3Vec = _mm256_setzero_ps();
                c6Vec = _mm256_setzero_ps();
                c9Vec = _mm256_setzero_ps();
                c1Vec = _mm256_setzero_ps();
                c2Vec = _mm256_setzero_ps();
                c4Vec = _mm256_setzero_ps();
                c5Vec = _mm256_setzero_ps();
                c7Vec = _mm256_setzero_ps();
                c8Vec = _mm256_setzero_ps();
                c10Vec = _mm256_setzero_ps();
                c11Vec = _mm256_setzero_ps();
            }

            {
                __m256 a0Vec;
                __m256 a1Vec;
                __m256 a2Vec;
                __m256 b0Vec;
                
                for (int lLoopIdx = 0; lLoopIdx < nonZeroCnt; lLoopIdx++) {
                    A += *dataOffsetMap;
                    dataOffsetMap++;
                    a0Vec = _mm256_loadu_ps(A + 0);
                    a1Vec = _mm256_loadu_ps(A + 8);
                    a2Vec = _mm256_loadu_ps(A + 16);

                    REMAIN_E_ONE_LP_ACT_E24(c0Vec, c1Vec, c2Vec);
                    REMAIN_E_ONE_LP_ACT_E24(c3Vec, c4Vec, c5Vec);
                    REMAIN_E_ONE_LP_ACT_E24(c6Vec, c7Vec, c8Vec);
                    REMAIN_E_ONE_LP_ACT_E24(c9Vec, c10Vec, c11Vec);
                }
            }
            {

                auto minVec = _mm256_set1_ps(minVal);
                auto maxVec = _mm256_set1_ps(maxVal);
                MIN_MAX_VEC(c0Vec);
                MIN_MAX_VEC(c1Vec);
                MIN_MAX_VEC(c2Vec);
                MIN_MAX_VEC(c3Vec);
                MIN_MAX_VEC(c4Vec);
                MIN_MAX_VEC(c5Vec);
                MIN_MAX_VEC(c6Vec);
                MIN_MAX_VEC(c7Vec);
                MIN_MAX_VEC(c8Vec);
                MIN_MAX_VEC(c9Vec);
                MIN_MAX_VEC(c10Vec);
                MIN_MAX_VEC(c11Vec);
            }
            REMAIN_TRANSPOSE_4x24_WITH_STORE(cTilePtrSt, unit);
        }

        // remained h part
        __m256 c0Vec;
        __m256 c1Vec;
        __m256 c2Vec;
        auto c0VecPtr = (float*)&c0Vec;
        auto c1VecPtr = (float*)&c1Vec;
        auto c2VecPtr = (float*)&c2Vec;
        auto minVec   = _mm256_set1_ps(minVal);
        auto maxVec   = _mm256_set1_ps(maxVal);

        for (; hLoopIdx < h; hLoopIdx++) {
            float* cTilePtrSt = C + (hLoopIdx / unit * cStride) + (hLoopIdx % unit);
            size_t nonZeroCnt = *NNZMap;
            NNZMap++;

            // inittialize mat_c tile with bias if existed.
            // [eP, hP] bias initialize.

            if (bias != nullptr) {
                c0Vec = _mm256_broadcast_ss(bias + hLoopIdx);
                c1Vec = c0Vec;
                c2Vec = c0Vec;
            } else {
                c0Vec = _mm256_setzero_ps();
                c1Vec = _mm256_setzero_ps();
                c2Vec = _mm256_setzero_ps();
            }
            __m256 a0Vec;
            __m256 a1Vec;
            __m256 a2Vec;
            for (int lLoopIdx = 0; lLoopIdx < nonZeroCnt; lLoopIdx++) {
                A += *dataOffsetMap;
                dataOffsetMap++;
                a0Vec      = _mm256_loadu_ps(A + 0);
                a1Vec      = _mm256_loadu_ps(A + 8);
                a2Vec      = _mm256_loadu_ps(A + 16);

                auto b0Vec = _mm256_broadcast_ss(B);
                B++;
                EMULATED_AVX2_FMA(c0Vec, a0Vec, b0Vec);
                EMULATED_AVX2_FMA(c1Vec, a1Vec, b0Vec);
                EMULATED_AVX2_FMA(c2Vec, a2Vec, b0Vec);
            }

            // min-max post process and store process.
            MIN_MAX_VEC(c0Vec);
            MIN_MAX_VEC(c1Vec);
            MIN_MAX_VEC(c2Vec);

            auto CStorePtr = cTilePtrSt;
            auto cxVecPtr  = c0VecPtr;
            if (eSize >= 8) {
                CStorePtr[8 * 0] = cxVecPtr[0];
                CStorePtr[8 * 1] = cxVecPtr[1];
                CStorePtr[8 * 2] = cxVecPtr[2];
                CStorePtr[8 * 3] = cxVecPtr[3];
                CStorePtr[8 * 4] = cxVecPtr[4];
                CStorePtr[8 * 5] = cxVecPtr[5];
                CStorePtr[8 * 6] = cxVecPtr[6];
                CStorePtr[8 * 7] = cxVecPtr[7];
                CStorePtr += 8 * unit;
                cxVecPtr = c1VecPtr;
            }
            if (eSize >= 16){
                CStorePtr[8 * 0] = cxVecPtr[0];
                CStorePtr[8 * 1] = cxVecPtr[1];
                CStorePtr[8 * 2] = cxVecPtr[2];
                CStorePtr[8 * 3] = cxVecPtr[3];
                CStorePtr[8 * 4] = cxVecPtr[4];
                CStorePtr[8 * 5] = cxVecPtr[5];
                CStorePtr[8 * 6] = cxVecPtr[6];
                CStorePtr[8 * 7] = cxVecPtr[7];
                CStorePtr += 8 * unit;
                cxVecPtr = c2VecPtr;
            }
            for (int i = 0; i < eSize % 8; i++) {
                CStorePtr[8 * i] = cxVecPtr[i];
            }
        }
        NNZMap -= h;
    }
    return;
#undef REMAIN_E_ONE_LP_ACT_E24
#undef ONE_LP_ACT_E24
}

#undef AVX2_SP_BLOCK4
#undef AVX2_SPARSE_EP
#undef FP32_BYTES
#undef EMULATED_AVX2_FMA
#undef MIN_MAX_VEC
#undef ONE_H_STORE_E24
#undef TRANSPOSE_4x4_WITH_STORE
#undef TRANSPOSE_4x24_WITH_STORE
#undef REMAIN_TRANSPOSE_4x24_WITH_STORE
