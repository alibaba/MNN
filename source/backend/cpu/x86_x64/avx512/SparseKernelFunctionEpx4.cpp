//
//  SparseKernelFunctionEpx8.cpp
//  MNN
//
//  Created by MNN on 2021/08/18.
//  Copyright © 2018 - 2021, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "SparseKernelFunction.hpp"

extern "C" {
    void _AVX512_MNNPackedSparseMatMulEpx4_ASM(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                    const float* postParameters, const float* bias, unsigned int* NNZMap,
                                    int* dataOffsetMap);
}

void _AVX512_MNNPackedSparseMatMulEpx4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                    const float* postParameters, const float* bias, unsigned int* NNZMap,
                                    int* dataOffsetMap) {

    auto eP = parameter[0] / sizeof(float);
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    auto aStride = eP * l; // sizeof(float);

    constexpr size_t packCUnit = 16;
    constexpr size_t packCUnitLog = 4;
    constexpr int sparseBlockOC = 4;
    // MNN_PRINT("eSize:%zu, eP:%zu, h:%zu, l:%zu\n", eSize, eP, h, l);
    // if (eSize == eP && (h % sparseBlockOC == 0)) {
    //     _AVX512_MNNPackedSparseMatMulEpx4_ASM(C, A, B, eSize, parameter, postParameters, bias, NNZMap, dataOffsetMap);
    //     return;
    // }

    __m512 vmin = _mm512_set1_ps(*(postParameters + 2));
    __m512 vmax = _mm512_set1_ps(*(postParameters + 3));
    // MNN_PRINT("begin caculate, eSize:%ld\n", eSize);
    const float* a = A;
    size_t ie = 0;
    for (ie = 0; ie + eP <= eSize; ie += eP) { // ep: 48
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << packCUnitLog);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~(sparseBlockOC - 1))); ih += sparseBlockOC) {
            auto c = blockC + (ih >> packCUnitLog) * cStride + (ih % packCUnit);

            __m512 vacc0, vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7, vacc8, vacc9, vacc10, vacc11;
            // tobe merged in to weight data
            if (bias) {
               vacc0 = _mm512_set1_ps(bias[ih]);
               vacc3 = _mm512_set1_ps(bias[ih + 1]);
               vacc6 = _mm512_set1_ps(bias[ih + 2]);
               vacc9 = _mm512_set1_ps(bias[ih + 3]);
            } else {
                vacc0 = _mm512_setzero_ps();
                vacc3 = _mm512_setzero_ps();
                vacc6 = _mm512_setzero_ps();
                vacc9 = _mm512_setzero_ps();
            }
            vacc1 = vacc0;
            vacc2 = vacc0;
            vacc4 = vacc3;
            vacc5 = vacc3;
            vacc7 = vacc6;
            vacc8 = vacc6;
            vacc10 = vacc9;
            vacc11 = vacc9;
            unsigned int lElement = *nnz++;

            __m512 va0_15_swap = _mm512_loadu_ps(a);
            __m512 va16_31_swap = _mm512_loadu_ps(a + 16);
            __m512 va32_48_swap = _mm512_loadu_ps(a + 32);
            const int diff = *dataOffset++;
            a = a + diff;

            // __m512 w0_swap = _mm512_set1_ps(*(w)); // donot work. should try 2-way segement iteration
            // __m512 w1_swap = _mm512_set1_ps(*(w + 1));
            // __m512 w2_swap = _mm512_set1_ps(*(w + 2));
            // __m512 w3_swap = _mm512_set1_ps(*(w + 3));

            for (auto il = 0; il < lElement; il++) {
              // __m512 va0_15_ = _mm512_loadu_ps(a);
              // __m512 va16_31_ = _mm512_loadu_ps(a + 16);
              // __m512 va32_48_ = _mm512_loadu_ps(a + 32);
                __m512 va0_15 = va0_15_swap;
                __m512 va16_31 = va16_31_swap;
                __m512 va32_48 = va32_48_swap;

                __m512 w0 = _mm512_set1_ps(*(w));
                __m512 w1 = _mm512_set1_ps(*(w + 1));
                __m512 w2 = _mm512_set1_ps(*(w + 2));
                __m512 w3 = _mm512_set1_ps(*(w + 3));
                w += sparseBlockOC;
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");

                vacc0 =  _mm512_fmadd_ps(va0_15, w0, vacc0);
                vacc1 =  _mm512_fmadd_ps(va16_31, w0, vacc1);
                vacc2 =  _mm512_fmadd_ps(va32_48, w0, vacc2);
                va0_15_swap = _mm512_loadu_ps(a);
                va16_31_swap = _mm512_loadu_ps(a + 16);
                va32_48_swap = _mm512_loadu_ps(a + 32);
                vacc3 =  _mm512_fmadd_ps(va0_15, w1, vacc3);
                vacc4 =  _mm512_fmadd_ps(va16_31, w1, vacc4);
                vacc5 =  _mm512_fmadd_ps(va32_48, w1, vacc5);

                const int diff = *dataOffset++;
                a = a + diff;

                vacc6 =  _mm512_fmadd_ps(va0_15, w2, vacc6);
                vacc7 =  _mm512_fmadd_ps(va16_31, w2, vacc7);
                vacc8 =  _mm512_fmadd_ps(va32_48, w2, vacc8);
                vacc9 =  _mm512_fmadd_ps(va0_15, w3, vacc9);
                vacc10 =  _mm512_fmadd_ps(va16_31, w3, vacc10);
                vacc11 =  _mm512_fmadd_ps(va32_48, w3, vacc11);

            }
            dataOffset--;
            a = a - (*dataOffset);
            vacc0 = _mm512_min_ps(vacc0, vmax);
            vacc1 = _mm512_min_ps(vacc1, vmax);
            vacc2 = _mm512_min_ps(vacc2, vmax);
            vacc3 = _mm512_min_ps(vacc3, vmax);
            vacc4 = _mm512_min_ps(vacc4, vmax);
            vacc5 = _mm512_min_ps(vacc5, vmax);
            vacc6 = _mm512_min_ps(vacc6, vmax);
            vacc7 = _mm512_min_ps(vacc7, vmax);
            vacc8 = _mm512_min_ps(vacc8, vmax);
            vacc9 = _mm512_min_ps(vacc9, vmax);
            vacc10 = _mm512_min_ps(vacc10, vmax);
            vacc11 = _mm512_min_ps(vacc11, vmax);

            vacc0 = _mm512_max_ps(vacc0, vmin);
            vacc1 = _mm512_max_ps(vacc1, vmin);
            vacc2 = _mm512_max_ps(vacc2, vmin);
            vacc3 = _mm512_max_ps(vacc3, vmin);
            vacc4 = _mm512_max_ps(vacc4, vmin);
            vacc5 = _mm512_max_ps(vacc5, vmin);
            vacc6 = _mm512_max_ps(vacc6, vmin);
            vacc7 = _mm512_max_ps(vacc7, vmin);
            vacc8 = _mm512_max_ps(vacc8, vmin);
            vacc9 = _mm512_max_ps(vacc9, vmin);
            vacc10 = _mm512_max_ps(vacc10, vmin);
            vacc11 = _mm512_max_ps(vacc11, vmin);


            TRANSPOSE4x4_STORE(c, 0, 0, packCUnit, vacc0, vacc3, vacc6, vacc9);
            TRANSPOSE4x4_STORE(c, 0, 1, packCUnit, vacc0, vacc3, vacc6, vacc9);
            TRANSPOSE4x4_STORE(c, 0, 2, packCUnit, vacc0, vacc3, vacc6, vacc9);
            TRANSPOSE4x4_STORE(c, 0, 3, packCUnit, vacc0, vacc3, vacc6, vacc9);
            TRANSPOSE4x4_STORE(c, 1, 0, packCUnit, vacc1, vacc4, vacc7, vacc10);
            TRANSPOSE4x4_STORE(c, 1, 1, packCUnit, vacc1, vacc4, vacc7, vacc10);
            TRANSPOSE4x4_STORE(c, 1, 2, packCUnit, vacc1, vacc4, vacc7, vacc10);
            TRANSPOSE4x4_STORE(c, 1, 3, packCUnit, vacc1, vacc4, vacc7, vacc10);
            TRANSPOSE4x4_STORE(c, 2, 0, packCUnit, vacc2, vacc5, vacc8, vacc11);
            TRANSPOSE4x4_STORE(c, 2, 1, packCUnit, vacc2, vacc5, vacc8, vacc11);
            TRANSPOSE4x4_STORE(c, 2, 2, packCUnit, vacc2, vacc5, vacc8, vacc11);
            TRANSPOSE4x4_STORE(c, 2, 3, packCUnit, vacc2, vacc5, vacc8, vacc11);
        }
        blockC += (h >> packCUnitLog) * cStride;
        for (; ih < h; ih++) {
            auto c = blockC + ih % packCUnit;
            __m512 vacc0 =  nullptr != bias ? _mm512_set1_ps(*(bias + ih)) : _mm512_setzero_ps();
            __m512 vacc1 = vacc0;
            __m512 vacc2 = vacc0;
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                __m512 va0_15 = _mm512_loadu_ps(a);
                __m512 va16_31 = _mm512_loadu_ps(a + 16);
                __m512 va32_48 = _mm512_loadu_ps(a + 32);
                __m512 w0 = _mm512_set1_ps(*w);

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc0 =  _mm512_fmadd_ps(va0_15, w0, vacc0);
                vacc1 =  _mm512_fmadd_ps(va16_31, w0, vacc1);
                vacc2 =  _mm512_fmadd_ps(va32_48, w0, vacc2);
            }
            vacc0 = _mm512_min_ps(vacc0, vmax);
            vacc1 = _mm512_min_ps(vacc1, vmax);
            vacc2 = _mm512_min_ps(vacc2, vmax);
            vacc0 = _mm512_max_ps(vacc0, vmin);
            vacc1 = _mm512_max_ps(vacc1, vmin);
            vacc2 = _mm512_max_ps(vacc2, vmin);

            // how to store faster: st4 / transpose
            STORE_VECTOR_AS_COLUMN(c, 0, packCUnit, vacc0);
            STORE_VECTOR_AS_COLUMN(c, 1, packCUnit, vacc1);
            STORE_VECTOR_AS_COLUMN(c, 2, packCUnit, vacc2);
        }
        a += aStride;
    }
    auto taileSize = eSize % eP;
    if (taileSize & 0x20) { // tail eSize bitmask 32
        // MNN_PRINT("caculate 32\n");
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << packCUnitLog);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~(sparseBlockOC - 1))); ih += sparseBlockOC) {
            auto c = blockC + (ih >> packCUnitLog) * cStride + (ih % packCUnit);

            __m512 vacc0, vacc1, vacc3, vacc4, vacc6, vacc7, vacc9, vacc10;
            // tobe merged in to weight data
            if (bias) {
               vacc0 = _mm512_set1_ps(bias[ih]);
               vacc3 = _mm512_set1_ps(bias[ih + 1]);
               vacc6 = _mm512_set1_ps(bias[ih + 2]);
               vacc9 = _mm512_set1_ps(bias[ih + 3]);
            } else {
                vacc0 = _mm512_setzero_ps();
                vacc3 = _mm512_setzero_ps();
                vacc6 = _mm512_setzero_ps();
                vacc9 = _mm512_setzero_ps();
            }
            vacc1 = vacc0;
            vacc4 = vacc3;
            vacc7 = vacc6;
            vacc10 = vacc9;
            unsigned int lElement = *nnz++;

            __m512 va0_15_swap = _mm512_loadu_ps(a);
            __m512 va16_31_swap = _mm512_loadu_ps(a + 16);
            const int diff = *dataOffset++;
            a = a + diff;

            // __m512 w0_swap = _mm512_set1_ps(*(w)); // donot work. should try 2-way segement iteration
            // __m512 w1_swap = _mm512_set1_ps(*(w + 1));
            // __m512 w2_swap = _mm512_set1_ps(*(w + 2));
            // __m512 w3_swap = _mm512_set1_ps(*(w + 3));

            for (auto il = 0; il < lElement; il++) {
              // __m512 va0_15_ = _mm512_loadu_ps(a);
              // __m512 va16_31_ = _mm512_loadu_ps(a + 16);
                __m512 va0_15 = va0_15_swap;
                __m512 va16_31 = va16_31_swap;

                __m512 w0 = _mm512_set1_ps(*(w));
                __m512 w1 = _mm512_set1_ps(*(w + 1));
                __m512 w2 = _mm512_set1_ps(*(w + 2));
                __m512 w3 = _mm512_set1_ps(*(w + 3));
                w += sparseBlockOC;
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");

                vacc0 =  _mm512_fmadd_ps(va0_15, w0, vacc0);
                vacc1 =  _mm512_fmadd_ps(va16_31, w0, vacc1);
                va0_15_swap = _mm512_loadu_ps(a);
                va16_31_swap = _mm512_loadu_ps(a + 16);
                vacc3 =  _mm512_fmadd_ps(va0_15, w1, vacc3);
                vacc4 =  _mm512_fmadd_ps(va16_31, w1, vacc4);

                const int diff = *dataOffset++;
                a = a + diff;

                vacc6 =  _mm512_fmadd_ps(va0_15, w2, vacc6);
                vacc7 =  _mm512_fmadd_ps(va16_31, w2, vacc7);
                vacc9 =  _mm512_fmadd_ps(va0_15, w3, vacc9);
                vacc10 =  _mm512_fmadd_ps(va16_31, w3, vacc10);

            }
            dataOffset--;
            a = a - (*dataOffset);
            vacc0 = _mm512_min_ps(vacc0, vmax);
            vacc1 = _mm512_min_ps(vacc1, vmax);
            vacc3 = _mm512_min_ps(vacc3, vmax);
            vacc4 = _mm512_min_ps(vacc4, vmax);
            vacc6 = _mm512_min_ps(vacc6, vmax);
            vacc7 = _mm512_min_ps(vacc7, vmax);
            vacc9 = _mm512_min_ps(vacc9, vmax);
            vacc10 = _mm512_min_ps(vacc10, vmax);

            vacc0 = _mm512_max_ps(vacc0, vmin);
            vacc1 = _mm512_max_ps(vacc1, vmin);
            vacc3 = _mm512_max_ps(vacc3, vmin);
            vacc4 = _mm512_max_ps(vacc4, vmin);
            vacc6 = _mm512_max_ps(vacc6, vmin);
            vacc7 = _mm512_max_ps(vacc7, vmin);
            vacc9 = _mm512_max_ps(vacc9, vmin);
            vacc10 = _mm512_max_ps(vacc10, vmin);

            TRANSPOSE4x4_STORE(c, 0, 0, packCUnit, vacc0, vacc3, vacc6, vacc9);
            TRANSPOSE4x4_STORE(c, 0, 1, packCUnit, vacc0, vacc3, vacc6, vacc9);
            TRANSPOSE4x4_STORE(c, 0, 2, packCUnit, vacc0, vacc3, vacc6, vacc9);
            TRANSPOSE4x4_STORE(c, 0, 3, packCUnit, vacc0, vacc3, vacc6, vacc9);

            TRANSPOSE4x4_STORE(c, 1, 0, packCUnit, vacc1, vacc4, vacc7, vacc10);
            TRANSPOSE4x4_STORE(c, 1, 1, packCUnit, vacc1, vacc4, vacc7, vacc10);
            TRANSPOSE4x4_STORE(c, 1, 2, packCUnit, vacc1, vacc4, vacc7, vacc10);
            TRANSPOSE4x4_STORE(c, 1, 3, packCUnit, vacc1, vacc4, vacc7, vacc10);
        }
        blockC += (h >> packCUnitLog) * cStride;
        for (; ih < h; ih++) {
            auto c = blockC + ih % packCUnit;
            __m512 vacc0 =  nullptr != bias ? _mm512_set1_ps(*(bias + ih)) : _mm512_setzero_ps();
            __m512 vacc1 = vacc0;
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                __m512 va0_15 = _mm512_loadu_ps(a);
                __m512 va16_31 = _mm512_loadu_ps(a + 16);
                __m512 w0 = _mm512_set1_ps(*w);

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc0 =  _mm512_fmadd_ps(va0_15, w0, vacc0);
                vacc1 =  _mm512_fmadd_ps(va16_31, w0, vacc1);
            }
            vacc0 = _mm512_min_ps(vacc0, vmax);
            vacc1 = _mm512_min_ps(vacc1, vmax);
            vacc0 = _mm512_max_ps(vacc0, vmin);
            vacc1 = _mm512_max_ps(vacc1, vmin);

            // how to store faster: st4 / transpose
            STORE_VECTOR_AS_COLUMN(c, 0, packCUnit, vacc0);
            STORE_VECTOR_AS_COLUMN(c, 1, packCUnit, vacc1);
        }
        ie += 32;
        a += 32;
    }
    if (taileSize & 0x10) { // tail eSize bitmask 16
        // MNN_PRINT("caculate 16\n");
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << packCUnitLog);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~(sparseBlockOC - 1))); ih += sparseBlockOC) {
            auto c = blockC + (ih >> packCUnitLog) * cStride + (ih % packCUnit);

            __m512 vacc0, vacc3, vacc6, vacc9;
            // tobe merged in to weight data
            if (bias) {
               vacc0 = _mm512_set1_ps(bias[ih]);
               vacc3 = _mm512_set1_ps(bias[ih + 1]);
               vacc6 = _mm512_set1_ps(bias[ih + 2]);
               vacc9 = _mm512_set1_ps(bias[ih + 3]);
            } else {
                vacc0 = _mm512_setzero_ps();
                vacc3 = _mm512_setzero_ps();
                vacc6 = _mm512_setzero_ps();
                vacc9 = _mm512_setzero_ps();
            }
            unsigned int lElement = *nnz++;

            __m512 va0_15_swap = _mm512_loadu_ps(a);
            const int diff = *dataOffset++;
            a = a + diff;

            // __m512 w0_swap = _mm512_set1_ps(*(w)); // donot work. should try 2-way segement iteration
            // __m512 w1_swap = _mm512_set1_ps(*(w + 1));
            // __m512 w2_swap = _mm512_set1_ps(*(w + 2));
            // __m512 w3_swap = _mm512_set1_ps(*(w + 3));

            for (auto il = 0; il < lElement; il++) {
              // __m512 va0_15_ = _mm512_loadu_ps(a);
                __m512 va0_15 = va0_15_swap;

                __m512 w0 = _mm512_set1_ps(*(w));
                __m512 w1 = _mm512_set1_ps(*(w + 1));
                __m512 w2 = _mm512_set1_ps(*(w + 2));
                __m512 w3 = _mm512_set1_ps(*(w + 3));
                w += sparseBlockOC;
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");

                vacc0 =  _mm512_fmadd_ps(va0_15, w0, vacc0);
                va0_15_swap = _mm512_loadu_ps(a);
                const int diff = *dataOffset++;
                a = a + diff;
                vacc3 =  _mm512_fmadd_ps(va0_15, w1, vacc3);
                vacc6 =  _mm512_fmadd_ps(va0_15, w2, vacc6);
                vacc9 =  _mm512_fmadd_ps(va0_15, w3, vacc9);

            }
            dataOffset--;
            a = a - (*dataOffset);
            vacc0 = _mm512_min_ps(vacc0, vmax);
            vacc3 = _mm512_min_ps(vacc3, vmax);
            vacc6 = _mm512_min_ps(vacc6, vmax);
            vacc9 = _mm512_min_ps(vacc9, vmax);

            vacc0 = _mm512_max_ps(vacc0, vmin);
            vacc3 = _mm512_max_ps(vacc3, vmin);
            vacc6 = _mm512_max_ps(vacc6, vmin);
            vacc9 = _mm512_max_ps(vacc9, vmin);

            TRANSPOSE4x4_STORE(c, 0, 0, packCUnit, vacc0, vacc3, vacc6, vacc9);
            TRANSPOSE4x4_STORE(c, 0, 1, packCUnit, vacc0, vacc3, vacc6, vacc9);
            TRANSPOSE4x4_STORE(c, 0, 2, packCUnit, vacc0, vacc3, vacc6, vacc9);
            TRANSPOSE4x4_STORE(c, 0, 3, packCUnit, vacc0, vacc3, vacc6, vacc9);
        }
        blockC += (h >> packCUnitLog) * cStride;
        for (; ih < h; ih++) {
            auto c = blockC + ih % packCUnit;
            __m512 vacc0 =  nullptr != bias ? _mm512_set1_ps(*(bias + ih)) : _mm512_setzero_ps();
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                __m512 va0_15 = _mm512_loadu_ps(a);
                __m512 w0 = _mm512_set1_ps(*w);

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc0 =  _mm512_fmadd_ps(va0_15, w0, vacc0);
            }
            vacc0 = _mm512_min_ps(vacc0, vmax);
            vacc0 = _mm512_max_ps(vacc0, vmin);

            // how to store faster: st4 / transpose
            STORE_VECTOR_AS_COLUMN(c, 0, packCUnit, vacc0);
        }
        ie += 16;
        a += 16;
    }
    if (taileSize & 0x08) { // tail eSize bitmask 8
        // MNN_PRINT("caculate 8\n");
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << packCUnitLog);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~(sparseBlockOC - 1))); ih += sparseBlockOC) {
            auto c = blockC + (ih >> packCUnitLog) * cStride + (ih % packCUnit);

            __m256 vacc0, vacc3, vacc6, vacc9;
            // tobe merged in to weight data
            if (bias) {
               vacc0 = _mm256_set1_ps(bias[ih]);
               vacc3 = _mm256_set1_ps(bias[ih + 1]);
               vacc6 = _mm256_set1_ps(bias[ih + 2]);
               vacc9 = _mm256_set1_ps(bias[ih + 3]);
            } else {
                vacc0 = _mm256_setzero_ps();
                vacc3 = _mm256_setzero_ps();
                vacc6 = _mm256_setzero_ps();
                vacc9 = _mm256_setzero_ps();
            }
            unsigned int lElement = *nnz++;

            __m256 va0_15_swap = _mm256_loadu_ps(a);
            const int diff = *dataOffset++;
            a = a + diff;

            // __m256 w0_swap = _mm256_set1_ps(*(w)); // donot work. should try 2-way segement iteration
            // __m256 w1_swap = _mm256_set1_ps(*(w + 1));
            // __m256 w2_swap = _mm256_set1_ps(*(w + 2));
            // __m256 w3_swap = _mm256_set1_ps(*(w + 3));

            for (auto il = 0; il < lElement; il++) {
              // __m256 va0_15_ = _mm256_loadu_ps(a);
                __m256 va0_15 = va0_15_swap;

                __m256 w0 = _mm256_set1_ps(*(w));
                __m256 w1 = _mm256_set1_ps(*(w + 1));
                __m256 w2 = _mm256_set1_ps(*(w + 2));
                __m256 w3 = _mm256_set1_ps(*(w + 3));
                w += sparseBlockOC;
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");

                vacc0 =  _mm256_fmadd_ps(va0_15, w0, vacc0);
                va0_15_swap = _mm256_loadu_ps(a);
                const int diff = *dataOffset++;
                a = a + diff;
                vacc3 =  _mm256_fmadd_ps(va0_15, w1, vacc3);
                vacc6 =  _mm256_fmadd_ps(va0_15, w2, vacc6);
                vacc9 =  _mm256_fmadd_ps(va0_15, w3, vacc9);

            }
            dataOffset--;
            a = a - (*dataOffset);
            vacc0 = _mm256_min_ps(vacc0, _mm512_extractf32x8_ps(vmax, 0));
            vacc3 = _mm256_min_ps(vacc3, _mm512_extractf32x8_ps(vmax, 0));
            vacc6 = _mm256_min_ps(vacc6, _mm512_extractf32x8_ps(vmax, 0));
            vacc9 = _mm256_min_ps(vacc9, _mm512_extractf32x8_ps(vmax, 0));

            vacc0 = _mm256_max_ps(vacc0, _mm512_extractf32x8_ps(vmin, 0));
            vacc3 = _mm256_max_ps(vacc3, _mm512_extractf32x8_ps(vmin, 0));
            vacc6 = _mm256_max_ps(vacc6, _mm512_extractf32x8_ps(vmin, 0));
            vacc9 = _mm256_max_ps(vacc9, _mm512_extractf32x8_ps(vmin, 0));

            TRANSPOSE_M256_4x4_STORE(c, 0, packCUnit, vacc0, vacc3, vacc6, vacc9);
            TRANSPOSE_M256_4x4_STORE(c, 1, packCUnit, vacc0, vacc3, vacc6, vacc9);

        }
        blockC += (h >> packCUnitLog) * cStride;
        for (; ih < h; ih++) {
            auto c = blockC + ih % packCUnit;
            __m256 vacc0 =  nullptr != bias ? _mm256_set1_ps(*(bias + ih)) : _mm256_setzero_ps();
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                __m256 va0_15 = _mm256_loadu_ps(a);
                __m256 w0 = _mm256_set1_ps(*w);

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc0 =  _mm256_fmadd_ps(va0_15, w0, vacc0);
            }
            vacc0 = _mm256_min_ps(vacc0, _mm512_extractf32x8_ps(vmax, 0));
            vacc0 = _mm256_max_ps(vacc0, _mm512_extractf32x8_ps(vmin, 0));

            // how to store faster: st4 / transpose
            STORE_M256_VECTOR_AS_COLUMN(c, packCUnit, vacc0);
        }
        ie += 8;
        a += 8;
    }
    if (taileSize & 0x04) { // tail eSize bitmask 4
        // MNN_PRINT("caculate 8\n");
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << packCUnitLog);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~(sparseBlockOC - 1))); ih += sparseBlockOC) {
            auto c = blockC + (ih >> packCUnitLog) * cStride + (ih % packCUnit);

            __m128 vacc0, vacc3, vacc6, vacc9;
            // tobe merged in to weight data
            if (bias) {
               vacc0 = _mm_set1_ps(bias[ih]);
               vacc3 = _mm_set1_ps(bias[ih + 1]);
               vacc6 = _mm_set1_ps(bias[ih + 2]);
               vacc9 = _mm_set1_ps(bias[ih + 3]);
            } else {
                vacc0 = _mm_setzero_ps();
                vacc3 = _mm_setzero_ps();
                vacc6 = _mm_setzero_ps();
                vacc9 = _mm_setzero_ps();
            }
            unsigned int lElement = *nnz++;

            __m128 va0_15_swap = _mm_loadu_ps(a);
            const int diff = *dataOffset++;
            a = a + diff;

            // __m128 w0_swap = _mm256_set1_ps(*(w)); // donot work. should try 2-way segement iteration
            // __m128 w1_swap = _mm256_set1_ps(*(w + 1));
            // __m128 w2_swap = _mm256_set1_ps(*(w + 2));
            // __m128 w3_swap = _mm256_set1_ps(*(w + 3));

            for (auto il = 0; il < lElement; il++) {
              // __m128 va0_15_ = _mm256_loadu_ps(a);
                __m128 va0_15 = va0_15_swap;

                __m128 w0 = _mm_set1_ps(*(w));
                __m128 w1 = _mm_set1_ps(*(w + 1));
                __m128 w2 = _mm_set1_ps(*(w + 2));
                __m128 w3 = _mm_set1_ps(*(w + 3));
                w += sparseBlockOC;
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");

                vacc0 =  _mm_fmadd_ps(va0_15, w0, vacc0);
                va0_15_swap = _mm_loadu_ps(a);
                const int diff = *dataOffset++;
                a = a + diff;
                vacc3 =  _mm_fmadd_ps(va0_15, w1, vacc3);
                vacc6 =  _mm_fmadd_ps(va0_15, w2, vacc6);
                vacc9 =  _mm_fmadd_ps(va0_15, w3, vacc9);

            }
            dataOffset--;
            a = a - (*dataOffset);
            vacc0 = _mm_min_ps(vacc0, _mm512_extractf32x4_ps(vmax, 0));
            vacc3 = _mm_min_ps(vacc3, _mm512_extractf32x4_ps(vmax, 0));
            vacc6 = _mm_min_ps(vacc6, _mm512_extractf32x4_ps(vmax, 0));
            vacc9 = _mm_min_ps(vacc9, _mm512_extractf32x4_ps(vmax, 0));

            vacc0 = _mm_max_ps(vacc0, _mm512_extractf32x4_ps(vmin, 0));
            vacc3 = _mm_max_ps(vacc3, _mm512_extractf32x4_ps(vmin, 0));
            vacc6 = _mm_max_ps(vacc6, _mm512_extractf32x4_ps(vmin, 0));
            vacc9 = _mm_max_ps(vacc9, _mm512_extractf32x4_ps(vmin, 0));

            _MM_TRANSPOSE4_PS(vacc0, vacc3, vacc6, vacc9);
            _mm_storeu_ps(c, vacc0);
            _mm_storeu_ps(c + packCUnit, vacc3);
            _mm_storeu_ps(c + packCUnit * 2, vacc6);
            _mm_storeu_ps(c + packCUnit * 3, vacc9);
        }

        blockC += (h >> packCUnitLog) * cStride;
        for (; ih < h; ih++) {
            auto c = blockC + ih % packCUnit;
            __m128 vacc0 =  nullptr != bias ? _mm_set1_ps(*(bias + ih)) : _mm_setzero_ps();
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                __m128 va0_15 = _mm_loadu_ps(a);
                __m128 w0 = _mm_set1_ps(*w);

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc0 =  _mm_fmadd_ps(va0_15, w0, vacc0);
            }
            vacc0 = _mm_min_ps(vacc0, _mm512_extractf32x4_ps(vmax, 0));
            vacc0 = _mm_max_ps(vacc0, _mm512_extractf32x4_ps(vmin, 0));

            union {
                __m128 v;
                float f[4];
            } vacc0_u;
            vacc0_u.v = vacc0;
            c[0] = vacc0_u.f[0];
            c[packCUnit] = vacc0_u.f[1];
            c[packCUnit * 2] = vacc0_u.f[2];
            c[+packCUnit * 3] = vacc0_u.f[3];
        }
        ie += 4;
        a += 4;
    }
    if (taileSize & 0x02) { // tail eSize bitmask 2
        // MNN_PRINT("caculate 8\n");
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << packCUnitLog);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~(sparseBlockOC - 1))); ih += sparseBlockOC) {
            auto c = blockC + (ih >> packCUnitLog) * cStride + (ih % packCUnit);

            __m128 vacc0, vacc1;
            // tobe merged in to weight data
            if (bias) {
               vacc0 = _mm_loadu_ps(bias + ih);
            } else {
                vacc0 = _mm_setzero_ps();
            }
            vacc1 = vacc0;
            unsigned int lElement = *nnz++;

            __m128 va0_swap = _mm_set1_ps(*(a));
            __m128 va1_swap = _mm_set1_ps(*(a + 1));
            const int diff = *dataOffset++;
            a = a + diff;

            for (auto il = 0; il < lElement; il++) {
              // __m128 va0_15_ = _mm256_loadu_ps(a);
                __m128 va0 = va0_swap;
                __m128 va1 = va1_swap;

                __m128 w0_4 = _mm_loadu_ps(w);
                w += sparseBlockOC;
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");

                vacc0 =  _mm_fmadd_ps(va0, w0_4, vacc0);
                va0_swap = _mm_set1_ps(*(a));
                va1_swap = _mm_set1_ps(*(a + 1));
                const int diff = *dataOffset++;
                a = a + diff;
                vacc1 = _mm_fmadd_ps(va1, w0_4, vacc1);
            }
            dataOffset--;
            a = a - (*dataOffset);
            vacc0 = _mm_min_ps(vacc0, _mm512_extractf32x4_ps(vmax, 0));
            vacc1 = _mm_min_ps(vacc1, _mm512_extractf32x4_ps(vmax, 0));

            vacc0 = _mm_max_ps(vacc0, _mm512_extractf32x4_ps(vmin, 0));
            vacc1 = _mm_max_ps(vacc1, _mm512_extractf32x4_ps(vmin, 0));

            // transpose is omitted
            _mm_storeu_ps(c, vacc0);
            _mm_storeu_ps(c + packCUnit, vacc1);
        }

        blockC += (h >> packCUnitLog) * cStride;
        for (; ih < h; ih++) {
            auto c = blockC + ih % packCUnit;
            __m128 vacc0 =  nullptr != bias ? _mm_set1_ps(*(bias + ih)) : _mm_setzero_ps();
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                __m128 va0_15 = _mm_loadu_ps(a);
                __m128 w0 = _mm_set1_ps(*w);

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B, c - C, *w);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                w++;
                a = a + diff;
                vacc0 =  _mm_fmadd_ps(va0_15, w0, vacc0);
            }
            vacc0 = _mm_min_ps(vacc0, _mm512_extractf32x4_ps(vmax, 0));
            vacc0 = _mm_max_ps(vacc0, _mm512_extractf32x4_ps(vmin, 0));

            union {
                __m128 v;
                float f[4];
            } vacc0_u;
            vacc0_u.v = vacc0;
            c[0] = vacc0_u.f[0];
            c[packCUnit] = vacc0_u.f[1];
        }
        ie += 2;
        a += 2;
    }
    if (taileSize & 0x01) { // tail eSize bitmask 1
        // MNN_PRINT("caculate 8\n");
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << packCUnitLog);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~(sparseBlockOC - 1))); ih += sparseBlockOC) {
            auto c = blockC + (ih >> packCUnitLog) * cStride + (ih % packCUnit);

            __m128 vacc0;
            // tobe merged in to weight data
            if (bias) {
               vacc0 = _mm_loadu_ps(bias + ih);
            } else {
                vacc0 = _mm_setzero_ps();
            }
            unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
              // __m128 va0_15_ = _mm256_loadu_ps(a);
                __m128 va0 = _mm_set1_ps(*(a));
                __m128 w0_4 = _mm_loadu_ps(w);
                w += sparseBlockOC;
                vacc0 =  _mm_fmadd_ps(va0, w0_4, vacc0);
                const int diff = *dataOffset++;
                a = a + diff;

            }
            vacc0 = _mm_min_ps(vacc0, _mm512_extractf32x4_ps(vmax, 0));
            vacc0 = _mm_max_ps(vacc0, _mm512_extractf32x4_ps(vmin, 0));

            // transpose is omitted
            _mm_storeu_ps(c, vacc0);
        }
        blockC += (h >> packCUnitLog) * cStride;
        for (; ih < h; ih++) {
            auto c = blockC + ih % packCUnit;
            float acc0 =  nullptr != bias ? *(bias + ih) : 0;
            const unsigned int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                acc0 += (*a) * (*w);
                w++;
                a = a + diff;
            }
            float minValue = *(postParameters + 2);
            float maxValue = *(postParameters + 3);
            acc0 =  std::max(std::min(maxValue, acc0), minValue);
            c[0] = acc0;
        }
    }

    return;
}

