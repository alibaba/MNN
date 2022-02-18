//
//  SparseKernelFunctionEpx1.cpp
//  MNN
//
//  Created by MNN on 2021/08/18.
//  Copyright © 2018 - 2021, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "SparseKernelFunction.hpp"

void _AVX512_MNNGetSparseMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 48;
    *lP = 1;
    *hP = 4;
    // hp is corresponding to sparse block along right matrix colum dimension. in ramdom sparse, it is 1.
    return;
}

void _AVX512_MNNPackedSparseMatMulEpx1(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                    const float* postParameters, const float* bias, unsigned int* NNZMap,
                                    int* dataOffsetMap) {

    auto eP = parameter[0] / sizeof(float);
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    auto aStride = eP * l; // sizeof(float);
    auto hC4 = UP_DIV(h, 4);

    constexpr size_t packCUnit = 16;
    constexpr size_t packCUnitLog = 4;
    constexpr int sparseBlockOC = 4;

    __m512 vmin = _mm512_set1_ps(*(postParameters + 2));
    __m512 vmax = _mm512_set1_ps(*(postParameters + 3));

    const float* a = A;
    size_t ie = 0;
    for (ie = 0; ie + eP <= eSize; ie += eP) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << packCUnitLog);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < h; ih++) {
            auto c = blockC + (ih >> packCUnitLog) * cStride + (ih % packCUnit);
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
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << packCUnitLog);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < h; ih++) {
            auto c = blockC + (ih >> packCUnitLog) * cStride + (ih % packCUnit);
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
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << packCUnitLog);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < h; ih++) {
            auto c = blockC + (ih >> packCUnitLog) * cStride + (ih % packCUnit);
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
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << packCUnitLog);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < h; ih++) {
            auto c = blockC + (ih >> packCUnitLog) * cStride + (ih % packCUnit);
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
    if (taileSize & 0x07) { // tail eSize bitmask 7, calculate all eP less than 8
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << packCUnitLog);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < h; ih++) {
            auto c = blockC + (ih >> packCUnitLog) * cStride + (ih % packCUnit);
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

            union {
                __m256 v;
                float f[8];
            } vacc0_u;
            vacc0_u.v = vacc0;
            // how to store faster: st4 / transpose
            for (auto iStore = 0; iStore < (taileSize & 0x07); iStore++) {
                 c[packCUnit * iStore] = vacc0_u.f[iStore];
            }
        }
        // ie += taileSize;
        // a += taileSize;
    }

  return;
}

