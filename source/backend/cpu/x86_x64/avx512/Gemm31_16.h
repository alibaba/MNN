
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdint.h>
#include "DynamicGemm.h"


template <int InputTile>
void _AVX512_MNNPackedMatMulO16FullLoadKernel(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias) {

#define REDUCE_MUL_ADD(ick)                                                                                \
    zmm0 = _mm512_loadu_ps(filterICPtr + ick * bStride);                                                   \
    if (InputTile > 8)                                                                                     \
        _mm_prefetch(filterICPtr + ick * bStride + AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT, _MM_HINT_T0);  \
    if (InputTile > 0)                                                                                     \
        zmm1 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 0 * AVX512_PACK_C_UNIT]), zmm0, zmm1);    \
    if (InputTile > 1)                                                                                     \
        zmm2 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 1 * AVX512_PACK_C_UNIT]), zmm0, zmm2);    \
    if (InputTile > 2)                                                                                     \
        zmm3 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 2 * AVX512_PACK_C_UNIT]), zmm0, zmm3);    \
    if (InputTile > 3)                                                                                     \
        zmm4 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 3 * AVX512_PACK_C_UNIT]), zmm0, zmm4);    \
    if (InputTile > 4)                                                                                     \
        zmm5 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 4 * AVX512_PACK_C_UNIT]), zmm0, zmm5);    \
    if (InputTile > 5)                                                                                     \
        zmm6 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 5 * AVX512_PACK_C_UNIT]), zmm0, zmm6);    \
    if (InputTile > 6)                                                                                     \
        zmm7 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 6 * AVX512_PACK_C_UNIT]), zmm0, zmm7);    \
    if (InputTile > 7)                                                                                     \
        zmm8 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 7 * AVX512_PACK_C_UNIT]), zmm0, zmm8);    \
    if (InputTile > 8)                                                                                     \
        zmm9 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 8 * AVX512_PACK_C_UNIT]), zmm0, zmm9);    \
    if (InputTile > 9)                                                                                     \
        zmm10 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 9 * AVX512_PACK_C_UNIT]), zmm0, zmm10);  \
    if (InputTile > 10)                                                                                    \
        zmm11 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 10 * AVX512_PACK_C_UNIT]), zmm0, zmm11); \
    if (InputTile > 11)                                                                                    \
        zmm12 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 11 * AVX512_PACK_C_UNIT]), zmm0, zmm12); \
    if (InputTile > 12)                                                                                    \
        zmm13 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 12 * AVX512_PACK_C_UNIT]), zmm0, zmm13); \
    if (InputTile > 13)                                                                                    \
        zmm14 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 13 * AVX512_PACK_C_UNIT]), zmm0, zmm14); \
    if (InputTile > 14)                                                                                    \
        zmm15 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 14 * AVX512_PACK_C_UNIT]), zmm0, zmm15); \
    if (InputTile > 15)                                                                                    \
        zmm16 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 15 * AVX512_PACK_C_UNIT]), zmm0, zmm16); \
    if (InputTile > 16)                                                                                    \
        zmm17 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 16 * AVX512_PACK_C_UNIT]), zmm0, zmm17); \
    if (InputTile > 17)                                                                                    \
        zmm18 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 17 * AVX512_PACK_C_UNIT]), zmm0, zmm18); \
    if (InputTile > 18)                                                                                    \
        zmm19 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 18 * AVX512_PACK_C_UNIT]), zmm0, zmm19); \
    if (InputTile > 19)                                                                                    \
        zmm20 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 19 * AVX512_PACK_C_UNIT]), zmm0, zmm20); \
    if (InputTile > 20)                                                                                    \
        zmm21 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 20 * AVX512_PACK_C_UNIT]), zmm0, zmm21); \
    if (InputTile > 21)                                                                                    \
        zmm22 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 21 * AVX512_PACK_C_UNIT]), zmm0, zmm22); \
    if (InputTile > 22)                                                                                    \
        zmm23 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 22 * AVX512_PACK_C_UNIT]), zmm0, zmm23); \
    if (InputTile > 23)                                                                                    \
        zmm24 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 23 * AVX512_PACK_C_UNIT]), zmm0, zmm24); \
    if (InputTile > 24)                                                                                    \
        zmm25 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 24 * AVX512_PACK_C_UNIT]), zmm0, zmm25); \
    if (InputTile > 25)                                                                                    \
        zmm26 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 25 * AVX512_PACK_C_UNIT]), zmm0, zmm26); \
    if (InputTile > 26)                                                                                    \
        zmm27 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 26 * AVX512_PACK_C_UNIT]), zmm0, zmm27); \
    if (InputTile > 27)                                                                                    \
        zmm28 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 27 * AVX512_PACK_C_UNIT]), zmm0, zmm28); \
    if (InputTile > 28)                                                                                    \
        zmm29 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 28 * AVX512_PACK_C_UNIT]), zmm0, zmm29); \
    if (InputTile > 29)                                                                                    \
        zmm30 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 29 * AVX512_PACK_C_UNIT]), zmm0, zmm30); \
    if (InputTile > 30)                                                                                    \
        zmm31 = _mm512_fmadd_ps(_mm512_set1_ps(inputICPtr[(ick) + 30 * AVX512_PACK_C_UNIT]), zmm0, zmm31);

    auto aStride      = parameter[0] / sizeof(float);
    auto l            = parameter[1];
    auto h            = parameter[2];
    auto cStride      = parameter[3] / sizeof(float);
    auto bStride      = parameter[5] / sizeof(float);
    int aTotal        = parameter[6];

    auto icTail = l % AVX512_PACK_C_UNIT;
    auto icPack = l - icTail;
    auto inputTilePtr = A;
    auto destPtr = C;
    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    for(; aTotal > 0; aTotal -= InputTile) {
        auto inputPtr = inputTilePtr;
        auto filterPtr = B;
        auto biasPtr = bias;
        if (biasPtr) {
            if (InputTile > 0 ) zmm1 = _mm512_loadu_ps(biasPtr);
            if (InputTile > 1 ) zmm2  = zmm1;
            if (InputTile > 2 ) zmm3  = zmm1;
            if (InputTile > 3 ) zmm4  = zmm1;
            if (InputTile > 4 ) zmm5  = zmm1;
            if (InputTile > 5 ) zmm6  = zmm1;
            if (InputTile > 6 ) zmm7  = zmm1;
            if (InputTile > 7 ) zmm8  = zmm1;
            if (InputTile > 8 ) zmm9  = zmm1;
            if (InputTile > 9 ) zmm10 = zmm1;
            if (InputTile > 10) zmm11 = zmm1;
            if (InputTile > 11) zmm12 = zmm1;
            if (InputTile > 12) zmm13 = zmm1;
            if (InputTile > 13) zmm14 = zmm1;
            if (InputTile > 14) zmm15 = zmm1;
            if (InputTile > 15) zmm16 = zmm1;
            if (InputTile > 16) zmm17 = zmm1;
            if (InputTile > 17) zmm18 = zmm1;
            if (InputTile > 18) zmm19 = zmm1;
            if (InputTile > 19) zmm20 = zmm1;
            if (InputTile > 20) zmm21 = zmm1;
            if (InputTile > 21) zmm22 = zmm1;
            if (InputTile > 22) zmm23 = zmm1;
            if (InputTile > 23) zmm24 = zmm1;
            if (InputTile > 24) zmm25 = zmm1;
            if (InputTile > 25) zmm26 = zmm1;
            if (InputTile > 26) zmm27 = zmm1;
            if (InputTile > 27) zmm28 = zmm1;
            if (InputTile > 28) zmm29 = zmm1;
            if (InputTile > 29) zmm30 = zmm1;
            if (InputTile > 30) zmm31 = zmm1;
        } else {
            if (InputTile > 0 ) zmm1  = _mm512_setzero_ps();
            if (InputTile > 1 ) zmm2  = _mm512_setzero_ps();
            if (InputTile > 2 ) zmm3  = _mm512_setzero_ps();
            if (InputTile > 3 ) zmm4  = _mm512_setzero_ps();
            if (InputTile > 4 ) zmm5  = _mm512_setzero_ps();
            if (InputTile > 5 ) zmm6  = _mm512_setzero_ps();
            if (InputTile > 6 ) zmm7  = _mm512_setzero_ps();
            if (InputTile > 7 ) zmm8  = _mm512_setzero_ps();
            if (InputTile > 8 ) zmm9  = _mm512_setzero_ps();
            if (InputTile > 9 ) zmm10 = _mm512_setzero_ps();
            if (InputTile > 10) zmm11 = _mm512_setzero_ps();
            if (InputTile > 11) zmm12 = _mm512_setzero_ps();
            if (InputTile > 12) zmm13 = _mm512_setzero_ps();
            if (InputTile > 13) zmm14 = _mm512_setzero_ps();
            if (InputTile > 14) zmm15 = _mm512_setzero_ps();
            if (InputTile > 15) zmm16 = _mm512_setzero_ps();
            if (InputTile > 16) zmm17 = _mm512_setzero_ps();
            if (InputTile > 17) zmm18 = _mm512_setzero_ps();
            if (InputTile > 18) zmm19 = _mm512_setzero_ps();
            if (InputTile > 19) zmm20 = _mm512_setzero_ps();
            if (InputTile > 20) zmm21 = _mm512_setzero_ps();
            if (InputTile > 21) zmm22 = _mm512_setzero_ps();
            if (InputTile > 22) zmm23 = _mm512_setzero_ps();
            if (InputTile > 23) zmm24 = _mm512_setzero_ps();
            if (InputTile > 24) zmm25 = _mm512_setzero_ps();
            if (InputTile > 25) zmm26 = _mm512_setzero_ps();
            if (InputTile > 26) zmm27 = _mm512_setzero_ps();
            if (InputTile > 27) zmm28 = _mm512_setzero_ps();
            if (InputTile > 28) zmm29 = _mm512_setzero_ps();
            if (InputTile > 29) zmm30 = _mm512_setzero_ps();
            if (InputTile > 30) zmm31 = _mm512_setzero_ps();
        }

        for(int il = 0; il < icPack; il += AVX512_PACK_C_UNIT) {
            auto inputICPtr = inputPtr;
            auto filterICPtr = filterPtr;

            // REDUCE_MUL_ADD(0 );
            // REDUCE_MUL_ADD(1 );
            // REDUCE_MUL_ADD(2 );
            // REDUCE_MUL_ADD(3 );
            // REDUCE_MUL_ADD(4 );
            // REDUCE_MUL_ADD(5 );
            // REDUCE_MUL_ADD(6 );
            // REDUCE_MUL_ADD(7 );
            // REDUCE_MUL_ADD(8 );
            // REDUCE_MUL_ADD(9 );
            // REDUCE_MUL_ADD(10);
            // REDUCE_MUL_ADD(11);
            // REDUCE_MUL_ADD(12);
            // REDUCE_MUL_ADD(13);
            // REDUCE_MUL_ADD(14);
            // REDUCE_MUL_ADD(15);

            for (int ick = 0; ick < AVX512_PACK_C_UNIT; ++ick) {
                REDUCE_MUL_ADD(ick);
            }

            inputPtr += InputTile * AVX512_PACK_C_UNIT;
            filterPtr += bStride * AVX512_PACK_C_UNIT;
        }

        auto inputICPtr = inputPtr;
        auto filterICPtr = filterPtr;
        float out[16] = {0};
        for(int ick = 0; ick < icTail; ++ick) {
            REDUCE_MUL_ADD(ick);
        }

        // write
        // oc < 16;
        if (InputTile > 0 ) _mm512_storeu_ps(destPtr + 0  * AVX512_PACK_C_UNIT, zmm1 );
        if (InputTile > 1 ) _mm512_storeu_ps(destPtr + 1  * AVX512_PACK_C_UNIT, zmm2 );
        if (InputTile > 2 ) _mm512_storeu_ps(destPtr + 2  * AVX512_PACK_C_UNIT, zmm3 );
        if (InputTile > 3 ) _mm512_storeu_ps(destPtr + 3  * AVX512_PACK_C_UNIT, zmm4 );
        if (InputTile > 4 ) _mm512_storeu_ps(destPtr + 4  * AVX512_PACK_C_UNIT, zmm5 );
        if (InputTile > 5 ) _mm512_storeu_ps(destPtr + 5  * AVX512_PACK_C_UNIT, zmm6 );
        if (InputTile > 6 ) _mm512_storeu_ps(destPtr + 6  * AVX512_PACK_C_UNIT, zmm7 );
        if (InputTile > 7 ) _mm512_storeu_ps(destPtr + 7  * AVX512_PACK_C_UNIT, zmm8 );
        if (InputTile > 8 ) _mm512_storeu_ps(destPtr + 8  * AVX512_PACK_C_UNIT, zmm9 );
        if (InputTile > 9 ) _mm512_storeu_ps(destPtr + 9  * AVX512_PACK_C_UNIT, zmm10);
        if (InputTile > 10) _mm512_storeu_ps(destPtr + 10 * AVX512_PACK_C_UNIT, zmm11);
        if (InputTile > 11) _mm512_storeu_ps(destPtr + 11 * AVX512_PACK_C_UNIT, zmm12);
        if (InputTile > 12) _mm512_storeu_ps(destPtr + 12 * AVX512_PACK_C_UNIT, zmm13);
        if (InputTile > 13) _mm512_storeu_ps(destPtr + 13 * AVX512_PACK_C_UNIT, zmm14);
        if (InputTile > 14) _mm512_storeu_ps(destPtr + 14 * AVX512_PACK_C_UNIT, zmm15);
        if (InputTile > 15) _mm512_storeu_ps(destPtr + 15 * AVX512_PACK_C_UNIT, zmm16);
        if (InputTile > 16) _mm512_storeu_ps(destPtr + 16 * AVX512_PACK_C_UNIT, zmm17);
        if (InputTile > 17) _mm512_storeu_ps(destPtr + 17 * AVX512_PACK_C_UNIT, zmm18);
        if (InputTile > 18) _mm512_storeu_ps(destPtr + 18 * AVX512_PACK_C_UNIT, zmm19);
        if (InputTile > 19) _mm512_storeu_ps(destPtr + 19 * AVX512_PACK_C_UNIT, zmm20);
        if (InputTile > 20) _mm512_storeu_ps(destPtr + 20 * AVX512_PACK_C_UNIT, zmm21);
        if (InputTile > 21) _mm512_storeu_ps(destPtr + 21 * AVX512_PACK_C_UNIT, zmm22);
        if (InputTile > 22) _mm512_storeu_ps(destPtr + 22 * AVX512_PACK_C_UNIT, zmm23);
        if (InputTile > 23) _mm512_storeu_ps(destPtr + 23 * AVX512_PACK_C_UNIT, zmm24);
        if (InputTile > 24) _mm512_storeu_ps(destPtr + 24 * AVX512_PACK_C_UNIT, zmm25);
        if (InputTile > 25) _mm512_storeu_ps(destPtr + 25 * AVX512_PACK_C_UNIT, zmm26);
        if (InputTile > 26) _mm512_storeu_ps(destPtr + 26 * AVX512_PACK_C_UNIT, zmm27);
        if (InputTile > 27) _mm512_storeu_ps(destPtr + 27 * AVX512_PACK_C_UNIT, zmm28);
        if (InputTile > 28) _mm512_storeu_ps(destPtr + 28 * AVX512_PACK_C_UNIT, zmm29);
        if (InputTile > 29) _mm512_storeu_ps(destPtr + 29 * AVX512_PACK_C_UNIT, zmm30);
        if (InputTile > 30) _mm512_storeu_ps(destPtr + 30 * AVX512_PACK_C_UNIT, zmm31);
        // oc < 32
        auto writeDestPtr = destPtr + cStride;
        inputTilePtr += aStride;
        destPtr += InputTile * AVX512_PACK_C_UNIT;
    }

#undef REDUCE_MUL_ADD
}


