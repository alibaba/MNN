#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdint.h>

template <int InputTile>
void _AVX512_MNNPackedMatMulO32FullLoadKernel(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias) {

#ifdef _MSC_VER
#define _mm_prefetch(ptr, hint) _mm_prefetch((const char*)(ptr), hint)
#endif // _MSC_VER

#define REDUCE_MUL_ADD(ick)                                                                               \
    zmm0 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 0 * AVX512_PACK_C_UNIT));                       \
    zmm1 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 1 * AVX512_PACK_C_UNIT));                       \
    if (InputTile > 10) {                                                                                 \
        _mm_prefetch(filterICPtr + ick * bStride + 0 * AVX512_PACK_C_UNIT + bStride * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                        \
        _mm_prefetch(filterICPtr + ick * bStride + 1 * AVX512_PACK_C_UNIT + bStride * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                        \
    }                                                                                                     \
    if (InputTile > 0) {                                                                                  \
        zmm2  = _mm512_set1_ps(inputICPtr[(ick) + 0 * AVX512_PACK_C_UNIT]);                               \
        zmm12 = _mm512_fmadd_ps(zmm2, zmm0, zmm12);                                                       \
        zmm22 = _mm512_fmadd_ps(zmm2, zmm1, zmm22);                                                       \
    }                                                                                                     \
    if (InputTile > 1) {                                                                                  \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 1 * AVX512_PACK_C_UNIT]);                               \
        zmm13 = _mm512_fmadd_ps(zmm3, zmm0, zmm13);                                                       \
        zmm23 = _mm512_fmadd_ps(zmm3, zmm1, zmm23);                                                       \
    }                                                                                                     \
    if (InputTile > 2) {                                                                                  \
        zmm4  = _mm512_set1_ps(inputICPtr[(ick) + 2 * AVX512_PACK_C_UNIT]);                               \
        zmm14 = _mm512_fmadd_ps(zmm4, zmm0, zmm14);                                                       \
        zmm24 = _mm512_fmadd_ps(zmm4, zmm1, zmm24);                                                       \
    }                                                                                                     \
    if (InputTile > 3) {                                                                                  \
        zmm5  = _mm512_set1_ps(inputICPtr[(ick) + 3 * AVX512_PACK_C_UNIT]);                               \
        zmm15 = _mm512_fmadd_ps(zmm5, zmm0, zmm15);                                                       \
        zmm25 = _mm512_fmadd_ps(zmm5, zmm1, zmm25);                                                       \
    }                                                                                                     \
    if (InputTile > 4) {                                                                                  \
        zmm6  = _mm512_set1_ps(inputICPtr[(ick) + 4 * AVX512_PACK_C_UNIT]);                               \
        zmm16 = _mm512_fmadd_ps(zmm6, zmm0, zmm16);                                                       \
        zmm26 = _mm512_fmadd_ps(zmm6, zmm1, zmm26);                                                       \
    }                                                                                                     \
    if (InputTile > 5) {                                                                                  \
        zmm7  = _mm512_set1_ps(inputICPtr[(ick) + 5 * AVX512_PACK_C_UNIT]);                               \
        zmm17 = _mm512_fmadd_ps(zmm7, zmm0, zmm17);                                                       \
        zmm27 = _mm512_fmadd_ps(zmm7, zmm1, zmm27);                                                       \
    }                                                                                                     \
    if (InputTile > 6) {                                                                                  \
        zmm8  = _mm512_set1_ps(inputICPtr[(ick) + 6 * AVX512_PACK_C_UNIT]);                               \
        zmm18 = _mm512_fmadd_ps(zmm8, zmm0, zmm18);                                                       \
        zmm28 = _mm512_fmadd_ps(zmm8, zmm1, zmm28);                                                       \
    }                                                                                                     \
    if (InputTile > 7) {                                                                                  \
        zmm9  = _mm512_set1_ps(inputICPtr[(ick) + 7 * AVX512_PACK_C_UNIT]);                               \
        zmm19 = _mm512_fmadd_ps(zmm9, zmm0, zmm19);                                                       \
        zmm29 = _mm512_fmadd_ps(zmm9, zmm1, zmm29);                                                       \
    }                                                                                                     \
    if (InputTile > 8) {                                                                                  \
        zmm10 = _mm512_set1_ps(inputICPtr[(ick) + 8 * AVX512_PACK_C_UNIT]);                               \
        zmm20 = _mm512_fmadd_ps(zmm10, zmm0, zmm20);                                                      \
        zmm30 = _mm512_fmadd_ps(zmm10, zmm1, zmm30);                                                      \
    }                                                                                                     \
    if (InputTile > 9) {                                                                                  \
        zmm11 = _mm512_set1_ps(inputICPtr[(ick) + 9 * AVX512_PACK_C_UNIT]);                               \
        zmm21 = _mm512_fmadd_ps(zmm11, zmm0, zmm21);                                                      \
        zmm31 = _mm512_fmadd_ps(zmm11, zmm1, zmm31);                                                      \
    }                                                                                                     \
    if (InputTile > 10)                                                                                   \
        printf("InputTile size too large. in function:%s\n", __FUNCTION__);

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;
    auto aStride      = parameter[0] / sizeof(float);
    auto l            = parameter[1];
    auto h            = parameter[2];
    auto cStride      = parameter[3] / sizeof(float);
    auto srcUnitStride = parameter[4] / sizeof(float);
    auto bStride      = parameter[5] / sizeof(float);
    int aTotal        = parameter[6];

    auto icTail = l % AVX512_PACK_C_UNIT;
    auto icPack = l - icTail;

    auto inputTilePtr = A;
    auto destPtr = C;


    for(; aTotal > 0; aTotal -= InputTile) {

        auto inputPtr = inputTilePtr;
        auto filterPtr = B;
        auto biasPtr = bias;

        if (biasPtr) {
            if (InputTile > 0 ) {
                zmm12 = _mm512_loadu_ps(biasPtr);
                zmm22 = _mm512_loadu_ps(biasPtr + AVX512_PACK_C_UNIT);
            }
            if (InputTile > 1 ) {
                zmm13 = zmm12;
                zmm23 = zmm22;
            }
            if (InputTile > 2 ) {
                zmm14 = zmm12;
                zmm24 = zmm22;
            }
            if (InputTile > 3 ) {
                zmm15 = zmm12;
                zmm25 = zmm22;
            }
            if (InputTile > 4 ) {
                zmm16 = zmm12;
                zmm26 = zmm22;
            }
            if (InputTile > 5 ) {
                zmm17 = zmm12;
                zmm27 = zmm22;
            }
            if (InputTile > 6 ) {
                zmm18 = zmm12;
                zmm28 = zmm22;
            }
            if (InputTile > 7 ) {
                zmm19 = zmm12;
                zmm29 = zmm22;
            }
            if (InputTile > 8 ) {
                zmm20 = zmm12;
                zmm30 = zmm22;
            }
            if (InputTile > 9 ) {
                zmm21 = zmm12;
                zmm31 = zmm22;
            }
        } else {
            if (InputTile > 0) zmm12 = _mm512_setzero_ps();
            if (InputTile > 1) zmm13 = _mm512_setzero_ps();
            if (InputTile > 2) zmm14 = _mm512_setzero_ps();
            if (InputTile > 3) zmm15 = _mm512_setzero_ps();
            if (InputTile > 4) zmm16 = _mm512_setzero_ps();
            if (InputTile > 5) zmm17 = _mm512_setzero_ps();
            if (InputTile > 6) zmm18 = _mm512_setzero_ps();
            if (InputTile > 7) zmm19 = _mm512_setzero_ps();
            if (InputTile > 8) zmm20 = _mm512_setzero_ps();
            if (InputTile > 9) zmm21 = _mm512_setzero_ps();
            if (InputTile > 0) zmm22 = _mm512_setzero_ps();
            if (InputTile > 1) zmm23 = _mm512_setzero_ps();
            if (InputTile > 2) zmm24 = _mm512_setzero_ps();
            if (InputTile > 3) zmm25 = _mm512_setzero_ps();
            if (InputTile > 4) zmm26 = _mm512_setzero_ps();
            if (InputTile > 5) zmm27 = _mm512_setzero_ps();
            if (InputTile > 6) zmm28 = _mm512_setzero_ps();
            if (InputTile > 7) zmm29 = _mm512_setzero_ps();
            if (InputTile > 8) zmm30 = _mm512_setzero_ps();
            if (InputTile > 9) zmm31 = _mm512_setzero_ps();
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

            for (int ick = 0; ick < AVX512_PACK_C_UNIT; ick++) {
                REDUCE_MUL_ADD(ick);
            }

            inputPtr += InputTile * AVX512_PACK_C_UNIT;
            // filterPtr += 2 * AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT;
            filterPtr += bStride * AVX512_PACK_C_UNIT;
        }

        auto inputICPtr = inputPtr;
        auto filterICPtr = filterPtr;
        for(int ick = 0; ick < icTail; ++ick) {
            REDUCE_MUL_ADD(ick);
        }

        // write
        // oc < 16;
        if (InputTile > 0 ) _mm512_storeu_ps(destPtr + 0  * AVX512_PACK_C_UNIT, zmm12);
        if (InputTile > 1 ) _mm512_storeu_ps(destPtr + 1  * AVX512_PACK_C_UNIT, zmm13);
        if (InputTile > 2 ) _mm512_storeu_ps(destPtr + 2  * AVX512_PACK_C_UNIT, zmm14);
        if (InputTile > 3 ) _mm512_storeu_ps(destPtr + 3  * AVX512_PACK_C_UNIT, zmm15);
        if (InputTile > 4 ) _mm512_storeu_ps(destPtr + 4  * AVX512_PACK_C_UNIT, zmm16);
        if (InputTile > 5 ) _mm512_storeu_ps(destPtr + 5  * AVX512_PACK_C_UNIT, zmm17);
        if (InputTile > 6 ) _mm512_storeu_ps(destPtr + 6  * AVX512_PACK_C_UNIT, zmm18);
        if (InputTile > 7 ) _mm512_storeu_ps(destPtr + 7  * AVX512_PACK_C_UNIT, zmm19);
        if (InputTile > 8 ) _mm512_storeu_ps(destPtr + 8  * AVX512_PACK_C_UNIT, zmm20);
        if (InputTile > 9 ) _mm512_storeu_ps(destPtr + 9  * AVX512_PACK_C_UNIT, zmm21);

        // oc < 32
        auto writeDestPtr = destPtr + cStride;
        if (InputTile > 0 ) _mm512_storeu_ps(writeDestPtr + 0  * AVX512_PACK_C_UNIT, zmm22);
        if (InputTile > 1 ) _mm512_storeu_ps(writeDestPtr + 1  * AVX512_PACK_C_UNIT, zmm23);
        if (InputTile > 2 ) _mm512_storeu_ps(writeDestPtr + 2  * AVX512_PACK_C_UNIT, zmm24);
        if (InputTile > 3 ) _mm512_storeu_ps(writeDestPtr + 3  * AVX512_PACK_C_UNIT, zmm25);
        if (InputTile > 4 ) _mm512_storeu_ps(writeDestPtr + 4  * AVX512_PACK_C_UNIT, zmm26);
        if (InputTile > 5 ) _mm512_storeu_ps(writeDestPtr + 5  * AVX512_PACK_C_UNIT, zmm27);
        if (InputTile > 6 ) _mm512_storeu_ps(writeDestPtr + 6  * AVX512_PACK_C_UNIT, zmm28);
        if (InputTile > 7 ) _mm512_storeu_ps(writeDestPtr + 7  * AVX512_PACK_C_UNIT, zmm29);
        if (InputTile > 8 ) _mm512_storeu_ps(writeDestPtr + 8  * AVX512_PACK_C_UNIT, zmm30);
        if (InputTile > 9 ) _mm512_storeu_ps(writeDestPtr + 9  * AVX512_PACK_C_UNIT, zmm31);
        inputTilePtr += aStride;
        destPtr += InputTile * AVX512_PACK_C_UNIT;
    }

#undef REDUCE_MUL_ADD

}

template <int InputTile>
void _AVX512_MNNPackedMatMulO32Swaped6Kernel(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias) {

#define REDUCE_MUL_ADD(ick)                                                                                          \
    zmm0 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 0 * AVX512_PACK_C_UNIT));                                  \
    zmm1 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 1 * AVX512_PACK_C_UNIT));                                  \
    if (InputTile > 10) {                                                                                            \
        _mm_prefetch(filterICPtr + ick * bStride + 0 * AVX512_PACK_C_UNIT + AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                                   \
        _mm_prefetch(filterICPtr + ick * bStride + 1 * AVX512_PACK_C_UNIT + AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                                   \
        _mm_prefetch(inputICPtr + ick * AVX512_PACK_C_UNIT, _MM_HINT_T0);                                            \
    }                                                                                                                \
    if (InputTile > 0) {                                                                                             \
        zmm2  = _mm512_set1_ps(inputICPtr[(ick) + 0 * AVX512_PACK_C_UNIT]);                                          \
        zmm8  = _mm512_fmadd_ps(zmm2, zmm0, zmm8);                                                                   \
        zmm20 = _mm512_fmadd_ps(zmm2, zmm1, zmm20);                                                                  \
    }                                                                                                                \
    if (InputTile > 1) {                                                                                             \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 1 * AVX512_PACK_C_UNIT]);                                          \
        zmm9  = _mm512_fmadd_ps(zmm3, zmm0, zmm9);                                                                   \
        zmm21 = _mm512_fmadd_ps(zmm3, zmm1, zmm21);                                                                  \
    }                                                                                                                \
    if (InputTile > 2) {                                                                                             \
        zmm4  = _mm512_set1_ps(inputICPtr[(ick) + 2 * AVX512_PACK_C_UNIT]);                                          \
        zmm10 = _mm512_fmadd_ps(zmm4, zmm0, zmm10);                                                                  \
        zmm22 = _mm512_fmadd_ps(zmm4, zmm1, zmm22);                                                                  \
    }                                                                                                                \
    if (InputTile > 3) {                                                                                             \
        zmm5  = _mm512_set1_ps(inputICPtr[(ick) + 3 * AVX512_PACK_C_UNIT]);                                          \
        zmm11 = _mm512_fmadd_ps(zmm5, zmm0, zmm11);                                                                  \
        zmm23 = _mm512_fmadd_ps(zmm5, zmm1, zmm23);                                                                  \
    }                                                                                                                \
    if (InputTile > 4) {                                                                                             \
        zmm6  = _mm512_set1_ps(inputICPtr[(ick) + 4 * AVX512_PACK_C_UNIT]);                                          \
        zmm12 = _mm512_fmadd_ps(zmm6, zmm0, zmm12);                                                                  \
        zmm24 = _mm512_fmadd_ps(zmm6, zmm1, zmm24);                                                                  \
    }                                                                                                                \
    if (InputTile > 5) {                                                                                             \
        zmm7  = _mm512_set1_ps(inputICPtr[(ick) + 5 * AVX512_PACK_C_UNIT]);                                          \
        zmm13 = _mm512_fmadd_ps(zmm7, zmm0, zmm13);                                                                  \
        zmm25 = _mm512_fmadd_ps(zmm7, zmm1, zmm25);                                                                  \
    }                                                                                                                \
    if (InputTile > 6) {                                                                                             \
        zmm2  = _mm512_set1_ps(inputICPtr[(ick) + 6 * AVX512_PACK_C_UNIT]);                                          \
        zmm14 = _mm512_fmadd_ps(zmm2, zmm0, zmm14);                                                                  \
        zmm26 = _mm512_fmadd_ps(zmm2, zmm1, zmm26);                                                                  \
    }                                                                                                                \
    if (InputTile > 7) {                                                                                             \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 7 * AVX512_PACK_C_UNIT]);                                          \
        zmm15 = _mm512_fmadd_ps(zmm3, zmm0, zmm15);                                                                  \
        zmm27 = _mm512_fmadd_ps(zmm3, zmm1, zmm27);                                                                  \
    }                                                                                                                \
    if (InputTile > 8) {                                                                                             \
        zmm4  = _mm512_set1_ps(inputICPtr[(ick) + 8 * AVX512_PACK_C_UNIT]);                                          \
        zmm16 = _mm512_fmadd_ps(zmm4, zmm0, zmm16);                                                                  \
        zmm28 = _mm512_fmadd_ps(zmm4, zmm1, zmm28);                                                                  \
    }                                                                                                                \
    if (InputTile > 9) {                                                                                             \
        zmm5  = _mm512_set1_ps(inputICPtr[(ick) + 9 * AVX512_PACK_C_UNIT]);                                          \
        zmm17 = _mm512_fmadd_ps(zmm5, zmm0, zmm17);                                                                  \
        zmm29 = _mm512_fmadd_ps(zmm5, zmm1, zmm29);                                                                  \
    }                                                                                                                \
    if (InputTile > 10) {                                                                                            \
        zmm6  = _mm512_set1_ps(inputICPtr[(ick) + 10 * AVX512_PACK_C_UNIT]);                                         \
        zmm18 = _mm512_fmadd_ps(zmm6, zmm0, zmm18);                                                                  \
        zmm30 = _mm512_fmadd_ps(zmm6, zmm1, zmm30);                                                                  \
    }                                                                                                                \
    if (InputTile > 11) {                                                                                            \
        zmm7  = _mm512_set1_ps(inputICPtr[(ick) + 11 * AVX512_PACK_C_UNIT]);                                         \
        zmm19 = _mm512_fmadd_ps(zmm7, zmm0, zmm19);                                                                  \
        zmm31 = _mm512_fmadd_ps(zmm7, zmm1, zmm31);                                                                  \
    }                                                                                                                \
    if (InputTile > 12)                                                                                              \
        printf("InputTile size too large. in function:%s\n", __FUNCTION__);

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;
    auto aStride      = parameter[0] / sizeof(float);
    auto l            = parameter[1];
    auto h            = parameter[2];
    auto cStride      = parameter[3] / sizeof(float);
    auto srcUnitStride = parameter[4] / sizeof(float);
    auto bStride      = parameter[5] / sizeof(float);
    int aTotal        = parameter[6];
    auto icTail = l % AVX512_PACK_C_UNIT;
    auto icPack = l - icTail;
    auto inputTilePtr = A;
    auto destPtr = C;
    for(; aTotal > 0; aTotal -= InputTile) {

        auto inputPtr = inputTilePtr;
        auto filterPtr = B;
        auto biasPtr = bias;
        if (biasPtr) {
            if (InputTile > 0 ) {
                zmm8  = _mm512_loadu_ps(biasPtr);
                zmm20 = _mm512_loadu_ps(biasPtr + AVX512_PACK_C_UNIT);
            }
            if (InputTile > 1 ) {
                zmm9  = zmm8 ;
                zmm21 = zmm20;
            }
            if (InputTile > 2 ) {
                zmm10 = zmm8 ;
                zmm22 = zmm20;
            }
            if (InputTile > 3 ) {
                zmm11 = zmm8 ;
                zmm23 = zmm20;
            }
            if (InputTile > 4 ) {
                zmm12 = zmm8 ;
                zmm24 = zmm20;
            }
            if (InputTile > 5 ) {
                zmm13 = zmm8 ;
                zmm25 = zmm20;
            }
            if (InputTile > 6 ) {
                zmm14 = zmm8 ;
                zmm26 = zmm20;
            }
            if (InputTile > 7 ) {
                zmm15 = zmm8 ;
                zmm27 = zmm20;
            }
            if (InputTile > 8 ) {
                zmm16 = zmm8 ;
                zmm28 = zmm20;
            }
            if (InputTile > 9 ) {
                zmm17 = zmm8 ;
                zmm29 = zmm20;
            }
            if (InputTile > 10 ) {
                zmm18 = zmm8 ;
                zmm30 = zmm20;
            }
            if (InputTile > 11 ) {
                zmm19 = zmm8 ;
                zmm31 = zmm20;
            }

        } else {
            if (InputTile > 0 ) zmm8  = _mm512_setzero_ps();
            if (InputTile > 1 ) zmm9  = _mm512_setzero_ps();
            if (InputTile > 2 ) zmm10 = _mm512_setzero_ps();
            if (InputTile > 3 ) zmm11 = _mm512_setzero_ps();
            if (InputTile > 4 ) zmm12 = _mm512_setzero_ps();
            if (InputTile > 5 ) zmm13 = _mm512_setzero_ps();
            if (InputTile > 6 ) zmm14 = _mm512_setzero_ps();
            if (InputTile > 7 ) zmm15 = _mm512_setzero_ps();
            if (InputTile > 8 ) zmm16 = _mm512_setzero_ps();
            if (InputTile > 9 ) zmm17 = _mm512_setzero_ps();
            if (InputTile > 10) zmm18 = _mm512_setzero_ps();
            if (InputTile > 11) zmm19 = _mm512_setzero_ps();

            if (InputTile > 0 ) zmm20 = _mm512_setzero_ps();
            if (InputTile > 1 ) zmm21 = _mm512_setzero_ps();
            if (InputTile > 2 ) zmm22 = _mm512_setzero_ps();
            if (InputTile > 3 ) zmm23 = _mm512_setzero_ps();
            if (InputTile > 4 ) zmm24 = _mm512_setzero_ps();
            if (InputTile > 5 ) zmm25 = _mm512_setzero_ps();
            if (InputTile > 6 ) zmm26 = _mm512_setzero_ps();
            if (InputTile > 7 ) zmm27 = _mm512_setzero_ps();
            if (InputTile > 8 ) zmm28 = _mm512_setzero_ps();
            if (InputTile > 9 ) zmm29 = _mm512_setzero_ps();
            if (InputTile > 10) zmm30 = _mm512_setzero_ps();
            if (InputTile > 11) zmm31 = _mm512_setzero_ps();
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

            // filterPtr += 2 * AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT;
            filterPtr += bStride * AVX512_PACK_C_UNIT;
        }

        auto inputICPtr = inputPtr;
        auto filterICPtr = filterPtr;
        for(int ick = 0; ick < icTail; ++ick) {
            REDUCE_MUL_ADD(ick);
        }

        // write
        // oc < 16;
        if (InputTile > 0 ) _mm512_storeu_ps(destPtr + 0   * AVX512_PACK_C_UNIT, zmm8 );
        if (InputTile > 1 ) _mm512_storeu_ps(destPtr + 1   * AVX512_PACK_C_UNIT, zmm9 );
        if (InputTile > 2 ) _mm512_storeu_ps(destPtr + 2   * AVX512_PACK_C_UNIT, zmm10);
        if (InputTile > 3 ) _mm512_storeu_ps(destPtr + 3   * AVX512_PACK_C_UNIT, zmm11);
        if (InputTile > 4 ) _mm512_storeu_ps(destPtr + 4   * AVX512_PACK_C_UNIT, zmm12);
        if (InputTile > 5 ) _mm512_storeu_ps(destPtr + 5   * AVX512_PACK_C_UNIT, zmm13);
        if (InputTile > 6 ) _mm512_storeu_ps(destPtr + 6   * AVX512_PACK_C_UNIT, zmm14);
        if (InputTile > 7 ) _mm512_storeu_ps(destPtr + 7   * AVX512_PACK_C_UNIT, zmm15);
        if (InputTile > 8 ) _mm512_storeu_ps(destPtr + 8   * AVX512_PACK_C_UNIT, zmm16);
        if (InputTile > 9 ) _mm512_storeu_ps(destPtr + 9   * AVX512_PACK_C_UNIT, zmm17);
        if (InputTile > 10) _mm512_storeu_ps(destPtr + 10  * AVX512_PACK_C_UNIT, zmm18);
        if (InputTile > 11) _mm512_storeu_ps(destPtr + 11  * AVX512_PACK_C_UNIT, zmm19);

        // oc < 32
        auto writeDestPtr = destPtr + cStride;
        if (InputTile > 0 ) _mm512_storeu_ps(writeDestPtr + 0   * AVX512_PACK_C_UNIT, zmm20);
        if (InputTile > 1 ) _mm512_storeu_ps(writeDestPtr + 1   * AVX512_PACK_C_UNIT, zmm21);
        if (InputTile > 2 ) _mm512_storeu_ps(writeDestPtr + 2   * AVX512_PACK_C_UNIT, zmm22);
        if (InputTile > 3 ) _mm512_storeu_ps(writeDestPtr + 3   * AVX512_PACK_C_UNIT, zmm23);
        if (InputTile > 4 ) _mm512_storeu_ps(writeDestPtr + 4   * AVX512_PACK_C_UNIT, zmm24);
        if (InputTile > 5 ) _mm512_storeu_ps(writeDestPtr + 5   * AVX512_PACK_C_UNIT, zmm25);
        if (InputTile > 6 ) _mm512_storeu_ps(writeDestPtr + 6   * AVX512_PACK_C_UNIT, zmm26);
        if (InputTile > 7 ) _mm512_storeu_ps(writeDestPtr + 7   * AVX512_PACK_C_UNIT, zmm27);
        if (InputTile > 8 ) _mm512_storeu_ps(writeDestPtr + 8   * AVX512_PACK_C_UNIT, zmm28);
        if (InputTile > 9 ) _mm512_storeu_ps(writeDestPtr + 9   * AVX512_PACK_C_UNIT, zmm29);
        if (InputTile > 10) _mm512_storeu_ps(writeDestPtr + 10  * AVX512_PACK_C_UNIT, zmm30);
        if (InputTile > 11) _mm512_storeu_ps(writeDestPtr + 11  * AVX512_PACK_C_UNIT, zmm31);
        inputTilePtr += aStride;
        destPtr += InputTile * AVX512_PACK_C_UNIT;
    }

#undef REDUCE_MUL_ADD

}


template <int InputTile>
void _AVX512_MNNPackedMatMulO32SwapedKernel(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias) {

#define REDUCE_MUL_ADD(ick)                                                                                          \
    zmm0 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 0 * AVX512_PACK_C_UNIT));                                  \
    zmm1 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 1 * AVX512_PACK_C_UNIT));                                  \
    if (InputTile > 10) {                                                                                            \
        _mm_prefetch(filterICPtr + ick * bStride + 0 * AVX512_PACK_C_UNIT + AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                                   \
        _mm_prefetch(filterICPtr + ick * bStride + 1 * AVX512_PACK_C_UNIT + AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                                   \
        _mm_prefetch(inputICPtr + ick * AVX512_PACK_C_UNIT, _MM_HINT_T0);                                            \
    }                                                                                                                \
    if (InputTile > 0) {                                                                                             \
        zmm2  = _mm512_set1_ps(inputICPtr[(ick) + 0 * AVX512_PACK_C_UNIT]);                                          \
        zmm4  = _mm512_fmadd_ps(zmm2, zmm0, zmm4);                                                                   \
        zmm18 = _mm512_fmadd_ps(zmm2, zmm1, zmm18);                                                                  \
    }                                                                                                                \
    if (InputTile > 1) {                                                                                             \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 1 * AVX512_PACK_C_UNIT]);                                          \
        zmm5  = _mm512_fmadd_ps(zmm3, zmm0, zmm5);                                                                   \
        zmm19 = _mm512_fmadd_ps(zmm3, zmm1, zmm19);                                                                  \
    }                                                                                                                \
    if (InputTile > 2) {                                                                                             \
        zmm2  = _mm512_set1_ps(inputICPtr[(ick) + 2 * AVX512_PACK_C_UNIT]);                                          \
        zmm6  = _mm512_fmadd_ps(zmm2, zmm0, zmm6);                                                                   \
        zmm20 = _mm512_fmadd_ps(zmm2, zmm1, zmm20);                                                                  \
    }                                                                                                                \
    if (InputTile > 3) {                                                                                             \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 3 * AVX512_PACK_C_UNIT]);                                          \
        zmm7  = _mm512_fmadd_ps(zmm3, zmm0, zmm7);                                                                   \
        zmm21 = _mm512_fmadd_ps(zmm3, zmm1, zmm21);                                                                  \
    }                                                                                                                \
    if (InputTile > 4) {                                                                                             \
        zmm2  = _mm512_set1_ps(inputICPtr[(ick) + 4 * AVX512_PACK_C_UNIT]);                                          \
        zmm8  = _mm512_fmadd_ps(zmm2, zmm0, zmm8);                                                                   \
        zmm22 = _mm512_fmadd_ps(zmm2, zmm1, zmm22);                                                                  \
    }                                                                                                                \
    if (InputTile > 5) {                                                                                             \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 5 * AVX512_PACK_C_UNIT]);                                          \
        zmm9  = _mm512_fmadd_ps(zmm3, zmm0, zmm9);                                                                   \
        zmm23 = _mm512_fmadd_ps(zmm3, zmm1, zmm23);                                                                  \
    }                                                                                                                \
    if (InputTile > 6) {                                                                                             \
        zmm2  = _mm512_set1_ps(inputICPtr[(ick) + 6 * AVX512_PACK_C_UNIT]);                                          \
        zmm10 = _mm512_fmadd_ps(zmm2, zmm0, zmm10);                                                                  \
        zmm24 = _mm512_fmadd_ps(zmm2, zmm1, zmm24);                                                                  \
    }                                                                                                                \
    if (InputTile > 7) {                                                                                             \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 7 * AVX512_PACK_C_UNIT]);                                          \
        zmm11 = _mm512_fmadd_ps(zmm3, zmm0, zmm11);                                                                  \
        zmm25 = _mm512_fmadd_ps(zmm3, zmm1, zmm25);                                                                  \
    }                                                                                                                \
    if (InputTile > 8) {                                                                                             \
        zmm2  = _mm512_set1_ps(inputICPtr[(ick) + 8 * AVX512_PACK_C_UNIT]);                                          \
        zmm12 = _mm512_fmadd_ps(zmm2, zmm0, zmm12);                                                                  \
        zmm26 = _mm512_fmadd_ps(zmm2, zmm1, zmm26);                                                                  \
    }                                                                                                                \
    if (InputTile > 9) {                                                                                             \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 9 * AVX512_PACK_C_UNIT]);                                          \
        zmm13 = _mm512_fmadd_ps(zmm3, zmm0, zmm13);                                                                  \
        zmm27 = _mm512_fmadd_ps(zmm3, zmm1, zmm27);                                                                  \
    }                                                                                                                \
    if (InputTile > 10) {                                                                                            \
        zmm2  = _mm512_set1_ps(inputICPtr[(ick) + 10 * AVX512_PACK_C_UNIT]);                                         \
        zmm14 = _mm512_fmadd_ps(zmm2, zmm0, zmm14);                                                                  \
        zmm28 = _mm512_fmadd_ps(zmm2, zmm1, zmm28);                                                                  \
    }                                                                                                                \
    if (InputTile > 11) {                                                                                            \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 11 * AVX512_PACK_C_UNIT]);                                         \
        zmm15 = _mm512_fmadd_ps(zmm3, zmm0, zmm15);                                                                  \
        zmm29 = _mm512_fmadd_ps(zmm3, zmm1, zmm29);                                                                  \
    }                                                                                                                \
    if (InputTile > 12) {                                                                                            \
        zmm2  = _mm512_set1_ps(inputICPtr[(ick) + 12 * AVX512_PACK_C_UNIT]);                                         \
        zmm16 = _mm512_fmadd_ps(zmm2, zmm0, zmm16);                                                                  \
        zmm30 = _mm512_fmadd_ps(zmm2, zmm1, zmm30);                                                                  \
    }                                                                                                                \
    if (InputTile > 13) {                                                                                            \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 13 * AVX512_PACK_C_UNIT]);                                         \
        zmm17 = _mm512_fmadd_ps(zmm3, zmm0, zmm17);                                                                  \
        zmm31 = _mm512_fmadd_ps(zmm3, zmm1, zmm31);                                                                  \
    }                                                                                                                \
    if (InputTile > 14)                                                                                              \
        printf("InputTile size too large. in function:%s\n", __FUNCTION__);

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;
    auto aStride      = parameter[0] / sizeof(float);
    auto l            = parameter[1];
    auto h            = parameter[2];
    auto cStride      = parameter[3] / sizeof(float);
    auto srcUnitStride = parameter[4] / sizeof(float);
    auto bStride      = parameter[5] / sizeof(float);
    int aTotal        = parameter[6];
    auto icTail = l % AVX512_PACK_C_UNIT;
    auto icPack = l - icTail;
    auto inputTilePtr = A;
    auto destPtr = C;
    for(; aTotal > 0; aTotal -= InputTile) {

        auto inputPtr = inputTilePtr;
        auto filterPtr = B;
        auto biasPtr = bias;

        if (biasPtr) {
            if (InputTile > 0 ) {
                zmm4  = _mm512_loadu_ps(biasPtr);
                zmm18 = _mm512_loadu_ps(biasPtr + AVX512_PACK_C_UNIT);
            }
            if (InputTile > 1 ) {
                zmm5  = zmm4;
                zmm19 = zmm18;
            }
            if (InputTile > 2 ) {
                zmm6  = zmm4;
                zmm20 = zmm18;
            }
            if (InputTile > 3 ) {
                zmm7  = zmm4;
                zmm21 = zmm18;
            }
            if (InputTile > 4 ) {
                zmm8  = zmm4;
                zmm22 = zmm18;
            }
            if (InputTile > 5 ) {
                zmm9  = zmm4;
                zmm23 = zmm18;
            }
            if (InputTile > 6 ) {
                zmm10 = zmm4;
                zmm24 = zmm18;
            }
            if (InputTile > 7 ) {
                zmm11 = zmm4;
                zmm25 = zmm18;
            }
            if (InputTile > 8 ) {
                zmm12 = zmm4;
                zmm26 = zmm18;
            }
            if (InputTile > 9 ) {
                zmm13 = zmm4;
                zmm27 = zmm18;
            }
            if (InputTile > 10 ) {
                zmm14 = zmm4;
                zmm28 = zmm18;
            }
            if (InputTile > 11 ) {
                zmm15 = zmm4;
                zmm29 = zmm18;
            }
            if (InputTile > 12 ) {
                zmm16 = zmm4;
                zmm30 = zmm18;
            }
            if (InputTile > 13 ) {
                zmm17 = zmm4;
                zmm31 = zmm18;
            }

        } else {
            if (InputTile > 0 ) zmm4  = _mm512_setzero_ps();
            if (InputTile > 1 ) zmm5  = _mm512_setzero_ps();
            if (InputTile > 2 ) zmm6  = _mm512_setzero_ps();
            if (InputTile > 3 ) zmm7  = _mm512_setzero_ps();
            if (InputTile > 4 ) zmm8  = _mm512_setzero_ps();
            if (InputTile > 5 ) zmm9  = _mm512_setzero_ps();
            if (InputTile > 6 ) zmm10 = _mm512_setzero_ps();
            if (InputTile > 7 ) zmm11 = _mm512_setzero_ps();
            if (InputTile > 8 ) zmm12 = _mm512_setzero_ps();
            if (InputTile > 9 ) zmm13 = _mm512_setzero_ps();
            if (InputTile > 10) zmm14 = _mm512_setzero_ps();
            if (InputTile > 11) zmm15 = _mm512_setzero_ps();
            if (InputTile > 12) zmm16 = _mm512_setzero_ps();
            if (InputTile > 13) zmm17 = _mm512_setzero_ps();

            if (InputTile > 0 ) zmm18 = _mm512_setzero_ps();
            if (InputTile > 1 ) zmm19 = _mm512_setzero_ps();
            if (InputTile > 2 ) zmm20 = _mm512_setzero_ps();
            if (InputTile > 3 ) zmm21 = _mm512_setzero_ps();
            if (InputTile > 4 ) zmm22 = _mm512_setzero_ps();
            if (InputTile > 5 ) zmm23 = _mm512_setzero_ps();
            if (InputTile > 6 ) zmm24 = _mm512_setzero_ps();
            if (InputTile > 7 ) zmm25 = _mm512_setzero_ps();
            if (InputTile > 8 ) zmm26 = _mm512_setzero_ps();
            if (InputTile > 9 ) zmm27 = _mm512_setzero_ps();
            if (InputTile > 10) zmm28 = _mm512_setzero_ps();
            if (InputTile > 11) zmm29 = _mm512_setzero_ps();
            if (InputTile > 12) zmm30 = _mm512_setzero_ps();
            if (InputTile > 13) zmm31 = _mm512_setzero_ps();
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

            // filterPtr += 2 * AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT;
            filterPtr += bStride * AVX512_PACK_C_UNIT;
        }

        auto inputICPtr = inputPtr;
        auto filterICPtr = filterPtr;
        for(int ick = 0; ick < icTail; ++ick) {
            REDUCE_MUL_ADD(ick);
        }

        // write
        // oc < 16;
        if (InputTile > 0 ) _mm512_storeu_ps(destPtr + 0   * AVX512_PACK_C_UNIT, zmm4 );
        if (InputTile > 1 ) _mm512_storeu_ps(destPtr + 1   * AVX512_PACK_C_UNIT, zmm5 );
        if (InputTile > 2 ) _mm512_storeu_ps(destPtr + 2   * AVX512_PACK_C_UNIT, zmm6 );
        if (InputTile > 3 ) _mm512_storeu_ps(destPtr + 3   * AVX512_PACK_C_UNIT, zmm7 );
        if (InputTile > 4 ) _mm512_storeu_ps(destPtr + 4   * AVX512_PACK_C_UNIT, zmm8 );
        if (InputTile > 5 ) _mm512_storeu_ps(destPtr + 5   * AVX512_PACK_C_UNIT, zmm9 );
        if (InputTile > 6 ) _mm512_storeu_ps(destPtr + 6   * AVX512_PACK_C_UNIT, zmm10);
        if (InputTile > 7 ) _mm512_storeu_ps(destPtr + 7   * AVX512_PACK_C_UNIT, zmm11);
        if (InputTile > 8 ) _mm512_storeu_ps(destPtr + 8   * AVX512_PACK_C_UNIT, zmm12);
        if (InputTile > 9 ) _mm512_storeu_ps(destPtr + 9   * AVX512_PACK_C_UNIT, zmm13);
        if (InputTile > 10) _mm512_storeu_ps(destPtr + 10  * AVX512_PACK_C_UNIT, zmm14);
        if (InputTile > 11) _mm512_storeu_ps(destPtr + 11  * AVX512_PACK_C_UNIT, zmm15);
        if (InputTile > 12) _mm512_storeu_ps(destPtr + 12  * AVX512_PACK_C_UNIT, zmm16);
        if (InputTile > 13) _mm512_storeu_ps(destPtr + 13  * AVX512_PACK_C_UNIT, zmm17);

        // oc < 32
        auto writeDestPtr = destPtr + cStride;
        if (InputTile > 0 ) _mm512_storeu_ps(writeDestPtr + 0   * AVX512_PACK_C_UNIT, zmm18);
        if (InputTile > 1 ) _mm512_storeu_ps(writeDestPtr + 1   * AVX512_PACK_C_UNIT, zmm19);
        if (InputTile > 2 ) _mm512_storeu_ps(writeDestPtr + 2   * AVX512_PACK_C_UNIT, zmm20);
        if (InputTile > 3 ) _mm512_storeu_ps(writeDestPtr + 3   * AVX512_PACK_C_UNIT, zmm21);
        if (InputTile > 4 ) _mm512_storeu_ps(writeDestPtr + 4   * AVX512_PACK_C_UNIT, zmm22);
        if (InputTile > 5 ) _mm512_storeu_ps(writeDestPtr + 5   * AVX512_PACK_C_UNIT, zmm23);
        if (InputTile > 6 ) _mm512_storeu_ps(writeDestPtr + 6   * AVX512_PACK_C_UNIT, zmm24);
        if (InputTile > 7 ) _mm512_storeu_ps(writeDestPtr + 7   * AVX512_PACK_C_UNIT, zmm25);
        if (InputTile > 8 ) _mm512_storeu_ps(writeDestPtr + 8   * AVX512_PACK_C_UNIT, zmm26);
        if (InputTile > 9 ) _mm512_storeu_ps(writeDestPtr + 9   * AVX512_PACK_C_UNIT, zmm27);
        if (InputTile > 10) _mm512_storeu_ps(writeDestPtr + 10  * AVX512_PACK_C_UNIT, zmm28);
        if (InputTile > 11) _mm512_storeu_ps(writeDestPtr + 11  * AVX512_PACK_C_UNIT, zmm29);
        if (InputTile > 12) _mm512_storeu_ps(writeDestPtr + 12  * AVX512_PACK_C_UNIT, zmm30);
        if (InputTile > 13) _mm512_storeu_ps(writeDestPtr + 13  * AVX512_PACK_C_UNIT, zmm31);

        inputTilePtr += aStride;
        destPtr += InputTile * AVX512_PACK_C_UNIT;
    }

#undef REDUCE_MUL_ADD

}

