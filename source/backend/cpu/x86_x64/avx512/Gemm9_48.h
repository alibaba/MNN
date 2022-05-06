#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdint.h>


#ifdef MNN_X86_USE_ASM
extern "C" {
void _AVX512_MNNPackedMatMulO48Swaped4KernelASM(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias);
}
#endif

template <int InputTile>
void _AVX512_MNNPackedMatMulO48FullLoadKernel(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias) {

#define REDUCE_MUL_ADD(ick)                                                                                          \
    zmm0 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 0 * AVX512_PACK_C_UNIT));                                  \
    zmm1 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 1 * AVX512_PACK_C_UNIT));                                  \
    zmm2 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 2 * AVX512_PACK_C_UNIT));                                  \
    if (InputTile > 5) { /*select the threashhold*/                                                                  \
        _mm_prefetch(filterICPtr + ick * bStride + 0 * AVX512_PACK_C_UNIT + AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                                   \
        _mm_prefetch(filterICPtr + ick * bStride + 1 * AVX512_PACK_C_UNIT + AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                                   \
        _mm_prefetch(filterICPtr + ick * bStride + 2 * AVX512_PACK_C_UNIT + AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                                   \
        _mm_prefetch(inputICPtr + ick * AVX512_PACK_C_UNIT, _MM_HINT_T0);                                            \
    }                                                                                                                \
    if (InputTile > 0) {                                                                                             \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 0 * AVX512_PACK_C_UNIT]);                                          \
        zmm10 = _mm512_fmadd_ps(zmm3, zmm0, zmm10);                                                                  \
        zmm17 = _mm512_fmadd_ps(zmm3, zmm1, zmm17);                                                                  \
        zmm24 = _mm512_fmadd_ps(zmm3, zmm2, zmm24);                                                                  \
    }                                                                                                                \
    if (InputTile > 1) {                                                                                             \
        zmm4  = _mm512_set1_ps(inputICPtr[(ick) + 1 * AVX512_PACK_C_UNIT]);                                          \
        zmm11 = _mm512_fmadd_ps(zmm4, zmm0, zmm11);                                                                  \
        zmm18 = _mm512_fmadd_ps(zmm4, zmm1, zmm18);                                                                  \
        zmm25 = _mm512_fmadd_ps(zmm4, zmm2, zmm25);                                                                  \
    }                                                                                                                \
    if (InputTile > 2) {                                                                                             \
        zmm5  = _mm512_set1_ps(inputICPtr[(ick) + 2 * AVX512_PACK_C_UNIT]);                                          \
        zmm12 = _mm512_fmadd_ps(zmm5, zmm0, zmm12);                                                                  \
        zmm19 = _mm512_fmadd_ps(zmm5, zmm1, zmm19);                                                                  \
        zmm26 = _mm512_fmadd_ps(zmm5, zmm2, zmm26);                                                                  \
    }                                                                                                                \
    if (InputTile > 3) {                                                                                             \
        zmm6  = _mm512_set1_ps(inputICPtr[(ick) + 3 * AVX512_PACK_C_UNIT]);                                          \
        zmm13 = _mm512_fmadd_ps(zmm6, zmm0, zmm13);                                                                  \
        zmm20 = _mm512_fmadd_ps(zmm6, zmm1, zmm20);                                                                  \
        zmm27 = _mm512_fmadd_ps(zmm6, zmm2, zmm27);                                                                  \
    }                                                                                                                \
    if (InputTile > 4) {                                                                                             \
        zmm7  = _mm512_set1_ps(inputICPtr[(ick) + 4 * AVX512_PACK_C_UNIT]);                                          \
        zmm14 = _mm512_fmadd_ps(zmm7, zmm0, zmm14);                                                                  \
        zmm21 = _mm512_fmadd_ps(zmm7, zmm1, zmm21);                                                                  \
        zmm28 = _mm512_fmadd_ps(zmm7, zmm2, zmm28);                                                                  \
    }                                                                                                                \
    if (InputTile > 5) {                                                                                             \
        zmm8  = _mm512_set1_ps(inputICPtr[(ick) + 5 * AVX512_PACK_C_UNIT]);                                          \
        zmm15 = _mm512_fmadd_ps(zmm8, zmm0, zmm15);                                                                  \
        zmm22 = _mm512_fmadd_ps(zmm8, zmm1, zmm22);                                                                  \
        zmm29 = _mm512_fmadd_ps(zmm8, zmm2, zmm29);                                                                  \
    }                                                                                                                \
    if (InputTile > 6) {                                                                                             \
        zmm9  = _mm512_set1_ps(inputICPtr[(ick) + 6 * AVX512_PACK_C_UNIT]);                                          \
        zmm16 = _mm512_fmadd_ps(zmm9, zmm0, zmm16);                                                                  \
        zmm23 = _mm512_fmadd_ps(zmm9, zmm1, zmm23);                                                                  \
        zmm30 = _mm512_fmadd_ps(zmm9, zmm2, zmm30);                                                                  \
    }                                                                                                                \
    if (InputTile > 7)                                                                                               \
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
                zmm10 = _mm512_loadu_ps(biasPtr);
                zmm17 = _mm512_loadu_ps(biasPtr + 1 * AVX512_PACK_C_UNIT);
                zmm24 = _mm512_loadu_ps(biasPtr + 2 * AVX512_PACK_C_UNIT);
            }
            if (InputTile > 1 ) {
                zmm11 = zmm10;
                zmm18 = zmm17;
                zmm25 = zmm24;
            }
            if (InputTile > 2 ) {
                zmm12 = zmm10;
                zmm19 = zmm17;
                zmm26 = zmm24;
            }
            if (InputTile > 3 ) {
                zmm13 = zmm10;
                zmm20 = zmm17;
                zmm27 = zmm24;
            }
            if (InputTile > 4 ) {
                zmm14 = zmm10;
                zmm21 = zmm17;
                zmm28 = zmm24;
            }
            if (InputTile > 5 ) {
                zmm15 = zmm10;
                zmm22 = zmm17;
                zmm29 = zmm24;
            }
            if (InputTile > 6 ) {
                zmm16 = zmm10;
                zmm23 = zmm17;
                zmm30 = zmm24;
            }
        } else {
            if (InputTile > 0 ) {
                zmm10 = _mm512_setzero_ps();
                zmm17 = _mm512_setzero_ps();
                zmm24 = _mm512_setzero_ps();
            }
            if (InputTile > 1 ) {
                zmm11 = _mm512_setzero_ps();
                zmm18 = _mm512_setzero_ps();
                zmm25 = _mm512_setzero_ps();
            }
            if (InputTile > 2 ) {
                zmm12 = _mm512_setzero_ps();
                zmm19 = _mm512_setzero_ps();
                zmm26 = _mm512_setzero_ps();
            }
            if (InputTile > 3 ) {
                zmm13 = _mm512_setzero_ps();
                zmm20 = _mm512_setzero_ps();
                zmm27 = _mm512_setzero_ps();
            }
            if (InputTile > 4 ) {
                zmm14 = _mm512_setzero_ps();
                zmm21 = _mm512_setzero_ps();
                zmm28 = _mm512_setzero_ps();
            }
            if (InputTile > 5 ) {
                zmm15 = _mm512_setzero_ps();
                zmm22 = _mm512_setzero_ps();
                zmm29 = _mm512_setzero_ps();
            }
            if (InputTile > 6 ) {
                zmm16 = _mm512_setzero_ps();
                zmm23 = _mm512_setzero_ps();
                zmm30 = _mm512_setzero_ps();
            }
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
        for(int ick = 0; ick < icTail; ++ick) {
            REDUCE_MUL_ADD(ick);
        }


        // write
        // oc = 16;
        if (InputTile > 0 ) _mm512_storeu_ps(destPtr + 0  * AVX512_PACK_C_UNIT, zmm10);
        if (InputTile > 1 ) _mm512_storeu_ps(destPtr + 1  * AVX512_PACK_C_UNIT, zmm11);
        if (InputTile > 2 ) _mm512_storeu_ps(destPtr + 2  * AVX512_PACK_C_UNIT, zmm12);
        if (InputTile > 3 ) _mm512_storeu_ps(destPtr + 3  * AVX512_PACK_C_UNIT, zmm13);
        if (InputTile > 4 ) _mm512_storeu_ps(destPtr + 4  * AVX512_PACK_C_UNIT, zmm14);
        if (InputTile > 5 ) _mm512_storeu_ps(destPtr + 5  * AVX512_PACK_C_UNIT, zmm15);
        if (InputTile > 6 ) _mm512_storeu_ps(destPtr + 6  * AVX512_PACK_C_UNIT, zmm16);

        // oc = 32
        auto destOC32Ptr = destPtr + 1 * cStride;
        if (InputTile > 0 ) _mm512_storeu_ps(destOC32Ptr + 0  * AVX512_PACK_C_UNIT, zmm17);
        if (InputTile > 1 ) _mm512_storeu_ps(destOC32Ptr + 1  * AVX512_PACK_C_UNIT, zmm18);
        if (InputTile > 2 ) _mm512_storeu_ps(destOC32Ptr + 2  * AVX512_PACK_C_UNIT, zmm19);
        if (InputTile > 3 ) _mm512_storeu_ps(destOC32Ptr + 3  * AVX512_PACK_C_UNIT, zmm20);
        if (InputTile > 4 ) _mm512_storeu_ps(destOC32Ptr + 4  * AVX512_PACK_C_UNIT, zmm21);
        if (InputTile > 5 ) _mm512_storeu_ps(destOC32Ptr + 5  * AVX512_PACK_C_UNIT, zmm22);
        if (InputTile > 6 ) _mm512_storeu_ps(destOC32Ptr + 6  * AVX512_PACK_C_UNIT, zmm23);

        // oc = 48
        auto destOC48Ptr = destPtr + 2 * cStride;
        if (InputTile > 0 ) _mm512_storeu_ps(destOC48Ptr + 0  * AVX512_PACK_C_UNIT, zmm24);
        if (InputTile > 1 ) _mm512_storeu_ps(destOC48Ptr + 1  * AVX512_PACK_C_UNIT, zmm25);
        if (InputTile > 2 ) _mm512_storeu_ps(destOC48Ptr + 2  * AVX512_PACK_C_UNIT, zmm26);
        if (InputTile > 3 ) _mm512_storeu_ps(destOC48Ptr + 3  * AVX512_PACK_C_UNIT, zmm27);
        if (InputTile > 4 ) _mm512_storeu_ps(destOC48Ptr + 4  * AVX512_PACK_C_UNIT, zmm28);
        if (InputTile > 5 ) _mm512_storeu_ps(destOC48Ptr + 5  * AVX512_PACK_C_UNIT, zmm29);
        if (InputTile > 6 ) _mm512_storeu_ps(destOC48Ptr + 6  * AVX512_PACK_C_UNIT, zmm30);

        inputTilePtr += aStride;
        destPtr += InputTile * AVX512_PACK_C_UNIT;
    }

#undef REDUCE_MUL_ADD

}
template <int InputTile>
void _AVX512_MNNPackedMatMulO48Swaped4Kernel(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias) {

// #ifdef MNN_X86_USE_ASM
//     if (InputTile == 8) {
//         _AVX512_MNNPackedMatMulO48Swaped4KernelASM(C, A, B, parameter, postParameters, bias);
//         return;
//     }
// #endif

#define REDUCE_MUL_ADD(ick)                                                                                          \
    zmm0 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 0 * AVX512_PACK_C_UNIT));                                  \
    zmm1 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 1 * AVX512_PACK_C_UNIT));                                  \
    zmm2 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 2 * AVX512_PACK_C_UNIT));                                  \
    if (InputTile > 7) { /*select the threashhold*/                                                                  \
        _mm_prefetch(filterICPtr + ick * bStride + 0 * AVX512_PACK_C_UNIT + AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                                   \
        _mm_prefetch(filterICPtr + ick * bStride + 1 * AVX512_PACK_C_UNIT + AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                                   \
        _mm_prefetch(filterICPtr + ick * bStride + 2 * AVX512_PACK_C_UNIT + AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                                   \
        _mm_prefetch(inputICPtr + ick * AVX512_PACK_C_UNIT, _MM_HINT_T0);                                            \
    }                                                                                                                \
    if (InputTile > 0) {                                                                                             \
        zmm4  = _mm512_set1_ps(inputICPtr[(ick) + 0 * AVX512_PACK_C_UNIT]);                                          \
        zmm8  = _mm512_fmadd_ps(zmm4, zmm0, zmm8);                                                                   \
        zmm16 = _mm512_fmadd_ps(zmm4, zmm1, zmm16);                                                                  \
        zmm24 = _mm512_fmadd_ps(zmm4, zmm2, zmm24);                                                                  \
    }                                                                                                                \
    if (InputTile > 1) {                                                                                             \
        zmm5  = _mm512_set1_ps(inputICPtr[(ick) + 1 * AVX512_PACK_C_UNIT]);                                          \
        zmm9  = _mm512_fmadd_ps(zmm5, zmm0, zmm9);                                                                   \
        zmm17 = _mm512_fmadd_ps(zmm5, zmm1, zmm17);                                                                  \
        zmm25 = _mm512_fmadd_ps(zmm5, zmm2, zmm25);                                                                  \
    }                                                                                                                \
    if (InputTile > 2) {                                                                                             \
        zmm6  = _mm512_set1_ps(inputICPtr[(ick) + 2 * AVX512_PACK_C_UNIT]);                                          \
        zmm10 = _mm512_fmadd_ps(zmm6, zmm0, zmm10);                                                                  \
        zmm18 = _mm512_fmadd_ps(zmm6, zmm1, zmm18);                                                                  \
        zmm26 = _mm512_fmadd_ps(zmm6, zmm2, zmm26);                                                                  \
    }                                                                                                                \
    if (InputTile > 3) {                                                                                             \
        zmm7  = _mm512_set1_ps(inputICPtr[(ick) + 3 * AVX512_PACK_C_UNIT]);                                          \
        zmm11 = _mm512_fmadd_ps(zmm7, zmm0, zmm11);                                                                  \
        zmm19 = _mm512_fmadd_ps(zmm7, zmm1, zmm19);                                                                  \
        zmm27 = _mm512_fmadd_ps(zmm7, zmm2, zmm27);                                                                  \
    }                                                                                                                \
    if (InputTile > 4) {                                                                                             \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 4 * AVX512_PACK_C_UNIT]);                                          \
        zmm12 = _mm512_fmadd_ps(zmm3, zmm0, zmm12);                                                                  \
        zmm20 = _mm512_fmadd_ps(zmm3, zmm1, zmm20);                                                                  \
        zmm28 = _mm512_fmadd_ps(zmm3, zmm2, zmm28);                                                                  \
    }                                                                                                                \
    if (InputTile > 5) {                                                                                             \
        zmm4  = _mm512_set1_ps(inputICPtr[(ick) + 5 * AVX512_PACK_C_UNIT]);                                          \
        zmm13 = _mm512_fmadd_ps(zmm4, zmm0, zmm13);                                                                  \
        zmm21 = _mm512_fmadd_ps(zmm4, zmm1, zmm21);                                                                  \
        zmm29 = _mm512_fmadd_ps(zmm4, zmm2, zmm29);                                                                  \
    }                                                                                                                \
    if (InputTile > 6) {                                                                                             \
        zmm5  = _mm512_set1_ps(inputICPtr[(ick) + 6 * AVX512_PACK_C_UNIT]);                                          \
        zmm14 = _mm512_fmadd_ps(zmm5, zmm0, zmm14);                                                                  \
        zmm22 = _mm512_fmadd_ps(zmm5, zmm1, zmm22);                                                                  \
        zmm30 = _mm512_fmadd_ps(zmm5, zmm2, zmm30);                                                                  \
    }                                                                                                                \
    if (InputTile > 7) {                                                                                             \
        zmm6  = _mm512_set1_ps(inputICPtr[(ick) + 7 * AVX512_PACK_C_UNIT]);                                          \
        zmm15 = _mm512_fmadd_ps(zmm6, zmm0, zmm15);                                                                  \
        zmm23 = _mm512_fmadd_ps(zmm6, zmm1, zmm23);                                                                  \
        zmm31 = _mm512_fmadd_ps(zmm6, zmm2, zmm31);                                                                  \
    }                                                                                                                \
    if (InputTile > 8)                                                                                               \
        printf("InputTile size too large. in function:%s\n", __FUNCTION__);

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;
    auto aStride       = parameter[0] / sizeof(float);
    auto l             = parameter[1];
    auto h             = parameter[2];
    auto cStride       = parameter[3] / sizeof(float);
    auto srcUnitStride = parameter[4] / sizeof(float);
    auto bStride       = parameter[5] / sizeof(float);
    int aTotal         = parameter[6];

    auto icTail = l % AVX512_PACK_C_UNIT;
    auto icPack = l - icTail;

    auto inputTilePtr = A;
    auto destPtr      = C;

    for (; aTotal > 0; aTotal -= InputTile) {
        auto inputPtr  = inputTilePtr;
        auto filterPtr = B;
        auto biasPtr   = bias;

        if (biasPtr) {
            if (InputTile > 0) {
                zmm8  = _mm512_loadu_ps(biasPtr);
                zmm16 = _mm512_loadu_ps(biasPtr + 1 * AVX512_PACK_C_UNIT);
                zmm24 = _mm512_loadu_ps(biasPtr + 2 * AVX512_PACK_C_UNIT);
            }
            if (InputTile > 1) {
                zmm9  = zmm8;
                zmm17 = zmm16;
                zmm25 = zmm24;
            }
            if (InputTile > 2) {
                zmm10 = zmm8;
                zmm18 = zmm16;
                zmm26 = zmm24;
            }
            if (InputTile > 3) {
                zmm11 = zmm8;
                zmm19 = zmm16;
                zmm27 = zmm24;
            }
            if (InputTile > 4) {
                zmm12 = zmm8;
                zmm20 = zmm16;
                zmm28 = zmm24;
            }
            if (InputTile > 5) {
                zmm13 = zmm8;
                zmm21 = zmm16;
                zmm29 = zmm24;
            }
            if (InputTile > 6) {
                zmm14 = zmm8;
                zmm22 = zmm16;
                zmm30 = zmm24;
            }
            if (InputTile > 7) {
                zmm15 = zmm8;
                zmm23 = zmm16;
                zmm31 = zmm24;
            }

        } else {
            if (InputTile > 0) {
                zmm8  = _mm512_setzero_ps();
                zmm16 = _mm512_setzero_ps();
                zmm24 = _mm512_setzero_ps();
            }
            if (InputTile > 1) {
                zmm9  = _mm512_setzero_ps();
                zmm17 = _mm512_setzero_ps();
                zmm25 = _mm512_setzero_ps();
            }
            if (InputTile > 2) {
                zmm10 = _mm512_setzero_ps();
                zmm18 = _mm512_setzero_ps();
                zmm26 = _mm512_setzero_ps();
            }
            if (InputTile > 3) {
                zmm11 = _mm512_setzero_ps();
                zmm19 = _mm512_setzero_ps();
                zmm27 = _mm512_setzero_ps();
            }
            if (InputTile > 4) {
                zmm12 = _mm512_setzero_ps();
                zmm20 = _mm512_setzero_ps();
                zmm28 = _mm512_setzero_ps();
            }
            if (InputTile > 5) {
                zmm13 = _mm512_setzero_ps();
                zmm21 = _mm512_setzero_ps();
                zmm29 = _mm512_setzero_ps();
            }
            if (InputTile > 6) {
                zmm14 = _mm512_setzero_ps();
                zmm22 = _mm512_setzero_ps();
                zmm30 = _mm512_setzero_ps();
            }
            if (InputTile > 7) {
                zmm15 = _mm512_setzero_ps();
                zmm23 = _mm512_setzero_ps();
                zmm31 = _mm512_setzero_ps();
            }
        }

        for (int il = 0; il < icPack; il += AVX512_PACK_C_UNIT) {
            auto inputICPtr  = inputPtr;
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
            // filterPtr += AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT;
            filterPtr += bStride * AVX512_PACK_C_UNIT;
        }

        auto inputICPtr  = inputPtr;
        auto filterICPtr = filterPtr;
        for (int ick = 0; ick < icTail; ++ick) {
            REDUCE_MUL_ADD(ick);
        }

        // write
        // oc = 16;
        if (InputTile > 0)
            _mm512_storeu_ps(destPtr + 0 * AVX512_PACK_C_UNIT, zmm8);
        if (InputTile > 1)
            _mm512_storeu_ps(destPtr + 1 * AVX512_PACK_C_UNIT, zmm9);
        if (InputTile > 2)
            _mm512_storeu_ps(destPtr + 2 * AVX512_PACK_C_UNIT, zmm10);
        if (InputTile > 3)
            _mm512_storeu_ps(destPtr + 3 * AVX512_PACK_C_UNIT, zmm11);
        if (InputTile > 4)
            _mm512_storeu_ps(destPtr + 4 * AVX512_PACK_C_UNIT, zmm12);
        if (InputTile > 5)
            _mm512_storeu_ps(destPtr + 5 * AVX512_PACK_C_UNIT, zmm13);
        if (InputTile > 6)
            _mm512_storeu_ps(destPtr + 6 * AVX512_PACK_C_UNIT, zmm14);
        if (InputTile > 7)
            _mm512_storeu_ps(destPtr + 7 * AVX512_PACK_C_UNIT, zmm15);

        // oc = 32
        auto destOC32Ptr = destPtr + 1 * cStride;
        if (InputTile > 0)
            _mm512_storeu_ps(destOC32Ptr + 0 * AVX512_PACK_C_UNIT, zmm16);
        if (InputTile > 1)
            _mm512_storeu_ps(destOC32Ptr + 1 * AVX512_PACK_C_UNIT, zmm17);
        if (InputTile > 2)
            _mm512_storeu_ps(destOC32Ptr + 2 * AVX512_PACK_C_UNIT, zmm18);
        if (InputTile > 3)
            _mm512_storeu_ps(destOC32Ptr + 3 * AVX512_PACK_C_UNIT, zmm19);
        if (InputTile > 4)
            _mm512_storeu_ps(destOC32Ptr + 4 * AVX512_PACK_C_UNIT, zmm20);
        if (InputTile > 5)
            _mm512_storeu_ps(destOC32Ptr + 5 * AVX512_PACK_C_UNIT, zmm21);
        if (InputTile > 6)
            _mm512_storeu_ps(destOC32Ptr + 6 * AVX512_PACK_C_UNIT, zmm22);
        if (InputTile > 7)
            _mm512_storeu_ps(destOC32Ptr + 7 * AVX512_PACK_C_UNIT, zmm23);

        // oc = 48
        auto destOC48Ptr = destPtr + 2 * cStride;
        if (InputTile > 0)
            _mm512_storeu_ps(destOC48Ptr + 0 * AVX512_PACK_C_UNIT, zmm24);
        if (InputTile > 1)
            _mm512_storeu_ps(destOC48Ptr + 1 * AVX512_PACK_C_UNIT, zmm25);
        if (InputTile > 2)
            _mm512_storeu_ps(destOC48Ptr + 2 * AVX512_PACK_C_UNIT, zmm26);
        if (InputTile > 3)
            _mm512_storeu_ps(destOC48Ptr + 3 * AVX512_PACK_C_UNIT, zmm27);
        if (InputTile > 4)
            _mm512_storeu_ps(destOC48Ptr + 4 * AVX512_PACK_C_UNIT, zmm28);
        if (InputTile > 5)
            _mm512_storeu_ps(destOC48Ptr + 5 * AVX512_PACK_C_UNIT, zmm29);
        if (InputTile > 6)
            _mm512_storeu_ps(destOC48Ptr + 6 * AVX512_PACK_C_UNIT, zmm30);
        if (InputTile > 7)
            _mm512_storeu_ps(destOC48Ptr + 7 * AVX512_PACK_C_UNIT, zmm31);

        inputTilePtr += aStride;
        destPtr += InputTile * AVX512_PACK_C_UNIT;
}

#undef REDUCE_MUL_ADD


}

template <int InputTile>
void _AVX512_MNNPackedMatMulO48Swaped2Kernel(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias) {

#define REDUCE_MUL_ADD(ick)                                                                                          \
    zmm0 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 0 * AVX512_PACK_C_UNIT));                                  \
    zmm1 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 1 * AVX512_PACK_C_UNIT));                                  \
    zmm2 = _mm512_loadu_ps(filterICPtr + (ick * bStride + 2 * AVX512_PACK_C_UNIT));                                  \
    if (InputTile > 7) { /*select the threashhold*/                                                                  \
        _mm_prefetch(filterICPtr + ick * bStride + 0 * AVX512_PACK_C_UNIT + AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                                   \
        _mm_prefetch(filterICPtr + ick * bStride + 1 * AVX512_PACK_C_UNIT + AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                                   \
        _mm_prefetch(filterICPtr + ick * bStride + 2 * AVX512_PACK_C_UNIT + AVX512_PACK_C_UNIT * AVX512_PACK_C_UNIT, \
                     _MM_HINT_T0);                                                                                   \
        _mm_prefetch(inputICPtr + ick * AVX512_PACK_C_UNIT, _MM_HINT_T0);                                            \
    }                                                                                                                \
    if (InputTile > 0) {                                                                                             \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 0 * AVX512_PACK_C_UNIT]);                                          \
        zmm5  = _mm512_fmadd_ps(zmm3, zmm0, zmm5);                                                                   \
        zmm14 = _mm512_fmadd_ps(zmm3, zmm1, zmm14);                                                                  \
        zmm23 = _mm512_fmadd_ps(zmm3, zmm2, zmm23);                                                                  \
    }                                                                                                                \
    if (InputTile > 1) {                                                                                             \
        zmm4  = _mm512_set1_ps(inputICPtr[(ick) + 1 * AVX512_PACK_C_UNIT]);                                          \
        zmm6  = _mm512_fmadd_ps(zmm4, zmm0, zmm6);                                                                   \
        zmm15 = _mm512_fmadd_ps(zmm4, zmm1, zmm15);                                                                  \
        zmm24 = _mm512_fmadd_ps(zmm4, zmm2, zmm24);                                                                  \
    }                                                                                                                \
    if (InputTile > 2) {                                                                                             \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 2 * AVX512_PACK_C_UNIT]);                                          \
        zmm7  = _mm512_fmadd_ps(zmm3, zmm0, zmm7);                                                                   \
        zmm16 = _mm512_fmadd_ps(zmm3, zmm1, zmm16);                                                                  \
        zmm25 = _mm512_fmadd_ps(zmm3, zmm2, zmm25);                                                                  \
    }                                                                                                                \
    if (InputTile > 3) {                                                                                             \
        zmm4  = _mm512_set1_ps(inputICPtr[(ick) + 3 * AVX512_PACK_C_UNIT]);                                          \
        zmm8  = _mm512_fmadd_ps(zmm4, zmm0, zmm8);                                                                   \
        zmm17 = _mm512_fmadd_ps(zmm4, zmm1, zmm17);                                                                  \
        zmm26 = _mm512_fmadd_ps(zmm4, zmm2, zmm26);                                                                  \
    }                                                                                                                \
    if (InputTile > 4) {                                                                                             \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 4 * AVX512_PACK_C_UNIT]);                                          \
        zmm9  = _mm512_fmadd_ps(zmm3, zmm0, zmm9);                                                                   \
        zmm18 = _mm512_fmadd_ps(zmm3, zmm1, zmm18);                                                                  \
        zmm27 = _mm512_fmadd_ps(zmm3, zmm2, zmm27);                                                                  \
    }                                                                                                                \
    if (InputTile > 5) {                                                                                             \
        zmm4  = _mm512_set1_ps(inputICPtr[(ick) + 5 * AVX512_PACK_C_UNIT]);                                          \
        zmm10 = _mm512_fmadd_ps(zmm4, zmm0, zmm10);                                                                  \
        zmm19 = _mm512_fmadd_ps(zmm4, zmm1, zmm19);                                                                  \
        zmm28 = _mm512_fmadd_ps(zmm4, zmm2, zmm28);                                                                  \
    }                                                                                                                \
    if (InputTile > 6) {                                                                                             \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 6 * AVX512_PACK_C_UNIT]);                                          \
        zmm11 = _mm512_fmadd_ps(zmm3, zmm0, zmm11);                                                                  \
        zmm20 = _mm512_fmadd_ps(zmm3, zmm1, zmm20);                                                                  \
        zmm29 = _mm512_fmadd_ps(zmm3, zmm2, zmm29);                                                                  \
    }                                                                                                                \
    if (InputTile > 7) {                                                                                             \
        zmm4  = _mm512_set1_ps(inputICPtr[(ick) + 7 * AVX512_PACK_C_UNIT]);                                          \
        zmm12 = _mm512_fmadd_ps(zmm4, zmm0, zmm12);                                                                  \
        zmm21 = _mm512_fmadd_ps(zmm4, zmm1, zmm21);                                                                  \
        zmm30 = _mm512_fmadd_ps(zmm4, zmm2, zmm30);                                                                  \
    }                                                                                                                \
    if (InputTile > 8) {                                                                                             \
        zmm3  = _mm512_set1_ps(inputICPtr[(ick) + 8 * AVX512_PACK_C_UNIT]);                                          \
        zmm13 = _mm512_fmadd_ps(zmm3, zmm0, zmm13);                                                                  \
        zmm22 = _mm512_fmadd_ps(zmm3, zmm1, zmm22);                                                                  \
        zmm31 = _mm512_fmadd_ps(zmm3, zmm2, zmm31);                                                                  \
    }                                                                                                                \
    if (InputTile > 9)                                                                                               \
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
                zmm5  = _mm512_loadu_ps(biasPtr);
                zmm14 = _mm512_loadu_ps(biasPtr + 1 * AVX512_PACK_C_UNIT);
                zmm23 = _mm512_loadu_ps(biasPtr + 2 * AVX512_PACK_C_UNIT);
                }
            if (InputTile > 1) {
                zmm6  = zmm5 ;
                zmm15 = zmm14;
                zmm24 = zmm23;
            }
            if (InputTile > 2) {
                zmm7  = zmm5 ;
                zmm16 = zmm14;
                zmm25 = zmm23;
            }
            if (InputTile > 3) {
                zmm8  = zmm5 ;
                zmm17 = zmm14;
                zmm26 = zmm23;
            }
            if (InputTile > 4) {
                zmm9  = zmm5 ;
                zmm18 = zmm14;
                zmm27 = zmm23;
            }
            if (InputTile > 5) {
                zmm10 = zmm5 ;
                zmm19 = zmm14;
                zmm28 = zmm23;
            }
            if (InputTile > 6) {
                zmm11 = zmm5 ;
                zmm20 = zmm14;
                zmm29 = zmm23;
            }
            if (InputTile > 7) {
                zmm12 = zmm5 ;
                zmm21 = zmm14;
                zmm30 = zmm23;
            }
            if (InputTile > 8) {
                zmm13 = zmm5 ;
                zmm22 = zmm14;
                zmm31 = zmm23;
            }

        } else {
            if (InputTile > 0 ) {
                zmm5  = _mm512_setzero_ps();
                zmm14 = _mm512_setzero_ps();
                zmm23 = _mm512_setzero_ps();
                }
            if (InputTile > 1) {
                zmm6  = _mm512_setzero_ps();
                zmm15 = _mm512_setzero_ps();
                zmm24 = _mm512_setzero_ps();
            }
            if (InputTile > 2) {
                zmm7  = _mm512_setzero_ps();
                zmm16 = _mm512_setzero_ps();
                zmm25 = _mm512_setzero_ps();
            }
            if (InputTile > 3) {
                zmm8  = _mm512_setzero_ps();
                zmm17 = _mm512_setzero_ps();
                zmm26 = _mm512_setzero_ps();
            }
            if (InputTile > 4) {
                zmm9  = _mm512_setzero_ps();
                zmm18 = _mm512_setzero_ps();
                zmm27 = _mm512_setzero_ps();
            }
            if (InputTile > 5) {
                zmm10 = _mm512_setzero_ps();
                zmm19 = _mm512_setzero_ps();
                zmm28 = _mm512_setzero_ps();
            }
            if (InputTile > 6) {
                zmm11 = _mm512_setzero_ps();
                zmm20 = _mm512_setzero_ps();
                zmm29 = _mm512_setzero_ps();
            }
            if (InputTile > 7) {
                zmm12 = _mm512_setzero_ps();
                zmm21 = _mm512_setzero_ps();
                zmm30 = _mm512_setzero_ps();
            }
            if (InputTile > 8) {
                zmm13 = _mm512_setzero_ps();
                zmm22 = _mm512_setzero_ps();
                zmm31 = _mm512_setzero_ps();
            }

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
        for(int ick = 0; ick < icTail; ++ick) {
            REDUCE_MUL_ADD(ick);
        }

        // write
        // oc = 16;
        if (InputTile > 0 ) _mm512_storeu_ps(destPtr + 0  * AVX512_PACK_C_UNIT, zmm5 );
        if (InputTile > 1 ) _mm512_storeu_ps(destPtr + 1  * AVX512_PACK_C_UNIT, zmm6 );
        if (InputTile > 2 ) _mm512_storeu_ps(destPtr + 2  * AVX512_PACK_C_UNIT, zmm7 );
        if (InputTile > 3 ) _mm512_storeu_ps(destPtr + 3  * AVX512_PACK_C_UNIT, zmm8 );
        if (InputTile > 4 ) _mm512_storeu_ps(destPtr + 4  * AVX512_PACK_C_UNIT, zmm9 );
        if (InputTile > 5 ) _mm512_storeu_ps(destPtr + 5  * AVX512_PACK_C_UNIT, zmm10);
        if (InputTile > 6 ) _mm512_storeu_ps(destPtr + 6  * AVX512_PACK_C_UNIT, zmm11);
        if (InputTile > 7 ) _mm512_storeu_ps(destPtr + 7  * AVX512_PACK_C_UNIT, zmm12);
        if (InputTile > 8 ) _mm512_storeu_ps(destPtr + 8  * AVX512_PACK_C_UNIT, zmm13);

        // oc = 32
        auto destOC32Ptr = destPtr + 1 * cStride;
        if (InputTile > 0 ) _mm512_storeu_ps(destOC32Ptr + 0  * AVX512_PACK_C_UNIT, zmm14);
        if (InputTile > 1 ) _mm512_storeu_ps(destOC32Ptr + 1  * AVX512_PACK_C_UNIT, zmm15);
        if (InputTile > 2 ) _mm512_storeu_ps(destOC32Ptr + 2  * AVX512_PACK_C_UNIT, zmm16);
        if (InputTile > 3 ) _mm512_storeu_ps(destOC32Ptr + 3  * AVX512_PACK_C_UNIT, zmm17);
        if (InputTile > 4 ) _mm512_storeu_ps(destOC32Ptr + 4  * AVX512_PACK_C_UNIT, zmm18);
        if (InputTile > 5 ) _mm512_storeu_ps(destOC32Ptr + 5  * AVX512_PACK_C_UNIT, zmm19);
        if (InputTile > 6 ) _mm512_storeu_ps(destOC32Ptr + 6  * AVX512_PACK_C_UNIT, zmm20);
        if (InputTile > 7 ) _mm512_storeu_ps(destOC32Ptr + 7  * AVX512_PACK_C_UNIT, zmm21);
        if (InputTile > 8 ) _mm512_storeu_ps(destOC32Ptr + 8  * AVX512_PACK_C_UNIT, zmm22);

        // oc = 48
        auto destOC48Ptr = destPtr + 2 * cStride;
        if (InputTile > 0 ) _mm512_storeu_ps(destOC48Ptr + 0  * AVX512_PACK_C_UNIT, zmm23);
        if (InputTile > 1 ) _mm512_storeu_ps(destOC48Ptr + 1  * AVX512_PACK_C_UNIT, zmm24);
        if (InputTile > 2 ) _mm512_storeu_ps(destOC48Ptr + 2  * AVX512_PACK_C_UNIT, zmm25);
        if (InputTile > 3 ) _mm512_storeu_ps(destOC48Ptr + 3  * AVX512_PACK_C_UNIT, zmm26);
        if (InputTile > 4 ) _mm512_storeu_ps(destOC48Ptr + 4  * AVX512_PACK_C_UNIT, zmm27);
        if (InputTile > 5 ) _mm512_storeu_ps(destOC48Ptr + 5  * AVX512_PACK_C_UNIT, zmm28);
        if (InputTile > 6 ) _mm512_storeu_ps(destOC48Ptr + 6  * AVX512_PACK_C_UNIT, zmm29);
        if (InputTile > 7 ) _mm512_storeu_ps(destOC48Ptr + 7  * AVX512_PACK_C_UNIT, zmm30);
        if (InputTile > 8 ) _mm512_storeu_ps(destOC48Ptr + 8  * AVX512_PACK_C_UNIT, zmm31);

        inputTilePtr += aStride;
        destPtr += InputTile * AVX512_PACK_C_UNIT;
    }

#undef REDUCE_MUL_ADD


}

