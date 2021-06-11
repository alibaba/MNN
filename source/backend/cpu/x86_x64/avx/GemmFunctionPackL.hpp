//
//  GemmFunctionPackL.hpp
//  MNN
//
//  Created by MNN on 2021/02/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

namespace {
static inline __m128i mm_loadu_si128(const void* addr) {
    return _mm_castps_si128(LOAD4((const float*)addr));
}

static inline __m256i mm256_broadcastsi128_si256(const void* addr) {
    return _mm256_broadcastsi128_si256(mm_loadu_si128(addr));
}
}  // namespace
//

template <typename TYPE>
static void _AVX_MNNPackedMatMul_3(TYPE* C, const TYPE* A, const TYPE* B, const size_t* parameter) {
    auto aStride      = 3;
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(TYPE);
    auto lC8          = UP_DIV(l, 8);
    auto hC4          = UP_DIV(h, 4);
    const int hC4Unit = 4;
    auto src = A;
    __m256 temp;
    for (int y = 0; y < hC4; ++y) {
        auto S00    = _mm256_xor_ps(temp, temp);
        auto S01    = _mm256_xor_ps(temp, temp);
        auto S02    = _mm256_xor_ps(temp, temp);
        auto S03    = _mm256_xor_ps(temp, temp);

        auto S10    = _mm256_xor_ps(temp, temp);
        auto S11    = _mm256_xor_ps(temp, temp);
        auto S12    = _mm256_xor_ps(temp, temp);
        auto S13    = _mm256_xor_ps(temp, temp);

        auto S20    = _mm256_xor_ps(temp, temp);
        auto S21    = _mm256_xor_ps(temp, temp);
        auto S22    = _mm256_xor_ps(temp, temp);
        auto S23    = _mm256_xor_ps(temp, temp);

        auto srcUse = src;
        for (int sy = 0; sy < lC8; ++sy) {
            auto s0 = LOAD8(srcUse + 0 * 8);
            auto s1 = LOAD8(srcUse + 1 * 8);
            auto s2 = LOAD8(srcUse + 2 * 8);
            temp = LOAD8(B + 0);
            S00 = MNNAVXFMA(s0, temp, S00);
            S10 = MNNAVXFMA(s1, temp, S10);
            S20 = MNNAVXFMA(s2, temp, S20);
            temp = LOAD8(B + 8);
            S01 = MNNAVXFMA(s0, temp, S01);
            S11 = MNNAVXFMA(s1, temp, S11);
            S21 = MNNAVXFMA(s2, temp, S21);
            temp = LOAD8(B + 16);
            S02 = MNNAVXFMA(s0, temp, S02);
            S12 = MNNAVXFMA(s1, temp, S12);
            S22 = MNNAVXFMA(s2, temp, S22);
            temp = LOAD8(B + 24);
            S03 = MNNAVXFMA(s0, temp, S03);
            S13 = MNNAVXFMA(s1, temp, S13);
            S23 = MNNAVXFMA(s2, temp, S23);

            B+=32;
            srcUse += aStride * 8;
        }
        
        // Hadd
        S00 = _mm256_hadd_ps(S00, S01);
        S02 = _mm256_hadd_ps(S02, S03);
        S00 = _mm256_hadd_ps(S00, S02);
        STORE_4(C + 0, _mm_add_ps(_mm256_extractf128_ps(S00, 0), _mm256_extractf128_ps(S00, 1)));

        S10 = _mm256_hadd_ps(S10, S11);
        S12 = _mm256_hadd_ps(S12, S13);
        S00 = _mm256_hadd_ps(S10, S12);
        STORE_4(C + 4, _mm_add_ps(_mm256_extractf128_ps(S00, 0), _mm256_extractf128_ps(S00, 1)));

        S20 = _mm256_hadd_ps(S20, S21);
        S22 = _mm256_hadd_ps(S22, S23);
        S00 = _mm256_hadd_ps(S20, S22);
        STORE_4(C + 8, _mm_add_ps(_mm256_extractf128_ps(S00, 0), _mm256_extractf128_ps(S00, 1)));
        
        B+=bExtraStride;
        C+=cStride;
    }
}


template <typename TYPE>
static void _AVX_MNNPackednMatMulRemainCommon(TYPE* C, const TYPE* A, const TYPE* B, size_t eSize,
                                              const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(TYPE);
    auto lC8          = UP_DIV(l, 8);
    auto hC4          = UP_DIV(h, 4);
    const int hC4Unit = 4;
    auto src = A;
    __m256 temp;
    if (eSize == 2) {
        for (int y = 0; y < hC4; ++y) {
            auto S00    = _mm256_xor_ps(temp, temp);
            auto S01    = _mm256_xor_ps(temp, temp);
            auto S02    = _mm256_xor_ps(temp, temp);
            auto S03    = _mm256_xor_ps(temp, temp);

            auto S10    = _mm256_xor_ps(temp, temp);
            auto S11    = _mm256_xor_ps(temp, temp);
            auto S12    = _mm256_xor_ps(temp, temp);
            auto S13    = _mm256_xor_ps(temp, temp);

            auto srcUse = src;
            for (int sy = 0; sy < lC8; ++sy) {
                auto s0 = LOAD8(srcUse + 0 * 8);
                auto s1 = LOAD8(srcUse + 1 * 8);
                temp = LOAD8(B + 0);
                S00 = MNNAVXFMA(s0, temp, S00);
                S10 = MNNAVXFMA(s1, temp, S10);
                temp = LOAD8(B + 8);
                S01 = MNNAVXFMA(s0, temp, S01);
                S11 = MNNAVXFMA(s1, temp, S11);
                temp = LOAD8(B + 16);
                S02 = MNNAVXFMA(s0, temp, S02);
                S12 = MNNAVXFMA(s1, temp, S12);
                temp = LOAD8(B + 24);
                S03 = MNNAVXFMA(s0, temp, S03);
                S13 = MNNAVXFMA(s1, temp, S13);

                B+=32;
                srcUse += aStride * 8;
            }
            
            // Hadd
            S00 = _mm256_hadd_ps(S00, S01);
            S02 = _mm256_hadd_ps(S02, S03);
            S00 = _mm256_hadd_ps(S00, S02);
            STORE_4(C + 0, _mm_add_ps(_mm256_extractf128_ps(S00, 0), _mm256_extractf128_ps(S00, 1)));

            S10 = _mm256_hadd_ps(S10, S11);
            S12 = _mm256_hadd_ps(S12, S13);
            S00 = _mm256_hadd_ps(S10, S12);
            STORE_4(C + 4, _mm_add_ps(_mm256_extractf128_ps(S00, 0), _mm256_extractf128_ps(S00, 1)));

            B+=bExtraStride;
            C+=cStride;
        }
    }
    if (eSize == 1) {
        for (int y = 0; y < hC4; ++y) {
            auto S00    = _mm256_xor_ps(temp, temp);
            auto S01    = _mm256_xor_ps(temp, temp);
            auto S02    = _mm256_xor_ps(temp, temp);
            auto S03    = _mm256_xor_ps(temp, temp);

            auto srcUse = src;
            for (int sy = 0; sy < lC8; ++sy) {
                auto s0 = LOAD8(srcUse + 0 * 8);
                temp = LOAD8(B + 0);
                S00 = MNNAVXFMA(s0, temp, S00);
                temp = LOAD8(B + 8);
                S01 = MNNAVXFMA(s0, temp, S01);
                temp = LOAD8(B + 16);
                S02 = MNNAVXFMA(s0, temp, S02);
                temp = LOAD8(B + 24);
                S03 = MNNAVXFMA(s0, temp, S03);

                B+=32;
                srcUse += aStride * 8;
            }
            
            // Hadd
            S00 = _mm256_hadd_ps(S00, S01);
            S02 = _mm256_hadd_ps(S02, S03);
            S00 = _mm256_hadd_ps(S00, S02);
            STORE_4(C + 0, _mm_add_ps(_mm256_extractf128_ps(S00, 0), _mm256_extractf128_ps(S00, 1)));

            B+=bExtraStride;
            C+=cStride;
        }
    }
}
