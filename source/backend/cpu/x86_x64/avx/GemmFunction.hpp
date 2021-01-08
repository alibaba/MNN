//
//  GemmFunction.hpp
//  MNN
//
//  Created by MNN on b'2020/09/22'.
//  Copyright © 2018, Alibaba Group Holding Limited

namespace {
static inline __m128i mm_loadu_si128(const void* addr) {
    return _mm_loadu_si128((__m128i const*)addr);
}

static inline __m256i mm256_broadcastsi128_si256(const void* addr) {
    return _mm256_broadcastsi128_si256(mm_loadu_si128(addr));
}
}  // namespace
//

#define INIT_MAIN_24_4                                  \
    auto s0  = _mm256_loadu_ps(A + 0 * 24);             \
    auto s1  = _mm256_loadu_ps(A + 0 * 24 + 8);         \
    auto s2  = _mm256_loadu_ps(A + 0 * 24 + 16);        \
    auto w0  = _mm256_broadcast_ss(weight + 0 * 4 + 0); \
    auto z0  = _mm256_mul_ps(s0, w0);                   \
    auto z1  = _mm256_mul_ps(s1, w0);                   \
    auto z2  = _mm256_mul_ps(s2, w0);                   \
    auto w1  = _mm256_broadcast_ss(weight + 0 * 4 + 1); \
    auto z3  = _mm256_mul_ps(s0, w1);                   \
    auto z4  = _mm256_mul_ps(s1, w1);                   \
    auto z5  = _mm256_mul_ps(s2, w1);                   \
    auto w2  = _mm256_broadcast_ss(weight + 0 * 4 + 2); \
    auto z6  = _mm256_mul_ps(s0, w2);                   \
    auto z7  = _mm256_mul_ps(s1, w2);                   \
    auto z8  = _mm256_mul_ps(s2, w2);                   \
    auto w3  = _mm256_broadcast_ss(weight + 0 * 4 + 3); \
    auto z9  = _mm256_mul_ps(s0, w3);                   \
    auto z10 = _mm256_mul_ps(s1, w3);                   \
    auto z11 = _mm256_mul_ps(s2, w3);

#define COMPUTE_24_4                                \
    s0  = _mm256_loadu_ps(A + sy * 24);             \
    s1  = _mm256_loadu_ps(A + sy * 24 + 8);         \
    s2  = _mm256_loadu_ps(A + sy * 24 + 16);        \
    w0  = _mm256_broadcast_ss(weight + sy * 4 + 0); \
    z0  = MNNAVXFMA(s0, w0, z0);                    \
    z1  = MNNAVXFMA(s1, w0, z1);                    \
    z2  = MNNAVXFMA(s2, w0, z2);                    \
    w1  = _mm256_broadcast_ss(weight + sy * 4 + 1); \
    z3  = MNNAVXFMA(s0, w1, z3);                    \
    z4  = MNNAVXFMA(s1, w1, z4);                    \
    z5  = MNNAVXFMA(s2, w1, z5);                    \
    w2  = _mm256_broadcast_ss(weight + sy * 4 + 2); \
    z6  = MNNAVXFMA(s0, w2, z6);                    \
    z7  = MNNAVXFMA(s1, w2, z7);                    \
    z8  = MNNAVXFMA(s2, w2, z8);                    \
    w3  = _mm256_broadcast_ss(weight + sy * 4 + 3); \
    z9  = MNNAVXFMA(s0, w3, z9);                    \
    z10 = MNNAVXFMA(s1, w3, z10);                   \
    z11 = MNNAVXFMA(s2, w3, z11);

static void _AVX_MNNPackedMatMul_24(float* C, const float* A, const float* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        INIT_MAIN_24_4;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_24_4;
        }
        TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
        TRANPOSE_SAVE(1, 0, z0, z3, z6, z9);
        TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
        TRANPOSE_SAVE(1, 1, z1, z4, z7, z10);
        TRANPOSE_SAVE(0, 2, z2, z5, z8, z11);
        TRANPOSE_SAVE(1, 2, z2, z5, z8, z11);
    }
}

#define INIT_MAIN_16_4                                  \
    auto s0  = _mm256_loadu_ps(A + 0 * aStride);        \
    auto s1  = _mm256_loadu_ps(A + 0 * aStride + 8);    \
    auto w0  = _mm256_broadcast_ss(weight + 0 * 4 + 0); \
    auto z0  = _mm256_mul_ps(s0, w0);                   \
    auto z1  = _mm256_mul_ps(s1, w0);                   \
    auto w1  = _mm256_broadcast_ss(weight + 0 * 4 + 1); \
    auto z3  = _mm256_mul_ps(s0, w1);                   \
    auto z4  = _mm256_mul_ps(s1, w1);                   \
    auto w2  = _mm256_broadcast_ss(weight + 0 * 4 + 2); \
    auto z6  = _mm256_mul_ps(s0, w2);                   \
    auto z7  = _mm256_mul_ps(s1, w2);                   \
    auto w3  = _mm256_broadcast_ss(weight + 0 * 4 + 3); \
    auto z9  = _mm256_mul_ps(s0, w3);                   \
    auto z10 = _mm256_mul_ps(s1, w3);

#define COMPUTE_16_4                                \
    s0  = _mm256_loadu_ps(A + sy * aStride);        \
    s1  = _mm256_loadu_ps(A + sy * aStride + 8);    \
    w0  = _mm256_broadcast_ss(weight + sy * 4 + 0); \
    z0  = MNNAVXFMA(s0, w0, z0);                    \
    z1  = MNNAVXFMA(s1, w0, z1);                    \
    w1  = _mm256_broadcast_ss(weight + sy * 4 + 1); \
    z3  = MNNAVXFMA(s0, w1, z3);                    \
    z4  = MNNAVXFMA(s1, w1, z4);                    \
    w2  = _mm256_broadcast_ss(weight + sy * 4 + 2); \
    z6  = MNNAVXFMA(s0, w2, z6);                    \
    z7  = MNNAVXFMA(s1, w2, z7);                    \
    w3  = _mm256_broadcast_ss(weight + sy * 4 + 3); \
    z9  = MNNAVXFMA(s0, w3, z9);                    \
    z10 = MNNAVXFMA(s1, w3, z10);

static void _AVX_MNNPackedMatMul_16(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        INIT_MAIN_16_4;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_16_4;
        }
        TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
        TRANPOSE_SAVE(1, 0, z0, z3, z6, z9);
        TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
        TRANPOSE_SAVE(1, 1, z1, z4, z7, z10);
    }
}

#define INIT_MAIN_8_4                                  \
    auto s0 = _mm256_loadu_ps(A + 0 * aStride);        \
    auto w0 = _mm256_broadcast_ss(weight + 0 * 4 + 0); \
    auto w1 = _mm256_broadcast_ss(weight + 0 * 4 + 1); \
    auto w2 = _mm256_broadcast_ss(weight + 0 * 4 + 2); \
    auto w3 = _mm256_broadcast_ss(weight + 0 * 4 + 3); \
    auto z0 = _mm256_mul_ps(s0, w0);                   \
    auto z3 = _mm256_mul_ps(s0, w1);                   \
    auto z6 = _mm256_mul_ps(s0, w2);                   \
    auto z9 = _mm256_mul_ps(s0, w3);

#define COMPUTE_8_4                                \
    s0 = _mm256_loadu_ps(A + sy * aStride);        \
    w0 = _mm256_broadcast_ss(weight + sy * 4 + 0); \
    w1 = _mm256_broadcast_ss(weight + sy * 4 + 1); \
    w2 = _mm256_broadcast_ss(weight + sy * 4 + 2); \
    w3 = _mm256_broadcast_ss(weight + sy * 4 + 3); \
    z0 = MNNAVXFMA(s0, w0, z0);                    \
    z3 = MNNAVXFMA(s0, w1, z3);                    \
    z6 = MNNAVXFMA(s0, w2, z6);                    \
    z9 = MNNAVXFMA(s0, w3, z9);

static void _AVX_MNNPackedMatMul_8(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        INIT_MAIN_8_4;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_8_4;
        }
        TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
        TRANPOSE_SAVE(1, 0, z0, z3, z6, z9);
    }
}
static void _AVX_MNNPackedMatMul_5(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    int lC4 = l / 4;
    int lR = lC4 * 4;
    const int hC4Unit = 4;
    int hC16 = hC4 / hC4Unit;
    int hR = hC16 * hC4Unit;
    auto src = A;
    for (int y = 0; y < hC16; ++y) {
        auto weight0 = B + (hC4Unit * y + 0) * bStride;
        auto dst0    = C + (hC4Unit * y + 0) * cStride;
        auto weight1 = B + (hC4Unit * y + 1) * bStride;
        auto dst1    = C + (hC4Unit * y + 1) * cStride;
        auto weight2 = B + (hC4Unit * y + 2) * bStride;
        auto dst2    = C + (hC4Unit * y + 2) * cStride;
        auto weight3 = B + (hC4Unit * y + 3) * bStride;
        auto dst3    = C + (hC4Unit * y + 3) * cStride;
        auto sumAvx00    = _mm256_set1_ps(0.0f);
        auto sumAvx01    = _mm256_set1_ps(0.0f);

        auto sumAvx10    = _mm256_set1_ps(0.0f);
        auto sumAvx11    = _mm256_set1_ps(0.0f);

        auto sumAvx20    = _mm256_set1_ps(0.0f);
        auto sumAvx21    = _mm256_set1_ps(0.0f);

        auto sumAvx30    = _mm256_set1_ps(0.0f);
        auto sumAvx31    = _mm256_set1_ps(0.0f);

        auto sumAvx40    = _mm256_set1_ps(0.0f);
        auto sumAvx41    = _mm256_set1_ps(0.0f);

        auto srcUse = src;
        for (int sy = 0; sy < l; ++sy) {
            auto S0 = _mm256_broadcast_ss(srcUse + 0);
            auto S1 = _mm256_broadcast_ss(srcUse + 1);
            auto S2 = _mm256_broadcast_ss(srcUse + 2);
            auto S3 = _mm256_broadcast_ss(srcUse + 3);
            auto S4 = _mm256_broadcast_ss(srcUse + 4);
            auto W0 =  _mm256_castsi256_ps(_mm256_insertf128_si256(mm256_broadcastsi128_si256(weight0), mm_loadu_si128(weight1), 1));
            auto W1 =  _mm256_castsi256_ps(_mm256_insertf128_si256(mm256_broadcastsi128_si256(weight2), mm_loadu_si128(weight3), 1));

            sumAvx00   = MNNAVXFMA(S0, W0, sumAvx00);
            sumAvx01   = MNNAVXFMA(S0, W1, sumAvx01);

            sumAvx10   = MNNAVXFMA(S1, W0, sumAvx10);
            sumAvx11   = MNNAVXFMA(S1, W1, sumAvx11);

            sumAvx20   = MNNAVXFMA(S2, W0, sumAvx20);
            sumAvx21   = MNNAVXFMA(S2, W1, sumAvx21);

            sumAvx30   = MNNAVXFMA(S3, W0, sumAvx30);
            sumAvx31   = MNNAVXFMA(S3, W1, sumAvx31);

            sumAvx40   = MNNAVXFMA(S4, W0, sumAvx40);
            sumAvx41   = MNNAVXFMA(S4, W1, sumAvx41);

            srcUse += aStride;
            weight0 += 4;
            weight1 += 4;
            weight2 += 4;
            weight3 += 4;
        }
        _mm256_storeu_ps(dst0 + 0, _mm256_permute2f128_ps(sumAvx00, sumAvx10, 32));
        _mm256_storeu_ps(dst0 + 8, _mm256_permute2f128_ps(sumAvx20, sumAvx30, 32));
        _mm_storeu_ps(dst0 + 16, _mm256_extractf128_ps(sumAvx40, 0));

        _mm256_storeu_ps(dst1 + 0, _mm256_permute2f128_ps(sumAvx00, sumAvx10, 49));
        _mm256_storeu_ps(dst1 + 8, _mm256_permute2f128_ps(sumAvx20, sumAvx30, 49));
        _mm_storeu_ps(dst1 + 16, _mm256_extractf128_ps(sumAvx40, 1));

        _mm256_storeu_ps(dst2 + 0, _mm256_permute2f128_ps(sumAvx01, sumAvx11, 32));
        _mm256_storeu_ps(dst2 + 8, _mm256_permute2f128_ps(sumAvx21, sumAvx31, 32));
        _mm_storeu_ps(dst2 + 16, _mm256_extractf128_ps(sumAvx41, 0));

        _mm256_storeu_ps(dst3 + 0, _mm256_permute2f128_ps(sumAvx01, sumAvx11, 49));
        _mm256_storeu_ps(dst3 + 8, _mm256_permute2f128_ps(sumAvx21, sumAvx31, 49));
        _mm_storeu_ps(dst3 + 16, _mm256_extractf128_ps(sumAvx41, 1));
    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        auto s0     = _mm_broadcast_ss(A + 0 * aStride + 0);
        auto s1     = _mm_broadcast_ss(A + 0 * aStride + 1);
        auto s2     = _mm_broadcast_ss(A + 0 * aStride + 2);
        auto s3     = _mm_broadcast_ss(A + 0 * aStride + 3);
        auto s4     = _mm_broadcast_ss(A + 0 * aStride + 4);
        auto w0     = _mm_loadu_ps(weight + 0 * 4);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z1     = _mm_mul_ps(s1, w0);
        auto z2     = _mm_mul_ps(s2, w0);
        auto z3     = _mm_mul_ps(s3, w0);
        auto z4     = _mm_mul_ps(s4, w0);

        for (int sy = 1; sy < l; ++sy) {
            s0 = _mm_broadcast_ss(A + sy * aStride + 0);
            s1 = _mm_broadcast_ss(A + sy * aStride + 1);
            s2 = _mm_broadcast_ss(A + sy * aStride + 2);
            s3 = _mm_broadcast_ss(A + sy * aStride + 3);
            s4 = _mm_broadcast_ss(A + sy * aStride + 4);
            w0 = _mm_loadu_ps(weight + sy * 4);
            z0 = MNNSSEFMA(s0, w0, z0);
            z1 = MNNSSEFMA(s1, w0, z1);
            z2 = MNNSSEFMA(s2, w0, z2);
            z3 = MNNSSEFMA(s3, w0, z3);
            z4 = MNNSSEFMA(s4, w0, z4);
        }
        _mm_store_ps(dst + 4 * 0, z0);
        _mm_store_ps(dst + 4 * 1, z1);
        _mm_store_ps(dst + 4 * 2, z2);
        _mm_store_ps(dst + 4 * 3, z3);
        _mm_store_ps(dst + 4 * 4, z4);
    }
}


static void _AVX_MNNPackedMatMul_3(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    int lC4 = l / 4;
    int lR = lC4 * 4;
    const int hC4Unit = 4;
    int hC16 = hC4 / hC4Unit;
    int hR = hC16 * hC4Unit;
    auto src = A;
    for (int y = 0; y < hC16; ++y) {
        auto weight0 = B + (hC4Unit * y + 0) * bStride;
        auto dst0    = C + (hC4Unit * y + 0) * cStride;
        auto weight1 = B + (hC4Unit * y + 1) * bStride;
        auto dst1    = C + (hC4Unit * y + 1) * cStride;
        auto weight2 = B + (hC4Unit * y + 2) * bStride;
        auto dst2    = C + (hC4Unit * y + 2) * cStride;
        auto weight3 = B + (hC4Unit * y + 3) * bStride;
        auto dst3    = C + (hC4Unit * y + 3) * cStride;
        auto sumAvx00    = _mm256_set1_ps(0.0f);
        auto sumAvx01    = _mm256_set1_ps(0.0f);

        auto sumAvx10    = _mm256_set1_ps(0.0f);
        auto sumAvx11    = _mm256_set1_ps(0.0f);

        auto sumAvx20    = _mm256_set1_ps(0.0f);
        auto sumAvx21    = _mm256_set1_ps(0.0f);

        auto srcUse = src;
        for (int sy = 0; sy < l; ++sy) {
            auto S0 = _mm256_broadcast_ss(srcUse + 0);
            auto S1 = _mm256_broadcast_ss(srcUse + 1);
            auto S2 = _mm256_broadcast_ss(srcUse + 2);
            auto W0 =  _mm256_castsi256_ps(_mm256_insertf128_si256(mm256_broadcastsi128_si256(weight0), mm_loadu_si128(weight1), 1));
            auto W1 =  _mm256_castsi256_ps(_mm256_insertf128_si256(mm256_broadcastsi128_si256(weight2), mm_loadu_si128(weight3), 1));

            sumAvx00   = MNNAVXFMA(S0, W0, sumAvx00);
            sumAvx01   = MNNAVXFMA(S0, W1, sumAvx01);

            sumAvx10   = MNNAVXFMA(S1, W0, sumAvx10);
            sumAvx11   = MNNAVXFMA(S1, W1, sumAvx11);

            sumAvx20   = MNNAVXFMA(S2, W0, sumAvx20);
            sumAvx21   = MNNAVXFMA(S2, W1, sumAvx21);

            srcUse += aStride;
            weight0 += 4;
            weight1 += 4;
            weight2 += 4;
            weight3 += 4;
        }
        _mm_storeu_ps(dst0 + 0, _mm256_extractf128_ps(sumAvx00, 0));
        _mm_storeu_ps(dst0 + 4, _mm256_extractf128_ps(sumAvx10, 0));
        _mm_storeu_ps(dst0 + 8, _mm256_extractf128_ps(sumAvx20, 0));

        _mm_storeu_ps(dst1 + 0, _mm256_extractf128_ps(sumAvx00, 1));
        _mm_storeu_ps(dst1 + 4, _mm256_extractf128_ps(sumAvx10, 1));
        _mm_storeu_ps(dst1 + 8, _mm256_extractf128_ps(sumAvx20, 1));

        _mm_storeu_ps(dst2 + 0, _mm256_extractf128_ps(sumAvx01, 0));
        _mm_storeu_ps(dst2 + 4, _mm256_extractf128_ps(sumAvx11, 0));
        _mm_storeu_ps(dst2 + 8, _mm256_extractf128_ps(sumAvx21, 0));

        _mm_storeu_ps(dst3 + 0, _mm256_extractf128_ps(sumAvx01, 1));
        _mm_storeu_ps(dst3 + 4, _mm256_extractf128_ps(sumAvx11, 1));
        _mm_storeu_ps(dst3 + 8, _mm256_extractf128_ps(sumAvx21, 1));

    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        auto s0     = _mm_broadcast_ss(A + 0 * aStride + 0);
        auto s1     = _mm_broadcast_ss(A + 0 * aStride + 1);
        auto s2     = _mm_broadcast_ss(A + 0 * aStride + 2);
        auto w0     = _mm_loadu_ps(weight + 0 * 4);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z1     = _mm_mul_ps(s1, w0);
        auto z2     = _mm_mul_ps(s2, w0);

        for (int sy = 1; sy < l; ++sy) {
            s0 = _mm_broadcast_ss(A + sy * aStride + 0);
            s1 = _mm_broadcast_ss(A + sy * aStride + 1);
            s2 = _mm_broadcast_ss(A + sy * aStride + 2);
            w0 = _mm_loadu_ps(weight + sy * 4);
            z0 = MNNSSEFMA(s0, w0, z0);
            z1 = MNNSSEFMA(s1, w0, z1);
            z2 = MNNSSEFMA(s2, w0, z2);
        }
        _mm_store_ps(dst + 4 * 0, z0);
        _mm_store_ps(dst + 4 * 1, z1);
        _mm_store_ps(dst + 4 * 2, z2);
    }
}

static void _AVX_MNNPackedMatMul_2(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    int lC4 = l / 4;
    int lR = lC4 * 4;
    const int hC4Unit = 4;
    int hC16 = hC4 / hC4Unit;
    int hR = hC16 * hC4Unit;
    auto src = A;
    for (int y = 0; y < hC16; ++y) {
        auto weight0 = B + (hC4Unit * y + 0) * bStride;
        auto dst0    = C + (hC4Unit * y + 0) * cStride;
        auto weight1 = B + (hC4Unit * y + 1) * bStride;
        auto dst1    = C + (hC4Unit * y + 1) * cStride;
        auto weight2 = B + (hC4Unit * y + 2) * bStride;
        auto dst2    = C + (hC4Unit * y + 2) * cStride;
        auto weight3 = B + (hC4Unit * y + 3) * bStride;
        auto dst3    = C + (hC4Unit * y + 3) * cStride;
        auto sumAvx00    = _mm256_set1_ps(0.0f);
        auto sumAvx01    = _mm256_set1_ps(0.0f);

        auto sumAvx10    = _mm256_set1_ps(0.0f);
        auto sumAvx11    = _mm256_set1_ps(0.0f);

        auto srcUse = src;
        for (int sy = 0; sy < l; ++sy) {
            auto S0 = _mm256_broadcast_ss(srcUse + 0);
            auto S1 = _mm256_broadcast_ss(srcUse + 1);
            auto W0 =  _mm256_castsi256_ps(_mm256_insertf128_si256(mm256_broadcastsi128_si256(weight0), mm_loadu_si128(weight1), 1));
            auto W1 =  _mm256_castsi256_ps(_mm256_insertf128_si256(mm256_broadcastsi128_si256(weight2), mm_loadu_si128(weight3), 1));

            sumAvx00   = MNNAVXFMA(S0, W0, sumAvx00);
            sumAvx01   = MNNAVXFMA(S0, W1, sumAvx01);

            sumAvx10   = MNNAVXFMA(S1, W0, sumAvx10);
            sumAvx11   = MNNAVXFMA(S1, W1, sumAvx11);

            srcUse += aStride;
            weight0 += 4;
            weight1 += 4;
            weight2 += 4;
            weight3 += 4;
        }
        _mm_storeu_ps(dst0 + 0, _mm256_extractf128_ps(sumAvx00, 0));
        _mm_storeu_ps(dst0 + 4, _mm256_extractf128_ps(sumAvx10, 0));

        _mm_storeu_ps(dst1 + 0, _mm256_extractf128_ps(sumAvx00, 1));
        _mm_storeu_ps(dst1 + 4, _mm256_extractf128_ps(sumAvx10, 1));

        _mm_storeu_ps(dst2 + 0, _mm256_extractf128_ps(sumAvx01, 0));
        _mm_storeu_ps(dst2 + 4, _mm256_extractf128_ps(sumAvx11, 0));

        _mm_storeu_ps(dst3 + 0, _mm256_extractf128_ps(sumAvx01, 1));
        _mm_storeu_ps(dst3 + 4, _mm256_extractf128_ps(sumAvx11, 1));

    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        auto s0     = _mm_broadcast_ss(A + 0 * aStride + 0);
        auto s1     = _mm_broadcast_ss(A + 0 * aStride + 1);
        auto w0     = _mm_loadu_ps(weight + 0 * 4);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z1     = _mm_mul_ps(s1, w0);

        for (int sy = 1; sy < l; ++sy) {
            s0 = _mm_broadcast_ss(A + sy * aStride + 0);
            s1 = _mm_broadcast_ss(A + sy * aStride + 1);
            w0 = _mm_loadu_ps(weight + sy * 4);
            z0 = MNNSSEFMA(s0, w0, z0);
            z1 = MNNSSEFMA(s1, w0, z1);
        }
        _mm_store_ps(dst + 4 * 0, z0);
        _mm_store_ps(dst + 4 * 1, z1);
    }
}

static void _AVX_MNNPackedMatMul_4(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    int lC4 = l / 4;
    int lR = lC4 * 4;
    const int hC4Unit = 4;
    int hC16 = hC4 / hC4Unit;
    int hR = hC16 * hC4Unit;
    auto src = A;
    for (int y = 0; y < hC16; ++y) {
        auto weight0 = B + (hC4Unit * y + 0) * bStride;
        auto dst0    = C + (hC4Unit * y + 0) * cStride;
        auto weight1 = B + (hC4Unit * y + 1) * bStride;
        auto dst1    = C + (hC4Unit * y + 1) * cStride;
        auto weight2 = B + (hC4Unit * y + 2) * bStride;
        auto dst2    = C + (hC4Unit * y + 2) * cStride;
        auto weight3 = B + (hC4Unit * y + 3) * bStride;
        auto dst3    = C + (hC4Unit * y + 3) * cStride;
        auto sumAvx00    = _mm256_set1_ps(0.0f);
        auto sumAvx01    = _mm256_set1_ps(0.0f);

        auto sumAvx10    = _mm256_set1_ps(0.0f);
        auto sumAvx11    = _mm256_set1_ps(0.0f);

        auto sumAvx20    = _mm256_set1_ps(0.0f);
        auto sumAvx21    = _mm256_set1_ps(0.0f);

        auto sumAvx30    = _mm256_set1_ps(0.0f);
        auto sumAvx31    = _mm256_set1_ps(0.0f);

        auto srcUse = src;
        for (int sy = 0; sy < l; ++sy) {
#define LOAD_S_4(i) \
auto s##i##0 = _mm256_castps_si256(_mm256_broadcast_ss(srcUse + (i) * aStride + 0));\
auto s##i##1 = _mm_castps_si128(_mm_broadcast_ss(srcUse + (i) * aStride + 1));\
auto S##i##0 = _mm256_castsi256_ps(_mm256_insertf128_si256(s##i##0, s##i##1, 1));\
s##i##0 = _mm256_castps_si256(_mm256_broadcast_ss(srcUse + (i) * aStride + 2));\
s##i##1 = _mm_castps_si128(_mm_broadcast_ss(srcUse + (i) * aStride + 3));\
auto S##i##1 = _mm256_castsi256_ps(_mm256_insertf128_si256(s##i##0, s##i##1, 1));\

            LOAD_S_4(0);
#undef LOAD_S_4
            auto W0 =  _mm256_castsi256_ps(mm256_broadcastsi128_si256(weight0));
            auto W1 =  _mm256_castsi256_ps(mm256_broadcastsi128_si256(weight1));
            auto W2 =  _mm256_castsi256_ps(mm256_broadcastsi128_si256(weight2));
            auto W3 =  _mm256_castsi256_ps(mm256_broadcastsi128_si256(weight3));

            sumAvx00   = MNNAVXFMA(S00, W0, sumAvx00);
            sumAvx01   = MNNAVXFMA(S01, W0, sumAvx01);

            sumAvx10   = MNNAVXFMA(S00, W1, sumAvx10);
            sumAvx11   = MNNAVXFMA(S01, W1, sumAvx11);

            sumAvx20   = MNNAVXFMA(S00, W2, sumAvx20);
            sumAvx21   = MNNAVXFMA(S01, W2, sumAvx21);

            sumAvx30   = MNNAVXFMA(S00, W3, sumAvx30);
            sumAvx31   = MNNAVXFMA(S01, W3, sumAvx31);
            srcUse += aStride;
            weight0 += 4;
            weight1 += 4;
            weight2 += 4;
            weight3 += 4;
        }
        _mm256_storeu_ps(dst0, sumAvx00);
        _mm256_storeu_ps(dst0 + 8, sumAvx01);
        _mm256_storeu_ps(dst1, sumAvx10);
        _mm256_storeu_ps(dst1 + 8, sumAvx11);
        _mm256_storeu_ps(dst2, sumAvx20);
        _mm256_storeu_ps(dst2 + 8, sumAvx21);
        _mm256_storeu_ps(dst3, sumAvx30);
        _mm256_storeu_ps(dst3 + 8, sumAvx31);
    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        auto s0     = _mm_loadu_ps(A + 0 * aStride);
        auto w0     = _mm_broadcast_ss(weight + 0 * 4 + 0);
        auto w1     = _mm_broadcast_ss(weight + 0 * 4 + 1);
        auto w2     = _mm_broadcast_ss(weight + 0 * 4 + 2);
        auto w3     = _mm_broadcast_ss(weight + 0 * 4 + 3);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z3     = _mm_mul_ps(s0, w1);
        auto z6     = _mm_mul_ps(s0, w2);
        auto z9     = _mm_mul_ps(s0, w3);

        for (int sy = 1; sy < l; ++sy) {
            s0 = _mm_loadu_ps(A + sy * aStride);
            w0 = _mm_broadcast_ss(weight + sy * 4 + 0);
            w1 = _mm_broadcast_ss(weight + sy * 4 + 1);
            w2 = _mm_broadcast_ss(weight + sy * 4 + 2);
            w3 = _mm_broadcast_ss(weight + sy * 4 + 3);
            z0 = MNNSSEFMA(s0, w0, z0);
            z3 = MNNSSEFMA(s0, w1, z3);
            z6 = MNNSSEFMA(s0, w2, z6);
            z9 = MNNSSEFMA(s0, w3, z9);
        }
        _MM_TRANSPOSE4_PS(z0, z3, z6, z9);
        _mm_store_ps(dst + 4 * 0, z0);
        _mm_store_ps(dst + 4 * 1, z3);
        _mm_store_ps(dst + 4 * 2, z6);
        _mm_store_ps(dst + 4 * 3, z9);
    }
}
static void _AVX_MNNPackednMatMulRemainCommon(float* C, const float* A, const float* B, size_t eSize,
                                              const size_t* parameter, float* cache, const float* postParameters,
                                              const float* bias) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    auto es           = eSize;
    auto oC           = C;
    auto aStride      = parameter[0] / sizeof(float);
    if (eSize >= 16) {
        _AVX_MNNPackedMatMul_16(C, A, B, parameter);
        eSize -= 16;
        C += 16 * 4;
        A += 16;
    }
    if (eSize >= 8) {
        _AVX_MNNPackedMatMul_8(C, A, B, parameter);
        eSize -= 8;
        C += 8 * 4;
        A += 8;
    }
    if (eSize >= 5) {
        _AVX_MNNPackedMatMul_5(C, A, B, parameter);
        eSize -= 5;
        C += 5 * 4;
        A += 5;
    }
    if (eSize == 4) {
        _AVX_MNNPackedMatMul_4(C, A, B, parameter);
        eSize -= 4;
        C += 4 * 4;
        A += 4;
    }
    if (eSize == 3) {
        _AVX_MNNPackedMatMul_3(C, A, B, parameter);
        eSize -= 3;
        C += 3 * 4;
        A += 3;
    }
    if (eSize == 2) {
        _AVX_MNNPackedMatMul_2(C, A, B, parameter);
        eSize -= 2;
        C += 2 * 4;
        A += 2;
    }
    if (eSize == 0) {
        return;
    }
    int lC4 = l / 4;
    int lR = lC4 * 4;
    const int hC4Unit = 4;
    int hC16 = hC4 / hC4Unit;
    int hR = hC16 * hC4Unit;
    auto src = A;
    int x = 0;
    for (int y = 0; y < hC16; ++y) {
        auto weight0 = B + (hC4Unit * y + 0) * bStride;
        auto dst0    = C + (hC4Unit * y + 0) * cStride + x * 4;
        auto weight1 = B + (hC4Unit * y + 1) * bStride;
        auto dst1    = C + (hC4Unit * y + 1) * cStride + x * 4;
        auto weight2 = B + (hC4Unit * y + 2) * bStride;
        auto dst2    = C + (hC4Unit * y + 2) * cStride + x * 4;
        auto weight3 = B + (hC4Unit * y + 3) * bStride;
        auto dst3    = C + (hC4Unit * y + 3) * cStride + x * 4;
        auto sumAvx00    = _mm256_set1_ps(0.0f);
        auto sumAvx01    = _mm256_set1_ps(0.0f);

        auto sumAvx10    = _mm256_set1_ps(0.0f);
        auto sumAvx11    = _mm256_set1_ps(0.0f);

        auto sumAvx20    = _mm256_set1_ps(0.0f);
        auto sumAvx21    = _mm256_set1_ps(0.0f);

        auto sumAvx30    = _mm256_set1_ps(0.0f);
        auto sumAvx31    = _mm256_set1_ps(0.0f);

        auto srcUse = src;
        for (int sy = 0; sy < lC4; ++sy) {
            auto s0 = _mm256_castps_si256(_mm256_broadcast_ss(srcUse + (0) * aStride));
            auto s1 = _mm_castps_si128(_mm_broadcast_ss(srcUse + (1) * aStride));
            auto S0 = _mm256_castsi256_ps(_mm256_insertf128_si256(s0, s1, 1));
            auto d0 = _mm256_castps_si256(_mm256_broadcast_ss(srcUse + (2) * aStride));
            auto d1 = _mm_castps_si128(_mm_broadcast_ss(srcUse + (3) * aStride));
            auto S1 = _mm256_castsi256_ps(_mm256_insertf128_si256(d0, d1, 1));
            auto W00 = _mm256_loadu_ps(weight0 + 16 * sy + 0);
            auto W01 = _mm256_loadu_ps(weight0 + 16 * sy + 8);
            auto W10 = _mm256_loadu_ps(weight1 + 16 * sy + 0);
            auto W11 = _mm256_loadu_ps(weight1 + 16 * sy + 8);

            auto W20 = _mm256_loadu_ps(weight2 + 16 * sy + 0);
            auto W21 = _mm256_loadu_ps(weight2 + 16 * sy + 8);
            auto W30 = _mm256_loadu_ps(weight3 + 16 * sy + 0);
            auto W31 = _mm256_loadu_ps(weight3 + 16 * sy + 8);

            sumAvx00   = MNNAVXFMA(S0, W00, sumAvx00);
            sumAvx01   = MNNAVXFMA(S1, W01, sumAvx01);

            sumAvx10   = MNNAVXFMA(S0, W10, sumAvx10);
            sumAvx11   = MNNAVXFMA(S1, W11, sumAvx11);

            sumAvx20   = MNNAVXFMA(S0, W20, sumAvx20);
            sumAvx21   = MNNAVXFMA(S1, W21, sumAvx21);

            sumAvx30   = MNNAVXFMA(S0, W30, sumAvx30);
            sumAvx31   = MNNAVXFMA(S1, W31, sumAvx31);
            srcUse += 4 * aStride;
        }
        sumAvx00 = _mm256_add_ps(sumAvx00, sumAvx01);
        sumAvx10 = _mm256_add_ps(sumAvx10, sumAvx11);
        sumAvx20 = _mm256_add_ps(sumAvx20, sumAvx21);
        sumAvx30 = _mm256_add_ps(sumAvx30, sumAvx31);
        auto sum00 = _mm256_extractf128_ps(sumAvx00, 0);
        auto sum01 = _mm256_extractf128_ps(sumAvx00, 1);
        auto sum0 = _mm_add_ps(sum00, sum01);
        auto sum10 = _mm256_extractf128_ps(sumAvx10, 0);
        auto sum11 = _mm256_extractf128_ps(sumAvx10, 1);
        auto sum1 = _mm_add_ps(sum10, sum11);

        auto sum20 = _mm256_extractf128_ps(sumAvx20, 0);
        auto sum21 = _mm256_extractf128_ps(sumAvx20, 1);
        auto sum2 = _mm_add_ps(sum20, sum21);
        auto sum30 = _mm256_extractf128_ps(sumAvx30, 0);
        auto sum31 = _mm256_extractf128_ps(sumAvx30, 1);
        auto sum3 = _mm_add_ps(sum30, sum31);
        for (int sy = lR; sy < l; ++sy) {
            auto s = _mm_broadcast_ss(srcUse);
            auto w0 = _mm_loadu_ps(weight0 + 4 * sy);
            auto w1 = _mm_loadu_ps(weight1 + 4 * sy);
            auto w2 = _mm_loadu_ps(weight2 + 4 * sy);
            auto w3 = _mm_loadu_ps(weight3 + 4 * sy);
            sum0    = MNNSSEFMA(s, w0, sum0);
            sum1    = MNNSSEFMA(s, w1, sum1);
            sum2    = MNNSSEFMA(s, w2, sum2);
            sum3    = MNNSSEFMA(s, w3, sum3);
            srcUse += aStride;
        }
        _mm_store_ps(dst0, sum0);
        _mm_store_ps(dst1, sum1);
        _mm_store_ps(dst2, sum2);
        _mm_store_ps(dst3, sum3);
    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride + x * 4;
        auto sumAvx0    = _mm256_set1_ps(0.0f);
        auto sumAvx1    = _mm256_set1_ps(0.0f);
        auto srcUse = src;
        for (int sy = 0; sy < lC4; ++sy) {
            auto s0 = _mm256_castps_si256(_mm256_broadcast_ss(srcUse + (0) * aStride));
            auto s1 = _mm_castps_si128(_mm_broadcast_ss(srcUse + (1) * aStride));
            auto S0 = _mm256_castsi256_ps(_mm256_insertf128_si256(s0, s1, 1));
            auto d0 = _mm256_castps_si256(_mm256_broadcast_ss(srcUse + (2) * aStride));
            auto d1 = _mm_castps_si128(_mm_broadcast_ss(srcUse + (3) * aStride));
            auto S1 = _mm256_castsi256_ps(_mm256_insertf128_si256(d0, d1, 1));
            auto W0 = _mm256_loadu_ps(weight + 16 * sy + 0);
            auto W1 = _mm256_loadu_ps(weight + 16 * sy + 8);
            sumAvx0   = MNNAVXFMA(S0, W0, sumAvx0);
            sumAvx1   = MNNAVXFMA(S1, W1, sumAvx1);
            srcUse += 4 * aStride;
        }
        sumAvx0 = _mm256_add_ps(sumAvx0, sumAvx1);
        auto sum0 = _mm256_extractf128_ps(sumAvx0, 0);
        auto sum1 = _mm256_extractf128_ps(sumAvx0, 1);
        auto sum = _mm_add_ps(sum0, sum1);
        for (int sy = lR; sy < l; ++sy) {
            auto s = _mm_broadcast_ss(srcUse);
            auto w = _mm_loadu_ps(weight + 4 * sy);
            sum    = MNNSSEFMA(s, w, sum);
            srcUse += aStride;
        }
        _mm_store_ps(dst, sum);
    }
}
