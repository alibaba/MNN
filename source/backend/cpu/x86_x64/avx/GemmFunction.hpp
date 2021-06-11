//
//  GemmFunction.hpp
//  MNN
//
//  Created by MNN on 2020/09/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define MNN_UNIT_E 24
#define TRANPOSE_SAVE(u, v, z0, z3, z6, z9)              \
    {                                                    \
        auto m0 = _mm256_extractf128_ps(z0, u);          \
        auto m1 = _mm256_extractf128_ps(z3, u);          \
        auto m2 = _mm256_extractf128_ps(z6, u);          \
        auto m3 = _mm256_extractf128_ps(z9, u);          \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);               \
        STORE_4(dst + 8 * (0 + 4 * u + 8 * v), m0); \
        STORE_4(dst + 8 * (1 + 4 * u + 8 * v), m1); \
        STORE_4(dst + 8 * (2 + 4 * u + 8 * v), m2); \
        STORE_4(dst + 8 * (3 + 4 * u + 8 * v), m3); \
    }

namespace {
static inline __m128i mm_loadu_si128(const void* addr) {
    return _mm_castps_si128(LOAD4((const float*)addr));
}

static inline __m256i mm256_broadcastsi128_si256(const void* addr) {
    return _mm256_broadcastsi128_si256(mm_loadu_si128(addr));
}
}  // namespace
//
#define INIT_MAIN_24_4                                  \
    auto s0  = LOAD8(A + 0 * 24);             \
    auto s1  = LOAD8(A + 0 * 24 + 8);         \
    auto s2  = LOAD8(A + 0 * 24 + 16);        \
    auto w0  = BROAD_LOAD(weight + 0 * 4 + 0); \
    auto z0  = _mm256_mul_ps(s0, w0);                   \
    auto z1  = _mm256_mul_ps(s1, w0);                   \
    auto z2  = _mm256_mul_ps(s2, w0);                   \
    auto w1  = BROAD_LOAD(weight + 0 * 4 + 1); \
    auto z3  = _mm256_mul_ps(s0, w1);                   \
    auto z4  = _mm256_mul_ps(s1, w1);                   \
    auto z5  = _mm256_mul_ps(s2, w1);                   \
    auto w2  = BROAD_LOAD(weight + 0 * 4 + 2); \
    auto z6  = _mm256_mul_ps(s0, w2);                   \
    auto z7  = _mm256_mul_ps(s1, w2);                   \
    auto z8  = _mm256_mul_ps(s2, w2);                   \
    auto w3  = BROAD_LOAD(weight + 0 * 4 + 3); \
    auto z9  = _mm256_mul_ps(s0, w3);                   \
    auto z10 = _mm256_mul_ps(s1, w3);                   \
    auto z11 = _mm256_mul_ps(s2, w3);

#define COMPUTE_24_4                                \
    s0  = LOAD8(A + sy * 24);             \
    s1  = LOAD8(A + sy * 24 + 8);         \
    s2  = LOAD8(A + sy * 24 + 16);        \
    w0  = BROAD_LOAD(weight + sy * 4 + 0); \
    z0  = MNNAVXFMA(s0, w0, z0);                    \
    z1  = MNNAVXFMA(s1, w0, z1);                    \
    z2  = MNNAVXFMA(s2, w0, z2);                    \
    w1  = BROAD_LOAD(weight + sy * 4 + 1); \
    z3  = MNNAVXFMA(s0, w1, z3);                    \
    z4  = MNNAVXFMA(s1, w1, z4);                    \
    z5  = MNNAVXFMA(s2, w1, z5);                    \
    w2  = BROAD_LOAD(weight + sy * 4 + 2); \
    z6  = MNNAVXFMA(s0, w2, z6);                    \
    z7  = MNNAVXFMA(s1, w2, z7);                    \
    z8  = MNNAVXFMA(s2, w2, z8);                    \
    w3  = BROAD_LOAD(weight + sy * 4 + 3); \
    z9  = MNNAVXFMA(s0, w3, z9);                    \
    z10 = MNNAVXFMA(s1, w3, z10);                   \
    z11 = MNNAVXFMA(s2, w3, z11);

template <typename TYPE>
static void _AVX_MNNPackedMatMul_Main(TYPE* C, const TYPE* A, const TYPE* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(TYPE);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
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


#define EXPAND_128(x) _mm256_castsi256_ps(_mm256_broadcastsi128_si256(_mm_castps_si128((x))))
//
#define INIT_MAIN_20_4                                  \
    auto s0  = LOAD8(A + 0 * aStride);             \
    auto s1  = LOAD8(A + 0 * aStride + 8);         \
    auto s2  = EXPAND_128(LOAD4(A + 0 * aStride + 16));        \
    auto w0  = BROAD_LOAD(weight + 0 * 4 + 0); \
    auto z0  = _mm256_mul_ps(s0, w0);                   \
    auto z1  = _mm256_mul_ps(s1, w0);                   \
    auto z2  = _mm256_mul_ps(s2, w0);                   \
    auto w1  = BROAD_LOAD(weight + 0 * 4 + 1); \
    auto z3  = _mm256_mul_ps(s0, w1);                   \
    auto z4  = _mm256_mul_ps(s1, w1);                   \
    auto z5  = _mm256_mul_ps(s2, w1);                   \
    auto w2  = BROAD_LOAD(weight + 0 * 4 + 2); \
    auto z6  = _mm256_mul_ps(s0, w2);                   \
    auto z7  = _mm256_mul_ps(s1, w2);                   \
    auto z8  = _mm256_mul_ps(s2, w2);                   \
    auto w3  = BROAD_LOAD(weight + 0 * 4 + 3); \
    auto z9  = _mm256_mul_ps(s0, w3);                   \
    auto z10 = _mm256_mul_ps(s1, w3);                   \
    auto z11 = _mm256_mul_ps(s2, w3);

#define COMPUTE_20_4                                \
    s0  = LOAD8(A + sy * aStride);             \
    s1  = LOAD8(A + sy * aStride + 8);         \
    s2  = EXPAND_128(LOAD4(A + sy * aStride + 16)); \
    w0  = BROAD_LOAD(weight + sy * 4 + 0); \
    z0  = MNNAVXFMA(s0, w0, z0);                    \
    z1  = MNNAVXFMA(s1, w0, z1);                    \
    z2  = MNNAVXFMA(s2, w0, z2);                    \
    w1  = BROAD_LOAD(weight + sy * 4 + 1); \
    z3  = MNNAVXFMA(s0, w1, z3);                    \
    z4  = MNNAVXFMA(s1, w1, z4);                    \
    z5  = MNNAVXFMA(s2, w1, z5);                    \
    w2  = BROAD_LOAD(weight + sy * 4 + 2); \
    z6  = MNNAVXFMA(s0, w2, z6);                    \
    z7  = MNNAVXFMA(s1, w2, z7);                    \
    z8  = MNNAVXFMA(s2, w2, z8);                    \
    w3  = BROAD_LOAD(weight + sy * 4 + 3); \
    z9  = MNNAVXFMA(s0, w3, z9);                    \
    z10 = MNNAVXFMA(s1, w3, z10);                   \
    z11 = MNNAVXFMA(s2, w3, z11);


template <typename TYPE>
static void _AVX_MNNPackedMatMul_20(TYPE* C, const TYPE* A, const TYPE* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(TYPE);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        INIT_MAIN_20_4;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_20_4;
        }
        TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
        TRANPOSE_SAVE(1, 0, z0, z3, z6, z9);
        TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
        TRANPOSE_SAVE(1, 1, z1, z4, z7, z10);
        TRANPOSE_SAVE(0, 2, z2, z5, z8, z11);
    }
}

#define INIT_MAIN_16_4                                  \
    auto s0  = LOAD8(A + 0 * aStride);        \
    auto s1  = LOAD8(A + 0 * aStride + 8);    \
    auto w0  = BROAD_LOAD(weight + 0 * 4 + 0); \
    auto z0  = _mm256_mul_ps(s0, w0);                   \
    auto z1  = _mm256_mul_ps(s1, w0);                   \
    auto w1  = BROAD_LOAD(weight + 0 * 4 + 1); \
    auto z3  = _mm256_mul_ps(s0, w1);                   \
    auto z4  = _mm256_mul_ps(s1, w1);                   \
    auto w2  = BROAD_LOAD(weight + 0 * 4 + 2); \
    auto z6  = _mm256_mul_ps(s0, w2);                   \
    auto z7  = _mm256_mul_ps(s1, w2);                   \
    auto w3  = BROAD_LOAD(weight + 0 * 4 + 3); \
    auto z9  = _mm256_mul_ps(s0, w3);                   \
    auto z10 = _mm256_mul_ps(s1, w3);

#define COMPUTE_16_4                                \
    s0  = LOAD8(A + sy * aStride);        \
    s1  = LOAD8(A + sy * aStride + 8);    \
    w0  = BROAD_LOAD(weight + sy * 4 + 0); \
    z0  = MNNAVXFMA(s0, w0, z0);                    \
    z1  = MNNAVXFMA(s1, w0, z1);                    \
    w1  = BROAD_LOAD(weight + sy * 4 + 1); \
    z3  = MNNAVXFMA(s0, w1, z3);                    \
    z4  = MNNAVXFMA(s1, w1, z4);                    \
    w2  = BROAD_LOAD(weight + sy * 4 + 2); \
    z6  = MNNAVXFMA(s0, w2, z6);                    \
    z7  = MNNAVXFMA(s1, w2, z7);                    \
    w3  = BROAD_LOAD(weight + sy * 4 + 3); \
    z9  = MNNAVXFMA(s0, w3, z9);                    \
    z10 = MNNAVXFMA(s1, w3, z10);

template <typename TYPE>
static void _AVX_MNNPackedMatMul_16(TYPE* C, const TYPE* A, const TYPE* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(TYPE);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
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

#define DST_ADDR_UNPACK4(x)\
auto dst0    = C + (hC4Unit * y / 2 + 0) * cStride + x * 8;\
auto dst1    = C + (hC4Unit * y / 2 + 0) * cStride + x * 8 + 4;\
auto dst2    = C + (hC4Unit * y / 2 + 1) * cStride + x * 8;\
auto dst3    = C + (hC4Unit * y / 2 + 1) * cStride + x * 8 + 4;\

template <typename TYPE>
static void _AVX_MNNPackedMatMul_5(TYPE* C, const TYPE* A, const TYPE* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(TYPE);
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
        auto weight1 = B + (hC4Unit * y + 1) * bStride;
        auto weight2 = B + (hC4Unit * y + 2) * bStride;
        auto weight3 = B + (hC4Unit * y + 3) * bStride;
        DST_ADDR_UNPACK4(0);
        auto sumAvx00    = _mm256_setzero_ps();
        auto sumAvx01    = _mm256_setzero_ps();

        auto sumAvx10    = _mm256_setzero_ps();
        auto sumAvx11    = _mm256_setzero_ps();

        auto sumAvx20    = _mm256_setzero_ps();
        auto sumAvx21    = _mm256_setzero_ps();

        auto sumAvx30    = _mm256_setzero_ps();
        auto sumAvx31    = _mm256_setzero_ps();

        auto sumAvx40    = _mm256_setzero_ps();
        auto sumAvx41    = _mm256_setzero_ps();

        auto srcUse = src;
        for (int sy = 0; sy < l; ++sy) {
            auto S0 = BROAD_LOAD(srcUse + 0);
            auto S1 = BROAD_LOAD(srcUse + 1);
            auto S2 = BROAD_LOAD(srcUse + 2);
            auto S3 = BROAD_LOAD(srcUse + 3);
            auto S4 = BROAD_LOAD(srcUse + 4);
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
        STORE_8(dst0, sumAvx00);
        STORE_8(dst0 + 8, sumAvx10);
        STORE_8(dst0 + 16, sumAvx20);
        STORE_8(dst0 + 24, sumAvx30);
        STORE_8(dst0 + 32, sumAvx40);

        STORE_8(dst2, sumAvx01);
        STORE_8(dst2 + 8, sumAvx11);
        STORE_8(dst2 + 16, sumAvx21);
        STORE_8(dst2 + 24, sumAvx31);
        STORE_8(dst2 + 32, sumAvx41);
    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto s0     = BROAD_LOAD_4(A + 0 * aStride + 0);
        auto s1     = BROAD_LOAD_4(A + 0 * aStride + 1);
        auto s2     = BROAD_LOAD_4(A + 0 * aStride + 2);
        auto s3     = BROAD_LOAD_4(A + 0 * aStride + 3);
        auto s4     = BROAD_LOAD_4(A + 0 * aStride + 4);
        auto w0     = LOAD4(weight + 0 * 4);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z1     = _mm_mul_ps(s1, w0);
        auto z2     = _mm_mul_ps(s2, w0);
        auto z3     = _mm_mul_ps(s3, w0);
        auto z4     = _mm_mul_ps(s4, w0);

        for (int sy = 1; sy < l; ++sy) {
            s0 = BROAD_LOAD_4(A + sy * aStride + 0);
            s1 = BROAD_LOAD_4(A + sy * aStride + 1);
            s2 = BROAD_LOAD_4(A + sy * aStride + 2);
            s3 = BROAD_LOAD_4(A + sy * aStride + 3);
            s4 = BROAD_LOAD_4(A + sy * aStride + 4);
            w0 = LOAD4(weight + sy * 4);
            z0 = MNNSSEFMA(s0, w0, z0);
            z1 = MNNSSEFMA(s1, w0, z1);
            z2 = MNNSSEFMA(s2, w0, z2);
            z3 = MNNSSEFMA(s3, w0, z3);
            z4 = MNNSSEFMA(s4, w0, z4);
        }
        STORE_4(dst + 8 * 0, z0);
        STORE_4(dst + 8 * 1, z1);
        STORE_4(dst + 8 * 2, z2);
        STORE_4(dst + 8 * 3, z3);
        STORE_4(dst + 8 * 4, z4);
    }
}

template <typename TYPE>
static void _AVX_MNNPackedMatMul_3(TYPE* C, const TYPE* A, const TYPE* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(TYPE);
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
        auto weight1 = B + (hC4Unit * y + 1) * bStride;
        auto weight2 = B + (hC4Unit * y + 2) * bStride;
        auto weight3 = B + (hC4Unit * y + 3) * bStride;
        auto sumAvx00    = _mm256_setzero_ps();
        auto sumAvx01    = _mm256_setzero_ps();

        auto sumAvx10    = _mm256_setzero_ps();
        auto sumAvx11    = _mm256_setzero_ps();

        auto sumAvx20    = _mm256_setzero_ps();
        auto sumAvx21    = _mm256_setzero_ps();

        DST_ADDR_UNPACK4(0);

        auto srcUse = src;
        for (int sy = 0; sy < l; ++sy) {
            auto S0 = BROAD_LOAD(srcUse + 0);
            auto S1 = BROAD_LOAD(srcUse + 1);
            auto S2 = BROAD_LOAD(srcUse + 2);
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
        STORE_4(dst0 + 0, _mm256_extractf128_ps(sumAvx00, 0));
        STORE_4(dst0 + 8, _mm256_extractf128_ps(sumAvx10, 0));
        STORE_4(dst0 + 16, _mm256_extractf128_ps(sumAvx20, 0));

        STORE_4(dst1 + 0, _mm256_extractf128_ps(sumAvx00, 1));
        STORE_4(dst1 + 8, _mm256_extractf128_ps(sumAvx10, 1));
        STORE_4(dst1 + 16, _mm256_extractf128_ps(sumAvx20, 1));

        STORE_4(dst2 + 0, _mm256_extractf128_ps(sumAvx01, 0));
        STORE_4(dst2 + 8, _mm256_extractf128_ps(sumAvx11, 0));
        STORE_4(dst2 + 16, _mm256_extractf128_ps(sumAvx21, 0));

        STORE_4(dst3 + 0, _mm256_extractf128_ps(sumAvx01, 1));
        STORE_4(dst3 + 8, _mm256_extractf128_ps(sumAvx11, 1));
        STORE_4(dst3 + 16, _mm256_extractf128_ps(sumAvx21, 1));

    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto s0     = BROAD_LOAD_4(A + 0 * aStride + 0);
        auto s1     = BROAD_LOAD_4(A + 0 * aStride + 1);
        auto s2     = BROAD_LOAD_4(A + 0 * aStride + 2);
        auto w0     = LOAD4(weight + 0 * 4);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z1     = _mm_mul_ps(s1, w0);
        auto z2     = _mm_mul_ps(s2, w0);

        for (int sy = 1; sy < l; ++sy) {
            s0 = BROAD_LOAD_4(A + sy * aStride + 0);
            s1 = BROAD_LOAD_4(A + sy * aStride + 1);
            s2 = BROAD_LOAD_4(A + sy * aStride + 2);
            w0 = LOAD4(weight + sy * 4);
            z0 = MNNSSEFMA(s0, w0, z0);
            z1 = MNNSSEFMA(s1, w0, z1);
            z2 = MNNSSEFMA(s2, w0, z2);
        }
        STORE_4(dst + 8 * 0, z0);
        STORE_4(dst + 8 * 1, z1);
        STORE_4(dst + 8 * 2, z2);
    }
}
template <typename TYPE>
static void _AVX_MNNPackedMatMul_2(TYPE* C, const TYPE* A, const TYPE* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(TYPE);
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
        auto weight1 = B + (hC4Unit * y + 1) * bStride;
        auto weight2 = B + (hC4Unit * y + 2) * bStride;
        auto weight3 = B + (hC4Unit * y + 3) * bStride;
        auto sumAvx00    = _mm256_setzero_ps();
        auto sumAvx01    = _mm256_setzero_ps();
        DST_ADDR_UNPACK4(0);

        auto sumAvx10    = _mm256_setzero_ps();
        auto sumAvx11    = _mm256_setzero_ps();

        auto srcUse = src;
        for (int sy = 0; sy < l; ++sy) {
            auto S0 = BROAD_LOAD(srcUse + 0);
            auto S1 = BROAD_LOAD(srcUse + 1);
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
        STORE_4(dst0 + 0, _mm256_extractf128_ps(sumAvx00, 0));
        STORE_4(dst0 + 8, _mm256_extractf128_ps(sumAvx10, 0));

        STORE_4(dst1 + 0, _mm256_extractf128_ps(sumAvx00, 1));
        STORE_4(dst1 + 8, _mm256_extractf128_ps(sumAvx10, 1));

        STORE_4(dst2 + 0, _mm256_extractf128_ps(sumAvx01, 0));
        STORE_4(dst2 + 8, _mm256_extractf128_ps(sumAvx11, 0));

        STORE_4(dst3 + 0, _mm256_extractf128_ps(sumAvx01, 1));
        STORE_4(dst3 + 8, _mm256_extractf128_ps(sumAvx11, 1));

    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto s0     = BROAD_LOAD_4(A + 0 * aStride + 0);
        auto s1     = BROAD_LOAD_4(A + 0 * aStride + 1);
        auto w0     = LOAD4(weight + 0 * 4);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z1     = _mm_mul_ps(s1, w0);

        for (int sy = 1; sy < l; ++sy) {
            s0 = BROAD_LOAD_4(A + sy * aStride + 0);
            s1 = BROAD_LOAD_4(A + sy * aStride + 1);
            w0 = LOAD4(weight + sy * 4);
            z0 = MNNSSEFMA(s0, w0, z0);
            z1 = MNNSSEFMA(s1, w0, z1);
        }
        STORE_4(dst + 8 * 0, z0);
        STORE_4(dst + 8 * 1, z1);
    }
}

template <typename TYPE>
static void _AVX_MNNPackedMatMul_4(TYPE* C, const TYPE* A, const TYPE* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(TYPE);
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
        auto weight1 = B + (hC4Unit * y + 1) * bStride;
        auto weight2 = B + (hC4Unit * y + 2) * bStride;
        auto weight3 = B + (hC4Unit * y + 3) * bStride;
        DST_ADDR_UNPACK4(0);

        auto sumAvx00    = _mm256_setzero_ps();
        auto sumAvx01    = _mm256_setzero_ps();

        auto sumAvx10    = _mm256_setzero_ps();
        auto sumAvx11    = _mm256_setzero_ps();

        auto sumAvx20    = _mm256_setzero_ps();
        auto sumAvx21    = _mm256_setzero_ps();

        auto sumAvx30    = _mm256_setzero_ps();
        auto sumAvx31    = _mm256_setzero_ps();

        auto srcUse = src;
        for (int sy = 0; sy < l; ++sy) {
            auto S0 = BROAD_LOAD(srcUse + 0);
            auto S1 = BROAD_LOAD(srcUse + 1);
            auto S2 = BROAD_LOAD(srcUse + 2);
            auto S3 = BROAD_LOAD(srcUse + 3);
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

            srcUse += aStride;
            weight0 += 4;
            weight1 += 4;
            weight2 += 4;
            weight3 += 4;
        }
        STORE_8(dst0, sumAvx00);
        STORE_8(dst0 + 8, sumAvx10);
        STORE_8(dst0 + 16, sumAvx20);
        STORE_8(dst0 + 24, sumAvx30);

        STORE_8(dst2, sumAvx01);
        STORE_8(dst2 + 8, sumAvx11);
        STORE_8(dst2 + 16, sumAvx21);
        STORE_8(dst2 + 24, sumAvx31);
    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto s0     = LOAD4(A + 0 * aStride);
        auto w0     = BROAD_LOAD_4(weight + 0 * 4 + 0);
        auto w1     = BROAD_LOAD_4(weight + 0 * 4 + 1);
        auto w2     = BROAD_LOAD_4(weight + 0 * 4 + 2);
        auto w3     = BROAD_LOAD_4(weight + 0 * 4 + 3);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z3     = _mm_mul_ps(s0, w1);
        auto z6     = _mm_mul_ps(s0, w2);
        auto z9     = _mm_mul_ps(s0, w3);

        for (int sy = 1; sy < l; ++sy) {
            s0 = LOAD4(A + sy * aStride);
            w0 = BROAD_LOAD_4(weight + sy * 4 + 0);
            w1 = BROAD_LOAD_4(weight + sy * 4 + 1);
            w2 = BROAD_LOAD_4(weight + sy * 4 + 2);
            w3 = BROAD_LOAD_4(weight + sy * 4 + 3);
            z0 = MNNSSEFMA(s0, w0, z0);
            z3 = MNNSSEFMA(s0, w1, z3);
            z6 = MNNSSEFMA(s0, w2, z6);
            z9 = MNNSSEFMA(s0, w3, z9);
        }
        _MM_TRANSPOSE4_PS(z0, z3, z6, z9);
        STORE_4(dst + 8 * 0, z0);
        STORE_4(dst + 8 * 1, z3);
        STORE_4(dst + 8 * 2, z6);
        STORE_4(dst + 8 * 3, z9);
    }
}
template <typename TYPE>
static void _AVX_MNNPackednMatMulRemainCommon(TYPE* C, const TYPE* A, const TYPE* B, size_t eSize,
                                              const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(TYPE);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    auto es           = eSize;
    auto oC           = C;
    auto aStride      = parameter[0] / sizeof(TYPE);
    if (eSize >= 20) {
        _AVX_MNNPackedMatMul_20<TYPE>(C, A, B, parameter);
        eSize -= 20;
        C += 20 * 8;
        A += 20;
    }
    if (eSize >= 16) {
        _AVX_MNNPackedMatMul_16<TYPE>(C, A, B, parameter);
        eSize -= 16;
        C += 16 * 8;
        A += 16;
    }
    while (eSize >= 5) {
        _AVX_MNNPackedMatMul_5<TYPE>(C, A, B, parameter);
        eSize -= 5;
        C += 5 * 8;
        A += 5;
    }
    if (eSize == 4) {
        _AVX_MNNPackedMatMul_4<TYPE>(C, A, B, parameter);
        return;
    }
    if (eSize == 3) {
        _AVX_MNNPackedMatMul_3<TYPE>(C, A, B, parameter);
        return;
    }
    if (eSize == 2) {
        _AVX_MNNPackedMatMul_2<TYPE>(C, A, B, parameter);
        return;
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
        auto weight1 = B + (hC4Unit * y + 1) * bStride;
        auto weight2 = B + (hC4Unit * y + 2) * bStride;
        auto weight3 = B + (hC4Unit * y + 3) * bStride;
        auto dst0    = C + (hC4Unit * y / 2 + 0) * cStride + x * 8;
        auto dst1    = C + (hC4Unit * y / 2 + 0) * cStride + x * 8 + 4;
        auto dst2    = C + (hC4Unit * y / 2 + 1) * cStride + x * 8;
        auto dst3    = C + (hC4Unit * y / 2 + 1) * cStride + x * 8 + 4;
        auto sumAvx00    = _mm256_setzero_ps();
        auto sumAvx01    = _mm256_setzero_ps();

        auto sumAvx10    = _mm256_setzero_ps();
        auto sumAvx11    = _mm256_setzero_ps();

        auto sumAvx20    = _mm256_setzero_ps();
        auto sumAvx21    = _mm256_setzero_ps();

        auto sumAvx30    = _mm256_setzero_ps();
        auto sumAvx31    = _mm256_setzero_ps();

        auto srcUse = src;
        for (int sy = 0; sy < lC4; ++sy) {
            auto s0 = _mm256_castps_si256(BROAD_LOAD(srcUse + (0) * aStride));
            auto s1 = _mm_castps_si128(BROAD_LOAD_4(srcUse + (1) * aStride));
            auto S0 = _mm256_castsi256_ps(_mm256_insertf128_si256(s0, s1, 1));
            auto d0 = _mm256_castps_si256(BROAD_LOAD(srcUse + (2) * aStride));
            auto d1 = _mm_castps_si128(BROAD_LOAD_4(srcUse + (3) * aStride));
            auto S1 = _mm256_castsi256_ps(_mm256_insertf128_si256(d0, d1, 1));
            auto W00 = LOAD8(weight0 + 16 * sy + 0);
            auto W01 = LOAD8(weight0 + 16 * sy + 8);
            auto W10 = LOAD8(weight1 + 16 * sy + 0);
            auto W11 = LOAD8(weight1 + 16 * sy + 8);

            auto W20 = LOAD8(weight2 + 16 * sy + 0);
            auto W21 = LOAD8(weight2 + 16 * sy + 8);
            auto W30 = LOAD8(weight3 + 16 * sy + 0);
            auto W31 = LOAD8(weight3 + 16 * sy + 8);

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
            auto s = BROAD_LOAD_4(srcUse);
            auto w0 = LOAD4(weight0 + 4 * sy);
            auto w1 = LOAD4(weight1 + 4 * sy);
            auto w2 = LOAD4(weight2 + 4 * sy);
            auto w3 = LOAD4(weight3 + 4 * sy);
            sum0    = MNNSSEFMA(s, w0, sum0);
            sum1    = MNNSSEFMA(s, w1, sum1);
            sum2    = MNNSSEFMA(s, w2, sum2);
            sum3    = MNNSSEFMA(s, w3, sum3);
            srcUse += aStride;
        }
        STORE_4(dst0, sum0);
        STORE_4(dst1, sum1);
        STORE_4(dst2, sum2);
        STORE_4(dst3, sum3);
    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + x * 8 + 4 * (y % 2);
        auto sumAvx0    = _mm256_setzero_ps();
        auto sumAvx1    = _mm256_setzero_ps();
        auto srcUse = src;
        for (int sy = 0; sy < lC4; ++sy) {
            auto s0 = _mm256_castps_si256(BROAD_LOAD(srcUse + (0) * aStride));
            auto s1 = _mm_castps_si128(BROAD_LOAD_4(srcUse + (1) * aStride));
            auto S0 = _mm256_castsi256_ps(_mm256_insertf128_si256(s0, s1, 1));
            auto d0 = _mm256_castps_si256(BROAD_LOAD(srcUse + (2) * aStride));
            auto d1 = _mm_castps_si128(BROAD_LOAD_4(srcUse + (3) * aStride));
            auto S1 = _mm256_castsi256_ps(_mm256_insertf128_si256(d0, d1, 1));
            auto W0 = LOAD8(weight + 16 * sy + 0);
            auto W1 = LOAD8(weight + 16 * sy + 8);
            sumAvx0   = MNNAVXFMA(S0, W0, sumAvx0);
            sumAvx1   = MNNAVXFMA(S1, W1, sumAvx1);
            srcUse += 4 * aStride;
        }
        sumAvx0 = _mm256_add_ps(sumAvx0, sumAvx1);
        auto sum0 = _mm256_extractf128_ps(sumAvx0, 0);
        auto sum1 = _mm256_extractf128_ps(sumAvx0, 1);
        auto sum = _mm_add_ps(sum0, sum1);
        for (int sy = lR; sy < l; ++sy) {
            auto s = BROAD_LOAD_4(srcUse);
            auto w = LOAD4(weight + 4 * sy);
            sum    = MNNSSEFMA(s, w, sum);
            srcUse += aStride;
        }
        STORE_4(dst, sum);
    }
}
