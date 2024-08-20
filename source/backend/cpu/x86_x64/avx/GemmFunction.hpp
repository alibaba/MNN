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

#define FMLA_TRANSPOSE_SAVE(u, v, z0, z3, z6, z9) \
    { \
        auto tmp_m0 = LOAD4(dst + 8 * (0 + 4 * u + 8 * v)); \
        auto tmp_m1 = LOAD4(dst + 8 * (1 + 4 * u + 8 * v)); \
        auto tmp_m2 = LOAD4(dst + 8 * (2 + 4 * u + 8 * v)); \
        auto tmp_m3 = LOAD4(dst + 8 * (3 + 4 * u + 8 * v)); \
        auto m0 = _mm256_extractf128_ps(z0, u);             \
        auto m1 = _mm256_extractf128_ps(z3, u);             \
        auto m2 = _mm256_extractf128_ps(z6, u);             \
        auto m3 = _mm256_extractf128_ps(z9, u);             \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);                  \
        m0 = _mm_add_ps(tmp_m0, m0);                    \
        m1 = _mm_add_ps(tmp_m1, m1);                    \
        m2 = _mm_add_ps(tmp_m2, m2);                    \
        m3 = _mm_add_ps(tmp_m3, m3);                    \
        STORE_4(dst + 8 * (0 + 4 * u + 8 * v), m0);     \
        STORE_4(dst + 8 * (1 + 4 * u + 8 * v), m1);     \
        STORE_4(dst + 8 * (2 + 4 * u + 8 * v), m2);     \
        STORE_4(dst + 8 * (3 + 4 * u + 8 * v), m3);     \
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

#ifdef MNN_LOW_MEMORY
//----------------------- MatMul(float, int4) Functions ---------------------------//

#define LOAD_WEIGHT_ALPHA_BIAS_int4x4 \
    auto weight0 = B + (hC4Unit * y + 0) * bStride / 2;\
    auto weight1 = B + (hC4Unit * y + 1) * bStride / 2;\
    auto weight2 = B + (hC4Unit * y + 2) * bStride / 2;\
    auto weight3 = B + (hC4Unit * y + 3) * bStride / 2;\
    auto alpha0  = _mm_loadu_ps(k + y * 16 + 0);\
    auto alpha1  = _mm_loadu_ps(k + y * 16 + 4);\
    auto alpha2  = _mm_loadu_ps(k + y * 16 + 8);\
    auto alpha3  = _mm_loadu_ps(k + y * 16 + 12);\
    auto bias0   = _mm_loadu_ps(b + y * 16 + 0);\
    auto bias1   = _mm_loadu_ps(b + y * 16 + 4);\
    auto bias2   = _mm_loadu_ps(b + y * 16 + 8);\
    auto bias3   = _mm_loadu_ps(b + y * 16 + 12);

#define LOAD_ALPHA_BIAS_DOUBLE \
    auto alpha0_2 = _mm256_set_m128(alpha0, alpha0);\
    auto alpha1_2 = _mm256_set_m128(alpha1, alpha1);\
    auto alpha2_2 = _mm256_set_m128(alpha2, alpha2);\
    auto alpha3_2 = _mm256_set_m128(alpha3, alpha3);\
    auto bias0_2  = _mm256_set_m128(bias0, bias0);\
    auto bias1_2  = _mm256_set_m128(bias1, bias1);\
    auto bias2_2  = _mm256_set_m128(bias2, bias2);\
    auto bias3_2  = _mm256_set_m128(bias3, bias3);

static inline __m128 _load_int4x4(const uint8_t* src, __m128 alpha, __m128 bias) {
    auto w01    = src[0];
    auto w23    = src[1];
    int iw01    = w01;
    int iw23    = w23;
    int iw0     = iw01 / 16;
    int iw1     = iw01 % 16;
    int iw2     = iw23 / 16;
    int iw3     = iw23 % 16;
    auto ws     = _mm_set_ps(iw3, iw2, iw1, iw0);
    ws          = _mm_add_ps(_mm_mul_ps(ws, alpha), bias);
    return ws;
}

static inline __m256 _load_int4x8(const uint8_t* src, __m256 alpha, __m256 bias) {
    float w[8];
    for (int i = 0; i < 4; i++) {
        int x = src[i];
        int a = x / 16;
        int b = x % 16;
        w[i * 2] = a;
        w[i * 2 + 1] = b;
    }
    auto w8 = LOAD8(w);
    return _mm256_add_ps(_mm256_mul_ps(w8, alpha), bias);
}


template <typename TYPE>
static void _AVX_MNNPackedMatMul_Main_int4(TYPE* C, const TYPE* A, const TYPE* fB, const size_t* parameter, const float* k, const float* b) {
    auto B            = reinterpret_cast<const uint8_t*>(fB);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    float weightBytes = 0.5; // sizeof(int4_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto hC4          = UP_DIV(h, 4);
    float ws_tmp[4];
    size_t blockId    = parameter[6];
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride / 2;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0  = LOAD8(A + 0 * 24);
        auto s1  = LOAD8(A + 0 * 24 + 8);
        auto s2  = LOAD8(A + 0 * 24 + 16);
        auto ws  = _load_int4x4(weight, alpha, bias);
        _mm_storeu_ps(ws_tmp, ws);
        auto w0  = _mm256_set1_ps(ws_tmp[0]);
        auto w1  = _mm256_set1_ps(ws_tmp[1]);
        auto w2  = _mm256_set1_ps(ws_tmp[2]);
        auto w3  = _mm256_set1_ps(ws_tmp[3]);
        auto z0  = _mm256_mul_ps(s0, w0);
        auto z1  = _mm256_mul_ps(s1, w0);
        auto z2  = _mm256_mul_ps(s2, w0);
        auto z3  = _mm256_mul_ps(s0, w1);
        auto z4  = _mm256_mul_ps(s1, w1);
        auto z5  = _mm256_mul_ps(s2, w1);
        auto z6  = _mm256_mul_ps(s0, w2);
        auto z7  = _mm256_mul_ps(s1, w2);
        auto z8  = _mm256_mul_ps(s2, w2);
        auto z9  = _mm256_mul_ps(s0, w3);
        auto z10 = _mm256_mul_ps(s1, w3);
        auto z11 = _mm256_mul_ps(s2, w3);

        for (int sy = 1; sy < l; ++sy) {
            s0  = LOAD8(A + sy * 24);
            s1  = LOAD8(A + sy * 24 + 8);
            s2  = LOAD8(A + sy * 24 + 16);
            ws  = _load_int4x4(weight + sy * 2, alpha, bias);
            _mm_storeu_ps(ws_tmp, ws);
            w0  = _mm256_set1_ps(ws_tmp[0]);
            w1  = _mm256_set1_ps(ws_tmp[1]);
            w2  = _mm256_set1_ps(ws_tmp[2]);
            w3  = _mm256_set1_ps(ws_tmp[3]);
            z0  = MNNAVXFMA(s0, w0, z0);
            z1  = MNNAVXFMA(s1, w0, z1);
            z2  = MNNAVXFMA(s2, w0, z2);
            z3  = MNNAVXFMA(s0, w1, z3);
            z4  = MNNAVXFMA(s1, w1, z4);
            z5  = MNNAVXFMA(s2, w1, z5);
            z6  = MNNAVXFMA(s0, w2, z6);
            z7  = MNNAVXFMA(s1, w2, z7);
            z8  = MNNAVXFMA(s2, w2, z8);
            z9  = MNNAVXFMA(s0, w3, z9);
            z10 = MNNAVXFMA(s1, w3, z10);
            z11 = MNNAVXFMA(s2, w3, z11);
        }
        if (blockId == 0) {
            TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(1, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
            TRANPOSE_SAVE(1, 1, z1, z4, z7, z10);
            TRANPOSE_SAVE(0, 2, z2, z5, z8, z11);
            TRANPOSE_SAVE(1, 2, z2, z5, z8, z11);
        } else {
            FMLA_TRANSPOSE_SAVE(0, 0, z0, z3, z6, z9);
            FMLA_TRANSPOSE_SAVE(1, 0, z0, z3, z6, z9);
            FMLA_TRANSPOSE_SAVE(0, 1, z1, z4, z7, z10);
            FMLA_TRANSPOSE_SAVE(1, 1, z1, z4, z7, z10);
            FMLA_TRANSPOSE_SAVE(0, 2, z2, z5, z8, z11);
            FMLA_TRANSPOSE_SAVE(1, 2, z2, z5, z8, z11);
        }
    }
}


template <typename TYPE>
static void _AVX_MNNPackedMatMul_int4_20(TYPE* C, const TYPE* A, const uint8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    float weightBytes = 0.5; // sizeof(int4_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
    auto hC4          = UP_DIV(h, 4);
    float ws_tmp[4];
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride / 2;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0  = LOAD8(A + 0 * aStride);
        auto s1  = LOAD8(A + 0 * aStride + 8);
        auto s2  = EXPAND_128(LOAD4(A + 0 * aStride + 16));
        auto ws  = _load_int4x4(weight, alpha, bias);
        _mm_storeu_ps(ws_tmp, ws);
        auto w0  = _mm256_set1_ps(ws_tmp[0]);
        auto w1  = _mm256_set1_ps(ws_tmp[1]);
        auto w2  = _mm256_set1_ps(ws_tmp[2]);
        auto w3  = _mm256_set1_ps(ws_tmp[3]);
        auto z0  = _mm256_mul_ps(s0, w0);
        auto z1  = _mm256_mul_ps(s1, w0);
        auto z2  = _mm256_mul_ps(s2, w0);
        auto z3  = _mm256_mul_ps(s0, w1);
        auto z4  = _mm256_mul_ps(s1, w1);
        auto z5  = _mm256_mul_ps(s2, w1);
        auto z6  = _mm256_mul_ps(s0, w2);
        auto z7  = _mm256_mul_ps(s1, w2);
        auto z8  = _mm256_mul_ps(s2, w2);
        auto z9  = _mm256_mul_ps(s0, w3);
        auto z10 = _mm256_mul_ps(s1, w3);
        auto z11 = _mm256_mul_ps(s2, w3);
        for (int sy = 1; sy < l; ++sy) {
            s0  = LOAD8(A + sy * aStride);
            s1  = LOAD8(A + sy * aStride + 8);
            s2  = EXPAND_128(LOAD4(A + sy * aStride + 16));
            ws  = _load_int4x4(weight + sy * 2, alpha, bias);
            _mm_storeu_ps(ws_tmp, ws);
            w0  = _mm256_set1_ps(ws_tmp[0]);
            w1  = _mm256_set1_ps(ws_tmp[1]);
            w2  = _mm256_set1_ps(ws_tmp[2]);
            w3  = _mm256_set1_ps(ws_tmp[3]);
            z0  = MNNAVXFMA(s0, w0, z0);
            z1  = MNNAVXFMA(s1, w0, z1);
            z2  = MNNAVXFMA(s2, w0, z2);
            z3  = MNNAVXFMA(s0, w1, z3);
            z4  = MNNAVXFMA(s1, w1, z4);
            z5  = MNNAVXFMA(s2, w1, z5);
            z6  = MNNAVXFMA(s0, w2, z6);
            z7  = MNNAVXFMA(s1, w2, z7);
            z8  = MNNAVXFMA(s2, w2, z8);
            z9  = MNNAVXFMA(s0, w3, z9);
            z10 = MNNAVXFMA(s1, w3, z10);
            z11 = MNNAVXFMA(s2, w3, z11);
        }
        if (0 == blockId) {
            TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(1, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
            TRANPOSE_SAVE(1, 1, z1, z4, z7, z10);
            TRANPOSE_SAVE(0, 2, z2, z5, z8, z11);
        } else {
            FMLA_TRANSPOSE_SAVE(0, 0, z0, z3, z6, z9);
            FMLA_TRANSPOSE_SAVE(1, 0, z0, z3, z6, z9);
            FMLA_TRANSPOSE_SAVE(0, 1, z1, z4, z7, z10);
            FMLA_TRANSPOSE_SAVE(1, 1, z1, z4, z7, z10);
            FMLA_TRANSPOSE_SAVE(0, 2, z2, z5, z8, z11);
        }
    }
}

template <typename TYPE>
static void _AVX_MNNPackedMatMul_int4_16(TYPE* C, const TYPE* A, const uint8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    float weightBytes = 0.5; // sizeof(int4_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
    auto hC4          = UP_DIV(h, 4);
    float ws_tmp[4];
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride / 2;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0  = LOAD8(A + 0 * aStride);
        auto s1  = LOAD8(A + 0 * aStride + 8);
        auto ws  = _load_int4x4(weight, alpha, bias);
        _mm_storeu_ps(ws_tmp, ws);
        auto w0  = _mm256_set1_ps(ws_tmp[0]);
        auto w1  = _mm256_set1_ps(ws_tmp[1]);
        auto w2  = _mm256_set1_ps(ws_tmp[2]);
        auto w3  = _mm256_set1_ps(ws_tmp[3]);
        auto z0  = _mm256_mul_ps(s0, w0);
        auto z1  = _mm256_mul_ps(s1, w0);
        auto z3  = _mm256_mul_ps(s0, w1);
        auto z4  = _mm256_mul_ps(s1, w1);
        auto z6  = _mm256_mul_ps(s0, w2);
        auto z7  = _mm256_mul_ps(s1, w2);
        auto z9  = _mm256_mul_ps(s0, w3);
        auto z10 = _mm256_mul_ps(s1, w3);
        for (int sy = 1; sy < l; ++sy) {
            s0  = LOAD8(A + sy * aStride);
            s1  = LOAD8(A + sy * aStride + 8);
            ws  = _load_int4x4(weight + sy * 2, alpha, bias);
            _mm_storeu_ps(ws_tmp, ws);
            w0  = _mm256_set1_ps(ws_tmp[0]);
            w1  = _mm256_set1_ps(ws_tmp[1]);
            w2  = _mm256_set1_ps(ws_tmp[2]);
            w3  = _mm256_set1_ps(ws_tmp[3]);
            z0  = MNNAVXFMA(s0, w0, z0);
            z1  = MNNAVXFMA(s1, w0, z1);
            z3  = MNNAVXFMA(s0, w1, z3);
            z4  = MNNAVXFMA(s1, w1, z4);
            z6  = MNNAVXFMA(s0, w2, z6);
            z7  = MNNAVXFMA(s1, w2, z7);
            z9  = MNNAVXFMA(s0, w3, z9);
            z10 = MNNAVXFMA(s1, w3, z10);
        }
        if (0 == blockId) {
            TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(1, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
            TRANPOSE_SAVE(1, 1, z1, z4, z7, z10);
        } else {
            FMLA_TRANSPOSE_SAVE(0, 0, z0, z3, z6, z9);
            FMLA_TRANSPOSE_SAVE(1, 0, z0, z3, z6, z9);
            FMLA_TRANSPOSE_SAVE(0, 1, z1, z4, z7, z10);
            FMLA_TRANSPOSE_SAVE(1, 1, z1, z4, z7, z10);
        }
    }
}

template <typename TYPE>
static void _AVX_MNNPackedMatMul_int4_5(TYPE* C, const TYPE* A, const uint8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    float weightBytes = 0.5;
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
    auto hC4          = UP_DIV(h, 4);
    int lC4 = l / 4;
    int lR = lC4 * 4;
    const int hC4Unit = 4;
    int hC16 = hC4 / hC4Unit;
    int hR = hC16 * hC4Unit;
    auto src = A;
    for (int y = 0; y < hC16; ++y) {
        LOAD_WEIGHT_ALPHA_BIAS_int4x4
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
            auto w0 = _load_int4x4(weight0, alpha0, bias0);
            auto w1 = _load_int4x4(weight1, alpha1, bias1);
            auto w2 = _load_int4x4(weight2, alpha2, bias2);
            auto w3 = _load_int4x4(weight3, alpha3, bias3);
            auto W0 = _mm256_set_m128(w1, w0);
            auto W1 = _mm256_set_m128(w3, w2);

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
            weight0 += 2;
            weight1 += 2;
            weight2 += 2;
            weight3 += 2;
        }
        if (0 == blockId) {
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
        } else {
            auto tmp0 = LOAD8(dst0);
            auto tmp1 = LOAD8(dst0 + 8);
            auto tmp2 = LOAD8(dst0 + 16);
            auto tmp3 = LOAD8(dst0 + 24);
            auto tmp4 = LOAD8(dst0 + 32);
            auto tmp5 = LOAD8(dst2);
            auto tmp6 = LOAD8(dst2 + 8);
            auto tmp7 = LOAD8(dst2 + 16);
            auto tmp8 = LOAD8(dst2 + 24);
            auto tmp9 = LOAD8(dst2 + 32);

            sumAvx00 = _mm256_add_ps(sumAvx00, tmp0);
            sumAvx10 = _mm256_add_ps(sumAvx10, tmp1);
            sumAvx20 = _mm256_add_ps(sumAvx20, tmp2);
            sumAvx30 = _mm256_add_ps(sumAvx30, tmp3);
            sumAvx40 = _mm256_add_ps(sumAvx40, tmp4);
            sumAvx01 = _mm256_add_ps(sumAvx01, tmp5);
            sumAvx11 = _mm256_add_ps(sumAvx11, tmp6);
            sumAvx21 = _mm256_add_ps(sumAvx21, tmp7);
            sumAvx31 = _mm256_add_ps(sumAvx31, tmp8);
            sumAvx41 = _mm256_add_ps(sumAvx41, tmp9);

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
    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride / 2;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0     = BROAD_LOAD_4(A + 0 * aStride + 0);
        auto s1     = BROAD_LOAD_4(A + 0 * aStride + 1);
        auto s2     = BROAD_LOAD_4(A + 0 * aStride + 2);
        auto s3     = BROAD_LOAD_4(A + 0 * aStride + 3);
        auto s4     = BROAD_LOAD_4(A + 0 * aStride + 4);
        auto w0     = _load_int4x4(weight, alpha, bias);
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
            w0 = _load_int4x4(weight + sy * 2, alpha, bias);
            z0 = MNNSSEFMA(s0, w0, z0);
            z1 = MNNSSEFMA(s1, w0, z1);
            z2 = MNNSSEFMA(s2, w0, z2);
            z3 = MNNSSEFMA(s3, w0, z3);
            z4 = MNNSSEFMA(s4, w0, z4);
        }
        if (0 == blockId) {
            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z1);
            STORE_4(dst + 8 * 2, z2);
            STORE_4(dst + 8 * 3, z3);
            STORE_4(dst + 8 * 4, z4);
        } else {
            auto tmp0 = LOAD4(dst + 8 * 0);
            auto tmp1 = LOAD4(dst + 8 * 1);
            auto tmp2 = LOAD4(dst + 8 * 2);
            auto tmp3 = LOAD4(dst + 8 * 3);
            auto tmp4 = LOAD4(dst + 8 * 4);

            z0 = _mm_add_ps(tmp0, z0);
            z1 = _mm_add_ps(tmp1, z1);
            z2 = _mm_add_ps(tmp2, z2);
            z3 = _mm_add_ps(tmp3, z3);
            z4 = _mm_add_ps(tmp4, z4);

            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z1);
            STORE_4(dst + 8 * 2, z2);
            STORE_4(dst + 8 * 3, z3);
            STORE_4(dst + 8 * 4, z4);
        }
    }
}


template <typename TYPE>
static void _AVX_MNNPackedMatMul_int4_4(TYPE* C, const TYPE* A, const uint8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    float weightBytes = 0.5; // sizeof(int4_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
    auto hC4          = UP_DIV(h, 4);
    int lC4 = l / 4;
    int lR = lC4 * 4;
    const int hC4Unit = 4;
    int hC16 = hC4 / hC4Unit;
    int hR = hC16 * hC4Unit;
    auto src = A;
    for (int y = 0; y < hC16; ++y) {
        LOAD_WEIGHT_ALPHA_BIAS_int4x4
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
            auto w0 = _load_int4x4(weight0, alpha0, bias0);
            auto w1 = _load_int4x4(weight1, alpha1, bias1);
            auto w2 = _load_int4x4(weight2, alpha2, bias2);
            auto w3 = _load_int4x4(weight3, alpha3, bias3);
            auto W0 = _mm256_set_m128(w1, w0);
            auto W1 = _mm256_set_m128(w3, w2);

            sumAvx00   = MNNAVXFMA(S0, W0, sumAvx00);
            sumAvx01   = MNNAVXFMA(S0, W1, sumAvx01);

            sumAvx10   = MNNAVXFMA(S1, W0, sumAvx10);
            sumAvx11   = MNNAVXFMA(S1, W1, sumAvx11);

            sumAvx20   = MNNAVXFMA(S2, W0, sumAvx20);
            sumAvx21   = MNNAVXFMA(S2, W1, sumAvx21);

            sumAvx30   = MNNAVXFMA(S3, W0, sumAvx30);
            sumAvx31   = MNNAVXFMA(S3, W1, sumAvx31);

            srcUse += aStride;
            weight0 += 2;
            weight1 += 2;
            weight2 += 2;
            weight3 += 2;
        }
        if (0 == blockId) {
            STORE_8(dst0, sumAvx00);
            STORE_8(dst0 + 8, sumAvx10);
            STORE_8(dst0 + 16, sumAvx20);
            STORE_8(dst0 + 24, sumAvx30);

            STORE_8(dst2, sumAvx01);
            STORE_8(dst2 + 8, sumAvx11);
            STORE_8(dst2 + 16, sumAvx21);
            STORE_8(dst2 + 24, sumAvx31);
        } else {
            auto tmp0 = LOAD8(dst0);
            auto tmp1 = LOAD8(dst0 + 8);
            auto tmp2 = LOAD8(dst0 + 16);
            auto tmp3 = LOAD8(dst0 + 24);

            auto tmp5 = LOAD8(dst2);
            auto tmp6 = LOAD8(dst2 + 8);
            auto tmp7 = LOAD8(dst2 + 16);
            auto tmp8 = LOAD8(dst2 + 24);

            sumAvx00 = _mm256_add_ps(sumAvx00, tmp0);
            sumAvx10 = _mm256_add_ps(sumAvx10, tmp1);
            sumAvx20 = _mm256_add_ps(sumAvx20, tmp2);
            sumAvx30 = _mm256_add_ps(sumAvx30, tmp3);

            sumAvx01 = _mm256_add_ps(sumAvx01, tmp5);
            sumAvx11 = _mm256_add_ps(sumAvx11, tmp6);
            sumAvx21 = _mm256_add_ps(sumAvx21, tmp7);
            sumAvx31 = _mm256_add_ps(sumAvx31, tmp8);

            STORE_8(dst0, sumAvx00);
            STORE_8(dst0 + 8, sumAvx10);
            STORE_8(dst0 + 16, sumAvx20);
            STORE_8(dst0 + 24, sumAvx30);

            STORE_8(dst2, sumAvx01);
            STORE_8(dst2 + 8, sumAvx11);
            STORE_8(dst2 + 16, sumAvx21);
            STORE_8(dst2 + 24, sumAvx31);
        }
    }
    float ws_tmp[4];
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride / 2;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0     = LOAD4(A + 0 * aStride);
        auto ws     = _load_int4x4(weight, alpha, bias);
        _mm_storeu_ps(ws_tmp, ws);
        auto w0     = _mm_set1_ps(ws_tmp[0]);
        auto w1     = _mm_set1_ps(ws_tmp[1]);
        auto w2     = _mm_set1_ps(ws_tmp[2]);
        auto w3     = _mm_set1_ps(ws_tmp[3]);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z3     = _mm_mul_ps(s0, w1);
        auto z6     = _mm_mul_ps(s0, w2);
        auto z9     = _mm_mul_ps(s0, w3);

        for (int sy = 1; sy < l; ++sy) {
            s0 = LOAD4(A + sy * aStride);
            ws = _load_int4x4(weight + sy * 2, alpha, bias);
            _mm_storeu_ps(ws_tmp, ws);
            w0 = _mm_set1_ps(ws_tmp[0]);
            w1 = _mm_set1_ps(ws_tmp[1]);
            w2 = _mm_set1_ps(ws_tmp[2]);
            w3 = _mm_set1_ps(ws_tmp[3]);
            z0 = MNNSSEFMA(s0, w0, z0);
            z3 = MNNSSEFMA(s0, w1, z3);
            z6 = MNNSSEFMA(s0, w2, z6);
            z9 = MNNSSEFMA(s0, w3, z9);
        }
        _MM_TRANSPOSE4_PS(z0, z3, z6, z9);
        if (0 == blockId) {
            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z3);
            STORE_4(dst + 8 * 2, z6);
            STORE_4(dst + 8 * 3, z9);
        } else {

            auto tmp0 = LOAD4(dst + 8 * 0);
            auto tmp1 = LOAD4(dst + 8 * 1);
            auto tmp2 = LOAD4(dst + 8 * 2);
            auto tmp3 = LOAD4(dst + 8 * 3);

            z0 = _mm_add_ps(tmp0, z0);
            z3 = _mm_add_ps(tmp1, z3);
            z6 = _mm_add_ps(tmp2, z6);
            z9 = _mm_add_ps(tmp3, z9);

            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z3);
            STORE_4(dst + 8 * 2, z6);
            STORE_4(dst + 8 * 3, z9);
        }
    }
}
template <typename TYPE>
static void _AVX_MNNPackedMatMul_int4_3(TYPE* C, const TYPE* A, const uint8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    float weightBytes = 0.5; // sizeof(int4_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
    auto hC4          = UP_DIV(h, 4);
    int lC4 = l / 4;
    int lR = lC4 * 4;
    const int hC4Unit = 4;
    int hC16 = hC4 / hC4Unit;
    int hR = hC16 * hC4Unit;
    auto src = A;
    for (int y = 0; y < hC16; ++y) {
        LOAD_WEIGHT_ALPHA_BIAS_int4x4
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
            auto w0 = _load_int4x4(weight0, alpha0, bias0);
            auto w1 = _load_int4x4(weight1, alpha1, bias1);
            auto w2 = _load_int4x4(weight2, alpha2, bias2);
            auto w3 = _load_int4x4(weight3, alpha3, bias3);
            auto W0 = _mm256_set_m128(w1, w0);
            auto W1 = _mm256_set_m128(w3, w2);

            sumAvx00   = MNNAVXFMA(S0, W0, sumAvx00);
            sumAvx01   = MNNAVXFMA(S0, W1, sumAvx01);

            sumAvx10   = MNNAVXFMA(S1, W0, sumAvx10);
            sumAvx11   = MNNAVXFMA(S1, W1, sumAvx11);

            sumAvx20   = MNNAVXFMA(S2, W0, sumAvx20);
            sumAvx21   = MNNAVXFMA(S2, W1, sumAvx21);

            srcUse += aStride;
            weight0 += 2;
            weight1 += 2;
            weight2 += 2;
            weight3 += 2;
        }
        if (0 == blockId) {
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
        } else {
            auto tmp00 = LOAD4(dst0 + 0);
            auto tmp01 = LOAD4(dst0 + 8);
            auto tmp02 = LOAD4(dst0 + 16);

            auto tmp10 = LOAD4(dst1 + 0);
            auto tmp11 = LOAD4(dst1 + 8);
            auto tmp12 = LOAD4(dst1 + 16);

            auto tmp20 = LOAD4(dst2 + 0);
            auto tmp21 = LOAD4(dst2 + 8);
            auto tmp22 = LOAD4(dst2 + 16);

            auto tmp30 = LOAD4(dst3 + 0);
            auto tmp31 = LOAD4(dst3 + 8);
            auto tmp32 = LOAD4(dst3 + 16);

            auto sum_tmp00 = _mm256_extractf128_ps(sumAvx00, 0);
            auto sum_tmp01 = _mm256_extractf128_ps(sumAvx10, 0);
            auto sum_tmp02 = _mm256_extractf128_ps(sumAvx20, 0);
            auto sum_tmp10 = _mm256_extractf128_ps(sumAvx00, 1);
            auto sum_tmp11 = _mm256_extractf128_ps(sumAvx10, 1);
            auto sum_tmp12 = _mm256_extractf128_ps(sumAvx20, 1);
            auto sum_tmp20 = _mm256_extractf128_ps(sumAvx01, 0);
            auto sum_tmp21 = _mm256_extractf128_ps(sumAvx11, 0);
            auto sum_tmp22 = _mm256_extractf128_ps(sumAvx21, 0);
            auto sum_tmp30 = _mm256_extractf128_ps(sumAvx01, 1);
            auto sum_tmp31 = _mm256_extractf128_ps(sumAvx11, 1);
            auto sum_tmp32 = _mm256_extractf128_ps(sumAvx21, 1);

            sum_tmp00 = _mm_add_ps(tmp00, sum_tmp00);
            sum_tmp01 = _mm_add_ps(tmp01, sum_tmp01);
            sum_tmp02 = _mm_add_ps(tmp02, sum_tmp02);
            sum_tmp10 = _mm_add_ps(tmp10, sum_tmp10);
            sum_tmp11 = _mm_add_ps(tmp11, sum_tmp11);
            sum_tmp12 = _mm_add_ps(tmp12, sum_tmp12);
            sum_tmp20 = _mm_add_ps(tmp20, sum_tmp20);
            sum_tmp21 = _mm_add_ps(tmp21, sum_tmp21);
            sum_tmp22 = _mm_add_ps(tmp22, sum_tmp22);
            sum_tmp30 = _mm_add_ps(tmp30, sum_tmp30);
            sum_tmp31 = _mm_add_ps(tmp31, sum_tmp31);
            sum_tmp32 = _mm_add_ps(tmp32, sum_tmp32);

            STORE_4(dst0 + 0,  sum_tmp00);
            STORE_4(dst0 + 8,  sum_tmp01);
            STORE_4(dst0 + 16, sum_tmp02);
            STORE_4(dst1 + 0,  sum_tmp10);
            STORE_4(dst1 + 8,  sum_tmp11);
            STORE_4(dst1 + 16, sum_tmp12);
            STORE_4(dst2 + 0,  sum_tmp20);
            STORE_4(dst2 + 8,  sum_tmp21);
            STORE_4(dst2 + 16, sum_tmp22);
            STORE_4(dst3 + 0,  sum_tmp30);
            STORE_4(dst3 + 8,  sum_tmp31);
            STORE_4(dst3 + 16, sum_tmp32);
        }

    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride / 2;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0     = BROAD_LOAD_4(A + 0 * aStride + 0);
        auto s1     = BROAD_LOAD_4(A + 0 * aStride + 1);
        auto s2     = BROAD_LOAD_4(A + 0 * aStride + 2);
        auto w0     = _load_int4x4(weight, alpha, bias);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z1     = _mm_mul_ps(s1, w0);
        auto z2     = _mm_mul_ps(s2, w0);

        for (int sy = 1; sy < l; ++sy) {
            s0 = BROAD_LOAD_4(A + sy * aStride + 0);
            s1 = BROAD_LOAD_4(A + sy * aStride + 1);
            s2 = BROAD_LOAD_4(A + sy * aStride + 2);
            w0 = _load_int4x4(weight + sy * 2, alpha, bias);
            z0 = MNNSSEFMA(s0, w0, z0);
            z1 = MNNSSEFMA(s1, w0, z1);
            z2 = MNNSSEFMA(s2, w0, z2);
        }
        if (0 == blockId) {
            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z1);
            STORE_4(dst + 8 * 2, z2);
        } else {
            auto tmp0 = LOAD4(dst + 8 * 0);
            auto tmp1 = LOAD4(dst + 8 * 1);
            auto tmp2 = LOAD4(dst + 8 * 2);

            z0 = _mm_add_ps(tmp0, z0);
            z1 = _mm_add_ps(tmp1, z1);
            z2 = _mm_add_ps(tmp2, z2);

            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z1);
            STORE_4(dst + 8 * 2, z2);
        }
    }
}

template <typename TYPE>
static void _AVX_MNNPackedMatMul_int4_2(TYPE* C, const TYPE* A, const uint8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    float weightBytes = 0.5;
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
    auto hC4          = UP_DIV(h, 4);
    int lC4 = l / 4;
    int lR = lC4 * 4;
    const int hC4Unit = 4;
    int hC16 = hC4 / hC4Unit;
    int hR = hC16 * hC4Unit;
    auto src = A;
    for (int y = 0; y < hC16; ++y) {
        LOAD_WEIGHT_ALPHA_BIAS_int4x4
        auto sumAvx00    = _mm256_setzero_ps();
        auto sumAvx01    = _mm256_setzero_ps();
        DST_ADDR_UNPACK4(0);

        auto sumAvx10    = _mm256_setzero_ps();
        auto sumAvx11    = _mm256_setzero_ps();

        auto srcUse = src;
        for (int sy = 0; sy < l; ++sy) {
            auto S0 = BROAD_LOAD(srcUse + 0);
            auto S1 = BROAD_LOAD(srcUse + 1);
            auto w0 = _load_int4x4(weight0, alpha0, bias0);
            auto w1 = _load_int4x4(weight1, alpha1, bias1);
            auto w2 = _load_int4x4(weight2, alpha2, bias2);
            auto w3 = _load_int4x4(weight3, alpha3, bias3);
            auto W0 = _mm256_set_m128(w1, w0);
            auto W1 = _mm256_set_m128(w3, w2);

            sumAvx00   = MNNAVXFMA(S0, W0, sumAvx00);
            sumAvx01   = MNNAVXFMA(S0, W1, sumAvx01);

            sumAvx10   = MNNAVXFMA(S1, W0, sumAvx10);
            sumAvx11   = MNNAVXFMA(S1, W1, sumAvx11);

            srcUse += aStride;
            weight0 += 2;
            weight1 += 2;
            weight2 += 2;
            weight3 += 2;
        }
        if (0 == blockId) {
            STORE_4(dst0 + 0, _mm256_extractf128_ps(sumAvx00, 0));
            STORE_4(dst0 + 8, _mm256_extractf128_ps(sumAvx10, 0));

            STORE_4(dst1 + 0, _mm256_extractf128_ps(sumAvx00, 1));
            STORE_4(dst1 + 8, _mm256_extractf128_ps(sumAvx10, 1));

            STORE_4(dst2 + 0, _mm256_extractf128_ps(sumAvx01, 0));
            STORE_4(dst2 + 8, _mm256_extractf128_ps(sumAvx11, 0));

            STORE_4(dst3 + 0, _mm256_extractf128_ps(sumAvx01, 1));
            STORE_4(dst3 + 8, _mm256_extractf128_ps(sumAvx11, 1));
        } else {
            auto tmp01 = LOAD4(dst0 + 0);
            auto tmp02 = LOAD4(dst0 + 8);
            auto tmp11 = LOAD4(dst1 + 0);
            auto tmp12 = LOAD4(dst1 + 8);
            auto tmp21 = LOAD4(dst2 + 0);
            auto tmp22 = LOAD4(dst2 + 8);
            auto tmp31 = LOAD4(dst3 + 0);
            auto tmp32 = LOAD4(dst3 + 8);

            auto x_tmp01 = _mm256_extractf128_ps(sumAvx00, 0);
            auto x_tmp02 = _mm256_extractf128_ps(sumAvx10, 0);
            auto x_tmp11 = _mm256_extractf128_ps(sumAvx00, 1);
            auto x_tmp12 = _mm256_extractf128_ps(sumAvx10, 1);
            auto x_tmp21 = _mm256_extractf128_ps(sumAvx01, 0);
            auto x_tmp22 = _mm256_extractf128_ps(sumAvx11, 0);
            auto x_tmp31 = _mm256_extractf128_ps(sumAvx01, 1);
            auto x_tmp32 = _mm256_extractf128_ps(sumAvx11, 1);

            x_tmp01 = _mm_add_ps(tmp01, x_tmp01);
            x_tmp02 = _mm_add_ps(tmp02, x_tmp02);
            x_tmp11 = _mm_add_ps(tmp11, x_tmp11);
            x_tmp12 = _mm_add_ps(tmp12, x_tmp12);
            x_tmp21 = _mm_add_ps(tmp21, x_tmp21);
            x_tmp22 = _mm_add_ps(tmp22, x_tmp22);
            x_tmp31 = _mm_add_ps(tmp31, x_tmp31);
            x_tmp32 = _mm_add_ps(tmp32, x_tmp32);

            STORE_4(dst0 + 0, x_tmp01);
            STORE_4(dst0 + 8, x_tmp02);
            STORE_4(dst1 + 0, x_tmp11);
            STORE_4(dst1 + 8, x_tmp12);
            STORE_4(dst2 + 0, x_tmp21);
            STORE_4(dst2 + 8, x_tmp22);
            STORE_4(dst3 + 0, x_tmp31);
            STORE_4(dst3 + 8, x_tmp32);
        }

    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride / 2;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0     = BROAD_LOAD_4(A + 0 * aStride + 0);
        auto s1     = BROAD_LOAD_4(A + 0 * aStride + 1);
        auto w0     = _load_int4x4(weight, alpha, bias);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z1     = _mm_mul_ps(s1, w0);

        for (int sy = 1; sy < l; ++sy) {
            s0 = BROAD_LOAD_4(A + sy * aStride + 0);
            s1 = BROAD_LOAD_4(A + sy * aStride + 1);
            w0 = _load_int4x4(weight + sy * 2, alpha, bias);
            z0 = MNNSSEFMA(s0, w0, z0);
            z1 = MNNSSEFMA(s1, w0, z1);
        }
        if (0 == blockId) {
            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z1);
        } else {
            auto t0 = LOAD4(dst + 8 * 0);
            auto t1 = LOAD4(dst + 8 * 1);
            z0 = _mm_add_ps(z0, t0);
            z1 = _mm_add_ps(z1, t1);
            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z1);
        }
    }
}

template <typename TYPE>
static void _AVX_MNNPackednMatMulRemainCommon_int4(TYPE* C, const TYPE* A, const TYPE* fB, size_t eSize,
                                                   const size_t* parameter, const float* k, const float* b) {
    auto B           = reinterpret_cast<const uint8_t*>(fB);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    float weightBytes = 0.5; // sizeof(int4_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride      = bExtraStride + 4 * l;
    auto hC4          = UP_DIV(h, 4);
    auto es           = eSize;
    auto oC           = C;
    auto aStride      = parameter[0] / sizeof(TYPE);
    size_t blockId    = parameter[6];
    if (eSize >= 20) {
        _AVX_MNNPackedMatMul_int4_20<TYPE>(C, A, B, parameter, k, b);
        eSize -= 20;
        C += 20 * 8;
        A += 20;
    }
    if (eSize >= 16) {
        _AVX_MNNPackedMatMul_int4_16<TYPE>(C, A, B, parameter, k, b);
        eSize -= 16;
        C += 16 * 8;
        A += 16;
    }
    while (eSize >= 5) {
        _AVX_MNNPackedMatMul_int4_5<TYPE>(C, A, B, parameter, k, b);
        eSize -= 5;
        C += 5 * 8;
        A += 5;
    }
    if (eSize == 4) {
        _AVX_MNNPackedMatMul_int4_4<TYPE>(C, A, B, parameter, k, b);
        return;
    }
    if (eSize == 3) {
        _AVX_MNNPackedMatMul_int4_3<TYPE>(C, A, B, parameter, k, b);
        return;
    }
    if (eSize == 2) {
        _AVX_MNNPackedMatMul_int4_2<TYPE>(C, A, B, parameter, k, b);
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
        auto dst0    = C + (hC4Unit * y / 2 + 0) * cStride + x * 8;
        auto dst1    = C + (hC4Unit * y / 2 + 0) * cStride + x * 8 + 4;
        auto dst2    = C + (hC4Unit * y / 2 + 1) * cStride + x * 8;
        auto dst3    = C + (hC4Unit * y / 2 + 1) * cStride + x * 8 + 4;
        LOAD_WEIGHT_ALPHA_BIAS_int4x4
        LOAD_ALPHA_BIAS_DOUBLE

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

            auto W00 = _load_int4x8(weight0 + 8 * sy + 0, alpha0_2, bias0_2);
            auto W01 = _load_int4x8(weight0 + 8 * sy + 4, alpha0_2, bias0_2);
            auto W10 = _load_int4x8(weight1 + 8 * sy + 0, alpha1_2, bias1_2);
            auto W11 = _load_int4x8(weight1 + 8 * sy + 4, alpha1_2, bias1_2);

            auto W20 = _load_int4x8(weight2 + 8 * sy + 0, alpha2_2, bias2_2);
            auto W21 = _load_int4x8(weight2 + 8 * sy + 4, alpha2_2, bias2_2);
            auto W30 = _load_int4x8(weight3 + 8 * sy + 0, alpha3_2, bias3_2);
            auto W31 = _load_int4x8(weight3 + 8 * sy + 4, alpha3_2, bias3_2);

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
            auto w0 = _load_int4x4(weight0 + 2 * sy, alpha0, bias0);
            auto w1 = _load_int4x4(weight1 + 2 * sy, alpha1, bias1);
            auto w2 = _load_int4x4(weight2 + 2 * sy, alpha2, bias2);
            auto w3 = _load_int4x4(weight3 + 2 * sy, alpha3, bias3);
            sum0    = MNNSSEFMA(s, w0, sum0);
            sum1    = MNNSSEFMA(s, w1, sum1);
            sum2    = MNNSSEFMA(s, w2, sum2);
            sum3    = MNNSSEFMA(s, w3, sum3);
            srcUse += aStride;
        }
        if (blockId == 0) {
            STORE_4(dst0, sum0);
            STORE_4(dst1, sum1);
            STORE_4(dst2, sum2);
            STORE_4(dst3, sum3);
        } else {
            auto tmp_0 = LOAD4(dst0);
            auto tmp_1 = LOAD4(dst1);
            auto tmp_2 = LOAD4(dst2);
            auto tmp_3 = LOAD4(dst3);
            sum0 = _mm_add_ps(tmp_0, sum0);
            sum1 = _mm_add_ps(tmp_1, sum1);
            sum2 = _mm_add_ps(tmp_2, sum2);
            sum3 = _mm_add_ps(tmp_3, sum3);
            STORE_4(dst0, sum0);
            STORE_4(dst1, sum1);
            STORE_4(dst2, sum2);
            STORE_4(dst3, sum3);
        }
    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride / 2;
        auto dst    = C + (y / 2) * cStride + x * 8 + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto alpha_2 = _mm256_set_m128(alpha, alpha);
        auto bias_2  = _mm256_set_m128(bias, bias);

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
            auto W0 = _load_int4x8(weight + 8 * sy + 0, alpha_2, bias_2);
            auto W1 = _load_int4x8(weight + 8 * sy + 4, alpha_2, bias_2);
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
            auto w = _load_int4x4(weight + sy * 2, alpha, bias);
            sum    = MNNSSEFMA(s, w, sum);
            srcUse += aStride;
        }
        if (blockId == 0) {
            STORE_4(dst, sum);
        } else {
            auto tmp_0 = LOAD4(dst);
            sum = _mm_add_ps(tmp_0, sum);
            STORE_4(dst, sum);
        }
    }
}

//----------------------- MatMul(float, int8) Functions ---------------------------//

#define LOAD_WEIGHT_ALPHA_BIAS_int8x4 \
    auto weight0 = B + (hC4Unit * y + 0) * bStride;\
    auto weight1 = B + (hC4Unit * y + 1) * bStride;\
    auto weight2 = B + (hC4Unit * y + 2) * bStride;\
    auto weight3 = B + (hC4Unit * y + 3) * bStride;\
    auto alpha0  = _mm_loadu_ps(k + y * 16 + 0);\
    auto alpha1  = _mm_loadu_ps(k + y * 16 + 4);\
    auto alpha2  = _mm_loadu_ps(k + y * 16 + 8);\
    auto alpha3  = _mm_loadu_ps(k + y * 16 + 12);\
    auto bias0   = _mm_loadu_ps(b + y * 16 + 0);\
    auto bias1   = _mm_loadu_ps(b + y * 16 + 4);\
    auto bias2   = _mm_loadu_ps(b + y * 16 + 8);\
    auto bias3   = _mm_loadu_ps(b + y * 16 + 12);

static inline __m128 _load_int8x4(const int8_t* src, __m128 alpha, __m128 bias) {
    int iw0     = src[0];
    int iw1     = src[1];
    int iw2     = src[2];
    int iw3     = src[3];
    auto ws     = _mm_set_ps(iw3, iw2, iw1, iw0);
    ws          = _mm_add_ps(_mm_mul_ps(ws, alpha), bias);
    return ws;
}

static inline __m256 _load_int8x8(const int8_t* src, __m256 alpha, __m256 bias) {
    float w[8];
    for (int i = 0; i < 8; i++) {
        w[i] = int(src[i]);
    }
    auto w8 = LOAD8(w);
    return _mm256_add_ps(_mm256_mul_ps(w8, alpha), bias);
}


template <typename TYPE>
static void _AVX_MNNPackedMatMul_Main_int8(TYPE* C, const TYPE* A, const TYPE* fB, const size_t* parameter, const float* k, const float* b) {
    auto B            = reinterpret_cast<const int8_t*>(fB);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    int weightBytes = sizeof(int8_t);
    auto bExtraStride = parameter[5] / weightBytes;
    auto bStride      = bExtraStride + 4 * l;
    auto hC4          = UP_DIV(h, 4);
    float ws_tmp[4];
    size_t blockId    = parameter[6];
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0  = LOAD8(A + 0 * 24);
        auto s1  = LOAD8(A + 0 * 24 + 8);
        auto s2  = LOAD8(A + 0 * 24 + 16);
        auto ws  = _load_int8x4(weight, alpha, bias);
        _mm_storeu_ps(ws_tmp, ws);
        auto w0  = _mm256_set1_ps(ws_tmp[0]);
        auto w1  = _mm256_set1_ps(ws_tmp[1]);
        auto w2  = _mm256_set1_ps(ws_tmp[2]);
        auto w3  = _mm256_set1_ps(ws_tmp[3]);
        auto z0  = _mm256_mul_ps(s0, w0);
        auto z1  = _mm256_mul_ps(s1, w0);
        auto z2  = _mm256_mul_ps(s2, w0);
        auto z3  = _mm256_mul_ps(s0, w1);
        auto z4  = _mm256_mul_ps(s1, w1);
        auto z5  = _mm256_mul_ps(s2, w1);
        auto z6  = _mm256_mul_ps(s0, w2);
        auto z7  = _mm256_mul_ps(s1, w2);
        auto z8  = _mm256_mul_ps(s2, w2);
        auto z9  = _mm256_mul_ps(s0, w3);
        auto z10 = _mm256_mul_ps(s1, w3);
        auto z11 = _mm256_mul_ps(s2, w3);

        for (int sy = 1; sy < l; ++sy) {
            s0  = LOAD8(A + sy * 24);
            s1  = LOAD8(A + sy * 24 + 8);
            s2  = LOAD8(A + sy * 24 + 16);
            ws  = _load_int8x4(weight + sy * 4, alpha, bias);
            _mm_storeu_ps(ws_tmp, ws);
            w0  = _mm256_set1_ps(ws_tmp[0]);
            w1  = _mm256_set1_ps(ws_tmp[1]);
            w2  = _mm256_set1_ps(ws_tmp[2]);
            w3  = _mm256_set1_ps(ws_tmp[3]);
            z0  = MNNAVXFMA(s0, w0, z0);
            z1  = MNNAVXFMA(s1, w0, z1);
            z2  = MNNAVXFMA(s2, w0, z2);
            z3  = MNNAVXFMA(s0, w1, z3);
            z4  = MNNAVXFMA(s1, w1, z4);
            z5  = MNNAVXFMA(s2, w1, z5);
            z6  = MNNAVXFMA(s0, w2, z6);
            z7  = MNNAVXFMA(s1, w2, z7);
            z8  = MNNAVXFMA(s2, w2, z8);
            z9  = MNNAVXFMA(s0, w3, z9);
            z10 = MNNAVXFMA(s1, w3, z10);
            z11 = MNNAVXFMA(s2, w3, z11);
        }
        if (blockId == 0) {
            TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(1, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
            TRANPOSE_SAVE(1, 1, z1, z4, z7, z10);
            TRANPOSE_SAVE(0, 2, z2, z5, z8, z11);
            TRANPOSE_SAVE(1, 2, z2, z5, z8, z11);
        } else {
            FMLA_TRANSPOSE_SAVE(0, 0, z0, z3, z6, z9);
            FMLA_TRANSPOSE_SAVE(1, 0, z0, z3, z6, z9);
            FMLA_TRANSPOSE_SAVE(0, 1, z1, z4, z7, z10);
            FMLA_TRANSPOSE_SAVE(1, 1, z1, z4, z7, z10);
            FMLA_TRANSPOSE_SAVE(0, 2, z2, z5, z8, z11);
            FMLA_TRANSPOSE_SAVE(1, 2, z2, z5, z8, z11);
        }
    }
}


template <typename TYPE>
static void _AVX_MNNPackedMatMul_int8_20(TYPE* C, const TYPE* A, const int8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    int weightBytes = sizeof(int8_t);
    auto bExtraStride = parameter[5] / weightBytes;
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
    auto hC4          = UP_DIV(h, 4);
    float ws_tmp[4];
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0  = LOAD8(A + 0 * aStride);
        auto s1  = LOAD8(A + 0 * aStride + 8);
        auto s2  = EXPAND_128(LOAD4(A + 0 * aStride + 16));
        auto ws  = _load_int8x4(weight, alpha, bias);
        _mm_storeu_ps(ws_tmp, ws);
        auto w0  = _mm256_set1_ps(ws_tmp[0]);
        auto w1  = _mm256_set1_ps(ws_tmp[1]);
        auto w2  = _mm256_set1_ps(ws_tmp[2]);
        auto w3  = _mm256_set1_ps(ws_tmp[3]);
        auto z0  = _mm256_mul_ps(s0, w0);
        auto z1  = _mm256_mul_ps(s1, w0);
        auto z2  = _mm256_mul_ps(s2, w0);
        auto z3  = _mm256_mul_ps(s0, w1);
        auto z4  = _mm256_mul_ps(s1, w1);
        auto z5  = _mm256_mul_ps(s2, w1);
        auto z6  = _mm256_mul_ps(s0, w2);
        auto z7  = _mm256_mul_ps(s1, w2);
        auto z8  = _mm256_mul_ps(s2, w2);
        auto z9  = _mm256_mul_ps(s0, w3);
        auto z10 = _mm256_mul_ps(s1, w3);
        auto z11 = _mm256_mul_ps(s2, w3);
        for (int sy = 1; sy < l; ++sy) {
            s0  = LOAD8(A + sy * aStride);
            s1  = LOAD8(A + sy * aStride + 8);
            s2  = EXPAND_128(LOAD4(A + sy * aStride + 16));
            ws  = _load_int8x4(weight + sy * 4, alpha, bias);
            _mm_storeu_ps(ws_tmp, ws);
            w0  = _mm256_set1_ps(ws_tmp[0]);
            w1  = _mm256_set1_ps(ws_tmp[1]);
            w2  = _mm256_set1_ps(ws_tmp[2]);
            w3  = _mm256_set1_ps(ws_tmp[3]);
            z0  = MNNAVXFMA(s0, w0, z0);
            z1  = MNNAVXFMA(s1, w0, z1);
            z2  = MNNAVXFMA(s2, w0, z2);
            z3  = MNNAVXFMA(s0, w1, z3);
            z4  = MNNAVXFMA(s1, w1, z4);
            z5  = MNNAVXFMA(s2, w1, z5);
            z6  = MNNAVXFMA(s0, w2, z6);
            z7  = MNNAVXFMA(s1, w2, z7);
            z8  = MNNAVXFMA(s2, w2, z8);
            z9  = MNNAVXFMA(s0, w3, z9);
            z10 = MNNAVXFMA(s1, w3, z10);
            z11 = MNNAVXFMA(s2, w3, z11);
        }
        if (0 == blockId) {
            TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(1, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
            TRANPOSE_SAVE(1, 1, z1, z4, z7, z10);
            TRANPOSE_SAVE(0, 2, z2, z5, z8, z11);
        } else {
            FMLA_TRANSPOSE_SAVE(0, 0, z0, z3, z6, z9);
            FMLA_TRANSPOSE_SAVE(1, 0, z0, z3, z6, z9);
            FMLA_TRANSPOSE_SAVE(0, 1, z1, z4, z7, z10);
            FMLA_TRANSPOSE_SAVE(1, 1, z1, z4, z7, z10);
            FMLA_TRANSPOSE_SAVE(0, 2, z2, z5, z8, z11);
        }
    }
}

template <typename TYPE>
static void _AVX_MNNPackedMatMul_int8_16(TYPE* C, const TYPE* A, const int8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(int8_t);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
    auto hC4          = UP_DIV(h, 4);
    float ws_tmp[4];
    for (int y = 0; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0  = LOAD8(A + 0 * aStride);
        auto s1  = LOAD8(A + 0 * aStride + 8);
        auto ws  = _load_int8x4(weight, alpha, bias);
        _mm_storeu_ps(ws_tmp, ws);
        auto w0  = _mm256_set1_ps(ws_tmp[0]);
        auto w1  = _mm256_set1_ps(ws_tmp[1]);
        auto w2  = _mm256_set1_ps(ws_tmp[2]);
        auto w3  = _mm256_set1_ps(ws_tmp[3]);
        auto z0  = _mm256_mul_ps(s0, w0);
        auto z1  = _mm256_mul_ps(s1, w0);
        auto z3  = _mm256_mul_ps(s0, w1);
        auto z4  = _mm256_mul_ps(s1, w1);
        auto z6  = _mm256_mul_ps(s0, w2);
        auto z7  = _mm256_mul_ps(s1, w2);
        auto z9  = _mm256_mul_ps(s0, w3);
        auto z10 = _mm256_mul_ps(s1, w3);
        for (int sy = 1; sy < l; ++sy) {
            s0  = LOAD8(A + sy * aStride);
            s1  = LOAD8(A + sy * aStride + 8);
            ws  = _load_int8x4(weight + sy * 4, alpha, bias);
            _mm_storeu_ps(ws_tmp, ws);
            w0  = _mm256_set1_ps(ws_tmp[0]);
            w1  = _mm256_set1_ps(ws_tmp[1]);
            w2  = _mm256_set1_ps(ws_tmp[2]);
            w3  = _mm256_set1_ps(ws_tmp[3]);
            z0  = MNNAVXFMA(s0, w0, z0);
            z1  = MNNAVXFMA(s1, w0, z1);
            z3  = MNNAVXFMA(s0, w1, z3);
            z4  = MNNAVXFMA(s1, w1, z4);
            z6  = MNNAVXFMA(s0, w2, z6);
            z7  = MNNAVXFMA(s1, w2, z7);
            z9  = MNNAVXFMA(s0, w3, z9);
            z10 = MNNAVXFMA(s1, w3, z10);
        }
        if (0 == blockId) {
            TRANPOSE_SAVE(0, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(1, 0, z0, z3, z6, z9);
            TRANPOSE_SAVE(0, 1, z1, z4, z7, z10);
            TRANPOSE_SAVE(1, 1, z1, z4, z7, z10);
        } else {
            FMLA_TRANSPOSE_SAVE(0, 0, z0, z3, z6, z9);
            FMLA_TRANSPOSE_SAVE(1, 0, z0, z3, z6, z9);
            FMLA_TRANSPOSE_SAVE(0, 1, z1, z4, z7, z10);
            FMLA_TRANSPOSE_SAVE(1, 1, z1, z4, z7, z10);
        }
    }
}

template <typename TYPE>
static void _AVX_MNNPackedMatMul_int8_5(TYPE* C, const TYPE* A, const int8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(int8_t);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
    auto hC4          = UP_DIV(h, 4);
    int lC4 = l / 4;
    int lR = lC4 * 4;
    const int hC4Unit = 4;
    int hC16 = hC4 / hC4Unit;
    int hR = hC16 * hC4Unit;
    auto src = A;
    for (int y = 0; y < hC16; ++y) {
        LOAD_WEIGHT_ALPHA_BIAS_int8x4
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
            auto w0 = _load_int8x4(weight0, alpha0, bias0);
            auto w1 = _load_int8x4(weight1, alpha1, bias1);
            auto w2 = _load_int8x4(weight2, alpha2, bias2);
            auto w3 = _load_int8x4(weight3, alpha3, bias3);
            auto W0 = _mm256_set_m128(w1, w0);
            auto W1 = _mm256_set_m128(w3, w2);

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
        if (0 == blockId) {
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
        } else {
            auto tmp0 = LOAD8(dst0);
            auto tmp1 = LOAD8(dst0 + 8);
            auto tmp2 = LOAD8(dst0 + 16);
            auto tmp3 = LOAD8(dst0 + 24);
            auto tmp4 = LOAD8(dst0 + 32);
            auto tmp5 = LOAD8(dst2);
            auto tmp6 = LOAD8(dst2 + 8);
            auto tmp7 = LOAD8(dst2 + 16);
            auto tmp8 = LOAD8(dst2 + 24);
            auto tmp9 = LOAD8(dst2 + 32);

            sumAvx00 = _mm256_add_ps(sumAvx00, tmp0);
            sumAvx10 = _mm256_add_ps(sumAvx10, tmp1);
            sumAvx20 = _mm256_add_ps(sumAvx20, tmp2);
            sumAvx30 = _mm256_add_ps(sumAvx30, tmp3);
            sumAvx40 = _mm256_add_ps(sumAvx40, tmp4);
            sumAvx01 = _mm256_add_ps(sumAvx01, tmp5);
            sumAvx11 = _mm256_add_ps(sumAvx11, tmp6);
            sumAvx21 = _mm256_add_ps(sumAvx21, tmp7);
            sumAvx31 = _mm256_add_ps(sumAvx31, tmp8);
            sumAvx41 = _mm256_add_ps(sumAvx41, tmp9);

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
    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0     = BROAD_LOAD_4(A + 0 * aStride + 0);
        auto s1     = BROAD_LOAD_4(A + 0 * aStride + 1);
        auto s2     = BROAD_LOAD_4(A + 0 * aStride + 2);
        auto s3     = BROAD_LOAD_4(A + 0 * aStride + 3);
        auto s4     = BROAD_LOAD_4(A + 0 * aStride + 4);
        auto w0     = _load_int8x4(weight, alpha, bias);
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
            w0 = _load_int8x4(weight + sy * 4, alpha, bias);
            z0 = MNNSSEFMA(s0, w0, z0);
            z1 = MNNSSEFMA(s1, w0, z1);
            z2 = MNNSSEFMA(s2, w0, z2);
            z3 = MNNSSEFMA(s3, w0, z3);
            z4 = MNNSSEFMA(s4, w0, z4);
        }
        if (0 == blockId) {
            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z1);
            STORE_4(dst + 8 * 2, z2);
            STORE_4(dst + 8 * 3, z3);
            STORE_4(dst + 8 * 4, z4);
        } else {
            auto tmp0 = LOAD4(dst + 8 * 0);
            auto tmp1 = LOAD4(dst + 8 * 1);
            auto tmp2 = LOAD4(dst + 8 * 2);
            auto tmp3 = LOAD4(dst + 8 * 3);
            auto tmp4 = LOAD4(dst + 8 * 4);

            z0 = _mm_add_ps(tmp0, z0);
            z1 = _mm_add_ps(tmp1, z1);
            z2 = _mm_add_ps(tmp2, z2);
            z3 = _mm_add_ps(tmp3, z3);
            z4 = _mm_add_ps(tmp4, z4);

            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z1);
            STORE_4(dst + 8 * 2, z2);
            STORE_4(dst + 8 * 3, z3);
            STORE_4(dst + 8 * 4, z4);
        }
    }
}


template <typename TYPE>
static void _AVX_MNNPackedMatMul_int8_4(TYPE* C, const TYPE* A, const int8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(int8_t);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
    auto hC4          = UP_DIV(h, 4);
    int lC4 = l / 4;
    int lR = lC4 * 4;
    const int hC4Unit = 4;
    int hC16 = hC4 / hC4Unit;
    int hR = hC16 * hC4Unit;
    auto src = A;
    for (int y = 0; y < hC16; ++y) {
        LOAD_WEIGHT_ALPHA_BIAS_int8x4
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
            auto w0 = _load_int8x4(weight0, alpha0, bias0);
            auto w1 = _load_int8x4(weight1, alpha1, bias1);
            auto w2 = _load_int8x4(weight2, alpha2, bias2);
            auto w3 = _load_int8x4(weight3, alpha3, bias3);
            auto W0 = _mm256_set_m128(w1, w0);
            auto W1 = _mm256_set_m128(w3, w2);

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
        if (0 == blockId) {
            STORE_8(dst0, sumAvx00);
            STORE_8(dst0 + 8, sumAvx10);
            STORE_8(dst0 + 16, sumAvx20);
            STORE_8(dst0 + 24, sumAvx30);

            STORE_8(dst2, sumAvx01);
            STORE_8(dst2 + 8, sumAvx11);
            STORE_8(dst2 + 16, sumAvx21);
            STORE_8(dst2 + 24, sumAvx31);
        } else {
            auto tmp0 = LOAD8(dst0);
            auto tmp1 = LOAD8(dst0 + 8);
            auto tmp2 = LOAD8(dst0 + 16);
            auto tmp3 = LOAD8(dst0 + 24);

            auto tmp5 = LOAD8(dst2);
            auto tmp6 = LOAD8(dst2 + 8);
            auto tmp7 = LOAD8(dst2 + 16);
            auto tmp8 = LOAD8(dst2 + 24);

            sumAvx00 = _mm256_add_ps(sumAvx00, tmp0);
            sumAvx10 = _mm256_add_ps(sumAvx10, tmp1);
            sumAvx20 = _mm256_add_ps(sumAvx20, tmp2);
            sumAvx30 = _mm256_add_ps(sumAvx30, tmp3);

            sumAvx01 = _mm256_add_ps(sumAvx01, tmp5);
            sumAvx11 = _mm256_add_ps(sumAvx11, tmp6);
            sumAvx21 = _mm256_add_ps(sumAvx21, tmp7);
            sumAvx31 = _mm256_add_ps(sumAvx31, tmp8);

            STORE_8(dst0, sumAvx00);
            STORE_8(dst0 + 8, sumAvx10);
            STORE_8(dst0 + 16, sumAvx20);
            STORE_8(dst0 + 24, sumAvx30);

            STORE_8(dst2, sumAvx01);
            STORE_8(dst2 + 8, sumAvx11);
            STORE_8(dst2 + 16, sumAvx21);
            STORE_8(dst2 + 24, sumAvx31);
        }
    }
    float ws_tmp[4];
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0     = LOAD4(A + 0 * aStride);
        auto ws     = _load_int8x4(weight, alpha, bias);
        _mm_storeu_ps(ws_tmp, ws);
        auto w0     = _mm_set1_ps(ws_tmp[0]);
        auto w1     = _mm_set1_ps(ws_tmp[1]);
        auto w2     = _mm_set1_ps(ws_tmp[2]);
        auto w3     = _mm_set1_ps(ws_tmp[3]);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z3     = _mm_mul_ps(s0, w1);
        auto z6     = _mm_mul_ps(s0, w2);
        auto z9     = _mm_mul_ps(s0, w3);

        for (int sy = 1; sy < l; ++sy) {
            s0 = LOAD4(A + sy * aStride);
            ws = _load_int8x4(weight + sy * 4, alpha, bias);
            _mm_storeu_ps(ws_tmp, ws);
            w0 = _mm_set1_ps(ws_tmp[0]);
            w1 = _mm_set1_ps(ws_tmp[1]);
            w2 = _mm_set1_ps(ws_tmp[2]);
            w3 = _mm_set1_ps(ws_tmp[3]);
            z0 = MNNSSEFMA(s0, w0, z0);
            z3 = MNNSSEFMA(s0, w1, z3);
            z6 = MNNSSEFMA(s0, w2, z6);
            z9 = MNNSSEFMA(s0, w3, z9);
        }
        _MM_TRANSPOSE4_PS(z0, z3, z6, z9);
        if (0 == blockId) {
            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z3);
            STORE_4(dst + 8 * 2, z6);
            STORE_4(dst + 8 * 3, z9);
        } else {

            auto tmp0 = LOAD4(dst + 8 * 0);
            auto tmp1 = LOAD4(dst + 8 * 1);
            auto tmp2 = LOAD4(dst + 8 * 2);
            auto tmp3 = LOAD4(dst + 8 * 3);

            z0 = _mm_add_ps(tmp0, z0);
            z3 = _mm_add_ps(tmp1, z3);
            z6 = _mm_add_ps(tmp2, z6);
            z9 = _mm_add_ps(tmp3, z9);

            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z3);
            STORE_4(dst + 8 * 2, z6);
            STORE_4(dst + 8 * 3, z9);
        }
    }
}
template <typename TYPE>
static void _AVX_MNNPackedMatMul_int8_3(TYPE* C, const TYPE* A, const int8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(int8_t);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
    auto hC4          = UP_DIV(h, 4);
    int lC4 = l / 4;
    int lR = lC4 * 4;
    const int hC4Unit = 4;
    int hC16 = hC4 / hC4Unit;
    int hR = hC16 * hC4Unit;
    auto src = A;
    for (int y = 0; y < hC16; ++y) {
        LOAD_WEIGHT_ALPHA_BIAS_int8x4
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
            auto w0 = _load_int8x4(weight0, alpha0, bias0);
            auto w1 = _load_int8x4(weight1, alpha1, bias1);
            auto w2 = _load_int8x4(weight2, alpha2, bias2);
            auto w3 = _load_int8x4(weight3, alpha3, bias3);
            auto W0 = _mm256_set_m128(w1, w0);
            auto W1 = _mm256_set_m128(w3, w2);

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
        if (0 == blockId) {
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
        } else {
            auto tmp00 = LOAD4(dst0 + 0);
            auto tmp01 = LOAD4(dst0 + 8);
            auto tmp02 = LOAD4(dst0 + 16);

            auto tmp10 = LOAD4(dst1 + 0);
            auto tmp11 = LOAD4(dst1 + 8);
            auto tmp12 = LOAD4(dst1 + 16);

            auto tmp20 = LOAD4(dst2 + 0);
            auto tmp21 = LOAD4(dst2 + 8);
            auto tmp22 = LOAD4(dst2 + 16);

            auto tmp30 = LOAD4(dst3 + 0);
            auto tmp31 = LOAD4(dst3 + 8);
            auto tmp32 = LOAD4(dst3 + 16);

            auto sum_tmp00 = _mm256_extractf128_ps(sumAvx00, 0);
            auto sum_tmp01 = _mm256_extractf128_ps(sumAvx10, 0);
            auto sum_tmp02 = _mm256_extractf128_ps(sumAvx20, 0);
            auto sum_tmp10 = _mm256_extractf128_ps(sumAvx00, 1);
            auto sum_tmp11 = _mm256_extractf128_ps(sumAvx10, 1);
            auto sum_tmp12 = _mm256_extractf128_ps(sumAvx20, 1);
            auto sum_tmp20 = _mm256_extractf128_ps(sumAvx01, 0);
            auto sum_tmp21 = _mm256_extractf128_ps(sumAvx11, 0);
            auto sum_tmp22 = _mm256_extractf128_ps(sumAvx21, 0);
            auto sum_tmp30 = _mm256_extractf128_ps(sumAvx01, 1);
            auto sum_tmp31 = _mm256_extractf128_ps(sumAvx11, 1);
            auto sum_tmp32 = _mm256_extractf128_ps(sumAvx21, 1);

            sum_tmp00 = _mm_add_ps(tmp00, sum_tmp00);
            sum_tmp01 = _mm_add_ps(tmp01, sum_tmp01);
            sum_tmp02 = _mm_add_ps(tmp02, sum_tmp02);
            sum_tmp10 = _mm_add_ps(tmp10, sum_tmp10);
            sum_tmp11 = _mm_add_ps(tmp11, sum_tmp11);
            sum_tmp12 = _mm_add_ps(tmp12, sum_tmp12);
            sum_tmp20 = _mm_add_ps(tmp20, sum_tmp20);
            sum_tmp21 = _mm_add_ps(tmp21, sum_tmp21);
            sum_tmp22 = _mm_add_ps(tmp22, sum_tmp22);
            sum_tmp30 = _mm_add_ps(tmp30, sum_tmp30);
            sum_tmp31 = _mm_add_ps(tmp31, sum_tmp31);
            sum_tmp32 = _mm_add_ps(tmp32, sum_tmp32);

            STORE_4(dst0 + 0,  sum_tmp00);
            STORE_4(dst0 + 8,  sum_tmp01);
            STORE_4(dst0 + 16, sum_tmp02);
            STORE_4(dst1 + 0,  sum_tmp10);
            STORE_4(dst1 + 8,  sum_tmp11);
            STORE_4(dst1 + 16, sum_tmp12);
            STORE_4(dst2 + 0,  sum_tmp20);
            STORE_4(dst2 + 8,  sum_tmp21);
            STORE_4(dst2 + 16, sum_tmp22);
            STORE_4(dst3 + 0,  sum_tmp30);
            STORE_4(dst3 + 8,  sum_tmp31);
            STORE_4(dst3 + 16, sum_tmp32);
        }

    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0     = BROAD_LOAD_4(A + 0 * aStride + 0);
        auto s1     = BROAD_LOAD_4(A + 0 * aStride + 1);
        auto s2     = BROAD_LOAD_4(A + 0 * aStride + 2);
        auto w0     = _load_int8x4(weight, alpha, bias);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z1     = _mm_mul_ps(s1, w0);
        auto z2     = _mm_mul_ps(s2, w0);

        for (int sy = 1; sy < l; ++sy) {
            s0 = BROAD_LOAD_4(A + sy * aStride + 0);
            s1 = BROAD_LOAD_4(A + sy * aStride + 1);
            s2 = BROAD_LOAD_4(A + sy * aStride + 2);
            w0 = _load_int8x4(weight + sy * 4, alpha, bias);
            z0 = MNNSSEFMA(s0, w0, z0);
            z1 = MNNSSEFMA(s1, w0, z1);
            z2 = MNNSSEFMA(s2, w0, z2);
        }
        if (0 == blockId) {
            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z1);
            STORE_4(dst + 8 * 2, z2);
        } else {
            auto tmp0 = LOAD4(dst + 8 * 0);
            auto tmp1 = LOAD4(dst + 8 * 1);
            auto tmp2 = LOAD4(dst + 8 * 2);

            z0 = _mm_add_ps(tmp0, z0);
            z1 = _mm_add_ps(tmp1, z1);
            z2 = _mm_add_ps(tmp2, z2);

            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z1);
            STORE_4(dst + 8 * 2, z2);
        }
    }
}

template <typename TYPE>
static void _AVX_MNNPackedMatMul_int8_2(TYPE* C, const TYPE* A, const int8_t* B, const size_t* parameter, const float* k, const float* b) {
    auto aStride      = parameter[0] / sizeof(TYPE);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(int8_t);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
    auto hC4          = UP_DIV(h, 4);
    int lC4 = l / 4;
    int lR = lC4 * 4;
    const int hC4Unit = 4;
    int hC16 = hC4 / hC4Unit;
    int hR = hC16 * hC4Unit;
    auto src = A;
    for (int y = 0; y < hC16; ++y) {
        LOAD_WEIGHT_ALPHA_BIAS_int8x4
        auto sumAvx00    = _mm256_setzero_ps();
        auto sumAvx01    = _mm256_setzero_ps();
        DST_ADDR_UNPACK4(0);

        auto sumAvx10    = _mm256_setzero_ps();
        auto sumAvx11    = _mm256_setzero_ps();

        auto srcUse = src;
        for (int sy = 0; sy < l; ++sy) {
            auto S0 = BROAD_LOAD(srcUse + 0);
            auto S1 = BROAD_LOAD(srcUse + 1);
            auto w0 = _load_int8x4(weight0, alpha0, bias0);
            auto w1 = _load_int8x4(weight1, alpha1, bias1);
            auto w2 = _load_int8x4(weight2, alpha2, bias2);
            auto w3 = _load_int8x4(weight3, alpha3, bias3);
            auto W0 = _mm256_set_m128(w1, w0);
            auto W1 = _mm256_set_m128(w3, w2);

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
        if (0 == blockId) {
            STORE_4(dst0 + 0, _mm256_extractf128_ps(sumAvx00, 0));
            STORE_4(dst0 + 8, _mm256_extractf128_ps(sumAvx10, 0));

            STORE_4(dst1 + 0, _mm256_extractf128_ps(sumAvx00, 1));
            STORE_4(dst1 + 8, _mm256_extractf128_ps(sumAvx10, 1));

            STORE_4(dst2 + 0, _mm256_extractf128_ps(sumAvx01, 0));
            STORE_4(dst2 + 8, _mm256_extractf128_ps(sumAvx11, 0));

            STORE_4(dst3 + 0, _mm256_extractf128_ps(sumAvx01, 1));
            STORE_4(dst3 + 8, _mm256_extractf128_ps(sumAvx11, 1));
        } else {
            auto tmp01 = LOAD4(dst0 + 0);
            auto tmp02 = LOAD4(dst0 + 8);
            auto tmp11 = LOAD4(dst1 + 0);
            auto tmp12 = LOAD4(dst1 + 8);
            auto tmp21 = LOAD4(dst2 + 0);
            auto tmp22 = LOAD4(dst2 + 8);
            auto tmp31 = LOAD4(dst3 + 0);
            auto tmp32 = LOAD4(dst3 + 8);

            auto x_tmp01 = _mm256_extractf128_ps(sumAvx00, 0);
            auto x_tmp02 = _mm256_extractf128_ps(sumAvx10, 0);
            auto x_tmp11 = _mm256_extractf128_ps(sumAvx00, 1);
            auto x_tmp12 = _mm256_extractf128_ps(sumAvx10, 1);
            auto x_tmp21 = _mm256_extractf128_ps(sumAvx01, 0);
            auto x_tmp22 = _mm256_extractf128_ps(sumAvx11, 0);
            auto x_tmp31 = _mm256_extractf128_ps(sumAvx01, 1);
            auto x_tmp32 = _mm256_extractf128_ps(sumAvx11, 1);

            x_tmp01 = _mm_add_ps(tmp01, x_tmp01);
            x_tmp02 = _mm_add_ps(tmp02, x_tmp02);
            x_tmp11 = _mm_add_ps(tmp11, x_tmp11);
            x_tmp12 = _mm_add_ps(tmp12, x_tmp12);
            x_tmp21 = _mm_add_ps(tmp21, x_tmp21);
            x_tmp22 = _mm_add_ps(tmp22, x_tmp22);
            x_tmp31 = _mm_add_ps(tmp31, x_tmp31);
            x_tmp32 = _mm_add_ps(tmp32, x_tmp32);

            STORE_4(dst0 + 0, x_tmp01);
            STORE_4(dst0 + 8, x_tmp02);
            STORE_4(dst1 + 0, x_tmp11);
            STORE_4(dst1 + 8, x_tmp12);
            STORE_4(dst2 + 0, x_tmp21);
            STORE_4(dst2 + 8, x_tmp22);
            STORE_4(dst3 + 0, x_tmp31);
            STORE_4(dst3 + 8, x_tmp32);
        }

    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto s0     = BROAD_LOAD_4(A + 0 * aStride + 0);
        auto s1     = BROAD_LOAD_4(A + 0 * aStride + 1);
        auto w0     = _load_int8x4(weight, alpha, bias);
        auto z0     = _mm_mul_ps(s0, w0);
        auto z1     = _mm_mul_ps(s1, w0);

        for (int sy = 1; sy < l; ++sy) {
            s0 = BROAD_LOAD_4(A + sy * aStride + 0);
            s1 = BROAD_LOAD_4(A + sy * aStride + 1);
            w0 = _load_int8x4(weight + sy * 4, alpha, bias);
            z0 = MNNSSEFMA(s0, w0, z0);
            z1 = MNNSSEFMA(s1, w0, z1);
        }
        if (0 == blockId) {
            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z1);
        } else {
            auto t0 = LOAD4(dst + 8 * 0);
            auto t1 = LOAD4(dst + 8 * 1);
            z0 = _mm_add_ps(z0, t0);
            z1 = _mm_add_ps(z1, t1);
            STORE_4(dst + 8 * 0, z0);
            STORE_4(dst + 8 * 1, z1);
        }
    }
}

template <typename TYPE>
static void _AVX_MNNPackednMatMulRemainCommon_int8(TYPE* C, const TYPE* A, const TYPE* fB, size_t eSize,
                                                   const size_t* parameter, const float* k, const float* b) {
    auto B           = reinterpret_cast<const int8_t*>(fB);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(int8_t);
    auto bStride      = bExtraStride + 4 * l;
    auto blockId      = parameter[6];
    auto hC4          = UP_DIV(h, 4);
    auto es           = eSize;
    auto oC           = C;
    auto aStride      = parameter[0] / sizeof(TYPE);
    if (eSize >= 20) {
        _AVX_MNNPackedMatMul_int8_20<TYPE>(C, A, B, parameter, k, b);
        eSize -= 20;
        C += 20 * 8;
        A += 20;
    }
    if (eSize >= 16) {
        _AVX_MNNPackedMatMul_int8_16<TYPE>(C, A, B, parameter, k, b);
        eSize -= 16;
        C += 16 * 8;
        A += 16;
    }
    while (eSize >= 5) {
        _AVX_MNNPackedMatMul_int8_5<TYPE>(C, A, B, parameter, k, b);
        eSize -= 5;
        C += 5 * 8;
        A += 5;
    }
    if (eSize == 4) {
        _AVX_MNNPackedMatMul_int8_4<TYPE>(C, A, B, parameter, k, b);
        return;
    }
    if (eSize == 3) {
        _AVX_MNNPackedMatMul_int8_3<TYPE>(C, A, B, parameter, k, b);
        return;
    }
    if (eSize == 2) {
        _AVX_MNNPackedMatMul_int8_2<TYPE>(C, A, B, parameter, k, b);
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
        auto dst0    = C + (hC4Unit * y / 2 + 0) * cStride + x * 8;
        auto dst1    = C + (hC4Unit * y / 2 + 0) * cStride + x * 8 + 4;
        auto dst2    = C + (hC4Unit * y / 2 + 1) * cStride + x * 8;
        auto dst3    = C + (hC4Unit * y / 2 + 1) * cStride + x * 8 + 4;
        LOAD_WEIGHT_ALPHA_BIAS_int8x4
        LOAD_ALPHA_BIAS_DOUBLE

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

            auto W00 = _load_int8x8(weight0 + 16 * sy + 0, alpha0_2, bias0_2);
            auto W01 = _load_int8x8(weight0 + 16 * sy + 8, alpha0_2, bias0_2);
            auto W10 = _load_int8x8(weight1 + 16 * sy + 0, alpha1_2, bias1_2);
            auto W11 = _load_int8x8(weight1 + 16 * sy + 8, alpha1_2, bias1_2);

            auto W20 = _load_int8x8(weight2 + 16 * sy + 0, alpha2_2, bias2_2);
            auto W21 = _load_int8x8(weight2 + 16 * sy + 8, alpha2_2, bias2_2);
            auto W30 = _load_int8x8(weight3 + 16 * sy + 0, alpha3_2, bias3_2);
            auto W31 = _load_int8x8(weight3 + 16 * sy + 8, alpha3_2, bias3_2);

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
            auto w0 = _load_int8x4(weight0 + 4 * sy, alpha0, bias0);
            auto w1 = _load_int8x4(weight1 + 4 * sy, alpha1, bias1);
            auto w2 = _load_int8x4(weight2 + 4 * sy, alpha2, bias2);
            auto w3 = _load_int8x4(weight3 + 4 * sy, alpha3, bias3);
            sum0    = MNNSSEFMA(s, w0, sum0);
            sum1    = MNNSSEFMA(s, w1, sum1);
            sum2    = MNNSSEFMA(s, w2, sum2);
            sum3    = MNNSSEFMA(s, w3, sum3);
            srcUse += aStride;
        }
        if (blockId == 0) {
            STORE_4(dst0, sum0);
            STORE_4(dst1, sum1);
            STORE_4(dst2, sum2);
            STORE_4(dst3, sum3);
        } else {
            auto tmp_0 = LOAD4(dst0);
            auto tmp_1 = LOAD4(dst1);
            auto tmp_2 = LOAD4(dst2);
            auto tmp_3 = LOAD4(dst3);
            sum0 = _mm_add_ps(tmp_0, sum0);
            sum1 = _mm_add_ps(tmp_1, sum1);
            sum2 = _mm_add_ps(tmp_2, sum2);
            sum3 = _mm_add_ps(tmp_3, sum3);
            STORE_4(dst0, sum0);
            STORE_4(dst1, sum1);
            STORE_4(dst2, sum2);
            STORE_4(dst3, sum3);
        }
    }
    for (int y = hR; y < hC4; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + x * 8 + 4 * (y % 2);
        auto alpha  = _mm_loadu_ps(k + y * 4);
        auto bias   = _mm_loadu_ps(b + y * 4);
        auto alpha_2 = _mm256_set_m128(alpha, alpha);
        auto bias_2  = _mm256_set_m128(bias, bias);

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
            auto W0 = _load_int8x8(weight + 16 * sy + 0, alpha_2, bias_2);
            auto W1 = _load_int8x8(weight + 16 * sy + 8, alpha_2, bias_2);
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
            auto w = _load_int8x4(weight + sy * 4, alpha, bias);
            sum    = MNNSSEFMA(s, w, sum);
            srcUse += aStride;
        }
        if (blockId == 0) {
            STORE_4(dst, sum);
        } else {
            auto tmp_0 = LOAD4(dst);
            sum = _mm_add_ps(tmp_0, sum);
            STORE_4(dst, sum);
            
        }
    }
}

#endif
