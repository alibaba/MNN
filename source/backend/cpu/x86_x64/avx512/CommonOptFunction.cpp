#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "math/Vec.hpp"
#include <limits>
#include <string.h>
#include <algorithm>
#include <vector>
#ifdef MNN_X86_USE_ASM
extern "C" {
void _AVX512_MNNGemmFloatUnitMainFMA(float* C, const float* A, const float* B, const size_t* parameter, size_t hC4);
}
#endif

using Vec8 = MNN::Math::Vec<float, 8>;
#define MNNAVXFMA _mm256_fmadd_ps
#define MNNAVX512FMA _mm512_fmadd_ps
#define MNNSSEFMA _mm_fmadd_ps

#define AVX512_TRANSPOSE_SAVE(u, v, z0, z3, z6, z9, z12, z15, z18, z21) \
    {                                                                   \
        auto m0 = _mm512_extractf32x4_ps(z0, u);                        \
        auto m1 = _mm512_extractf32x4_ps(z3, u);                        \
        auto m2 = _mm512_extractf32x4_ps(z6, u);                        \
        auto m3 = _mm512_extractf32x4_ps(z9, u);                        \
        auto m4 = _mm512_extractf32x4_ps(z12, u);                       \
        auto m5 = _mm512_extractf32x4_ps(z15, u);                       \
        auto m6 = _mm512_extractf32x4_ps(z18, u);                       \
        auto m7 = _mm512_extractf32x4_ps(z21, u);                       \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);                              \
        _MM_TRANSPOSE4_PS(m4, m5, m6, m7);                              \
        _mm_storeu_ps(dst + 4 * (0 + 4 * u + 16 * v), m0);               \
        _mm_storeu_ps(dst + 4 * (1 + 4 * u + 16 * v), m1);               \
        _mm_storeu_ps(dst + 4 * (2 + 4 * u + 16 * v), m2);               \
        _mm_storeu_ps(dst + 4 * (3 + 4 * u + 16 * v), m3);               \
        _mm_storeu_ps(dst + cStride + 4 * (0 + 4 * u + 16 * v), m4);     \
        _mm_storeu_ps(dst + cStride + 4 * (1 + 4 * u + 16 * v), m5);     \
        _mm_storeu_ps(dst + cStride + 4 * (2 + 4 * u + 16 * v), m6);     \
        _mm_storeu_ps(dst + cStride + 4 * (3 + 4 * u + 16 * v), m7);     \
    }

#define AVX512_TRANSPOSE_SAVE_HALF(u, v, z0, z3, z6, z9, z12, z15, z18, z21) \
    {                                                                   \
        auto m0 = _mm512_extractf32x4_ps(z0, u);                        \
        auto m1 = _mm512_extractf32x4_ps(z3, u);                        \
        auto m2 = _mm512_extractf32x4_ps(z6, u);                        \
        auto m3 = _mm512_extractf32x4_ps(z9, u);                        \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);                              \
        _mm_storeu_ps(dst + 4 * (0 + 4 * u + 16 * v), m0);               \
        _mm_storeu_ps(dst + 4 * (1 + 4 * u + 16 * v), m1);               \
        _mm_storeu_ps(dst + 4 * (2 + 4 * u + 16 * v), m2);               \
        _mm_storeu_ps(dst + 4 * (3 + 4 * u + 16 * v), m3);               \
    }

#define AVX2_TRANSPOSE_SAVE(u, z0, z3, z6, z9, z12, z15, z18, z21)   \
    {                                                                \
        auto m0 = _mm256_extractf128_ps(z0, u);                      \
        auto m1 = _mm256_extractf128_ps(z3, u);                      \
        auto m2 = _mm256_extractf128_ps(z6, u);                      \
        auto m3 = _mm256_extractf128_ps(z9, u);                      \
        auto m4 = _mm256_extractf128_ps(z12, u);                     \
        auto m5 = _mm256_extractf128_ps(z15, u);                     \
        auto m6 = _mm256_extractf128_ps(z18, u);                     \
        auto m7 = _mm256_extractf128_ps(z21, u);                     \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);                           \
        _MM_TRANSPOSE4_PS(m4, m5, m6, m7);                           \
        _mm_storeu_ps(dst + 4 * (0 + 4 * u), m0);                     \
        _mm_storeu_ps(dst + 4 * (1 + 4 * u), m1);                     \
        _mm_storeu_ps(dst + 4 * (2 + 4 * u), m2);                     \
        _mm_storeu_ps(dst + 4 * (3 + 4 * u), m3);                     \
        _mm_storeu_ps(dst + cStride + 4 * (0 + 4 * u), m4);           \
        _mm_storeu_ps(dst + cStride + 4 * (1 + 4 * u), m5);           \
        _mm_storeu_ps(dst + cStride + 4 * (2 + 4 * u), m6);           \
        _mm_storeu_ps(dst + cStride + 4 * (3 + 4 * u), m7);           \
    }

#define AVX2_TRANSPOSE_SAVE_HALF(u, z0, z3, z6, z9, z12, z15, z18, z21)   \
    {                                                                \
        auto m0 = _mm256_extractf128_ps(z0, u);                      \
        auto m1 = _mm256_extractf128_ps(z3, u);                      \
        auto m2 = _mm256_extractf128_ps(z6, u);                      \
        auto m3 = _mm256_extractf128_ps(z9, u);                      \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);                           \
        _mm_storeu_ps(dst + 4 * (0 + 4 * u), m0);                     \
        _mm_storeu_ps(dst + 4 * (1 + 4 * u), m1);                     \
        _mm_storeu_ps(dst + 4 * (2 + 4 * u), m2);                     \
        _mm_storeu_ps(dst + 4 * (3 + 4 * u), m3);                     \
    }

#define INIT_MAIN_4_8                                        \
        auto s0 = _mm256_loadu_ps(weight + 0 * 8);             \
        auto w0 = _mm256_broadcast_ss(A + 0 * aStride + 0);      \
        auto w1 = _mm256_broadcast_ss(A + 0 * aStride + 1);      \
        auto w2 = _mm256_broadcast_ss(A + 0 * aStride + 2);      \
        auto w3 = _mm256_broadcast_ss(A + 0 * aStride + 3);      \
        auto z0 = _mm256_mul_ps(s0, w0);                        \
        auto z1 = _mm256_mul_ps(s0, w1);                        \
        auto z2 = _mm256_mul_ps(s0, w2);                        \
        auto z3 = _mm256_mul_ps(s0, w3);                        \

#define COMPUTE_4_8                                          \
        s0 = _mm256_loadu_ps(weight + sy * 8);                 \
        w0 = _mm256_broadcast_ss(A + sy * aStride + 0);          \
        w1 = _mm256_broadcast_ss(A + sy * aStride + 1);          \
        w2 = _mm256_broadcast_ss(A + sy * aStride + 2);          \
        w3 = _mm256_broadcast_ss(A + sy * aStride + 3);          \
        z0 = MNNAVXFMA(s0, w0, z0);                          \
        z1 = MNNAVXFMA(s0, w1, z1);                          \
        z2 = MNNAVXFMA(s0, w2, z2);                          \
        z3 = MNNAVXFMA(s0, w3, z3);                          \

#define INIT_MAIN_5_8                                        \
        auto s0 = _mm256_loadu_ps(weight + 0 * 8);             \
        auto w0 = _mm256_broadcast_ss(A + 0 * aStride + 0);      \
        auto w1 = _mm256_broadcast_ss(A + 0 * aStride + 1);      \
        auto w2 = _mm256_broadcast_ss(A + 0 * aStride + 2);      \
        auto w3 = _mm256_broadcast_ss(A + 0 * aStride + 3);      \
        auto w4 = _mm256_broadcast_ss(A + 0 * aStride + 4);      \
        auto z0 = _mm256_mul_ps(s0, w0);                        \
        auto z1 = _mm256_mul_ps(s0, w1);                        \
        auto z2 = _mm256_mul_ps(s0, w2);                        \
        auto z3 = _mm256_mul_ps(s0, w3);                        \
        auto z4 = _mm256_mul_ps(s0, w4);                        \

#define COMPUTE_5_8                                          \
        s0 = _mm256_loadu_ps(weight + sy * 8);                 \
        w0 = _mm256_broadcast_ss(A + sy * aStride + 0);          \
        w1 = _mm256_broadcast_ss(A + sy * aStride + 1);          \
        w2 = _mm256_broadcast_ss(A + sy * aStride + 2);          \
        w3 = _mm256_broadcast_ss(A + sy * aStride + 3);          \
        w4 = _mm256_broadcast_ss(A + sy * aStride + 4);      \
        z0 = MNNAVXFMA(s0, w0, z0);                          \
        z1 = MNNAVXFMA(s0, w1, z1);                          \
        z2 = MNNAVXFMA(s0, w2, z2);                          \
        z3 = MNNAVXFMA(s0, w3, z3);                          \
        z4 = MNNAVXFMA(s0, w4, z4);                          \


#define INIT_MAIN_8_8                                  \
    auto s0 = _mm256_loadu_ps(A + 0 * aStride);        \
    auto w0 = _mm256_broadcast_ss(weight + 0 * 8 + 0); \
    auto w1 = _mm256_broadcast_ss(weight + 0 * 8 + 1); \
    auto w2 = _mm256_broadcast_ss(weight + 0 * 8 + 2); \
    auto w3 = _mm256_broadcast_ss(weight + 0 * 8 + 3); \
    auto w4 = _mm256_broadcast_ss(weight + 0 * 8 + 4); \
    auto w5 = _mm256_broadcast_ss(weight + 0 * 8 + 5); \
    auto w6 = _mm256_broadcast_ss(weight + 0 * 8 + 6); \
    auto w7 = _mm256_broadcast_ss(weight + 0 * 8 + 7); \
    auto z0 = _mm256_mul_ps(s0, w0);                   \
    auto z1 = _mm256_mul_ps(s0, w1);                   \
    auto z2 = _mm256_mul_ps(s0, w2);                   \
    auto z3 = _mm256_mul_ps(s0, w3);                   \
    auto z4 = _mm256_mul_ps(s0, w4);                   \
    auto z5 = _mm256_mul_ps(s0, w5);                   \
    auto z6 = _mm256_mul_ps(s0, w6);                   \
    auto z7 = _mm256_mul_ps(s0, w7);

#define COMPUTE_8_8                                \
    s0 = _mm256_loadu_ps(A + sy * aStride);        \
    w0 = _mm256_broadcast_ss(weight + sy * 8 + 0); \
    w1 = _mm256_broadcast_ss(weight + sy * 8 + 1); \
    w2 = _mm256_broadcast_ss(weight + sy * 8 + 2); \
    w3 = _mm256_broadcast_ss(weight + sy * 8 + 3); \
    w4 = _mm256_broadcast_ss(weight + sy * 8 + 4); \
    w5 = _mm256_broadcast_ss(weight + sy * 8 + 5); \
    w6 = _mm256_broadcast_ss(weight + sy * 8 + 6); \
    w7 = _mm256_broadcast_ss(weight + sy * 8 + 7); \
    z0 = MNNAVXFMA(s0, w0, z0);                    \
    z1 = MNNAVXFMA(s0, w1, z1);                    \
    z2 = MNNAVXFMA(s0, w2, z2);                    \
    z3 = MNNAVXFMA(s0, w3, z3);                    \
    z4 = MNNAVXFMA(s0, w4, z4);                    \
    z5 = MNNAVXFMA(s0, w5, z5);                    \
    z6 = MNNAVXFMA(s0, w6, z6);                    \
    z7 = MNNAVXFMA(s0, w7, z7);

#define INIT_MAIN_16_8                                  \
    auto s0  = _mm512_loadu_ps(A + 0 * aStride);        \
    auto wt  = _mm_load_ss(weight + 0 * 8 + 0);         \
    auto w0  = _mm512_broadcastss_ps(wt);               \
    auto z0  = _mm512_mul_ps(s0, w0);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 1);               \
    auto w1  = _mm512_broadcastss_ps(wt);               \
    auto z1  = _mm512_mul_ps(s0, w1);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 2);               \
    auto w2  = _mm512_broadcastss_ps(wt);               \
    auto z2  = _mm512_mul_ps(s0, w2);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 3);               \
    auto w3  = _mm512_broadcastss_ps(wt);               \
    auto z3  = _mm512_mul_ps(s0, w3);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 4);               \
    auto w4  = _mm512_broadcastss_ps(wt);               \
    auto z4  = _mm512_mul_ps(s0, w4);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 5);               \
    auto w5  = _mm512_broadcastss_ps(wt);               \
    auto z5  = _mm512_mul_ps(s0, w5);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 6);               \
    auto w6  = _mm512_broadcastss_ps(wt);               \
    auto z6  = _mm512_mul_ps(s0, w6);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 7);               \
    auto w7  = _mm512_broadcastss_ps(wt);               \
    auto z7  = _mm512_mul_ps(s0, w7);


#define COMPUTE_16_8                                \
    s0  = _mm512_loadu_ps(A + sy * aStride);        \
    wt  = _mm_load_ss(weight + sy * 8 + 0);         \
    w0  = _mm512_broadcastss_ps(wt);                \
    z0  = MNNAVX512FMA(s0, w0, z0);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 1);         \
    w1  = _mm512_broadcastss_ps(wt);                \
    z1  = MNNAVX512FMA(s0, w1, z1);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 2);         \
    w2  = _mm512_broadcastss_ps(wt);                \
    z2  = MNNAVX512FMA(s0, w2, z2);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 3);         \
    w3  = _mm512_broadcastss_ps(wt);                \
    z3  = MNNAVX512FMA(s0, w3, z3);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 4);         \
    w4  = _mm512_broadcastss_ps(wt);                \
    z4  = MNNAVX512FMA(s0, w4, z4);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 5);         \
    w5  = _mm512_broadcastss_ps(wt);                \
    z5  = MNNAVX512FMA(s0, w5, z5);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 6);         \
    w6  = _mm512_broadcastss_ps(wt);                \
    z6  = MNNAVX512FMA(s0, w6, z6);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 7);         \
    w7  = _mm512_broadcastss_ps(wt);                \
    z7  = MNNAVX512FMA(s0, w7, z7);

#define INIT_MAIN_48_8                                  \
    auto s0  = _mm512_loadu_ps(A + 0 * 48);             \
    auto s1  = _mm512_loadu_ps(A + 0 * 48 + 16);        \
    auto s2  = _mm512_loadu_ps(A + 0 * 48 + 32);        \
    auto wt  = _mm_load_ss(weight + 0 * 8 + 0);         \
    auto w0  = _mm512_broadcastss_ps(wt);               \
    auto z0  = _mm512_mul_ps(s0, w0);                   \
    auto z1  = _mm512_mul_ps(s1, w0);                   \
    auto z2  = _mm512_mul_ps(s2, w0);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 1);               \
    auto w1  = _mm512_broadcastss_ps(wt);               \
    auto z3  = _mm512_mul_ps(s0, w1);                   \
    auto z4  = _mm512_mul_ps(s1, w1);                   \
    auto z5  = _mm512_mul_ps(s2, w1);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 2);               \
    auto w2  = _mm512_broadcastss_ps(wt);               \
    auto z6  = _mm512_mul_ps(s0, w2);                   \
    auto z7  = _mm512_mul_ps(s1, w2);                   \
    auto z8  = _mm512_mul_ps(s2, w2);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 3);               \
    auto w3  = _mm512_broadcastss_ps(wt);               \
    auto z9  = _mm512_mul_ps(s0, w3);                   \
    auto z10 = _mm512_mul_ps(s1, w3);                   \
    auto z11 = _mm512_mul_ps(s2, w3);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 4);               \
    auto w4  = _mm512_broadcastss_ps(wt);               \
    auto z12 = _mm512_mul_ps(s0, w4);                   \
    auto z13 = _mm512_mul_ps(s1, w4);                   \
    auto z14 = _mm512_mul_ps(s2, w4);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 5);               \
    auto w5  = _mm512_broadcastss_ps(wt);               \
    auto z15 = _mm512_mul_ps(s0, w5);                   \
    auto z16 = _mm512_mul_ps(s1, w5);                   \
    auto z17 = _mm512_mul_ps(s2, w5);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 6);               \
    auto w6  = _mm512_broadcastss_ps(wt);               \
    auto z18 = _mm512_mul_ps(s0, w6);                   \
    auto z19 = _mm512_mul_ps(s1, w6);                   \
    auto z20 = _mm512_mul_ps(s2, w6);                   \
    wt = _mm_load_ss(weight + 0 * 8 + 7);               \
    auto w7  = _mm512_broadcastss_ps(wt);               \
    auto z21 = _mm512_mul_ps(s0, w7);                   \
    auto z22 = _mm512_mul_ps(s1, w7);                   \
    auto z23 = _mm512_mul_ps(s2, w7);

#define COMPUTE_48_8                                \
    s0  = _mm512_loadu_ps(A + sy * 48);             \
    s1  = _mm512_loadu_ps(A + sy * 48 + 16);        \
    s2  = _mm512_loadu_ps(A + sy * 48 + 32);        \
    wt  = _mm_load_ss(weight + sy * 8 + 0);         \
    w0  = _mm512_broadcastss_ps(wt);                \
    z0  = MNNAVX512FMA(s0, w0, z0);                 \
    z1  = MNNAVX512FMA(s1, w0, z1);                 \
    z2  = MNNAVX512FMA(s2, w0, z2);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 1);         \
    w1  = _mm512_broadcastss_ps(wt);                \
    z3  = MNNAVX512FMA(s0, w1, z3);                 \
    z4  = MNNAVX512FMA(s1, w1, z4);                 \
    z5  = MNNAVX512FMA(s2, w1, z5);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 2);         \
    w2  = _mm512_broadcastss_ps(wt);                \
    z6  = MNNAVX512FMA(s0, w2, z6);                 \
    z7  = MNNAVX512FMA(s1, w2, z7);                 \
    z8  = MNNAVX512FMA(s2, w2, z8);                 \
    wt  = _mm_load_ss(weight + sy * 8 + 3);         \
    w3  = _mm512_broadcastss_ps(wt);                \
    z9  = MNNAVX512FMA(s0, w3, z9);                 \
    z10  = MNNAVX512FMA(s1, w3, z10);               \
    z11  = MNNAVX512FMA(s2, w3, z11);               \
    wt  = _mm_load_ss(weight + sy * 8 + 4);         \
    w4  = _mm512_broadcastss_ps(wt);                \
    z12  = MNNAVX512FMA(s0, w4, z12);               \
    z13  = MNNAVX512FMA(s1, w4, z13);               \
    z14  = MNNAVX512FMA(s2, w4, z14);               \
    wt  = _mm_load_ss(weight + sy * 8 + 5);         \
    w5  = _mm512_broadcastss_ps(wt);                \
    z15  = MNNAVX512FMA(s0, w5, z15);               \
    z16  = MNNAVX512FMA(s1, w5, z16);               \
    z17  = MNNAVX512FMA(s2, w5, z17);               \
    wt  = _mm_load_ss(weight + sy * 8 + 6);         \
    w6  = _mm512_broadcastss_ps(wt);                \
    z18  = MNNAVX512FMA(s0, w6, z18);               \
    z19  = MNNAVX512FMA(s1, w6, z19);               \
    z20  = MNNAVX512FMA(s2, w6, z20);               \
    wt  = _mm_load_ss(weight + sy * 8 + 7);         \
    w7  = _mm512_broadcastss_ps(wt);                \
    z21  = MNNAVX512FMA(s0, w7, z21);               \
    z22  = MNNAVX512FMA(s1, w7, z22);               \
    z23  = MNNAVX512FMA(s2, w7, z23);

// TODO: this function is not implemented for avx512 yet.
void AVX512GemmPostTreat(float* C, size_t eSize, const size_t* parameter, const float* postParameters,
                       const float* bias) {
    if (nullptr == postParameters) {
        return;
    }
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    auto minValue     = _mm_broadcast_ss(postParameters + 2);
    auto maxValue     = _mm_broadcast_ss(postParameters + 3);
    int eC2           = eSize / 2;
    int eR            = eSize % 2;
    auto minV2        = _mm256_broadcast_ss(postParameters + 2);
    auto maxV2        = _mm256_broadcast_ss(postParameters + 3);
    if (nullptr != bias) {
        if (eR > 0) {
            for (int y = 0; y < hC4; ++y) {
                auto biasValue = _mm_loadu_ps(bias + 4 * y);
                auto bias2     = _mm256_broadcast_ps((__m128*)(bias + 4 * y));
                auto dst       = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = _mm256_add_ps(bias2, _mm256_loadu_ps(dst));
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    _mm256_storeu_ps(dst, sum);
                    dst += 8;
                }
                auto sum = _mm_add_ps(biasValue, _mm_loadu_ps(dst));
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                _mm_storeu_ps(dst, sum);
            }
        } else {
            for (int y = 0; y < hC4; ++y) {
                auto biasValue = _mm_loadu_ps(bias + 4 * y);
                auto bias2     = _mm256_broadcast_ps((__m128*)(bias + 4 * y));
                auto dst       = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = _mm256_add_ps(bias2, _mm256_loadu_ps(dst));
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    _mm256_storeu_ps(dst, sum);
                    dst += 8;
                }
            }
        }
    } else {
        if (eR > 0) {
            for (int y = 0; y < hC4; ++y) {
                auto dst = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = _mm256_loadu_ps(dst);
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    _mm256_storeu_ps(dst, sum);
                    dst += 8;
                }
                auto sum = _mm_loadu_ps(dst);
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                _mm_storeu_ps(dst, sum);
            }
        } else {
            for (int y = 0; y < hC4; ++y) {
                auto dst = C + y * cStride;
                for (int x = 0; x < eC2; ++x) {
                    auto sum = _mm256_loadu_ps(dst);
                    sum      = _mm256_max_ps(sum, minV2);
                    sum      = _mm256_min_ps(sum, maxV2);
                    _mm256_storeu_ps(dst, sum);
                    dst += 8;
                }
            }
        }
    }
}

static void _AVX512_MNNPackedMatMul_48(float* C, const float* A, const float* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;  //hP=8
    auto hC4 = UP_DIV(h, 4);
    auto hC8 = hC4 / 2;
    auto hR = hC4 % 2;

    for (int y = 0; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + 2 * y * cStride;
        INIT_MAIN_48_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_48_8;
        }

        AVX512_TRANSPOSE_SAVE(0, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE(1, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE(2, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE(3, 0, z0, z3, z6, z9, z12, z15, z18, z21);

        AVX512_TRANSPOSE_SAVE(0, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE(1, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE(2, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE(3, 1, z1, z4, z7, z10, z13, z16, z19, z22);

        AVX512_TRANSPOSE_SAVE(0, 2, z2, z5, z8, z11, z14, z17, z20, z23);
        AVX512_TRANSPOSE_SAVE(1, 2, z2, z5, z8, z11, z14, z17, z20, z23);
        AVX512_TRANSPOSE_SAVE(2, 2, z2, z5, z8, z11, z14, z17, z20, z23);
        AVX512_TRANSPOSE_SAVE(3, 2, z2, z5, z8, z11, z14, z17, z20, z23);
    }
    if (hR > 0) {
        auto weight = B + hC8 * bStride;
        auto dst    = C + 2 * hC8 * cStride;
        INIT_MAIN_48_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_48_8;
        }

        AVX512_TRANSPOSE_SAVE_HALF(0, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE_HALF(1, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE_HALF(2, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE_HALF(3, 0, z0, z3, z6, z9, z12, z15, z18, z21);

        AVX512_TRANSPOSE_SAVE_HALF(0, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE_HALF(1, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE_HALF(2, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE_HALF(3, 1, z1, z4, z7, z10, z13, z16, z19, z22);

        AVX512_TRANSPOSE_SAVE_HALF(0, 2, z2, z5, z8, z11, z14, z17, z20, z23);
        AVX512_TRANSPOSE_SAVE_HALF(1, 2, z2, z5, z8, z11, z14, z17, z20, z23);
        AVX512_TRANSPOSE_SAVE_HALF(2, 2, z2, z5, z8, z11, z14, z17, z20, z23);
        AVX512_TRANSPOSE_SAVE_HALF(3, 2, z2, z5, z8, z11, z14, z17, z20, z23);
    }
}

static void _AVX512_MNNPackedMatMul_16(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC4 = UP_DIV(h, 4);
    auto hC8 = hC4 / 2;
    auto hR = hC4 % 2;

    for (int y = 0; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + 2 * y * cStride;
        INIT_MAIN_16_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_16_8;
        }
        AVX512_TRANSPOSE_SAVE(0, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE(1, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE(2, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE(3, 0, z0, z1, z2, z3, z4, z5, z6, z7);
    }
    if (hR > 0) {
        auto weight = B + hC8 * bStride;
        auto dst    = C + 2 * hC8 * cStride;
        INIT_MAIN_16_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_16_8;
        }
        AVX512_TRANSPOSE_SAVE_HALF(0, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE_HALF(1, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE_HALF(2, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE_HALF(3, 0, z0, z1, z2, z3, z4, z5, z6, z7);
    }
}

static void _AVX2_MNNPackedMatMul_8(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC4 = UP_DIV(h, 4);
    auto hC8 = hC4 / 2;
    auto hR = hC4 % 2;

    for (int y = 0; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + 2 * y * cStride;
        INIT_MAIN_8_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_8_8;
        }
        AVX2_TRANSPOSE_SAVE(0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX2_TRANSPOSE_SAVE(1, z0, z1, z2, z3, z4, z5, z6, z7);
    }
    if (hR > 0) {
        auto weight = B + hC8 * bStride;
        auto dst    = C + 2 * hC8 * cStride;
        INIT_MAIN_8_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_8_8;
        }
        AVX2_TRANSPOSE_SAVE_HALF(0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX2_TRANSPOSE_SAVE_HALF(1, z0, z1, z2, z3, z4, z5, z6, z7);
    }
}

static void _AVX2_MNNPackedMatMul_5(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC4 = UP_DIV(h, 4);
    auto hC8 = hC4 / 2;
    auto hR = hC4 % 2;
    auto lC2 = l / 2;
    auto lR  = l % 2;

    for (int y = 0; y < hC8; ++y) {
        auto Z0 = _mm512_setzero_ps();
        auto Z1 = _mm512_setzero_ps();
        auto Z2 = _mm512_setzero_ps();
        auto Z3 = _mm512_setzero_ps();
        auto Z4 = _mm512_setzero_ps();
        auto a = A;
        for (int sy = 0; sy < lC2; ++sy) {
            auto W = _mm512_loadu_ps(B + 16 * sy);
            auto s00 = _mm256_broadcast_ss(a);
            auto s01 = _mm256_broadcast_ss(a + aStride);
            auto S0 = _mm512_insertf32x8(_mm512_castps256_ps512(s00), s01, 1);

            auto s10 = _mm256_broadcast_ss(a + 1);
            auto s11 = _mm256_broadcast_ss(a + 1 + aStride);
            auto S1 = _mm512_insertf32x8(_mm512_castps256_ps512(s10), s11, 1);

            auto s20 = _mm256_broadcast_ss(a + 2);
            auto s21 = _mm256_broadcast_ss(a + 2 + aStride);
            auto S2 = _mm512_insertf32x8(_mm512_castps256_ps512(s20), s21, 1);

            auto s30 = _mm256_broadcast_ss(a + 3);
            auto s31 = _mm256_broadcast_ss(a + 3 + aStride);
            auto S3 = _mm512_insertf32x8(_mm512_castps256_ps512(s30), s31, 1);

            auto s40 = _mm256_broadcast_ss(a + 4);
            auto s41 = _mm256_broadcast_ss(a + 4 + aStride);
            auto S4 = _mm512_insertf32x8(_mm512_castps256_ps512(s40), s41, 1);
            
            Z0 = MNNAVX512FMA(S0, W, Z0);
            Z1 = MNNAVX512FMA(S1, W, Z1);
            Z2 = MNNAVX512FMA(S2, W, Z2);
            Z3 = MNNAVX512FMA(S3, W, Z3);
            Z4 = MNNAVX512FMA(S4, W, Z4);

            a += 2 * aStride;
        }
        auto z0 = _mm256_add_ps(_mm512_extractf32x8_ps(Z0, 0), _mm512_extractf32x8_ps(Z0, 1));
        auto z1 = _mm256_add_ps(_mm512_extractf32x8_ps(Z1, 0), _mm512_extractf32x8_ps(Z1, 1));
        auto z2 = _mm256_add_ps(_mm512_extractf32x8_ps(Z2, 0), _mm512_extractf32x8_ps(Z2, 1));
        auto z3 = _mm256_add_ps(_mm512_extractf32x8_ps(Z3, 0), _mm512_extractf32x8_ps(Z3, 1));
        auto z4 = _mm256_add_ps(_mm512_extractf32x8_ps(Z4, 0), _mm512_extractf32x8_ps(Z4, 1));
        if (lR > 0) {
            int sy = l - 1;
            __m256 s0;
            __m256 w0;
            __m256 w1;
            __m256 w2;
            __m256 w3;
            __m256 w4;
            auto weight = B;
            COMPUTE_5_8;
        }
        auto p0 = _mm256_permute2f128_ps(z0, z1, 32);
        auto p2 = _mm256_permute2f128_ps(z0, z1, 49);
        auto p1 = _mm256_permute2f128_ps(z2, z3, 32);
        auto p3 = _mm256_permute2f128_ps(z2, z3, 49);
        auto p4 = _mm256_extractf128_ps(z4, 0);
        auto p5 = _mm256_extractf128_ps(z4, 1);
        _mm256_storeu_ps(C + 8 * 0, p0);
        _mm256_storeu_ps(C + 8 * 1, p1);
        _mm_storeu_ps(C + 8 * 2, p4);
        _mm256_storeu_ps(C + 8 * 0 + cStride, p2);
        _mm256_storeu_ps(C + 8 * 1 + cStride, p3);
        _mm_storeu_ps(C + 8 * 2 + cStride, p5);

        B += bStride;
        C += 2 * cStride;
    }
    if (hR > 0) {
        auto weight = B;
        auto dst    = C;
        INIT_MAIN_5_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_5_8;
        }
        auto p0 = _mm256_permute2f128_ps(z0, z1, 32);
        auto p2 = _mm256_permute2f128_ps(z0, z1, 49);
        auto p1 = _mm256_permute2f128_ps(z2, z3, 32);
        auto p3 = _mm256_permute2f128_ps(z2, z3, 49);
        auto p4 = _mm256_extractf128_ps(z4, 0);
        _mm256_storeu_ps(dst + 8 * 0, p0);
        _mm256_storeu_ps(dst + 8 * 1, p1);
        _mm_storeu_ps(dst + 8 * 2, p4);
    }
}

static void _AVX2_MNNPackedMatMul_4(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC4 = UP_DIV(h, 4);
    auto hC8 = hC4 / 2;
    auto hR = hC4 % 2;

    for (int y = 0; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + 2 * y * cStride;
        INIT_MAIN_4_8;
        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_4_8;
        }
        auto p0 = _mm256_permute2f128_ps(z0, z1, 32);
        auto p2 = _mm256_permute2f128_ps(z0, z1, 49);
        auto p1 = _mm256_permute2f128_ps(z2, z3, 32);
        auto p3 = _mm256_permute2f128_ps(z2, z3, 49);
        _mm256_storeu_ps(dst + 8 * 0, p0);
        _mm256_storeu_ps(dst + 8 * 1, p1);
        _mm256_storeu_ps(dst + cStride + 8 * 0, p2);
        _mm256_storeu_ps(dst + cStride + 8 * 1, p3);
    }
    if (hR > 0) {
        auto weight = B + hC8 * bStride;
        auto dst    = C + 2 * hC8 * cStride;
        INIT_MAIN_4_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_4_8;
        }
        auto p0 = _mm256_permute2f128_ps(z0, z1, 32);
        auto p2 = _mm256_permute2f128_ps(z0, z1, 49);
        auto p1 = _mm256_permute2f128_ps(z2, z3, 32);
        auto p3 = _mm256_permute2f128_ps(z2, z3, 49);
        _mm256_storeu_ps(dst + 8 * 0, p0);
        _mm256_storeu_ps(dst + 8 * 1, p1);
    }
}

static void _AVX512_MNNPackednMatMulRemainCommon(float* C, const float* A, const float* B, size_t eSize,
                                              const size_t* parameter, float* cache, const float* postParameters,
                                              const float* bias) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC4 = UP_DIV(h, 4);
    auto hC8 = hC4 / 2;
    auto hR = hC4 % 2;
    auto es           = eSize;
    auto oC           = C;
    auto aStride      = parameter[0] / sizeof(float);

    while (eSize >= 16) {
        _AVX512_MNNPackedMatMul_16(C, A, B, parameter);
        eSize -= 16;
        C += 16 * 4;
        A += 16;
    }

    while (eSize >= 8) {
        _AVX2_MNNPackedMatMul_8(C, A, B, parameter);
        eSize -= 8;
        C += 8 * 4;
        A += 8;
    }
    if (eSize >= 5) {
        _AVX2_MNNPackedMatMul_5(C, A, B, parameter);
        eSize -= 5;
        C += 5 * 4;
        A += 5;
    }
    if (eSize >= 4) {
        _AVX2_MNNPackedMatMul_4(C, A, B, parameter);
        eSize -= 4;
        C += 4 * 4;
        A += 4;
    }

    if (eSize == 0) {
        return;
    }

    int valid = 1 << 31;
    __m128i mask;
    switch (eSize) {
        case 1:
            mask = _mm_set_epi32(0, 0, 0, valid);
            break;
        case 2:
            mask = _mm_set_epi32(0, 0, valid, valid);
            break;
        case 3:
            mask = _mm_set_epi32(0, valid, valid, valid);
            break;
    }

    //TODO: further optimize
    // Remain x = 1..3
    for (int y = 0; y < hC8; y++) {
        auto weight = B + y * bStride;
        auto dst = C + 2 * y * cStride;
        //INIT_MAIN_x_8
        auto s0 = _mm_maskload_ps(A + 0 * aStride, mask);
        auto w0 = _mm_broadcast_ss(weight + 0 * 8 + 0);
        auto w1 = _mm_broadcast_ss(weight + 0 * 8 + 1);
        auto w2 = _mm_broadcast_ss(weight + 0 * 8 + 2);
        auto w3 = _mm_broadcast_ss(weight + 0 * 8 + 3);
        auto w4 = _mm_broadcast_ss(weight + 0 * 8 + 4);
        auto w5 = _mm_broadcast_ss(weight + 0 * 8 + 5);
        auto w6 = _mm_broadcast_ss(weight + 0 * 8 + 6);
        auto w7 = _mm_broadcast_ss(weight + 0 * 8 + 7);
        auto z0 = _mm_mul_ps(s0, w0);
        auto z1 = _mm_mul_ps(s0, w1);
        auto z2 = _mm_mul_ps(s0, w2);
        auto z3 = _mm_mul_ps(s0, w3);
        auto z4 = _mm_mul_ps(s0, w4);
        auto z5 = _mm_mul_ps(s0, w5);
        auto z6 = _mm_mul_ps(s0, w6);
        auto z7 = _mm_mul_ps(s0, w7);
        //COMPUTE_x_8
        for (int sy = 1; sy < l; sy++) {
            s0 = _mm_maskload_ps(A + sy * aStride, mask);
            w0 = _mm_broadcast_ss(weight + sy * 8 + 0);
            w1 = _mm_broadcast_ss(weight + sy * 8 + 1);
            w2 = _mm_broadcast_ss(weight + sy * 8 + 2);
            w3 = _mm_broadcast_ss(weight + sy * 8 + 3);
            w4 = _mm_broadcast_ss(weight + sy * 8 + 4);
            w5 = _mm_broadcast_ss(weight + sy * 8 + 5);
            w6 = _mm_broadcast_ss(weight + sy * 8 + 6);
            w7 = _mm_broadcast_ss(weight + sy * 8 + 7);
            z0 = MNNSSEFMA(s0, w0, z0);
            z1 = MNNSSEFMA(s0, w1, z1);
            z2 = MNNSSEFMA(s0, w2, z2);
            z3 = MNNSSEFMA(s0, w3, z3);
            z4 = MNNSSEFMA(s0, w4, z4);
            z5 = MNNSSEFMA(s0, w5, z5);
            z6 = MNNSSEFMA(s0, w6, z6);
            z7 = MNNSSEFMA(s0, w7, z7);
        }
        //TRANSPOSE_SAVE
        _MM_TRANSPOSE4_PS(z0, z1, z2, z3);
        _MM_TRANSPOSE4_PS(z4, z5, z6, z7);
        if (eSize == 1) {
            _mm_storeu_ps(dst + 4 * 0, z0);
            _mm_storeu_ps(dst + cStride + 4 * 0, z4);
        } else if (eSize == 2) {
            _mm_storeu_ps(dst + 4 * 0, z0);
            _mm_storeu_ps(dst + 4 * 1, z1);
            _mm_storeu_ps(dst + cStride + 4 * 0, z4);
            _mm_storeu_ps(dst + cStride + 4 * 1, z5);
        } else {
            _mm_storeu_ps(dst + 4 * 0, z0);
            _mm_storeu_ps(dst + 4 * 1, z1);
            _mm_storeu_ps(dst + 4 * 2, z2);
            _mm_storeu_ps(dst + cStride + 4 * 0, z4);
            _mm_storeu_ps(dst + cStride + 4 * 1, z5);
            _mm_storeu_ps(dst + cStride + 4 * 2, z6);
        }
    }
    if (hR > 0) {
        auto weight = B + hC8 * bStride;
        auto dst    = C + 2 * hC8 * cStride;
        auto s0 = _mm_maskload_ps(A + 0 * aStride, mask);
        auto w0 = _mm_broadcast_ss(weight + 0 * 8 + 0);
        auto w1 = _mm_broadcast_ss(weight + 0 * 8 + 1);
        auto w2 = _mm_broadcast_ss(weight + 0 * 8 + 2);
        auto w3 = _mm_broadcast_ss(weight + 0 * 8 + 3);
        auto z0 = _mm_mul_ps(s0, w0);
        auto z1 = _mm_mul_ps(s0, w1);
        auto z2 = _mm_mul_ps(s0, w2);
        auto z3 = _mm_mul_ps(s0, w3);
        //COMPUTE_x_8
        for (int sy = 1; sy < l; sy++) {
            s0 = _mm_maskload_ps(A + sy * aStride, mask);
            w0 = _mm_broadcast_ss(weight + sy * 8 + 0);
            w1 = _mm_broadcast_ss(weight + sy * 8 + 1);
            w2 = _mm_broadcast_ss(weight + sy * 8 + 2);
            w3 = _mm_broadcast_ss(weight + sy * 8 + 3);
            z0 = MNNSSEFMA(s0, w0, z0);
            z1 = MNNSSEFMA(s0, w1, z1);
            z2 = MNNSSEFMA(s0, w2, z2);
            z3 = MNNSSEFMA(s0, w3, z3);
        }
        //TRANSPOSE_SAVE
        _MM_TRANSPOSE4_PS(z0, z1, z2, z3);
        if (eSize == 1) {
            _mm_storeu_ps(dst + 4 * 0, z0);
        } else if (eSize == 2) {
            _mm_storeu_ps(dst + 4 * 0, z0);
            _mm_storeu_ps(dst + 4 * 1, z1);
        } else {
            _mm_storeu_ps(dst + 4 * 0, z0);
            _mm_storeu_ps(dst + 4 * 1, z1);
            _mm_storeu_ps(dst + 4 * 2, z2);
        }
    }
}

void _AVX512_MNNPackC4ForMatMul_A(float* dest, const float* source, size_t e, size_t l, size_t eReal) {
#define MAIN_COMPUTE                            \
    auto s00 = _mm_loadu_ps(srcX + 0 * 4);      \
    auto s01 = _mm_loadu_ps(srcX + 1 * 4);      \
    auto s02 = _mm_loadu_ps(srcX + 2 * 4);      \
    auto s03 = _mm_loadu_ps(srcX + 3 * 4);      \
    auto s10 = _mm_loadu_ps(srcX + 4 * 4);      \
    auto s11 = _mm_loadu_ps(srcX + 5 * 4);      \
    auto s12 = _mm_loadu_ps(srcX + 6 * 4);      \
    auto s13 = _mm_loadu_ps(srcX + 7 * 4);      \
    auto s20 = _mm_loadu_ps(srcX + 8 * 4);      \
    auto s21 = _mm_loadu_ps(srcX + 9 * 4);      \
    auto s22 = _mm_loadu_ps(srcX + 10 * 4);     \
    auto s23 = _mm_loadu_ps(srcX + 11 * 4);     \
    auto s30 = _mm_loadu_ps(srcX + 12 * 4);     \
    auto s31 = _mm_loadu_ps(srcX + 13 * 4);     \
    auto s32 = _mm_loadu_ps(srcX + 14 * 4);     \
    auto s33 = _mm_loadu_ps(srcX + 15 * 4);     \
    auto s40 = _mm_loadu_ps(srcX + 16 * 4);     \
    auto s41 = _mm_loadu_ps(srcX + 17 * 4);     \
    auto s42 = _mm_loadu_ps(srcX + 18 * 4);     \
    auto s43 = _mm_loadu_ps(srcX + 19 * 4);     \
    auto s50 = _mm_loadu_ps(srcX + 20 * 4);     \
    auto s51 = _mm_loadu_ps(srcX + 21 * 4);     \
    auto s52 = _mm_loadu_ps(srcX + 22 * 4);     \
    auto s53 = _mm_loadu_ps(srcX + 23 * 4);     \
    auto s60 = _mm_loadu_ps(srcX + 24 * 4);     \
    auto s61 = _mm_loadu_ps(srcX + 25 * 4);     \
    auto s62 = _mm_loadu_ps(srcX + 26 * 4);     \
    auto s63 = _mm_loadu_ps(srcX + 27 * 4);     \
    auto s70 = _mm_loadu_ps(srcX + 28 * 4);     \
    auto s71 = _mm_loadu_ps(srcX + 29 * 4);     \
    auto s72 = _mm_loadu_ps(srcX + 30 * 4);     \
    auto s73 = _mm_loadu_ps(srcX + 31 * 4);     \
    auto s80 = _mm_loadu_ps(srcX + 32 * 4);     \
    auto s81 = _mm_loadu_ps(srcX + 33 * 4);     \
    auto s82 = _mm_loadu_ps(srcX + 34 * 4);     \
    auto s83 = _mm_loadu_ps(srcX + 35 * 4);     \
    auto s90 = _mm_loadu_ps(srcX + 36 * 4);     \
    auto s91 = _mm_loadu_ps(srcX + 37 * 4);     \
    auto s92 = _mm_loadu_ps(srcX + 38 * 4);     \
    auto s93 = _mm_loadu_ps(srcX + 39 * 4);     \
    auto s100 = _mm_loadu_ps(srcX + 40 * 4);    \
    auto s101 = _mm_loadu_ps(srcX + 41 * 4);    \
    auto s102 = _mm_loadu_ps(srcX + 42 * 4);    \
    auto s103 = _mm_loadu_ps(srcX + 43 * 4);    \
    auto s110 = _mm_loadu_ps(srcX + 44 * 4);    \
    auto s111 = _mm_loadu_ps(srcX + 45 * 4);    \
    auto s112 = _mm_loadu_ps(srcX + 46 * 4);    \
    auto s113 = _mm_loadu_ps(srcX + 47 * 4);    \
    _MM_TRANSPOSE4_PS(s00, s01, s02, s03);      \
    _MM_TRANSPOSE4_PS(s10, s11, s12, s13);      \
    _MM_TRANSPOSE4_PS(s20, s21, s22, s23);      \
    _MM_TRANSPOSE4_PS(s30, s31, s32, s33);      \
    _MM_TRANSPOSE4_PS(s40, s41, s42, s43);      \
    _MM_TRANSPOSE4_PS(s50, s51, s52, s53);      \
    _MM_TRANSPOSE4_PS(s60, s61, s62, s63);      \
    _MM_TRANSPOSE4_PS(s70, s71, s72, s73);      \
    _MM_TRANSPOSE4_PS(s80, s81, s82, s83);      \
    _MM_TRANSPOSE4_PS(s90, s91, s92, s93);      \
    _MM_TRANSPOSE4_PS(s100, s101, s102, s103);  \
    _MM_TRANSPOSE4_PS(s110, s111, s112, s113);

#define STORE_TEMP(i)                                   \
    _mm_storeu_ps(dstX + 4 * (12 * i + 0), s##0##i);    \
    _mm_storeu_ps(dstX + 4 * (12 * i + 1), s##1##i);    \
    _mm_storeu_ps(dstX + 4 * (12 * i + 2), s##2##i);    \
    _mm_storeu_ps(dstX + 4 * (12 * i + 3), s##3##i);    \
    _mm_storeu_ps(dstX + 4 * (12 * i + 4), s##4##i);    \
    _mm_storeu_ps(dstX + 4 * (12 * i + 5), s##5##i);    \
    _mm_storeu_ps(dstX + 4 * (12 * i + 6), s##6##i);    \
    _mm_storeu_ps(dstX + 4 * (12 * i + 7), s##7##i);    \
    _mm_storeu_ps(dstX + 4 * (12 * i + 8), s##8##i);    \
    _mm_storeu_ps(dstX + 4 * (12 * i + 9), s##9##i);    \
    _mm_storeu_ps(dstX + 4 * (12 * i + 10), s##10##i);  \
    _mm_storeu_ps(dstX + 4 * (12 * i + 11), s##11##i);

    const int pack   = 48;   //eP=48  Hardcode here?
    const int packC4 = pack / 4;
    auto ePack       = e / pack;
    auto lC4         = l / 4;
    auto lDiv        = UP_DIV(l, 4);
    auto eRemain     = ePack * pack;
    auto lRemain     = lC4 * 4;
    auto lRes        = l - lRemain;
    for (int y = 0; y < ePack; ++y) {
        auto dstY = dest + y * l * pack;
        auto srcY = source + y * pack * 4;
        for (int x = 0; x < lC4; ++x) {
            auto srcX = srcY + x * 4 * eReal;
            auto dstX = dstY + x * pack * 4;
            MAIN_COMPUTE;

            STORE_TEMP(0);
            STORE_TEMP(1);
            STORE_TEMP(2);
            STORE_TEMP(3);
        }
    }
    auto lastLc4Src = source + lC4 * 4 * eReal;
    auto lastLc4Dst = dest + lC4 * pack * 4;
    if (lRes == 3) {
        for (int y = 0; y < ePack; ++y) {
            auto dstX = lastLc4Dst + y * l * pack;
            auto srcX = lastLc4Src + y * pack * 4;
            MAIN_COMPUTE;
            STORE_TEMP(0);
            STORE_TEMP(1);
            STORE_TEMP(2);
        }
    } else if (lRes == 2) {
        for (int y = 0; y < ePack; ++y) {
            auto dstX = lastLc4Dst + y * l * pack;
            auto srcX = lastLc4Src + y * pack * 4;
            MAIN_COMPUTE;
            STORE_TEMP(0);
            STORE_TEMP(1);
        }
    } else if (lRes == 1) {
        for (int y = 0; y < ePack; ++y) {
            auto dstX = lastLc4Dst + y * l * pack;
            auto srcX = lastLc4Src + y * pack * 4;
            MAIN_COMPUTE;
            STORE_TEMP(0);
        }
    }
    // Down
    {
        auto eLast    = e - eRemain;
        auto lastDest = dest + ePack * pack * l;
        for (int xC = 0; xC < lC4; ++xC) {
            for (int y = eRemain; y < e; ++y) {
                auto yR = y - eRemain;
                for (int xR = 0; xR < 4; ++xR) {
                    lastDest[(xC * 4 + xR) * eLast + yR] = source[xC * eReal * 4 + y * 4 + xR];
                }
            }
        }
        for (int x = lC4 * 4; x < l; ++x) {
            auto xR = x % 4;
            auto xC = lC4;
            for (int y = eRemain; y < e; ++y) {
                auto yR                  = y - eRemain;
                lastDest[x * eLast + yR] = source[xC * eReal * 4 + y * 4 + xR];
            }
        }
    }
}

void _AVX512_MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    {
        auto hP = h / 8;
        auto hR = hP * 8;
        if (hR != h) {
            ::memset(dest, 0, UP_DIV(h, 8) * 8 * l * sizeof(float));
        }
        if (!transpose) {
            for (int y=0; y<hP; ++y) {
                auto destY = dest + y * 8 * l;
                auto sourceY = source + y * 8;
                for (int x=0; x<l; ++x) {
                    ::memcpy(destY + 8 * x, sourceY + x * h, 8 * sizeof(float));
                }
            }
            auto hRemain = h - hR;
            if (hRemain > 0) {
                auto destY = dest + hP * 8 * l;
                auto sourceY = source + hP * 8;
                for (int x=0; x<l; ++x) {
                    ::memcpy(destY + 8 * x, sourceY + x * h, hRemain * sizeof(float));
                }
            }
            return;
        }
    }

    int lStride = h;
    int hStride = 1;
    const int hP = 8;
    if (transpose) {
        lStride = 1;
        hStride = l;
    }
    for (int y=0; y<h; ++y) {
        int yC = y / hP;
        int yR = y % hP;
        for (int x=0; x<l; ++x) {
            dest[yC * l * hP + yR + x * hP] = source[x*lStride + y*hStride];
        }
    }
}

void _AVX512_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, float* cache, const float* postParameters, const float* bias) {
    auto h       = parameter[2];
    auto hC4     = UP_DIV(h, 4);
    auto cStride = parameter[3] / sizeof(float);
//#ifdef MNN_X86_USE_ASM
//    _AVX512_MNNGemmFloatUnitMainFMA(C, A, B, parameter, hC4);
//#else
    _AVX512_MNNPackedMatMul_48(C, A, B, parameter);
//#endif
    AVX512GemmPostTreat(C, 48, parameter, postParameters, bias);
}

void _AVX512_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, float* cache, const float* postParameters, const float* bias) {
    _AVX512_MNNPackednMatMulRemainCommon(C, A, B, eSize, parameter, cache, postParameters, bias);
    AVX512GemmPostTreat(C, eSize, parameter, postParameters, bias);
}
