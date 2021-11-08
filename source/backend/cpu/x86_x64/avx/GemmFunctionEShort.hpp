//
//  GemmFunctionEShort.hpp
//  MNN
//
//  Created by MNN on 2020/09/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define MNN_UNIT_E 6
#define TRANPOSE_SAVE(u, v, z0, z3, z6, z9)              \
    {                                                    \
        auto m0 = _mm256_extractf128_ps(z0, u);          \
        auto m1 = _mm256_extractf128_ps(z3, u);          \
        auto m2 = _mm256_extractf128_ps(z6, u);          \
        auto m3 = _mm256_extractf128_ps(z9, u);          \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);               \
        STORE_4(dst + 4 * (0 + 4 * u + 8 * v), m0); \
        STORE_4(dst + 4 * (1 + 4 * u + 8 * v), m1); \
        STORE_4(dst + 4 * (2 + 4 * u + 8 * v), m2); \
        STORE_4(dst + 4 * (3 + 4 * u + 8 * v), m3); \
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
#define INIT_MAIN                                  \
auto w0 = LOAD8(TB + 0 * 8); \
auto w1 = LOAD8(TB + 1 * 8); \
auto s0  = BROAD_LOAD(TA + 0);             \
auto s1  = BROAD_LOAD(TA + 1);             \
auto z00 = _mm256_mul_ps(s0, w0);                   \
auto z10 = _mm256_mul_ps(s1, w0);                   \
auto z01 = _mm256_mul_ps(s0, w1);                   \
auto z11 = _mm256_mul_ps(s1, w1);                   \
s0  = BROAD_LOAD(TA + 2);             \
s1  = BROAD_LOAD(TA + 3);             \
auto z20 = _mm256_mul_ps(s0, w0);                   \
auto z30 = _mm256_mul_ps(s1, w0);                   \
auto z21 = _mm256_mul_ps(s0, w1);                   \
auto z31 = _mm256_mul_ps(s1, w1);                   \
s0  = BROAD_LOAD(TA + 4);             \
s1  = BROAD_LOAD(TA + 5);             \
auto z40 = _mm256_mul_ps(s0, w0);                   \
auto z50 = _mm256_mul_ps(s1, w0);                   \
auto z41 = _mm256_mul_ps(s0, w1);                   \
auto z51 = _mm256_mul_ps(s1, w1);                   \

#define COMPUTE_MAIN                                \
w0 = LOAD8(TB + 0 * 8); \
w1 = LOAD8(TB + 1 * 8); \
s0  = BROAD_LOAD(TA + 0);             \
s1  = BROAD_LOAD(TA + 1);             \
z00 = MNNAVXFMA(s0, w0, z00);                   \
z10 = MNNAVXFMA(s1, w0, z10);                   \
z01 = MNNAVXFMA(s0, w1, z01);                   \
z11 = MNNAVXFMA(s1, w1, z11);                   \
s0  = BROAD_LOAD(TA + 2);             \
s1  = BROAD_LOAD(TA + 3);             \
z20 = MNNAVXFMA(s0, w0, z20);                   \
z30 = MNNAVXFMA(s1, w0, z30);                   \
z21 = MNNAVXFMA(s0, w1, z21);                   \
z31 = MNNAVXFMA(s1, w1, z31);                   \
s0  = BROAD_LOAD(TA + 4);             \
s1  = BROAD_LOAD(TA + 5);             \
z40 = MNNAVXFMA(s0, w0, z40);                   \
z50 = MNNAVXFMA(s1, w0, z50);                   \
z41 = MNNAVXFMA(s0, w1, z41);                   \
z51 = MNNAVXFMA(s1, w1, z51);                   \

#define INIT_MAIN_S                                  \
auto w0 = LOAD8(TB + 0 * 8); \
auto s0  = BROAD_LOAD(TA + 0);             \
auto s1  = BROAD_LOAD(TA + 1);             \
auto z00 = _mm256_mul_ps(s0, w0);                   \
auto z10 = _mm256_mul_ps(s1, w0);                   \
s0  = BROAD_LOAD(TA + 2);             \
s1  = BROAD_LOAD(TA + 3);             \
auto z20 = _mm256_mul_ps(s0, w0);                   \
auto z30 = _mm256_mul_ps(s1, w0);                   \
s0  = BROAD_LOAD(TA + 4);             \
s1  = BROAD_LOAD(TA + 5);             \
auto z40 = _mm256_mul_ps(s0, w0);                   \
auto z50 = _mm256_mul_ps(s1, w0);                   \

#define COMPUTE_MAIN_S                                \
w0 = LOAD8(TB + 0 * 8); \
s0  = BROAD_LOAD(TA + 0);             \
s1  = BROAD_LOAD(TA + 1);             \
z00 = MNNAVXFMA(s0, w0, z00);                   \
z10 = MNNAVXFMA(s1, w0, z10);                   \
s0  = BROAD_LOAD(TA + 2);             \
s1  = BROAD_LOAD(TA + 3);             \
z20 = MNNAVXFMA(s0, w0, z20);                   \
z30 = MNNAVXFMA(s1, w0, z30);                   \
s0  = BROAD_LOAD(TA + 4);             \
s1  = BROAD_LOAD(TA + 5);             \
z40 = MNNAVXFMA(s0, w0, z40);                   \
z50 = MNNAVXFMA(s1, w0, z50);                   \

#define STORE_MAIN \
_mm256_storeu_ps(dst + 0 * 8 + 0 * cStride, z00);\
_mm256_storeu_ps(dst + 1 * 8 + 0 * cStride, z10);\
_mm256_storeu_ps(dst + 2 * 8 + 0 * cStride, z20);\
_mm256_storeu_ps(dst + 3 * 8 + 0 * cStride, z30);\
_mm256_storeu_ps(dst + 4 * 8 + 0 * cStride, z40);\
_mm256_storeu_ps(dst + 5 * 8 + 0 * cStride, z50);\
_mm256_storeu_ps(dst + 0 * 8 + 1 * cStride, z01);\
_mm256_storeu_ps(dst + 1 * 8 + 1 * cStride, z11);\
_mm256_storeu_ps(dst + 2 * 8 + 1 * cStride, z21);\
_mm256_storeu_ps(dst + 3 * 8 + 1 * cStride, z31);\
_mm256_storeu_ps(dst + 4 * 8 + 1 * cStride, z41);\
_mm256_storeu_ps(dst + 5 * 8 + 1 * cStride, z51);\

#define STORE_MAIN_S \
_mm256_storeu_ps(dst + 0 * 8 + 0 * cStride, z00);\
_mm256_storeu_ps(dst + 1 * 8 + 0 * cStride, z10);\
_mm256_storeu_ps(dst + 2 * 8 + 0 * cStride, z20);\
_mm256_storeu_ps(dst + 3 * 8 + 0 * cStride, z30);\
_mm256_storeu_ps(dst + 4 * 8 + 0 * cStride, z40);\
_mm256_storeu_ps(dst + 5 * 8 + 0 * cStride, z50);\

#define MAIN_C \
auto cStride      = parameter[3] / sizeof(TYPE);\
auto bExtraStride = parameter[5] / sizeof(TYPE);\
auto bStride      = bExtraStride + l * 16;\
auto hC4          = UP_DIV(h, 8);\
auto hC8 = hC4 / 2;\
auto hR = hC4 % 2;\
auto lC2 = (l - 1) / 2;\
auto lR = (l-1) % 2;\
    for (int y = 0; y < hC8; ++y) {\
        auto dst    = C;\
        auto TA = A;\
        auto TB = B;\
        INIT_MAIN;\
        TA += aStride;\
        TB += 16;\
        for (int sy = 0; sy < lC2; ++sy) {\
            COMPUTE_MAIN;\
            TA += aStride;\
            TB += 16;\
            COMPUTE_MAIN;\
            TA += aStride;\
            TB += 16;\
        }\
        if (lR > 0) {\
        COMPUTE_MAIN;\
        TA += aStride;\
        TB += 16;\
        }\
        STORE_MAIN;\
        B+= bStride;\
        C+= cStride * 2;\
    }\
    if (hR > 0) {\
        auto dst    = C;\
        auto TA = A;\
        auto TB = B;\
        INIT_MAIN_S;\
        TA += aStride;\
        TB += 16;\
        for (int sy = 1; sy < l; ++sy) {\
            COMPUTE_MAIN_S;\
            TA += aStride;\
            TB += 16;\
        }\
        STORE_MAIN_S;\
    }\


template <typename TYPE>
static void _AVX_MNNPackedMatMul_Main(TYPE* C, const TYPE* A, const TYPE* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    int aStride = 6;
    MAIN_C;
}

/**........  5*/

#undef INIT_MAIN
#undef INIT_MAIN_S
#undef COMPUTE_MAIN
#undef COMPUTE_MAIN_S
#undef STORE_MAIN
#undef STORE_MAIN_S

#define INIT_MAIN                                  \
auto w0 = LOAD8(TB + 0 * 8); \
auto w1 = LOAD8(TB + 1 * 8); \
auto s0  = BROAD_LOAD(TA + 0);             \
auto s1  = BROAD_LOAD(TA + 1);             \
auto z00 = _mm256_mul_ps(s0, w0);                   \
auto z10 = _mm256_mul_ps(s1, w0);                   \
auto z01 = _mm256_mul_ps(s0, w1);                   \
auto z11 = _mm256_mul_ps(s1, w1);                   \
s0  = BROAD_LOAD(TA + 2);             \
s1  = BROAD_LOAD(TA + 3);             \
auto z20 = _mm256_mul_ps(s0, w0);                   \
auto z30 = _mm256_mul_ps(s1, w0);                   \
auto z21 = _mm256_mul_ps(s0, w1);                   \
auto z31 = _mm256_mul_ps(s1, w1);                   \
s0  = BROAD_LOAD(TA + 4);             \
auto z40 = _mm256_mul_ps(s0, w0);                   \
auto z41 = _mm256_mul_ps(s0, w1);                   \

#define COMPUTE_MAIN                                \
w0 = LOAD8(TB + 0 * 8); \
w1 = LOAD8(TB + 1 * 8); \
s0  = BROAD_LOAD(TA + 0);             \
s1  = BROAD_LOAD(TA + 1);             \
z00 = MNNAVXFMA(s0, w0, z00);                   \
z10 = MNNAVXFMA(s1, w0, z10);                   \
z01 = MNNAVXFMA(s0, w1, z01);                   \
z11 = MNNAVXFMA(s1, w1, z11);                   \
s0  = BROAD_LOAD(TA + 2);             \
s1  = BROAD_LOAD(TA + 3);             \
z20 = MNNAVXFMA(s0, w0, z20);                   \
z30 = MNNAVXFMA(s1, w0, z30);                   \
z21 = MNNAVXFMA(s0, w1, z21);                   \
z31 = MNNAVXFMA(s1, w1, z31);                   \
s0  = BROAD_LOAD(TA + 4);             \
z40 = MNNAVXFMA(s0, w0, z40);                   \
z41 = MNNAVXFMA(s0, w1, z41);                   \

#define INIT_MAIN_S                                  \
auto w0 = LOAD8(TB + 0 * 8); \
auto s0  = BROAD_LOAD(TA + 0);             \
auto s1  = BROAD_LOAD(TA + 1);             \
auto z00 = _mm256_mul_ps(s0, w0);                   \
auto z10 = _mm256_mul_ps(s1, w0);                   \
s0  = BROAD_LOAD(TA + 2);             \
s1  = BROAD_LOAD(TA + 3);             \
auto z20 = _mm256_mul_ps(s0, w0);                   \
auto z30 = _mm256_mul_ps(s1, w0);                   \
s0  = BROAD_LOAD(TA + 4);             \
auto z40 = _mm256_mul_ps(s0, w0);                   \

#define COMPUTE_MAIN_S                                \
w0 = LOAD8(TB + 0 * 8); \
s0  = BROAD_LOAD(TA + 0);             \
s1  = BROAD_LOAD(TA + 1);             \
z00 = MNNAVXFMA(s0, w0, z00);                   \
z10 = MNNAVXFMA(s1, w0, z10);                   \
s0  = BROAD_LOAD(TA + 2);             \
s1  = BROAD_LOAD(TA + 3);             \
z20 = MNNAVXFMA(s0, w0, z20);                   \
z30 = MNNAVXFMA(s1, w0, z30);                   \
s0  = BROAD_LOAD(TA + 4);             \
z40 = MNNAVXFMA(s0, w0, z40);                   \

#define STORE_MAIN \
_mm256_storeu_ps(dst + 0 * 8 + 0 * cStride, z00);\
_mm256_storeu_ps(dst + 1 * 8 + 0 * cStride, z10);\
_mm256_storeu_ps(dst + 2 * 8 + 0 * cStride, z20);\
_mm256_storeu_ps(dst + 3 * 8 + 0 * cStride, z30);\
_mm256_storeu_ps(dst + 4 * 8 + 0 * cStride, z40);\
_mm256_storeu_ps(dst + 0 * 8 + 1 * cStride, z01);\
_mm256_storeu_ps(dst + 1 * 8 + 1 * cStride, z11);\
_mm256_storeu_ps(dst + 2 * 8 + 1 * cStride, z21);\
_mm256_storeu_ps(dst + 3 * 8 + 1 * cStride, z31);\
_mm256_storeu_ps(dst + 4 * 8 + 1 * cStride, z41);\

#define STORE_MAIN_S \
_mm256_storeu_ps(dst + 0 * 8 + 0 * cStride, z00);\
_mm256_storeu_ps(dst + 1 * 8 + 0 * cStride, z10);\
_mm256_storeu_ps(dst + 2 * 8 + 0 * cStride, z20);\
_mm256_storeu_ps(dst + 3 * 8 + 0 * cStride, z30);\
_mm256_storeu_ps(dst + 4 * 8 + 0 * cStride, z40);\



template <typename TYPE>
static void _AVX_MNNPackedMatMul_5(TYPE* C, const TYPE* A, const TYPE* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto aStride      = parameter[0] / sizeof(TYPE);
    MAIN_C;
}

/**........  4*/
#undef INIT_MAIN
#undef INIT_MAIN_S
#undef COMPUTE_MAIN
#undef COMPUTE_MAIN_S
#undef STORE_MAIN
#undef STORE_MAIN_S

#define INIT_MAIN                                  \
auto w0 = LOAD8(TB + 0 * 8); \
auto w1 = LOAD8(TB + 1 * 8); \
auto s0  = BROAD_LOAD(TA + 0);             \
auto s1  = BROAD_LOAD(TA + 1);             \
auto z00 = _mm256_mul_ps(s0, w0);                   \
auto z10 = _mm256_mul_ps(s1, w0);                   \
auto z01 = _mm256_mul_ps(s0, w1);                   \
auto z11 = _mm256_mul_ps(s1, w1);                   \
s0  = BROAD_LOAD(TA + 2);             \
s1  = BROAD_LOAD(TA + 3);             \
auto z20 = _mm256_mul_ps(s0, w0);                   \
auto z30 = _mm256_mul_ps(s1, w0);                   \
auto z21 = _mm256_mul_ps(s0, w1);                   \
auto z31 = _mm256_mul_ps(s1, w1);                   \

#define COMPUTE_MAIN                                \
w0 = LOAD8(TB + 0 * 8); \
w1 = LOAD8(TB + 1 * 8); \
s0  = BROAD_LOAD(TA + 0);             \
s1  = BROAD_LOAD(TA + 1);             \
z00 = MNNAVXFMA(s0, w0, z00);                   \
z10 = MNNAVXFMA(s1, w0, z10);                   \
z01 = MNNAVXFMA(s0, w1, z01);                   \
z11 = MNNAVXFMA(s1, w1, z11);                   \
s0  = BROAD_LOAD(TA + 2);             \
s1  = BROAD_LOAD(TA + 3);             \
z20 = MNNAVXFMA(s0, w0, z20);                   \
z30 = MNNAVXFMA(s1, w0, z30);                   \
z21 = MNNAVXFMA(s0, w1, z21);                   \
z31 = MNNAVXFMA(s1, w1, z31);                   \

#define INIT_MAIN_S                                  \
auto w0 = LOAD8(TB + 0 * 8); \
auto s0  = BROAD_LOAD(TA + 0);             \
auto s1  = BROAD_LOAD(TA + 1);             \
auto z00 = _mm256_mul_ps(s0, w0);                   \
auto z10 = _mm256_mul_ps(s1, w0);                   \
s0  = BROAD_LOAD(TA + 2);             \
s1  = BROAD_LOAD(TA + 3);             \
auto z20 = _mm256_mul_ps(s0, w0);                   \
auto z30 = _mm256_mul_ps(s1, w0);                   \

#define COMPUTE_MAIN_S                                \
w0 = LOAD8(TB + 0 * 8); \
s0  = BROAD_LOAD(TA + 0);             \
s1  = BROAD_LOAD(TA + 1);             \
z00 = MNNAVXFMA(s0, w0, z00);                   \
z10 = MNNAVXFMA(s1, w0, z10);                   \
s0  = BROAD_LOAD(TA + 2);             \
s1  = BROAD_LOAD(TA + 3);             \
z20 = MNNAVXFMA(s0, w0, z20);                   \
z30 = MNNAVXFMA(s1, w0, z30);                   \

#define STORE_MAIN \
_mm256_storeu_ps(dst + 0 * 8 + 0 * cStride, z00);\
_mm256_storeu_ps(dst + 1 * 8 + 0 * cStride, z10);\
_mm256_storeu_ps(dst + 2 * 8 + 0 * cStride, z20);\
_mm256_storeu_ps(dst + 3 * 8 + 0 * cStride, z30);\
_mm256_storeu_ps(dst + 0 * 8 + 1 * cStride, z01);\
_mm256_storeu_ps(dst + 1 * 8 + 1 * cStride, z11);\
_mm256_storeu_ps(dst + 2 * 8 + 1 * cStride, z21);\
_mm256_storeu_ps(dst + 3 * 8 + 1 * cStride, z31);\

#define STORE_MAIN_S \
_mm256_storeu_ps(dst + 0 * 8 + 0 * cStride, z00);\
_mm256_storeu_ps(dst + 1 * 8 + 0 * cStride, z10);\
_mm256_storeu_ps(dst + 2 * 8 + 0 * cStride, z20);\
_mm256_storeu_ps(dst + 3 * 8 + 0 * cStride, z30);\



template <typename TYPE>
static void _AVX_MNNPackedMatMul_4(TYPE* C, const TYPE* A, const TYPE* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto aStride      = parameter[0] / sizeof(TYPE);
    MAIN_C;
}


/**........  3*/
#undef INIT_MAIN
#undef INIT_MAIN_S
#undef COMPUTE_MAIN
#undef COMPUTE_MAIN_S
#undef STORE_MAIN
#undef STORE_MAIN_S

#define INIT_MAIN                                  \
auto w0 = LOAD8(TB + 0 * 8); \
auto w1 = LOAD8(TB + 1 * 8); \
auto s0  = BROAD_LOAD(TA + 0);             \
auto s1  = BROAD_LOAD(TA + 1);             \
auto z00 = _mm256_mul_ps(s0, w0);                   \
auto z10 = _mm256_mul_ps(s1, w0);                   \
auto z01 = _mm256_mul_ps(s0, w1);                   \
auto z11 = _mm256_mul_ps(s1, w1);                   \
s0  = BROAD_LOAD(TA + 2);             \
auto z20 = _mm256_mul_ps(s0, w0);                   \
auto z21 = _mm256_mul_ps(s0, w1);                   \

#define COMPUTE_MAIN                                \
w0 = LOAD8(TB + 0 * 8); \
w1 = LOAD8(TB + 1 * 8); \
s0  = BROAD_LOAD(TA + 0);             \
s1  = BROAD_LOAD(TA + 1);             \
z00 = MNNAVXFMA(s0, w0, z00);                   \
z10 = MNNAVXFMA(s1, w0, z10);                   \
z01 = MNNAVXFMA(s0, w1, z01);                   \
z11 = MNNAVXFMA(s1, w1, z11);                   \
s0  = BROAD_LOAD(TA + 2);             \
z20 = MNNAVXFMA(s0, w0, z20);                   \
z21 = MNNAVXFMA(s0, w1, z21);                   \

#define INIT_MAIN_S                                  \
auto w0 = LOAD8(TB + 0 * 8); \
auto s0  = BROAD_LOAD(TA + 0);             \
auto s1  = BROAD_LOAD(TA + 1);             \
auto z00 = _mm256_mul_ps(s0, w0);                   \
auto z10 = _mm256_mul_ps(s1, w0);                   \
s0  = BROAD_LOAD(TA + 2);             \
auto z20 = _mm256_mul_ps(s0, w0);                   \

#define COMPUTE_MAIN_S                                \
w0 = LOAD8(TB + 0 * 8); \
s0  = BROAD_LOAD(TA + 0);             \
s1  = BROAD_LOAD(TA + 1);             \
z00 = MNNAVXFMA(s0, w0, z00);                   \
z10 = MNNAVXFMA(s1, w0, z10);                   \
s0  = BROAD_LOAD(TA + 2);             \
z20 = MNNAVXFMA(s0, w0, z20);                   \

#define STORE_MAIN \
_mm256_storeu_ps(dst + 0 * 8 + 0 * cStride, z00);\
_mm256_storeu_ps(dst + 1 * 8 + 0 * cStride, z10);\
_mm256_storeu_ps(dst + 2 * 8 + 0 * cStride, z20);\
_mm256_storeu_ps(dst + 0 * 8 + 1 * cStride, z01);\
_mm256_storeu_ps(dst + 1 * 8 + 1 * cStride, z11);\
_mm256_storeu_ps(dst + 2 * 8 + 1 * cStride, z21);\

#define STORE_MAIN_S \
_mm256_storeu_ps(dst + 0 * 8 + 0 * cStride, z00);\
_mm256_storeu_ps(dst + 1 * 8 + 0 * cStride, z10);\
_mm256_storeu_ps(dst + 2 * 8 + 0 * cStride, z20);\

template <typename TYPE>
static void _AVX_MNNPackedMatMul_3(TYPE* C, const TYPE* A, const TYPE* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto aStride      = parameter[0] / sizeof(TYPE);
    MAIN_C;
}

/**.... 2*/
#undef INIT_MAIN
#undef INIT_MAIN_S
#undef COMPUTE_MAIN
#undef COMPUTE_MAIN_S
#undef STORE_MAIN
#undef STORE_MAIN_S

#define INIT_MAIN                                  \
auto w0 = LOAD8(TB + 0 * 8); \
auto w1 = LOAD8(TB + 1 * 8); \
auto s0  = BROAD_LOAD(TA + 0);             \
auto s1  = BROAD_LOAD(TA + 1);             \
auto z00 = _mm256_mul_ps(s0, w0);                   \
auto z10 = _mm256_mul_ps(s1, w0);                   \
auto z01 = _mm256_mul_ps(s0, w1);                   \
auto z11 = _mm256_mul_ps(s1, w1);                   \

#define COMPUTE_MAIN                                \
w0 = LOAD8(TB + 0 * 8); \
w1 = LOAD8(TB + 1 * 8); \
s0  = BROAD_LOAD(TA + 0);             \
s1  = BROAD_LOAD(TA + 1);             \
z00 = MNNAVXFMA(s0, w0, z00);                   \
z10 = MNNAVXFMA(s1, w0, z10);                   \
z01 = MNNAVXFMA(s0, w1, z01);                   \
z11 = MNNAVXFMA(s1, w1, z11);                   \

#define INIT_MAIN_S                                  \
auto w0 = LOAD8(TB + 0 * 8); \
auto s0  = BROAD_LOAD(TA + 0);             \
auto s1  = BROAD_LOAD(TA + 1);             \
auto z00 = _mm256_mul_ps(s0, w0);                   \
auto z10 = _mm256_mul_ps(s1, w0);                   \

#define COMPUTE_MAIN_S                                \
w0 = LOAD8(TB + 0 * 8); \
s0  = BROAD_LOAD(TA + 0);             \
s1  = BROAD_LOAD(TA + 1);             \
z00 = MNNAVXFMA(s0, w0, z00);                   \
z10 = MNNAVXFMA(s1, w0, z10);                   \

#define STORE_MAIN \
_mm256_storeu_ps(dst + 0 * 8 + 0 * cStride, z00);\
_mm256_storeu_ps(dst + 1 * 8 + 0 * cStride, z10);\
_mm256_storeu_ps(dst + 0 * 8 + 1 * cStride, z01);\
_mm256_storeu_ps(dst + 1 * 8 + 1 * cStride, z11);\

#define STORE_MAIN_S \
_mm256_storeu_ps(dst + 0 * 8 + 0 * cStride, z00);\
_mm256_storeu_ps(dst + 1 * 8 + 0 * cStride, z10);\

template <typename TYPE>
static void _AVX_MNNPackedMatMul_2(TYPE* C, const TYPE* A, const TYPE* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto aStride      = parameter[0] / sizeof(TYPE);
    MAIN_C;
}

/**.... 1*/
#undef INIT_MAIN
#undef INIT_MAIN_S
#undef COMPUTE_MAIN
#undef COMPUTE_MAIN_S
#undef STORE_MAIN
#undef STORE_MAIN_S

#define INIT_MAIN                                  \
auto w0 = LOAD8(TB + 0 * 8); \
auto w1 = LOAD8(TB + 1 * 8); \
auto s0  = BROAD_LOAD(TA + 0);             \
auto z00 = _mm256_mul_ps(s0, w0);                   \
auto z01 = _mm256_mul_ps(s0, w1);                   \

#define COMPUTE_MAIN                                \
w0 = LOAD8(TB + 0 * 8); \
w1 = LOAD8(TB + 1 * 8); \
s0  = BROAD_LOAD(TA + 0);             \
z00 = MNNAVXFMA(s0, w0, z00);                   \
z01 = MNNAVXFMA(s0, w1, z01);                   \

#define INIT_MAIN_S                                  \
auto w0 = LOAD8(TB + 0 * 8); \
auto s0  = BROAD_LOAD(TA + 0);             \
auto z00 = _mm256_mul_ps(s0, w0);                   \

#define COMPUTE_MAIN_S                                \
w0 = LOAD8(TB + 0 * 8); \
s0  = BROAD_LOAD(TA + 0);             \
z00 = MNNAVXFMA(s0, w0, z00);                   \

#define STORE_MAIN \
_mm256_storeu_ps(dst + 0 * 8 + 0 * cStride, z00);\
_mm256_storeu_ps(dst + 0 * 8 + 1 * cStride, z01);\

#define STORE_MAIN_S \
_mm256_storeu_ps(dst + 0 * 8 + 0 * cStride, z00);\

template <typename TYPE>
static void _AVX_MNNPackedMatMul_1(TYPE* C, const TYPE* A, const TYPE* B, const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto aStride      = parameter[0] / sizeof(TYPE);
    MAIN_C;
}

template <typename TYPE>
static void _AVX_MNNPackednMatMulRemainCommon(TYPE* C, const TYPE* A, const TYPE* B, size_t eSize,
                                              const size_t* parameter) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(TYPE);
    auto bExtraStride = parameter[5] / sizeof(TYPE);
    auto bStride      = bExtraStride + l * 16;
    auto hC4          = UP_DIV(h, 8);
    auto es           = eSize;
    auto oC           = C;
    switch (eSize) {
        case 5:
            _AVX_MNNPackedMatMul_5(C, A, B, parameter);
            break;
        case 4:
            _AVX_MNNPackedMatMul_4(C, A, B, parameter);
            break;
        case 3:
            _AVX_MNNPackedMatMul_3(C, A, B, parameter);
            break;
        case 2:
            _AVX_MNNPackedMatMul_2(C, A, B, parameter);
            break;
        case 1:
            _AVX_MNNPackedMatMul_1(C, A, B, parameter);
            break;
        default:
            break;
    }
}
