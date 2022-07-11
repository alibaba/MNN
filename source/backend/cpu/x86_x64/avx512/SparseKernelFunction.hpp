//
//  SparseKernelFunction.hpp
//  MNN
//
//  Created by MNN on 2021/08/18.
//  Copyright © 2018 - 2021, Alibaba Group Holding Limited
//

#ifndef _SPARSE_KERNEL_FUNCTION_H_
#define _SPARSE_KERNEL_FUNCTION_H_
#include "FunctionSummary.hpp"
#include "core/Macro.h"

constexpr int AVX512F32 = 16;

#define TRANSPOSE4x4_STORE(dest, ablock, aSegment, packCUnit, vacc0, vacc3, vacc6, vacc9)                       \
    {                                                                                                           \
        auto m128_0 = _mm512_extractf32x4_ps(vacc0, aSegment);                                                  \
        auto m128_1 = _mm512_extractf32x4_ps(vacc3, aSegment);                                                  \
        auto m128_2 = _mm512_extractf32x4_ps(vacc6, aSegment);                                                  \
        auto m128_3 = _mm512_extractf32x4_ps(vacc9, aSegment);                                                  \
        _MM_TRANSPOSE4_PS(m128_0, m128_1, m128_2, m128_3);                                                      \
        _mm_storeu_ps(dest + AVX512F32 * packCUnit * ablock + 4 * packCUnit * aSegment, m128_0);                 \
        _mm_storeu_ps(dest + AVX512F32 * packCUnit * ablock + 4 * packCUnit * aSegment + packCUnit, m128_1);     \
        _mm_storeu_ps(dest + AVX512F32 * packCUnit * ablock + 4 * packCUnit * aSegment + packCUnit * 2, m128_2); \
        _mm_storeu_ps(dest + AVX512F32 * packCUnit * ablock + 4 * packCUnit * aSegment + packCUnit * 3, m128_3); \
    }

inline void STORE_VECTOR_AS_COLUMN(float* dest, size_t ablock, size_t packCUnit, __m512 vacc) {
    union {
        __m512 v;
        float f[16];
    } vacc_u;
    vacc_u.v = vacc;
    dest[AVX512F32 * packCUnit * ablock + 0]              = vacc_u.f[0];
    dest[AVX512F32 * packCUnit * ablock + packCUnit]      = vacc_u.f[1];
    dest[AVX512F32 * packCUnit * ablock + packCUnit * 2]  = vacc_u.f[2];
    dest[AVX512F32 * packCUnit * ablock + packCUnit * 3]  = vacc_u.f[3];
    dest[AVX512F32 * packCUnit * ablock + packCUnit * 4]  = vacc_u.f[4];
    dest[AVX512F32 * packCUnit * ablock + packCUnit * 5]  = vacc_u.f[5];
    dest[AVX512F32 * packCUnit * ablock + packCUnit * 6]  = vacc_u.f[6];
    dest[AVX512F32 * packCUnit * ablock + packCUnit * 7]  = vacc_u.f[7];
    dest[AVX512F32 * packCUnit * ablock + packCUnit * 8]  = vacc_u.f[8];
    dest[AVX512F32 * packCUnit * ablock + packCUnit * 9]  = vacc_u.f[9];
    dest[AVX512F32 * packCUnit * ablock + packCUnit * 10] = vacc_u.f[10];
    dest[AVX512F32 * packCUnit * ablock + packCUnit * 11] = vacc_u.f[11];
    dest[AVX512F32 * packCUnit * ablock + packCUnit * 12] = vacc_u.f[12];
    dest[AVX512F32 * packCUnit * ablock + packCUnit * 13] = vacc_u.f[13];
    dest[AVX512F32 * packCUnit * ablock + packCUnit * 14] = vacc_u.f[14];
    dest[AVX512F32 * packCUnit * ablock + packCUnit * 15] = vacc_u.f[15];
}

#define TRANSPOSE4x8_STORE(dest, ablock, aSegment, packCUnit, v0, v3, v6, v9, v12, v15, v18, v21) {        \
    auto m0 = _mm512_extractf32x4_ps(v0, aSegment);                                                         \
    auto m1 = _mm512_extractf32x4_ps(v3, aSegment);                                                         \
    auto m2 = _mm512_extractf32x4_ps(v6, aSegment);                                                         \
    auto m3 = _mm512_extractf32x4_ps(v9, aSegment);                                                         \
    auto m4 = _mm512_extractf32x4_ps(v12, aSegment);                                                        \
    auto m5 = _mm512_extractf32x4_ps(v15, aSegment);                                                        \
    auto m6 = _mm512_extractf32x4_ps(v18, aSegment);                                                        \
    auto m7 = _mm512_extractf32x4_ps(v21, aSegment);                                                        \
    _MM_TRANSPOSE4_PS(m0, m1, m2, m3);                                                                      \
    _MM_TRANSPOSE4_PS(m4, m5, m6, m7);                                                                      \
    auto tmp0 = _mm256_castps128_ps256(m0);                                                                 \
    auto tmp4 = _mm256_castps128_ps256(m4);                                                                 \
    auto s0   = _mm256_permute2f128_ps(tmp0, tmp4, 0x20);                                                   \
    auto tmp1 = _mm256_castps128_ps256(m1);                                                                 \
    auto tmp5 = _mm256_castps128_ps256(m5);                                                                 \
    auto s1   = _mm256_permute2f128_ps(tmp1, tmp5, 0x20);                                                   \
    auto tmp2 = _mm256_castps128_ps256(m2);                                                                 \
    auto tmp6 = _mm256_castps128_ps256(m6);                                                                 \
    auto s2   = _mm256_permute2f128_ps(tmp2, tmp6, 0x20);                                                   \
    auto tmp3 = _mm256_castps128_ps256(m3);                                                                 \
    auto tmp7 = _mm256_castps128_ps256(m7);                                                                 \
    auto s3   = _mm256_permute2f128_ps(tmp3, tmp7, 0x20);                                                   \
    _mm256_storeu_ps(dest + AVX512F32 * packCUnit * ablock + 4 * packCUnit * aSegment + packCUnit * 0, s0); \
    _mm256_storeu_ps(dest + AVX512F32 * packCUnit * ablock + 4 * packCUnit * aSegment + packCUnit * 1, s1); \
    _mm256_storeu_ps(dest + AVX512F32 * packCUnit * ablock + 4 * packCUnit * aSegment + packCUnit * 2, s2); \
    _mm256_storeu_ps(dest + AVX512F32 * packCUnit * ablock + 4 * packCUnit * aSegment + packCUnit * 3, s3); \
}


#define TRANSPOSE_M256_4x4_STORE(dest, aSegment, packCUnit, vacc0, vacc3, vacc6, vacc9) \
    {                                                                                   \
        auto m128_0 = _mm256_extractf128_ps(vacc0, aSegment);                           \
        auto m128_1 = _mm256_extractf128_ps(vacc3, aSegment);                           \
        auto m128_2 = _mm256_extractf128_ps(vacc6, aSegment);                           \
        auto m128_3 = _mm256_extractf128_ps(vacc9, aSegment);                           \
        _MM_TRANSPOSE4_PS(m128_0, m128_1, m128_2, m128_3);                              \
        _mm_storeu_ps(dest + 4 * packCUnit * aSegment, m128_0);                          \
        _mm_storeu_ps(dest + 4 * packCUnit * aSegment + packCUnit, m128_1);              \
        _mm_storeu_ps(dest + 4 * packCUnit * aSegment + packCUnit * 2, m128_2);          \
        _mm_storeu_ps(dest + 4 * packCUnit * aSegment + packCUnit * 3, m128_3);          \
    }

#define TRANSPOSE_M256_8x8_STORE(dest, packCUnit, v0, v3, v6, v9, v12, v15, v18, v21)           \
    {                                                                                           \
        auto t0 = _mm256_unpacklo_ps(v0, v3);                                                   \
        auto t1 = _mm256_unpackhi_ps(v0, v3);                                                   \
        auto t2 = _mm256_unpacklo_ps(v6, v9);                                                   \
        auto t3 = _mm256_unpackhi_ps(v6, v9);                                                   \
        auto t4 = _mm256_unpacklo_ps(v12, v15);                                                 \
        auto t5 = _mm256_unpackhi_ps(v12, v15);                                                 \
        auto t6 = _mm256_unpacklo_ps(v18, v21);                                                 \
        auto t7 = _mm256_unpackhi_ps(v18, v21);                                                 \
                                                                                                \
        v0  = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));                               \
        v3  = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));                               \
        v6  = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));                               \
        v9  = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));                               \
        v12 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));                               \
        v15 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));                               \
        v18 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));                               \
        v21 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));                               \
                                                                                                \
        t0 = _mm256_permute2f128_ps(v0, v12, 0x20);                                             \
        t1 = _mm256_permute2f128_ps(v3, v15, 0x20);                                             \
        t2 = _mm256_permute2f128_ps(v6, v18, 0x20);                                             \
        t3 = _mm256_permute2f128_ps(v9, v21, 0x20);                                             \
        t4 = _mm256_permute2f128_ps(v0, v12, 0x31);                                             \
        t5 = _mm256_permute2f128_ps(v3, v15, 0x31);                                             \
        t6 = _mm256_permute2f128_ps(v6, v18, 0x31);                                             \
        t7 = _mm256_permute2f128_ps(v9, v21, 0x31);                                             \
        _mm256_storeu_ps(dest, t0);                                                             \
        _mm256_storeu_ps(dest + packCUnit, t1);                                                 \
        _mm256_storeu_ps(dest + packCUnit * 2, t2);                                             \
        _mm256_storeu_ps(dest + packCUnit * 3, t3);                                             \
        _mm256_storeu_ps(dest + packCUnit * 4, t4);                                             \
        _mm256_storeu_ps(dest + packCUnit * 5, t5);                                             \
        _mm256_storeu_ps(dest + packCUnit * 6, t6);                                             \
        _mm256_storeu_ps(dest + packCUnit * 7, t7);                                             \
    }

inline void STORE_M256_VECTOR_AS_COLUMN(float* dest, size_t packCUnit, __m256 vacc) {
    union {
        __m256 v;
        float f[8];
    } vacc_u;
    vacc_u.v = vacc;
    dest[0]             = vacc_u.f[0];
    dest[packCUnit]     = vacc_u.f[1];
    dest[packCUnit * 2] = vacc_u.f[2];
    dest[packCUnit * 3] = vacc_u.f[3];
    dest[packCUnit * 4] = vacc_u.f[4];
    dest[packCUnit * 5] = vacc_u.f[5];
    dest[packCUnit * 6] = vacc_u.f[6];
    dest[packCUnit * 7] = vacc_u.f[7];
}

#endif