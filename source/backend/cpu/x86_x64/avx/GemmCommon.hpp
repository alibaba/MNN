//
//  GemmCommon.hpp
//  MNN
//
//  Created by MNN on b'2020/09/22'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GemmCommon_hpp
#define GemmCommon_hpp
#include <MNN/MNNDefine.h>
#include <stdint.h>

#define TRANPOSE_SAVE(u, v, z0, z3, z6, z9)              \
    {                                                    \
        auto m0 = _mm256_extractf128_ps(z0, u);          \
        auto m1 = _mm256_extractf128_ps(z3, u);          \
        auto m2 = _mm256_extractf128_ps(z6, u);          \
        auto m3 = _mm256_extractf128_ps(z9, u);          \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);               \
        _mm_store_ps(dst + 4 * (0 + 4 * u + 8 * v), m0); \
        _mm_store_ps(dst + 4 * (1 + 4 * u + 8 * v), m1); \
        _mm_store_ps(dst + 4 * (2 + 4 * u + 8 * v), m2); \
        _mm_store_ps(dst + 4 * (3 + 4 * u + 8 * v), m3); \
    }

void AVX2GemmPostTreat(float* C, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias);
#endif
