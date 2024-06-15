//
//  GemmCommon.hpp
//  MNN
//
//  Created by MNN on 2020/09/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GemmCommon_hpp
#define GemmCommon_hpp
#include <MNN/MNNDefine.h>
#include <stdint.h>

#define TRANPOSE_SAVE(u, v, z0, z3, z6, z9)      \
    {                                            \
        auto m0 = z0;                            \
        auto m1 = z3;                            \
        auto m2 = z6;                            \
        auto m3 = z9;                            \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);       \
        _mm_storeu_ps(dst + 4 * (0 + 4 * v), m0); \
        _mm_storeu_ps(dst + 4 * (1 + 4 * v), m1); \
        _mm_storeu_ps(dst + 4 * (2 + 4 * v), m2); \
        _mm_storeu_ps(dst + 4 * (3 + 4 * v), m3); \
    }

#define FMLA_TRANPOSE_SAVE(u, v, z0, z3, z6, z9)      \
    {                                            \
        auto m0 = z0;                            \
        auto m1 = z3;                            \
        auto m2 = z6;                            \
        auto m3 = z9;                            \
        _MM_TRANSPOSE4_PS(m0, m1, m2, m3);       \
        auto t0 = _mm_loadu_ps(dst + 4 * (0 + 4 * v));\
        auto t1 = _mm_loadu_ps(dst + 4 * (1 + 4 * v));\
        auto t2 = _mm_loadu_ps(dst + 4 * (2 + 4 * v));\
        auto t3 = _mm_loadu_ps(dst + 4 * (3 + 4 * v));\
        m0 = _mm_add_ps(m0, t0);\
        m1 = _mm_add_ps(m1, t1);\
        m2 = _mm_add_ps(m2, t2);\
        m3 = _mm_add_ps(m3, t3);\
        _mm_storeu_ps(dst + 4 * (0 + 4 * v), m0); \
        _mm_storeu_ps(dst + 4 * (1 + 4 * v), m1); \
        _mm_storeu_ps(dst + 4 * (2 + 4 * v), m2); \
        _mm_storeu_ps(dst + 4 * (3 + 4 * v), m3); \
    }

void _SSE_GemmPostTreat(float* C, size_t eSize, const size_t* parameter, const float* postParameters,
                        const float* bias);
#endif
