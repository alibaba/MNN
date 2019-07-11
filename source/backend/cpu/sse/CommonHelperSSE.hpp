//
//  CommonHelperSSE.hpp
//  MNN
//
//  Created by MNN on 2018/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef COMMON_HELPER_SSE_HPP
#define COMMON_HELPER_SSE_HPP
#ifdef MNN_USE_SSE
#define PRAGMA(X) _Pragma(#X)
#if defined(_MSC_VER)
#define TargetBegin(X) __pragma(push_macro(X))
#define TargetEnd(X) __pragma(pop_macro(X))
#elif defined(__clang__)
#define TargetBegin(X) PRAGMA(clang attribute push (__attribute__((target(X))), apply_to=function))
#define TargetEnd(X) PRAGMA(clang attribute pop)
#elif defined(__GNUC__)
#define TargetBegin(X) \
    PRAGMA(GCC push_options) \
    PRAGMA(GCC target(X))
#define TargetEnd(X) PRAGMA(GCC pop_options)
#endif

enum CPU_FEATURE {SSE, AVX};

bool cpu_feature_available(CPU_FEATURE feature);

#endif // MNN_USE_SSE
#endif // COMMON_HELPER_SSE_HPP