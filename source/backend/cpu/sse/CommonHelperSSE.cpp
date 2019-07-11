//
//  CommonHelperSSE.cpp
//  MNN
//
//  Created by MNN on 2018/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_USE_SSE
#include "CommonHelperSSE.hpp"
#if defined(_MSC_VER)
#include <intrin.h>
#endif

static void mnn_cpuid(int data[4], int funcNumber) {
#if defined(_MSC_VER)
    __cpuidex(data, funcNumber, 0);
#elif defined(__clang__) || defined(__GNUC__)
    __asm("cpuid" : "=a"(data[0]), "=b"(data[1]), "=c"(data[2]), "=d"(data[3]) : "a"(funcNumber), "c"(0) : );
#endif
}

static bool mnn_xgetbvAndDetect(int ctr, int bitToDetect) {
#if defined(_MSC_VER)
    return (_xgetbv(0) >> bitToDetect) & 1;
#elif defined(__clang__) || defined(__GNUC__)
    int a, d;
    __asm("xgetbv" : "=a"(a), "=d"(d) : "c"(ctr) : );
    return (bitToDetect < 32 ? (a >> bitToDetect) : d >> (bitToDetect - 32)) & 1;
#else
    return false;
#endif
}

static bool cpu_feature_available_internal(CPU_FEATURE feature) {
#if !(defined(_MSC_VER) || defined(__clang__) || defined(__GNUC__))
    return false;
#else
    int data[4];
    mnn_cpuid(data, 0);
    if (data[0] < 1) return false;
    mnn_cpuid(data, 1);
    bool xsaveSupport = (data[2] >> 27) & 1;
    if (!xsaveSupport) {
        return false;
    }
    bool cpuSupport = false, osSupport = false;
    if (feature == AVX) {
        cpuSupport = (data[2] >> 28) & 1;
        osSupport = mnn_xgetbvAndDetect(0, 2);
    } else if (feature == SSE) {
        cpuSupport = (data[3] >> 25) & 1;
        osSupport = mnn_xgetbvAndDetect(0, 1);
    }
    return cpuSupport && osSupport;
#endif
}

static bool featureArray[] = {
    cpu_feature_available_internal(SSE),
    cpu_feature_available_internal(AVX)
}; // static variable initialization, only run once.

bool cpu_feature_available(CPU_FEATURE feature) {
    if (feature != SSE && feature != AVX) return false;
    return featureArray[feature];
}

#endif // MNN_USE_SSE