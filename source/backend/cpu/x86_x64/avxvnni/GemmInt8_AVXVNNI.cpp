//
//  GemmInt8_AVXVNNI.cpp
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../avx/GemmCommon.hpp"
#include "../avx/FunctionSummary.hpp"
#include "core/Macro.h"
#include <math.h>

#ifdef MNN_LOW_MEMORY
#define AVX2_PACKINT8 8
#define GEMMINT8_AVX2_E 4
#define GEMMINT8_AVX2_L 4
#define GEMMINT8_AVX2_H 8

// vpdpbusd sums four u8 x s8 products straight into int32. The AVX2 path goes through an int16 maddubs that
// cannot saturate for 4-bit weights, so both paths give the same integers.
#define MNN_GEMMINT8_W4_NAME _AVX_MNNGemmInt8AddBiasScale_16x4_w4_AVXVNNI
#define MNN_GEMMINT8_W4_DOT(acc, s, w) _mm256_dpbusd_avx_epi32(acc, s, w)
#include "../avx/GemmInt8W4.inl"
#undef MNN_GEMMINT8_W4_DOT
#undef MNN_GEMMINT8_W4_NAME
#endif
