//
//  WinogradOptFunctionInt8.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <limits>
#include <vector>
#include <map>
#include <tuple>
#include <functional>
#include <numeric>
#include "WinogradOptFunctionInt8.hpp"
#include "Int8FunctionsOpt.h"
#include "core/TensorUtils.hpp"
#include "math/Vec.hpp"
#include "core/Macro.h"

namespace MNN {

#define VecType MNN::Math::Vec<int8_t, 16>
#ifdef MNN_USE_SSE
static inline void LOADX_C4(const int8_t* src, size_t srcZStep, VecType& vec0, VecType& vec1, VecType& vec2, VecType& vec3) {
    vec0 = VecType::load(src + 0 * srcZStep);
    vec1 = VecType::load(src + 1 * srcZStep);
    vec2 = VecType::load(src + 2 * srcZStep);
    vec3 = VecType::load(src + 3 * srcZStep);
    auto m0 = _mm_castsi128_ps(vec0.value);
    auto m1 = _mm_castsi128_ps(vec1.value);
    auto m2 = _mm_castsi128_ps(vec2.value);
    auto m3 = _mm_castsi128_ps(vec3.value);
    _MM_TRANSPOSE4_PS(m0, m1, m2, m3);
    vec0.value = _mm_castps_si128(m0);
    vec1.value = _mm_castps_si128(m1);
    vec2.value = _mm_castps_si128(m2);
    vec3.value = _mm_castps_si128(m3);
    vec0 = vec0 - 128;
    vec1 = vec1 - 128;
    vec2 = vec2 - 128;
    vec3 = vec3 - 128;
}
static inline void LOADY_C4(const int8_t* src, size_t srcXStep, size_t srcZStep,
                            VecType& vec0, VecType& vec1, VecType& vec2, VecType& vec3) {
#define LOADZ(ptr, vec) \
vec.value = _mm_set_epi32(*(int32_t*)(ptr + 3 * srcZStep), *(int32_t*)(ptr + 2 * srcZStep), *(int32_t*)(ptr + srcZStep), *(int32_t*)(ptr)); \
vec = vec - 128;
    LOADZ(src + 0 * srcXStep, vec0);
    LOADZ(src + 1 * srcXStep, vec1);
    LOADZ(src + 2 * srcXStep, vec2);
    LOADZ(src + 3 * srcXStep, vec3);
#undef LOADZ
}
#endif // MNN_USE_SSE

static inline void LOAD_C16(const int8_t* src, size_t step,
                            VecType& vec0, VecType& vec1, VecType& vec2, VecType& vec3) {
#ifdef MNN_USE_SSE
#define _LOAD(ptr, vec) vec = VecType::load(ptr) - 128
#else
#define _LOAD(ptr, vec) vec = VecType::load(ptr)
#endif
    _LOAD(src + 0 * step, vec0);
    _LOAD(src + 1 * step, vec1);
    _LOAD(src + 2 * step, vec2);
    _LOAD(src + 3 * step, vec3);
}

static inline void SAVE_C16(int8_t* dst, size_t dstXStep, const VecType& m0, const VecType& m1, const VecType& m2, const VecType& m3) {
#ifdef MNN_USE_SSE
#define _SAVE(ptr, vec) VecType::save(ptr, vec + 128)
#else
#define _SAVE(ptr, vec) VecType::save(ptr, vec)
#endif
    _SAVE(dst + 0 * dstXStep, m0);
    _SAVE(dst + 1 * dstXStep, m1);
    _SAVE(dst + 2 * dstXStep, m2);
    _SAVE(dst + 3 * dstXStep, m3);
}

static inline void SRC_TRANS_4X4(const VecType& in0, const VecType& in1, const VecType& in2, const VecType& in3,
                             VecType& m0, VecType& m1, VecType& m2, VecType& m3) {
    m0 = in0 - in2;
    m1 = in1 + in2;
    m2 = in2 - in1;
    m3 = in3 - in1;
}

static void _sourceTrans1Slow(const int8_t* srcStart, int8_t* dstStart, size_t srcXStep, size_t srcZStep,
                              size_t dstXStep, size_t dstZStep, size_t count, size_t inPack, size_t outPack, size_t xC, size_t unit,
#ifdef MNN_USE_SSE
                              int32_t offset_i = -128, int32_t offset_o = 128
#else
                              int32_t offset_i = 0, int32_t offset_o = 0
#endif
                              
                              ) {
    for (int i = 0; i < count; ++i) {
        for (int x = 0; x < xC; ++x) {
            auto srcZ = srcStart + (i / inPack) * srcZStep + (i % inPack) + x * unit * inPack;
            auto dstZ = dstStart + (i / outPack) * dstZStep + (i % outPack) + x * outPack;
            int8_t src[4];
            for (int j = 0; j < 4; ++j) {
                src[j] = srcZ[j * srcXStep] + offset_i;
            }
            dstZ[0 * dstXStep] = src[0] - src[2] + offset_o;
            dstZ[1 * dstXStep] = src[1] + src[2] + offset_o;
            dstZ[2 * dstXStep] = src[2] - src[1] + offset_o;
            dstZ[3 * dstXStep] = src[3] - src[1] + offset_o;
        }
    }
}
static void _sourceTrans2Slow(const int8_t* srcStart, int8_t* dstStart, size_t srcYStep, size_t srcZStep,
                              size_t dstXStep, size_t dstZStep, size_t count, size_t inPack, size_t outPack, size_t xC, size_t unit) {
#ifdef MNN_USE_SSE
    constexpr int offset_i = -128, offset_o = 128;
#else
    constexpr int offset_i = 0, offset_o = 0;
#endif
    int8_t midBuffer[4 * 4];
    for (int i = 0; i < count; ++i) {
        for (int x = 0; x < xC; ++x) {
            auto srcZ = srcStart + (i / inPack) * srcZStep + (i % inPack) + x * unit * inPack;
            auto dstZ = dstStart + (i / outPack) * dstZStep + (i % outPack) + x * outPack;
            for (int h = 0; h < 4; ++h) {
                _sourceTrans1Slow(srcZ + h * srcYStep, midBuffer + h * 4, inPack, 0, 1, 0, 1, 1, 1, 1, 0, offset_i, 0);
            }
            for (int w = 0; w < 4; ++w) {
                _sourceTrans1Slow(midBuffer + w, dstZ + w * dstXStep, 4, 0, dstXStep * 4, 0, 1, 1, 1, 1, 0, 0, offset_o);
            }
        }
    }
}
/*
 2D winograd: SrcTransXFunc, then SrcTransYFunc
 SrcTransXFunc: read along X-dim and elements is continuous (C4 then X4), so load(simd) then transpose, then compute(simd) as int8x16 (C16)
 SrcTransYFunc: read alone Y-dim and elements is not continuous(only int8x4), so load by lane to reduce IO (because not need repack data before transform)
 ARM82:
 4(UNIT) -> 4(SRC_UNIT)
 1xN ~ _sourceTransXPack4x4
 Nx1 ~ _sourceTransYPack4x4
 NxN ~ _sourceTrans2Pack4x4
 ARM32/64/SSE: 4(UNIT) -> 16(SRC_UNIT)
 1xN ~ _sourceTransXPack4x16
 Nx1 ~ _sourceTransYPack4x16
 NxM ~ _sourceTrans2Pack4x16
 AVX: 8(UNIT) -> 16(SRC_UNIT)
 1xN ~ _sourceTransXPack8x16
 Nx1 ~ _sourceTransYPack8x16
 NxM ~ _sourceTrans2Pack8x16
 AVX512: 16(UNIT) -> 16(SRC_UNIT)
 1xN ~ _sourceTransXPack16x16
 Nx1 ~ _sourceTransYPack16x16
 NxM ~ _sourceTrans2Pack16x16
 
*/

#if defined(MNN_USE_NEON)
extern "C" {
#ifdef MNN_USE_ARMV82
void _sourceTransXPack4x4Int8(const int8_t*, int8_t*, size_t, size_t, size_t, size_t, size_t, size_t);
void _sourceTransYPack4x4Int8(const int8_t*, int8_t*, size_t, size_t, size_t, size_t, size_t, size_t);
void _sourceTrans2Pack4x4Int8(const int8_t*, int8_t*, size_t, size_t, size_t, size_t, size_t, size_t, size_t);
#endif // MNN_USE_ARMV82
void _sourceTransXPack4x16Int8(const int8_t*, int8_t*, size_t, size_t, size_t, size_t, size_t, size_t);
void _sourceTransYPack4x16Int8(const int8_t*, int8_t*, size_t, size_t, size_t, size_t, size_t, size_t);
void _sourceTrans2Pack4x16Int8(const int8_t*, int8_t*, size_t, size_t, size_t, size_t, size_t, size_t, size_t);
}
#endif // MNN_USE_NEON

// winograd source transform with simd, C4 -> C4
static void _sourceTransXPack4x4(const int8_t* srcStart, int8_t* dstStart, size_t srcZStep,
                                 size_t dstXStep, size_t dstZStep, size_t countC4, size_t xC, size_t unit) {
#if defined(MNN_USE_ARMV82) && defined(MNN_USE_NEON)
    size_t countUnit = countC4 / 4, countRemain = countC4 % 4;
    if (countUnit > 0) {
        _sourceTransXPack4x4Int8(srcStart, dstStart, srcZStep, dstXStep, dstZStep, countUnit, xC, unit);
    }
    srcStart += countUnit * 4 * srcZStep;
    dstStart += countUnit * 4 * dstZStep;
#else
    size_t countRemain = countC4;
#endif
    _sourceTrans1Slow(srcStart, dstStart, 4, srcZStep, dstXStep, dstZStep, countRemain * 4, 4, 4, xC, unit);
}

// winograd source transform with simd, fused C4 -> C16 pack
static void _sourceTransXPack4x16(const int8_t* srcStart, int8_t* dstStart, size_t srcZStep,
                                  size_t dstXStep, size_t dstZStep, size_t countC4, size_t xC, size_t unit) {
#if defined(MNN_USE_NEON)
    size_t countUnit = countC4 / 4, countRemain = countC4 % 4;
    if (countUnit > 0) {
        _sourceTransXPack4x16Int8(srcStart, dstStart, srcZStep, dstXStep, dstZStep, countUnit, xC, unit);
    }
    srcStart += countUnit * 4 * srcZStep;
    dstStart += countUnit * dstZStep;
#elif defined(MNN_USE_SSE)
    size_t countUnit = countC4 / 4, countRemain = countC4 % 4;
    for (int z = 0; z < countUnit; ++z) {
        for (int x = 0; x < xC; ++x) {
            // load, then 4x int8x4 => 1x int8x16, then do simd compute, save
            VecType in0, in1, in2, in3;
            LOADX_C4(srcStart + x * unit * 4, srcZStep, in0, in1, in2, in3);
            VecType m0, m1, m2, m3;
            SRC_TRANS_4X4(in0, in1, in2, in3, m0, m1, m2, m3);
            SAVE_C16(dstStart + x * 16, dstXStep, m0, m1, m2, m3);
        }
        srcStart += srcZStep * 4;
        dstStart += dstZStep;
    }
#else
    size_t countRemain = countC4;
#endif
    _sourceTrans1Slow(srcStart, dstStart, 4, srcZStep, dstXStep, dstZStep, countRemain * 4, 4, 16, xC, unit);
}

// AVX, just run correctly, because winograd int8 compute on AVX is wrose choice
static void _sourceTransXPack8x16(const int8_t* srcStart, int8_t* dstStart, size_t srcZStep,
                                  size_t dstXStep, size_t dstZStep, size_t countC8, size_t xC, size_t unit) {
    _sourceTrans1Slow(srcStart, dstStart, 8, srcZStep, dstXStep, dstZStep, countC8 * 8, 8, 16, xC, unit);
}

// AVX512
static void _sourceTransXPack16x16(const int8_t* srcStart, int8_t* dstStart, size_t srcZStep,
                                  size_t dstXStep, size_t dstZStep, size_t countC16, size_t xC, size_t unit) {
    for (int z = 0; z < countC16; ++z) {
        for (int x = 0; x < xC; ++x) {
            // load, then 4x int8x4 => 1x int8x16, then do simd compute, save
            VecType in0, in1, in2, in3;
            LOAD_C16(srcStart + x * unit * 16, 16, in0, in1, in2, in3);
            VecType m0, m1, m2, m3;
            SRC_TRANS_4X4(in0, in1, in2, in3, m0, m1, m2, m3);
            SAVE_C16(dstStart + x * 16, dstXStep, m0, m1, m2, m3);
        }
        srcStart += srcZStep;
        dstStart += dstZStep;
    }
}

// winograd source transform with simd, C4 -> C4
static void _sourceTransYPack4x4(const int8_t* srcStart, int8_t* dstStart, size_t srcXStep, size_t srcZStep,
                                 size_t dstXStep, size_t dstZStep, size_t countC4, size_t xC) {
#if defined(MNN_USE_ARMV82)
    size_t countUnit = countC4 / 4, countRemain = countC4 % 4;
    if (countUnit > 0) {
        _sourceTransYPack4x4Int8(srcStart, dstStart, srcXStep, srcZStep, dstXStep, dstZStep, countUnit, xC);
    }
    srcStart += countUnit * 4 * srcZStep;
    dstStart += countUnit * 4 * dstZStep;
#else
    size_t countRemain = countC4;
#endif
    // simd accelerate can't be used
    _sourceTrans1Slow(srcStart, dstStart, srcXStep, srcZStep, dstXStep, dstZStep, countRemain * 4, 4, 4, xC, 1);
}

// winograd source transform with simd, fused C4 -> C16 pack
static void _sourceTransYPack4x16(const int8_t* srcStart, int8_t* dstStart, size_t srcXStep, size_t srcZStep,
                                  size_t dstXStep, size_t dstZStep, size_t countC4, size_t xC) {
#if defined(MNN_USE_NEON)
    size_t countUnit = countC4 / 4, countRemain = countC4 % 4;
    if (countUnit > 0) {
        _sourceTransYPack4x16Int8(srcStart, dstStart, srcXStep, srcZStep, dstXStep, dstZStep, countUnit, xC);
    }
    srcStart += countUnit * 4 * srcZStep;
    dstStart += countUnit * dstZStep;
#elif defined(MNN_USE_SSE)
    size_t countUnit = countC4 / 4, countRemain = countC4 % 4;
    for (int z = 0; z < countUnit; ++z) {
        for (int x = 0; x < xC; ++x) {
            // load, then 4x int8x4 => 1x int8x16, then do simd compute, save
            VecType in0, in1, in2, in3;
            LOADY_C4(srcStart + x * 4, srcXStep, srcZStep, in0, in1, in2, in3);
            VecType m0, m1, m2, m3;
            SRC_TRANS_4X4(in0, in1, in2, in3, m0, m1, m2, m3);
            SAVE_C16(dstStart + x * 16, dstXStep, m0, m1, m2, m3);
        }
        srcStart += srcZStep * 4;
        dstStart += dstZStep;
    }
#else
    size_t countRemain = countC4;
#endif
    // simd accelerate can't be used
    _sourceTrans1Slow(srcStart, dstStart, srcXStep, srcZStep, dstXStep, dstZStep, countRemain * 4, 4, 16, xC, 1);
}

// AVX, just run correctly, because winograd int8 compute on AVX is wrose choice
static void _sourceTransYPack8x16(const int8_t* srcStart, int8_t* dstStart, size_t srcXStep, size_t srcZStep,
                                 size_t dstXStep, size_t dstZStep, size_t countC8, size_t xC) {
    _sourceTrans1Slow(srcStart, dstStart, srcXStep, srcZStep, dstXStep, dstZStep, countC8 * 8, 8, 16, xC, 1);
}

// AVX512
static void _sourceTransYPack16x16(const int8_t* srcStart, int8_t* dstStart, size_t srcXStep, size_t srcZStep,
                                 size_t dstXStep, size_t dstZStep, size_t countC16, size_t xC) {
    for (int z = 0; z < countC16; ++z) {
        for (int x = 0; x < xC; ++x) {
            // load, then 4x int8x4 => 1x int8x16, then do simd compute, save
            VecType in0, in1, in2, in3;
            LOAD_C16(srcStart + x * 16, srcXStep, in0, in1, in2, in3);
            VecType m0, m1, m2, m3;
            SRC_TRANS_4X4(in0, in1, in2, in3, m0, m1, m2, m3);
            SAVE_C16(dstStart + x * 16, dstXStep, m0, m1, m2, m3);
        }
        srcStart += srcZStep;
        dstStart += dstZStep;
    }
}

static void _sourceTrans2Pack4x4(const int8_t* srcStart, int8_t* dstStart, size_t srcYStep, size_t srcZStep,
                                 size_t dstXStep, size_t dstZStep, size_t countC4, size_t xC, size_t unit) {
#if defined(MNN_USE_ARMV82) && defined(MNN_USE_NEON)
    size_t countUnit = countC4 / 4, countRemain = countC4 % 4;
    if (countUnit > 0) {
        _sourceTrans2Pack4x4Int8(srcStart, dstStart, srcYStep, srcZStep, dstXStep, dstZStep, countUnit, xC, unit);
    }
    srcStart += countUnit * 4 * srcZStep;
    dstStart += countUnit * 4 * dstZStep;
#else
    size_t countRemain = countC4;
#endif
    _sourceTrans2Slow(srcStart, dstStart, srcYStep, srcZStep, dstXStep, dstZStep, countRemain * 4, 4, 4, xC, unit);
}

static void _sourceTrans2Pack4x16(const int8_t* srcStart, int8_t* dstStart, size_t srcYStep, size_t srcZStep,
                                 size_t dstXStep, size_t dstZStep, size_t countC4, size_t xC, size_t unit) {
#if defined(MNN_USE_NEON)
    size_t countUnit = countC4 / 4, countRemain = countC4 % 4;
    if (countUnit > 0) {
        _sourceTrans2Pack4x16Int8(srcStart, dstStart, srcYStep, srcZStep, dstXStep, dstZStep, countUnit, xC, unit);
    }
    srcStart += countUnit * 4 * srcZStep;
    dstStart += countUnit * dstZStep;
#elif defined(MNN_USE_SSE)
    size_t countUnit = countC4 / 4, countRemain = countC4 % 4;
    for (int z = 0; z < countUnit; ++z) {
        for (int x = 0; x < xC; ++x) {
            // load, then 4x int8x4 => 1x int8x16, then do simd compute, save
            VecType in[4][4], mid[4][4], out[4];
            for (int i = 0; i < 4; ++i) {
                LOADX_C4(srcStart + x * unit * 4 + i * srcYStep, srcZStep, in[i][0], in[i][1], in[i][2], in[i][3]);
            }
            for (int i = 0; i < 4; ++i) {
                SRC_TRANS_4X4(in[0][i], in[1][i], in[2][i], in[3][i], mid[0][i], mid[1][i], mid[2][i], mid[3][i]);
            }
            for (int i = 0; i < 4; ++i) {
                SRC_TRANS_4X4(mid[i][0], mid[i][1], mid[i][2], mid[i][3], out[0], out[1], out[2], out[3]);
                SAVE_C16(dstStart + i * dstXStep * 4 + x * 16, dstXStep, out[0], out[1], out[2], out[3]);
            }
        }
        srcStart += srcZStep * 4;
        dstStart += dstZStep;
    }
#else
    size_t countRemain = countC4;
#endif
    _sourceTrans2Slow(srcStart, dstStart, srcYStep, srcZStep, dstXStep, dstZStep, countRemain * 4, 4, 16, xC, unit);
}

static void _sourceTrans2Pack8x16(const int8_t* srcStart, int8_t* dstStart, size_t srcYStep, size_t srcZStep,
                                 size_t dstXStep, size_t dstZStep, size_t countC8, size_t xC, size_t unit) {
    _sourceTrans2Slow(srcStart, dstStart, srcYStep, srcZStep, dstXStep, dstZStep, countC8 * 8, 8, 16, xC, unit);
}

static void _sourceTrans2Pack16x16(const int8_t* srcStart, int8_t* dstStart, size_t srcYStep, size_t srcZStep,
                                 size_t dstXStep, size_t dstZStep, size_t countC16, size_t xC, size_t unit) {
    for (int z = 0; z < countC16; ++z) {
        for (int x = 0; x < xC; ++x) {
            // load, then 4x int8x4 => 1x int8x16, then do simd compute, save
            VecType in[4][4], mid[4][4], out[4];
            for (int i = 0; i < 4; ++i) {
                LOAD_C16(srcStart + x * unit * 16 + i * srcYStep, 16, in[i][0], in[i][1], in[i][2], in[i][3]);
            }
            for (int i = 0; i < 4; ++i) {
                SRC_TRANS_4X4(in[0][i], in[1][i], in[2][i], in[3][i], mid[0][i], mid[1][i], mid[2][i], mid[3][i]);
            }
            for (int i = 0; i < 4; ++i) {
                SRC_TRANS_4X4(mid[i][0], mid[i][1], mid[i][2], mid[i][3], out[0], out[1], out[2], out[3]);
                SAVE_C16(dstStart + i * dstXStep * 4 + x * 16, dstXStep, out[0], out[1], out[2], out[3]);
            }
        }
        srcStart += srcZStep;
        dstStart += dstZStep;
    }
}

WinogradFunctionInt8::SrcTransXFunc WinogradFunctionInt8::chooseSourceTransformX(int alpha, int inPack, int outPack) {
    std::map<std::tuple<int, int, int>, WinogradFunctionInt8::SrcTransXFunc> func_table = {
        {std::make_tuple(4, 4, 4), _sourceTransXPack4x4},
        {std::make_tuple(4, 4, 16), _sourceTransXPack4x16},
        {std::make_tuple(4, 8, 16), _sourceTransXPack8x16},
        {std::make_tuple(4, 16, 16), _sourceTransXPack16x16},
    };
    auto iter = func_table.find(std::make_tuple(alpha, inPack, outPack));
    if (iter != func_table.end()) {
        return iter->second;
    }
    return nullptr;
}

WinogradFunctionInt8::SrcTransYFunc WinogradFunctionInt8::chooseSourceTransformY(int alpha, int inPack, int outPack) {
    std::map<std::tuple<int, int, int>, WinogradFunctionInt8::SrcTransYFunc> func_table = {
        {std::make_tuple(4, 4, 4), _sourceTransYPack4x4},
        {std::make_tuple(4, 4, 16), _sourceTransYPack4x16},
        {std::make_tuple(4, 8, 16), _sourceTransYPack8x16},
        {std::make_tuple(4, 16, 16), _sourceTransYPack16x16},
    };
    auto iter = func_table.find(std::make_tuple(alpha, inPack, outPack));
    if (iter != func_table.end()) {
        return iter->second;
    }
    return nullptr;
}

WinogradFunctionInt8::SrcTrans2Func WinogradFunctionInt8::chooseSourceTransform2(int alpha, int inPack, int outPack) {
    std::map<std::tuple<int, int, int>, WinogradFunctionInt8::SrcTrans2Func> func_table = {
        {std::make_tuple(4, 4, 4), _sourceTrans2Pack4x4},
        {std::make_tuple(4, 4, 16), _sourceTrans2Pack4x16},
        {std::make_tuple(4, 8, 16), _sourceTrans2Pack8x16},
        {std::make_tuple(4, 16, 16), _sourceTrans2Pack16x16},
    };
    auto iter = func_table.find(std::make_tuple(alpha, inPack, outPack));
    if (iter != func_table.end()) {
        return iter->second;
    }
    return nullptr;
}

}
