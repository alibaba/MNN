//
//  WinogradInt8Helper.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if __GNUC__ == 4
#pragma GCC optimize("-flax-vector-conversions")
#endif

#include <limits>
#include <vector>
#include <map>
#include <tuple>
#include <functional>
#include "WinogradInt8Helper.hpp"
#include "Int8FunctionsOpt.h"
#include "core/TensorUtils.hpp"
#include "math/Vec.hpp"

namespace MNN {

#if (defined(MNN_USE_NEON) && defined(__aarch64__)) || defined(MNN_USE_SSE)
using VecType = MNN::Math::Vec<int8_t, 16>;
static inline void TRANS_4x4(VecType& vec0, VecType& vec1, VecType& vec2, VecType& vec3) {
#if defined(MNN_USE_SSE)
    auto m0 = _mm_castsi128_ps(vec0.value);
    auto m1 = _mm_castsi128_ps(vec1.value);
    auto m2 = _mm_castsi128_ps(vec2.value);
    auto m3 = _mm_castsi128_ps(vec3.value);
    _MM_TRANSPOSE4_PS(m0, m1, m2, m3);
    vec0.value = _mm_castps_si128(m0);
    vec1.value = _mm_castps_si128(m1);
    vec2.value = _mm_castps_si128(m2);
    vec3.value = _mm_castps_si128(m3);
#else
    auto m0 = vtrn1q_s32(vec0.value, vec1.value), m1 = vtrn2q_s32(vec0.value, vec1.value);
    auto m2 = vtrn1q_s32(vec2.value, vec3.value), m3 = vtrn2q_s32(vec2.value, vec3.value);
    vec0.value = vtrn1q_s64(m0, m2);
    vec1.value = vtrn1q_s64(m1, m3);
    vec2.value = vtrn2q_s64(m0, m2);
    vec3.value = vtrn2q_s64(m1, m3);
#endif
}
#endif

// winograd source transform with simd, C4 -> C4
static void _sourceTransUnit4x4Pack4x4(const int8_t* srcStart, int8_t* dstStart, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countC4) {
#if defined(MNN_USE_NEON) && defined(__aarch64__)
    using VecType = MNN::Math::Vec<int8_t, 16>;
    int countUnit = countC4 / 4, countRemain = countC4 % 4;
    for (int z = 0; z < countUnit; ++z) {
        // load, then 4x int8x4 => 1x int8x16, then do simd compute, save
        VecType in[4] = {
            VecType::load(srcStart + 0 * srcZStep),
            VecType::load(srcStart + 1 * srcZStep),
            VecType::load(srcStart + 2 * srcZStep),
            VecType::load(srcStart + 3 * srcZStep)
        };
        TRANS_4x4(in[0], in[1], in[2], in[3]);
        VecType m[4] = {
            in[0] - in[2],
            in[1] + in[2],
            in[2] - in[1],
            in[3] - in[1]
        };
        for (int i = 0; i < 4; ++i) {
            auto tmp = vreinterpretq_s32_s8(m[i].value);
            vst1q_lane_s32((int32_t*)(dstStart + 0 * dstZStep), tmp, 0);
            vst1q_lane_s32((int32_t*)(dstStart + 1 * dstZStep), tmp, 1);
            vst1q_lane_s32((int32_t*)(dstStart + 2 * dstZStep), tmp, 2);
            vst1q_lane_s32((int32_t*)(dstStart + 3 * dstZStep), tmp, 3);
            dstStart += dstXStep;
        }
        dstStart -= dstXStep * 4;
        srcStart += srcZStep * 4;
    }
#else
    int countUnit = 0, countRemain = countC4;
#endif
    // simd accelerate can't be used
    using VecType1 = MNN::Math::Vec<int8_t, 4>;
    for (int i = 0; i < countRemain; ++i) {
        auto s0 = VecType1::load(srcStart + 0 * 4);
        auto s1 = VecType1::load(srcStart + 1 * 4);
        auto s2 = VecType1::load(srcStart + 2 * 4);
        auto s3 = VecType1::load(srcStart + 3 * 4);
        VecType1::save(dstStart + 0 * dstXStep, s0 - s2);
        VecType1::save(dstStart + 1 * dstXStep, s1 + s2);
        VecType1::save(dstStart + 2 * dstXStep, s2 - s1);
        VecType1::save(dstStart + 3 * dstXStep, s3 - s1);
        
        srcStart += srcZStep;
        dstStart += dstZStep;
    }
}

// winograd source transform with simd, fused C4 -> C16 pack
static void _sourceTransUnit4x4Pack4x16(const int8_t* srcStart, int8_t* dstStart, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countC4) {
    using VecType = MNN::Math::Vec<int8_t, 16>;
#if (defined(MNN_USE_NEON) && defined(__aarch64__)) || defined(MNN_USE_SSE)
    int countUnit = countC4 / 4, countRemain = countC4 % 4;
    for (int z = 0; z < countUnit; ++z) {
        // load, then 4x int8x4 => 1x int8x16, then do simd compute, save
        auto in0 = VecType::load(srcStart + 0 * srcZStep);
        auto in1 = VecType::load(srcStart + 1 * srcZStep);
        auto in2 = VecType::load(srcStart + 2 * srcZStep);
        auto in3 = VecType::load(srcStart + 3 * srcZStep);
        TRANS_4x4(in0, in1, in2, in3);
        VecType::save(dstStart + 0 * dstXStep, in0 - in2);
        VecType::save(dstStart + 1 * dstXStep, in1 + in2);
        VecType::save(dstStart + 2 * dstXStep, in2 - in1);
        VecType::save(dstStart + 3 * dstXStep, in3 - in1);
        srcStart += srcZStep * 4;
        dstStart += dstZStep;
    }
#else
    int countUnit = 0, countRemain = countC4;
#endif
    // simd accelerate can't be used
    for (int i = 0; i < countRemain * 4; ++i) {
        auto srcZ = srcStart + (i / 4) * srcZStep + (i % 4);
        auto dstZ = dstStart + (i / 16) * dstZStep + (i % 16);
        int8_t src[4];
        for (int j = 0; j < 4; ++j) {
            src[j] = srcZ[j * 4];
        }
        dstZ[0 * dstXStep] = src[0] - src[2];
        dstZ[1 * dstXStep] = src[1] + src[2];
        dstZ[2 * dstXStep] = src[2] - src[1];
        dstZ[3 * dstXStep] = src[3] - src[1];
    }
}

// winograd source transform with simd, fused C16 -> C4 pack
static void _sourceTransUnit4x4Pack16x4(const int8_t* srcStart, int8_t* dstStart, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countC4) {
    using VecType = MNN::Math::Vec<int8_t, 16>;
#if defined(MNN_USE_NEON) && defined(__aarch64__)
    int countUnit = countC4 / 4, countRemain = countC4 % 4;
    for (int z = 0; z < countUnit; ++z) {
        // load 1x int8x16, then do simd compute, then 1x int8x16 => 4x int8x4, save
        VecType in[4] = {
            VecType::load(srcStart + 0 * srcZStep),
            VecType::load(srcStart + 1 * srcZStep),
            VecType::load(srcStart + 2 * srcZStep),
            VecType::load(srcStart + 3 * srcZStep)
        };
        VecType m[4] = {
            in[0] - in[2],
            in[1] + in[2],
            in[2] - in[1],
            in[3] - in[1]
        };
        for (int i = 0; i < 4; ++i) {
            auto tmp = vreinterpretq_s32_s8(m[i].value);
            vst1q_lane_s32((int32_t*)(dstStart + 0 * dstZStep), tmp, 0);
            vst1q_lane_s32((int32_t*)(dstStart + 1 * dstZStep), tmp, 1);
            vst1q_lane_s32((int32_t*)(dstStart + 2 * dstZStep), tmp, 2);
            vst1q_lane_s32((int32_t*)(dstStart + 3 * dstZStep), tmp, 3);
            dstStart += dstXStep;
        }
        dstStart -= dstXStep * 4;
        srcStart += srcZStep * 4;
    }
#else
    int countUnit = 0, countRemain = countC4;
#endif
    // simd accelerate can't be used
    for (int i = 0; i < countRemain * 4; ++i) {
        auto srcZ = srcStart + (i / 4) * srcZStep + (i % 4);
        auto dstZ = dstStart + (i / 16) * dstZStep + (i % 16);
        int8_t src[4];
        for (int j = 0; j < 4; ++j) {
            src[j] = srcZ[j * 4];
        }
        dstZ[0 * dstXStep] = src[0] - src[2];
        dstZ[1 * dstXStep] = src[1] + src[2];
        dstZ[2 * dstXStep] = src[2] - src[1];
        dstZ[3 * dstXStep] = src[3] - src[1];
    }
}

// winograd source transform with simd, C16 -> C16, countC4 = UP_DIV(count, 16)
static void _sourceTransUnit4x4Pack16x16(const int8_t* srcStart, int8_t* dstStart, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countC4) {
    using VecType = MNN::Math::Vec<int8_t, 16>;
    for (int i = 0; i < countC4; ++i) {
        auto s0 = VecType::load(srcStart + 0 * 16);
        auto s1 = VecType::load(srcStart + 1 * 16);
        auto s2 = VecType::load(srcStart + 2 * 16);
        auto s3 = VecType::load(srcStart + 3 * 16);
        VecType::save(dstStart + 0 * dstXStep, s0 - s2);
        VecType::save(dstStart + 1 * dstXStep, s1 + s2);
        VecType::save(dstStart + 2 * dstXStep, s2 - s1);
        VecType::save(dstStart + 3 * dstXStep, s3 - s1);
        
        srcStart += srcZStep;
        dstStart += dstZStep;
    }
}

WinogradInt8Helper::SrcTransFunc WinogradInt8Helper::chooseSourceTransform(int alpha, int inPack, int outPack) {
    std::map<std::tuple<int, int, int>, WinogradInt8Helper::SrcTransFunc> func_table = {
        {std::make_tuple(4, 4, 16), _sourceTransUnit4x4Pack4x16},
        {std::make_tuple(4, 16, 4), _sourceTransUnit4x4Pack16x4},
        {std::make_tuple(4, 4, 4), _sourceTransUnit4x4Pack4x4},
        {std::make_tuple(4, 16, 16), _sourceTransUnit4x4Pack16x16}
    };
    auto func_iter = func_table.find(std::make_tuple(alpha, inPack, outPack));
    if (func_iter == func_table.end()) {
        return nullptr;
    }
    return func_iter->second;
}

static void _destTransformUnit4x2(const float* srcStart, float* dstStart, size_t srcXStep, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countC4) {
    using VecType = MNN::Math::Vec<float, 4>;
    VecType c0(0.5f);
    for (int i = 0; i < countC4; ++i) {
        auto x0 = VecType::load(srcStart + srcXStep * 0);
        auto x1 = VecType::load(srcStart + srcXStep * 1);
        auto x2 = VecType::load(srcStart + srcXStep * 2);
        auto x3 = VecType::load(srcStart + srcXStep * 3);
        auto m0 = x0, m1 = x3;
        VecType::mla(m0, x1 + x2, c0);
        VecType::mla(m1, x1 - x2, c0);
        VecType::save(dstStart + dstXStep * 0, m0);
        VecType::save(dstStart + dstXStep * 1, m1);
        
        srcStart += srcZStep;
        dstStart += dstZStep;
    }
}

static void _destTransformUnit4x3(const float* srcStart, float* dstStart, size_t srcXStep, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countC4) {
    using VecType = MNN::Math::Vec<float, 4>;
    VecType c0(0.5f);
    for (int i = 0; i < countC4; ++i) {
        auto x0 = VecType::load(srcStart + srcXStep * 0);
        auto x1 = VecType::load(srcStart + srcXStep * 1);
        auto x2 = VecType::load(srcStart + srcXStep * 2);
        auto x3 = VecType::load(srcStart + srcXStep * 3);
        auto m0 = x0 + (x1 + x2) * 0.5;
        auto m1 = (x1 - x2) * 0.5;
        auto m2 = x3 + (x1 + x2) * 0.5;
        VecType::save(dstStart + dstXStep * 0, m0);
        VecType::save(dstStart + dstXStep * 1, m1);
        VecType::save(dstStart + dstXStep * 2, m2);
        
        srcStart += srcZStep;
        dstStart += dstZStep;
    }
}

WinogradInt8Helper::DstTransFunc WinogradInt8Helper::chooseDestTransform(int alpha, int unit) {
    std::map<std::tuple<int, int>, WinogradInt8Helper::DstTransFunc> func_table = {
        {std::make_tuple(4, 2), _destTransformUnit4x2},
        {std::make_tuple(4, 3), _destTransformUnit4x3},
    };
    auto func_iter = func_table.find(std::make_tuple(alpha, unit));
    if (func_iter == func_table.end()) {
        return nullptr;
    }
    return func_iter->second;
}

typedef bool(*WeightTransFunc)(const int8_t* srcStart, int8_t* dstStart, size_t srcStep, size_t dstStep);

static bool _weightTransUnit3x4(const int8_t* srcStart, int8_t* dstStart, size_t srcStep, size_t dstStep) {
    int32_t x[3], m[4];
    for (int i = 0; i < 3; ++i) {
        x[i] = (int32_t)(srcStart[i * srcStep]);
    }
    m[0] = x[0];
    m[1] = x[0] + x[1] + x[2];
    m[2] = x[0] - x[1] + x[2];
    m[3] = x[2];
    bool overflow = false;
    for (int i = 0; i < 4; ++i) {
        overflow |= (m[i] < std::numeric_limits<int8_t>::min() || m[i] > std::numeric_limits<int8_t>::max());
        dstStart[i * dstStep] = (int8_t)m[i];
    }
    return overflow;
}

static bool _weightTransUnit2x4(const int8_t* srcStart, int8_t* dstStart, size_t srcStep, size_t dstStep) {
    int32_t x[2], m[4];
    for (int i = 0; i < 2; ++i) {
        x[i] = (int32_t)(srcStart[i * srcStep]);
    }
    m[0] = x[0];
    m[1] = x[0] + x[1];
    m[2] = x[0] - x[1];
    m[3] = x[1];
    bool overflow = false;
    for (int i = 0; i < 4; ++i) {
        overflow |= (m[i] < std::numeric_limits<int8_t>::min() || m[i] > std::numeric_limits<int8_t>::max());
        dstStart[i * dstStep] = (int8_t)m[i];
    }
    return overflow;
}

static WeightTransFunc _chooseWeightTransform(int alpha, int kernel) {
    std::map<std::tuple<int, int>, WeightTransFunc> func_table = {
        {std::make_tuple(4, 3), _weightTransUnit3x4},
        {std::make_tuple(4, 2), _weightTransUnit2x4},
    };
    auto func_iter = func_table.find(std::make_tuple(alpha, kernel));
    if (func_iter == func_table.end()) {
        return nullptr;
    }
    return func_iter->second;
}

WinogradInt8Helper::WinogradInt8Helper(int unitY, int unitX, const Convolution2DCommon* common, const CoreInt8Functions* core) {
    mCommon = common;
    mAlphaY = unitY + common->kernelY() - 1;
    mAlphaX = unitX + common->kernelX() - 1;
    mInt8Core = core;
}

std::shared_ptr<Tensor> WinogradInt8Helper::allocTransformWeight(const Tensor* weightSrc) {
    int UNIT, SRC_UNIT, DST_XUNIT;
    mInt8Core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int oc4 = UP_DIV(mCommon->outputCount(), UNIT), ic4 = UP_DIV(mCommon->inputCount(), SRC_UNIT);
    return std::shared_ptr<Tensor>(Tensor::createDevice<int8_t>({mAlphaY, mAlphaX, oc4, ic4, UNIT, SRC_UNIT}));
}
// whether transform success without overflow, only detect overflow when weightDst == nullptr
bool WinogradInt8Helper::transformWeight(const Tensor* weightSrc, Tensor* weightDst) {
    bool fake = (weightDst == nullptr); // fake transform, only for detect overflow
    int UNIT, SRC_UNIT, DST_XUNIT;
    mInt8Core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int oc = mCommon->outputCount(), ic = mCommon->inputCount();
    int kernelY = mCommon->kernelY(), kernelX = mCommon->kernelX();
    auto transFuncY = _chooseWeightTransform(mAlphaY, kernelY);
    auto transFuncX = _chooseWeightTransform(mAlphaX, kernelX);
    mValid = (transFuncY != nullptr || kernelY == 1);
    mValid &= (transFuncX != nullptr || kernelX == 1);
    if (!mValid) {
        return mValid;
    }
    // assign new T[xx] to shared_ptr<T[]> is not support due to bug of some compiler (c++11)
    // so not use: std::shared_ptr<int8_t[]> cache(new int8_t[xx])
    std::shared_ptr<int8_t> cache(new int8_t[mAlphaY * kernelX + mAlphaY * mAlphaX * UNIT * SRC_UNIT],
                                  [](int8_t* ptr) { delete[] ptr; });
    int dstYStep = (fake ? 0 : weightDst->stride(0)), dstXStep = (fake ? 0 : weightDst->stride(1));
    int dstOZStep = (fake ? 0 : weightDst->stride(2)), dstSZStep = (fake ? 0 : weightDst->stride(3));
    int8_t* dataDstOrigin;
    if (fake) {
        dataDstOrigin = cache.get() + mAlphaY * kernelX;
        memset(dataDstOrigin, 0, mAlphaY * mAlphaX * UNIT * SRC_UNIT);
    } else {
        dataDstOrigin = weightDst->host<int8_t>();
        memset(dataDstOrigin, 0, weightDst->size());
    }
    
    bool overflow = false;
    for (int oz = 0; oz < oc; ++oz) {
        int oz4 = oz / UNIT, ozRemain = oz % UNIT;
        for (int sz = 0; sz < ic; ++sz) {
            int sz4 = sz / SRC_UNIT, szRemain = sz % SRC_UNIT;
            auto dataSrcZ = weightSrc->host<int8_t>() + (ic * oz + sz) * kernelY * kernelX;
            auto dataDstZ = dataDstOrigin + oz4 * dstOZStep + sz4 * dstSZStep + ozRemain * SRC_UNIT + szRemain;
            for (int i = 0; i < kernelX; ++i) {
                if (kernelY != 1) {
                    overflow |= transFuncY(dataSrcZ + i, cache.get() + i, kernelX, kernelX);
                } else {
                    cache.get()[i] = dataSrcZ[i];
                }
            }
            int yLen = (kernelY == 1 ? 1 : mAlphaY);
            for (int i = 0; i < yLen; ++i) {
                if (kernelX != 1) {
                    overflow |= transFuncX(cache.get() + i * kernelX, dataDstZ + i * dstYStep, 1, dstXStep);
                } else {
                    dataDstZ[i * dstYStep] = cache.get()[i * kernelX];
                }
            }
        }
    }
    return !overflow;
}
// when overflow occur, return true
bool WinogradInt8Helper::weightOverflow(const Tensor* weight, int unitY, int unitX, const Convolution2DCommon* common, const CoreInt8Functions* core) {
    WinogradInt8Helper helper(unitY, unitX, common, core);
    return !(helper.transformWeight(weight, nullptr));
}
// when overflow occur or not support, return true
bool WinogradInt8Helper::featureOverflow(const Tensor* input, int alphaY, int alphaX) {
    std::map<int, std::pair<int8_t, int8_t>> limit2D = {
#ifdef MNN_USE_SSE
        {4, {-32, 31}} // int6
#else
        {4, {-64, 63}} // int6
#endif
    }, limit1D = {
#ifdef MNN_USE_SSE
        {4, {-64, 63}} // int7
#else
        {4, {-128, 127}} // int7
#endif
    };
    auto quantAttr = TensorUtils::getDescribe(input)->quantAttr;
    if (quantAttr == nullptr) {
        MNN_ERROR("Tensor quantAttr should not be nullptr\n");
        return true;
    }
    auto iter = limit2D.end();
    if (alphaY == 1 || alphaX == 1) {
        iter = limit1D.find(ALIMAX(alphaY, alphaX));
    } else if (alphaY == alphaX) {
        iter = limit2D.find(alphaY);
    }
    
    bool overflow = (quantAttr->min < iter->second.first || quantAttr->max > iter->second.second);
    return overflow;
}

}
