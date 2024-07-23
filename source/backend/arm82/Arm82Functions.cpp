#if defined(__ANDROID__) || defined(__aarch64__)
#include <math.h>
#include <float.h>
#include "Arm82Functions.hpp"
#include "Arm82OptFunc.hpp"
#include "Arm82WinogradOptFunc.hpp"
#include "Arm82Vec.hpp"
#include "Arm82Binary.hpp"
#include "Arm82Unary.hpp"
#include "Arm82Relu.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/CPUPool.hpp"
#include "backend/cpu/CPURuntime.hpp"

#define FLOAT FLOAT16
#define PACK 8
using Vec = MNN::Math::Vec<FLOAT16, 8>;

#include "backend/cpu/GridSampler.hpp"

#if defined(MNN_USE_NEON)
#include <arm_neon.h>
#endif

extern "C" {
// (UP_DIV(l,8), e, 8) -> (UP_DIV(e,eP), l, eP)
void Arm82MNNPackForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el);

// C(UP_DIV(h,8), e, h8) = B(UP_DIV(h,hP), l, hP) * A(l, eP), hP = 24
// parameter: [aStride, l, h, cStride, bExtraStride]
// aStride in parameter is deprecated (useless), but for code clean, just retain it
void MNNPackedMatMulFP16(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias);

// C(UP_DIV(h,8), e, h8) = B(UP_DIV(h,hP), l, hP) * A(l, e), hP = 24, e >= 1
// parameter: [aStride, l, h, cStride, bExtraStride]
void MNNPackedMatMulRemainFP16(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);

#ifdef MNN_LOW_MEMORY
void MNNPackedMatMulFP16_int4(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
void MNNPackedMatMulRemainFP16_int4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
void MNNPackedMatMulFP16_int8(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
void MNNPackedMatMulRemainFP16_int8(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);

void MNNAbsMaxFP16(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack);
void MNNQuantScaleFP16(float* sum, float* absmax, float* quant_scale, float* dequant_scale, size_t thread, size_t batch);
void MNNDynamicQuantFP16(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack);
void MNNQuantSumFP16(float* sum, const float* dequant_scale, size_t thread, size_t batch);
#endif
#if defined(__aarch64__)
void CountMinMaxValue_FP16(float* source, float* minVal, float* maxVal, size_t sizeQuad);
void MNNSumByAxisLForMatmul_A_ARM86(float* dest, int8_t* source, const float* dequantScale, ssize_t realDstCount, SumByAxisParams sumParams);
void MNNSumByAxisLForMatmul_A_ARM82(float* dest, int8_t* source, const float* dequantScale, ssize_t realDstCount, SumByAxisParams sumParams);
#endif
void MNNConvDwF23MulTransUnitFP16(FLOAT16 **cacheLine, const FLOAT16 *weight, FLOAT16 *dest, size_t ow);

void MNNConvDwF23SourceTransUnitFP16(const FLOAT16 *source, FLOAT16 *dest, size_t unit);

void MNNConvRunForLineDepthwiseFP16(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height, size_t srcHStep, size_t dstHStep);
}


namespace MNN {

static void MNNMatrixAddFP16(FLOAT16* C, const FLOAT16* A, const FLOAT16* B, size_t widthC8, size_t cStride, size_t aStride, size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y, b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC8; ++x) {
            vst1q_f16(c + x * 8, vaddq_f16(vld1q_f16(a + x * 8), vld1q_f16(b + x * 8)));
        }
    }
}
static void MNNMatrixSubFP16(FLOAT16* C, const FLOAT16* A, const FLOAT16* B, size_t widthC8, size_t cStride, size_t aStride, size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y, b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC8; ++x) {
            vst1q_f16(c + x * 8, vsubq_f16(vld1q_f16(a + x * 8), vld1q_f16(b + x * 8)));
        }
    }
}
#if defined(__aarch64__)
static void ARM82CountMinMaxValue(float* source, float* minVal, float* maxVal, size_t size) {
    if (size % 8 == 0) {
        CountMinMaxValue_FP16(source, minVal, maxVal, size / 8);
    } else {
        auto remain = size - 8 * (size / 8);
        auto max_ = ((__fp16*)source)[0];
        auto min_ = max_;
        if (size >= 8) {
            CountMinMaxValue_FP16(source, minVal, maxVal, size / 8);
            max_ = ((__fp16*)maxVal)[0];
            min_ = ((__fp16*)minVal)[0];
        }
        if (remain > 0) {
            int16_t tmp[8] = {0};
            auto srcRemain = reinterpret_cast<uint8_t*>(source) + 8 * (size / 8) * 2;
            ::memcpy(tmp, srcRemain, remain * 2);
            CountMinMaxValue_FP16((float*)tmp, (float*)tmp, (float*)((uint8_t*)tmp + 2), 1);
            max_ = ALIMAX(((__fp16*)tmp)[1], max_);
            min_ = ALIMIN(((__fp16*)tmp)[0], min_);
        }
        reinterpret_cast<__fp16*>(minVal)[0] = min_;
        reinterpret_cast<__fp16*>(maxVal)[0] = max_;
    }
}
#endif

static void Arm82MNNPackForMatMul_B(float* destC, const float* sourceC, size_t h, size_t l, bool transpose) {
    auto dest = (int16_t*)destC;
    auto source = (int16_t*)sourceC;
    int ePack, lPack, hPack;
    Arm82MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    auto hP = (int)h / hPack;
    auto hR = (int)hP * hPack;
    if (hR != h) {
        ::memset(dest, 0, UP_DIV(h, hPack) * hPack * l * sizeof(FLOAT16));
    }
    if (!transpose) {
        for (int y = 0; y < hP; ++y) {
            auto destY = dest + y * hPack * l;
            auto sourceY = source + y * hPack;
            for (int x = 0; x < l; ++x) {
                ::memcpy(destY + hPack * x, sourceY + x * h, hPack * sizeof(FLOAT16));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * hPack * l;
            auto sourceY = source + hP * hPack;
            for (int x = 0; x < l; ++x) {
                ::memcpy(destY + hPack * x, sourceY + x * h, hRemain * sizeof(FLOAT16));
            }
        }
        return;
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < l; ++x) {
            dest[(y / hPack * l + x) * hPack + y % hPack] = source[y * l + x];
        }
    }
}

static void MNNScaleAndAddBiasFP16(FLOAT16* dst, const FLOAT16* src, const FLOAT16* bias, const FLOAT16* alpha, size_t planeNumber,
                        size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        FLOAT16* dstZ         = dst + planeNumber * 8 * z;
        const FLOAT16* srcZ   = src + planeNumber * 8 * z;
        auto biasZ = vld1q_f16(bias + 8 * z), alphaZ = vld1q_f16(alpha + 8 * z);
        for (int p = 0; p < planeNumber; ++p) {
            FLOAT16* dstX       = dstZ + 8 * p;
            const FLOAT16* srcX = srcZ + 8 * p;
            auto res = vaddq_f16(vmulq_f16(vld1q_f16(srcX), alphaZ), biasZ);
            vst1q_f16(dstX, res);
        }
    }
}

static void MNNGridSampleComputeCordFP16(FLOAT16* dst, const FLOAT16* src, size_t inH, size_t inW, size_t outH, size_t outW, size_t stride, bool alignCorners) {
    float16x8_t zero = vdupq_n_f16(0);
    float16x8_t one = vdupq_n_f16(1);
    float16x8_t half = vdupq_n_f16(0.5f);
    float16x8_t a = alignCorners ? one : zero;
    float16x8_t b = alignCorners ? zero : one;
    float16x8_t inW_sub_a = vsubq_f16(vdupq_n_f16(inW), a);
    float16x8_t inH_sub_a = vsubq_f16(vdupq_n_f16(inH), a);

    int area = outH * outW;
    int areaC8 = area / 8;
    int areaRemain = area - areaC8 * 8;
    for (int i = 0; i < areaC8; ++i) {
        auto cordH = vld2q_f16(src);
        // float16x8_t x = cordH.val[0];
        // float16x8_t y = cordH.val[1];
        cordH.val[0] = vmulq_f16(half, vsubq_f16(vmulq_f16(vaddq_f16(one, cordH.val[0]), inW_sub_a), b));
        cordH.val[1] = vmulq_f16(half, vsubq_f16(vmulq_f16(vaddq_f16(one, cordH.val[1]), inH_sub_a), b));
        vst2q_f16(dst, cordH);

        src += 16;
        dst += 16;
    }
    if (areaRemain == 0) {
        return;
    }

    // areaRemain
    FLOAT16 tempDst[16];
    ::memcpy(tempDst, src, areaRemain * 2 * sizeof(int16_t));
    auto cordH = vld2q_f16(tempDst);
    cordH.val[0] = vmulq_f16(half, vsubq_f16(vmulq_f16(vaddq_f16(one, cordH.val[0]), inW_sub_a), b));
    cordH.val[1] = vmulq_f16(half, vsubq_f16(vmulq_f16(vaddq_f16(one, cordH.val[1]), inH_sub_a), b));
    vst2q_f16(tempDst, cordH);
    ::memcpy(dst, tempDst, areaRemain * 2 * sizeof(int16_t));
}

static void MNNGridSampleComputeCord3DFp16(FLOAT* dst, const FLOAT* src, size_t inD, size_t inH, size_t inW, size_t outD, size_t outH, size_t outW, size_t strideD, size_t strideH, bool alignCorners) {
    float16x8_t zero = vdupq_n_f16(0);
    float16x8_t one = vdupq_n_f16(1);
    float16x8_t half = vdupq_n_f16(0.5f);
    float16x8_t a = alignCorners ? one : zero;
    float16x8_t b = alignCorners ? zero : one;
    float16x8_t inW_sub_a = vsubq_f16(vdupq_n_f16(inW), a);
    float16x8_t inH_sub_a = vsubq_f16(vdupq_n_f16(inH), a);
    float16x8_t inD_sub_a = vsubq_f16(vdupq_n_f16(inD), a);
    size_t area = outH * outW * outD;
    size_t areaC8 = area / 8;
    size_t areaRemain = area - areaC8 * 8;

    for (int i = 0; i < areaC8; ++i) {
        auto cordH = vld3q_f16(src);
        // float16x8_t x = cordH.val[0];
        // float16x8_t y = cordH.val[1];
        cordH.val[0] = vmulq_f16(half, vsubq_f16(vmulq_f16(vaddq_f16(one, cordH.val[0]), inW_sub_a), b));
        cordH.val[1] = vmulq_f16(half, vsubq_f16(vmulq_f16(vaddq_f16(one, cordH.val[1]), inH_sub_a), b));
        cordH.val[2] = vmulq_f16(half, vsubq_f16(vmulq_f16(vaddq_f16(one, cordH.val[2]), inD_sub_a), b));
        vst3q_f16(dst, cordH);
        src += 24;
        dst += 24;
    }
    if (areaRemain == 0) {
        return;
    }

    // areaRemain
    FLOAT16 tempDst[24];
    ::memcpy(tempDst, src, areaRemain * 3 * sizeof(int16_t));
    auto cordH = vld3q_f16(tempDst);
    cordH.val[0] = vmulq_f16(half, vsubq_f16(vmulq_f16(vaddq_f16(one, cordH.val[0]), inW_sub_a), b));
    cordH.val[1] = vmulq_f16(half, vsubq_f16(vmulq_f16(vaddq_f16(one, cordH.val[1]), inH_sub_a), b));
    cordH.val[2] = vmulq_f16(half, vsubq_f16(vmulq_f16(vaddq_f16(one, cordH.val[2]), inD_sub_a), b));
    vst3q_f16(tempDst, cordH);
    ::memcpy(dst, tempDst, areaRemain * 3 * sizeof(int16_t));
}
static void MNNRoiPoolingMaxFP16(FLOAT16* dst, const FLOAT16* src, int hLen, int wLen, int iw) {
    Vec max = Vec(-65504.0f);
    for (int h = 0; h < hLen; h++, src += iw * 8) {
        for (int w = 0; w < wLen; w++) {
            Vec in = Vec::load(src + w * 8);
            max = Vec::max(max, in);
        }
    }
    Vec::save(dst, max);
}

static void MNNRoiAlignMaxFP16(FLOAT16* dst, const FLOAT16* src, const std::vector<std::vector<int>> &vecPos, const std::vector<std::vector<float>> &vecArea, int samplingRatioArea, int pooledHeight, int pooledWidth) {
    for (int h = 0; h < pooledHeight; ++h, dst += pooledWidth * 8) {
        int preCalcIdx = h * pooledWidth * samplingRatioArea;
        for (int w = 0; w < pooledWidth; ++w) {
            Vec res = Vec(-65504.0f);
            for (int i = 0; i < samplingRatioArea; ++i) {
                const std::vector<int>& pos    = vecPos[preCalcIdx];
                const std::vector<float>& area = vecArea[preCalcIdx];

                Vec val0 = Vec::load(src + pos[0] * 8);
                Vec val1 = Vec::load(src + pos[1] * 8);
                Vec val2 = Vec::load(src + pos[2] * 8);
                Vec val3 = Vec::load(src + pos[3] * 8);
                Vec mla  = val0 * area[0];
                mla       = Vec::fma(mla, val1, area[1]);
                mla       = Vec::fma(mla, val2, area[2]);
                mla       = Vec::fma(mla, val3, area[3]);
                res       = Vec::max(res, mla);
                preCalcIdx++;
            }
            Vec::save(dst + w * 8, res);
        }
    }
}

static void MNNRoiAlignAvgFP16(FLOAT16* dst, const FLOAT16* src, const std::vector<std::vector<int>> &vecPos, const std::vector<std::vector<float>> &vecArea, int samplingRatioArea, int pooledHeight, int pooledWidth) {
    float invSamplingCnt = 1.f / samplingRatioArea;
    for (int h = 0; h < pooledHeight; ++h, dst += pooledWidth * 8) {
        int preCalcIdx = h * pooledWidth * samplingRatioArea;
        for (int w = 0; w < pooledWidth; ++w) {
            Vec res = Vec(0.f);
            for (int i = 0; i < samplingRatioArea; ++i) {
                const std::vector<int>& pos    = vecPos[preCalcIdx];
                const std::vector<float>& area = vecArea[preCalcIdx];

                Vec val0 = Vec::load(src + pos[0] * 8);
                Vec val1 = Vec::load(src + pos[1] * 8);
                Vec val2 = Vec::load(src + pos[2] * 8);
                Vec val3 = Vec::load(src + pos[3] * 8);
                Vec mla  = val0 * area[0];
                mla       = Vec::fma(mla, val1, area[1]);
                mla       = Vec::fma(mla, val2, area[2]);
                mla       = Vec::fma(mla, val3, area[3]);
                res       += mla;
                preCalcIdx++;
            }
            res = res * invSamplingCnt;
            Vec::save(dst + w * 8, res);
        }
    }
}

static void MNNCopyC8WithStrideFP16(const FLOAT16* source, FLOAT16* dest, size_t srcStride, size_t dstStride, size_t count) {
    using Vec = MNN::Math::Vec<FLOAT16, 8>;
    for (int i = 0; i < count; ++i) {
        auto srcPtr = source + i * srcStride;
        auto dstPtr = dest + i * dstStride;
        Vec::save(dstPtr, Vec::load(srcPtr));
    }
}

static void MNNAddC8WithStrideFP16(const FLOAT16* source, FLOAT16* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto srcPtr = source + i * srcStride;
        auto dstPtr = dest + i * dstStride;
        auto value = Vec::load(dstPtr) + Vec::load(srcPtr);
        Vec::save(dstPtr, value);
    }
}

static void MNNAxByClampBroadcastC8FP16(float* CF, const float* AF, const float* BF, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters) {
    auto C = (FLOAT16*)CF;
    auto A = (FLOAT16*)AF;
    auto B = (FLOAT16*)BF;
    using Vec = MNN::Math::Vec<FLOAT16, 8>;
    auto minF = Vec(parameters[2]);
    auto maxF = Vec(parameters[3]);
    auto beta = Vec(parameters[1]);
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + 8 * y;
        auto bv = Vec::load(b);
        auto c = C + cStride * y;
        for (int x = 0; x < width; ++x) {
            auto av = Vec::load(a + 8 * x);
            auto cv = av + bv * beta;
            cv = Vec::min(cv, maxF);
            cv = Vec::max(cv, minF);
            Vec::save(c + 8 * x, cv);
        }
    }
}

void ARM82MultiAndDestTransformCommon(FLOAT16 **cacheLine, const FLOAT16 *weight, FLOAT16 *dest, int cacheLineSize, int ow, const float* bias, const float* parameters) {
    constexpr int pack = 8;
    int unit = ow / 2;
    auto biasF = Vec::load((const float16_t*)bias);
    auto minF = Vec(parameters[2]);
    auto maxF = Vec(parameters[3]);
    MNN_ASSERT(cacheLineSize >= 1);
    for (int x = 0; x < unit; ++x) {
        int offset = 4 * pack * x, i = 0;
        Vec m0 = Vec::load(weight + i * 4 * pack) * Vec::load(cacheLine[i] + offset);
        Vec m1 = Vec::load(weight + (i * 4 + 1) * pack) * Vec::load(cacheLine[i] + offset + pack * 1);
        Vec m2 = Vec::load(weight + (i * 4 + 2) * pack) * Vec::load(cacheLine[i] + offset + pack * 2);
        Vec m3 = Vec::load(weight + (i * 4 + 3) * pack) * Vec::load(cacheLine[i] + offset + pack * 3);
        for (i = 1; i < cacheLineSize; ++i) {
            m0 = m0 + Vec::load(weight + i * 4 * pack) * Vec::load(cacheLine[i] + offset);
            m1 = m1 + Vec::load(weight + (i * 4 + 1) * pack) * Vec::load(cacheLine[i] + offset + pack * 1);
            m2 = m2 + Vec::load(weight + (i * 4 + 2) * pack) * Vec::load(cacheLine[i] + offset + pack * 2);
            m3 = m3 + Vec::load(weight + (i * 4 + 3) * pack) * Vec::load(cacheLine[i] + offset + pack * 3);
        }
        auto o0 = m0 + m1 + m2 + biasF;
        auto o1 = m1 - m2 + m3 + biasF;
        o0 = Vec::min(maxF, o0);
        o1 = Vec::min(maxF, o1);
        o0 = Vec::max(minF, o0);
        o1 = Vec::max(minF, o1);
        Vec::save(dest + (2 * x + 0) * pack, o0);
        Vec::save(dest + (2 * x + 1) * pack, o1);
    }
    if (unit * 2 < ow) {
        int offset = 4 * pack * unit, i = 0;
        Vec m0 = Vec::load(weight + i * 4 * pack) * Vec::load(cacheLine[i] + offset);
        Vec m1 = Vec::load(weight + (i * 4 + 1) * pack) * Vec::load(cacheLine[i] + offset + pack);
        Vec m2 = Vec::load(weight + (i * 4 + 2) * pack) * Vec::load(cacheLine[i] + offset + pack * 2);
        for (i = 1; i < cacheLineSize; ++i) {
            m0 = m0 + Vec::load(weight + i * 4 * pack) * Vec::load(cacheLine[i] + offset);
            m1 = m1 + Vec::load(weight + (i * 4 + 1) * pack) * Vec::load(cacheLine[i] + offset + pack);
            m2 = m2 + Vec::load(weight + (i * 4 + 2) * pack) * Vec::load(cacheLine[i] + offset + pack * 2);
        }
        auto o0 = m0 + m1 + m2 + biasF;
        o0 = Vec::min(maxF, o0);
        o0 = Vec::max(minF, o0);
        Vec::save(dest + 2 * unit * pack, o0);
    }
}
// unit: winograd unit (output is w/2)
void ARM82SourceTransformCommon(const FLOAT16 *source, FLOAT16 *dest, int unit, int iw, int pad, int su, int eu) {
    constexpr int pack = 8; // float16x8
    for (int x = 0; x < su; ++x) {
        auto dstX = dest + 4 * pack * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;
        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);
        Vec v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec::load(source + pack * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];
        Vec::save(dstX + pack * 0, m0);
        Vec::save(dstX + pack * 1, m1);
        Vec::save(dstX + pack * 2, m2);
        Vec::save(dstX + pack * 3, m3);
    }
    MNNConvDwF23SourceTransUnitFP16(source + pack * (su * 2 - pad), dest + 4 * pack * su, eu - su);
    for (int x = eu; x < unit; ++x) {
        auto dstX = dest + 4 * pack * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;
        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);
        Vec v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec::load(source + pack * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];
        Vec::save(dstX + pack * 0, m0);
        Vec::save(dstX + pack * 1, m1);
        Vec::save(dstX + pack * 2, m2);
        Vec::save(dstX + pack * 3, m3);
    }
}

void ARM82StrassenMerge(FLOAT16* c11, FLOAT16* c12, FLOAT16* c21, FLOAT16* c22, FLOAT16* xAddr,
                          size_t cStride, size_t eSub, size_t hSub) {
    const int pack = 8;
    for (int y = 0; y < hSub; ++y) {
        auto c11Y = c11 + y * cStride;
        auto c12Y = c12 + y * cStride;
        auto c22Y = c22 + y * cStride;
        auto c21Y = c21 + y * cStride;
        auto xY = xAddr + y * eSub * pack;
        for (int x = 0; x < eSub; ++x) {
            auto xv = vld1q_f16(xY + x * pack);
            auto c21v = vld1q_f16(c21Y + x * pack);
            auto c11v = vld1q_f16(c11Y + x * pack);
            auto c22v = vld1q_f16(c22Y + x * pack);
            auto c12v = vld1q_f16(c12Y + x * pack);
            c12v = c12v + xv;
            c21v = c12v + c21v;
            c12v = c22v + c12v;
            c22v = c22v + c21v;
            c12v = c11v + c12v;
            vst1q_f16(c12Y + x * pack, c12v);
            vst1q_f16(c22Y + x * pack, c22v);
            vst1q_f16(c21Y + x * pack, c21v);
        }
    }
}

void MNNUnpackTransposeInt16C8(int16_t* dst, const int16_t* src, size_t area, size_t depth, int32_t* areaOffset) {
    int srcAreaOffset = areaOffset[0];
    int c      = (int)depth;
    int cDiv4  = c / 8;
    int cAlign = cDiv4 * 8;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = src + hi * 8;
        auto dstHeight = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            vst1q_s16(dstHeight + ci * 8, vld1q_s16(srcHeight + 8 * ci * srcAreaOffset));
        }
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + srcAreaOffset * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * 8;
        auto dstHeight = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNPackTransposeInt16C8(int16_t* dst, const int16_t* src, size_t area, size_t depth, int32_t* areaOffset) {
    if (depth == 8) {
        ::memcpy(dst, src, area * depth * sizeof(int16_t));
        return;
    }
    int dstAreaOffset = areaOffset[1];
    int c      = (int)depth;
    int cDiv4  = c / 8;
    int cAlign = cDiv4 * 8;
    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = (src + hi * c);
        auto dstHeight = (dst + hi * 8);
        for (int ci = 0; ci < cDiv4; ++ci) {
            vst1q_s16(dstHeight + ci * dstAreaOffset * 8, vld1q_s16(srcHeight + 8 * ci));
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + dstAreaOffset * cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * c;
        auto dstHeight = dstAlign + hi * 8;
        for (int i = 0; i < 8; ++i) {
            dstHeight[i] = 0;
        }
        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

static void MNNConvRunForUnitDepthWiseFP16(float* dst, const float* src, const float* weight, size_t fw, size_t fh,
                                           size_t weight_y_step, size_t dilateX_step, size_t dilateY_step) {
    int fx, fy;
    Vec dstValue(0.0f);
    auto src_z    = (const FLOAT16*)src;
    auto weight_z = (const FLOAT16*)weight;
    for (fy = 0; fy < fh; ++fy) {
        auto src_y    = src_z + fy * dilateY_step;
        auto weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            auto weight_x = weight_y + 8 * fx;
            auto src_x    = src_y + fx * dilateX_step;
            dstValue = dstValue + Vec::load(src_x) * Vec::load(weight_x);
        }
    }
    Vec::save((FLOAT16*)dst, dstValue);
}

static void _MNNDeconvRunForUnitDepthWise(const FLOAT16* dst, FLOAT16* src, const FLOAT16* weight, size_t fw, size_t fh,
                                  size_t weight_y_step, size_t dilateX_step, size_t dilateY_step) {
    int fx, fy;
    auto src_z          = src;
    auto weight_z = weight;
    Vec dstV           = Vec::load(dst);
    for (fy = 0; fy < fh; ++fy) {
        auto src_y          = src_z + fy * dilateY_step;
        auto weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            Vec weight_x = Vec::load(weight_y + 8 * fx);
            Vec src_x    = Vec::load(src_y + fx * dilateX_step);
            Vec::save(src_y + fx * dilateX_step, src_x + weight_x * dstV);
        }
    }
}
static void _MNNDeconvRunForLineDepthwise(const FLOAT16* dst, FLOAT16* src, const FLOAT16* weight, size_t width, size_t src_w_setup,
                                  size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) {
    int dx;
    for (dx = 0; dx < width; ++dx) {
        auto dst_x = dst + dx * 8;
        auto src_dx      = src + src_w_setup * dx;
        _MNNDeconvRunForUnitDepthWise(dst_x, src_dx, weight, fw, fh, fw * 8, dilateX_step, dilateY_step);
    }
}

static void _MNNComputeMatMulForH_1_FP16(const float* AF, const float* BF, float* CF, const float* biasPtrF, const MatMulParam* param, size_t tId) {
    auto A = (const FLOAT16*)AF;
    auto B = (const FLOAT16*)BF;
    auto C = (FLOAT16*)CF;
    auto biasPtr = (const FLOAT16*)biasPtrF;
    int e = param->e;
    int l = param->l;
    int numberThread = param->numberThread;
    float biasValue = 0.0f;
    if (nullptr != biasPtr) {
        biasValue = biasPtr[0];
    }
    if (param->ATranspose) {
        auto eC4 = e / 8;
        auto eR = e % 8;
        for (int y=tId; y<eC4; y+=numberThread) {
            Vec sumValue = Vec(biasValue);
            auto srcY = A + y * 8;
            for (int x=0; x<l; ++x) {
                sumValue = sumValue + Vec::load(srcY + x * e) * Vec(B[x]);
            }
            Vec::save(C + 8 * y, sumValue);
        }
        if (0 == tId && eR > 0) {
            Vec sumValue = Vec(biasValue);
            auto srcY = A + eC4 * 8;
            FLOAT16 AR[8];
            for (int x=0; x<l; ++x) {
                ::memcpy(AR, srcY + x * e, eR * sizeof(int16_t));
                sumValue = sumValue + Vec::load(AR) * Vec(B[x]);
            }
            FLOAT16 CR[8];
            Vec::save(CR, sumValue);
            ::memcpy(C + 8 * eC4, CR, eR * sizeof(int16_t));
        }
        return;
    }
    auto lC4 = l / 8;
    auto lR = l % 8;
    for (int y=tId; y<e; y+=numberThread) {
        Vec sumValue = Vec(biasValue);
        auto srcY = A + y * l;
        for (int x=0; x<lC4; ++x) {
            sumValue = sumValue + Vec::load(srcY + 8 * x) * Vec::load(B + 8 * x);
        }
        if (lR > 0) {
            FLOAT16 AR[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            FLOAT16 BR[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            ::memcpy(AR, srcY + lC4 * 8, lR * sizeof(int16_t));
            ::memcpy(BR, B + 8 * lC4, lR * sizeof(int16_t));
            sumValue = sumValue + Vec::load(AR) * Vec::load(BR);
        }
        float sumSingle = sumValue[0] + sumValue[1] + sumValue[2] + sumValue[3] + sumValue[4] + sumValue[5] + sumValue[6] + sumValue[7];
        C[y] = sumSingle;
    }
}

static void _MNNComputeMatMulForE_1_FP16(const float* AF, const float* BF, float* CF, const float* biasPtrF, const MatMulParam* param, size_t tId) {
    auto l = param->l;
    auto h = param->h;
    auto numberThread = param->numberThread;
    auto lC4 = l / 8;
    auto lR = l % 8;
    auto A = (const FLOAT16*)AF;
    auto B = (const FLOAT16*)BF;
    auto C = (FLOAT16*)CF;
    auto biasPtr = (const FLOAT16*)biasPtrF;
    if (param->BTranspose) {
        for (int y=tId; y<h; y+=numberThread) {
            Vec sumValue = Vec(0.0f);
            auto by = B + y * l;
            for (int x=0; x<lC4; ++x) {
                sumValue = sumValue + Vec::load(A + x * 8) * Vec::load(by + x * 8);
            }
            if (lR > 0) {
                FLOAT16 AR[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                FLOAT16 BR[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                ::memcpy(AR, A + lC4 * 8, lR * sizeof(int16_t));
                ::memcpy(BR, by + 8 * lC4, lR * sizeof(int16_t));
                sumValue = sumValue + Vec::load(AR) * Vec::load(BR);
            }
            float sumRemain = sumValue[0] + sumValue[1] + sumValue[2] + sumValue[3] + sumValue[4] + sumValue[5] + sumValue[6] + sumValue[7];
            if (nullptr != biasPtr) {
                sumRemain += biasPtr[y];
            }
            C[y] = sumRemain;
        }
    } else {
        auto hC4 = h / 8;
        auto hR = h % 8;
        for (int y=tId; y<hC4; y+=numberThread) {
            auto bs = B + 8 * y;
            Vec sumValue = Vec(0.0f);
            if (biasPtr != nullptr) {
                sumValue = Vec::load(biasPtr + 8 * y);
            }
            auto srcY = A + y * l * 8;
            for (int x=0; x<l; ++x) {
                sumValue = sumValue + Vec(A[x]) * Vec::load(bs + h * x);
            }
            Vec::save(C + 8 * y, sumValue);
        }
        if (tId == 0 && hR > 0) {
            auto bs = B + 8 * hC4;
            Vec sumValue = Vec(0.0f);
            if (biasPtr != nullptr) {
                FLOAT16 biasTemp[8];
                ::memcpy(biasTemp, biasPtr + 8 * hC4, hR * sizeof(int16_t));
                sumValue = Vec::load(biasTemp);
            }
            auto srcY = A + 8 * hC4 * l;
            FLOAT16 bTemp[8];
            for (int x=0; x<l; ++x) {
                ::memcpy(bTemp, bs + h * x, hR * sizeof(int16_t));
                sumValue = sumValue + Vec(A[x]) * Vec::load(bTemp);
            }
            FLOAT16 cTemp[8];
            Vec::save(cTemp, sumValue);
            ::memcpy(C + 8 * hC4, cTemp, hR * sizeof(int16_t));
        }
    }
}

static CoreFunctions* gInstance = nullptr;

bool Arm82Functions::init() {
    using Vec = MNN::Math::Vec<FLOAT16, 8>;
    auto origin = MNNGetCoreFunctions();
#define FUNC_PTR_ASSIGN(dst, src) dst = (decltype(dst))(src)
    gInstance = new CoreFunctions;

    FUNC_PTR_ASSIGN(gInstance->MNNFp32ToLowp, MNNQuantizeFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNLowpToFp32, MNNDequantizeFP16);
    gInstance->bytes = 2;

    // Packed
    gInstance->pack = 8;
    FUNC_PTR_ASSIGN(gInstance->MNNPackCUnit, MNNPackC8FP16);
    FUNC_PTR_ASSIGN(gInstance->MNNUnpackCUnit, MNNUnPackC8FP16);
    FUNC_PTR_ASSIGN(gInstance->MNNPackCUnitTranspose, MNNPackTransposeInt16C8);
    FUNC_PTR_ASSIGN(gInstance->MNNUnpackCUnitTranspose, MNNUnpackTransposeInt16C8);
    FUNC_PTR_ASSIGN(gInstance->MNNConvRunForUnitDepthWise, MNNConvRunForUnitDepthWiseFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNConvRunForLineDepthwise, MNNConvRunForLineDepthwiseFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNAxByClampBroadcastUnit, MNNAxByClampBroadcastC8FP16);
    FUNC_PTR_ASSIGN(gInstance->MNNConvDwF23MulTransUnit, MNNConvDwF23MulTransUnitFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNSourceTransformCommonF23, ARM82SourceTransformCommon);
    FUNC_PTR_ASSIGN(gInstance->MNNMultiAndDestTransformCommon23, ARM82MultiAndDestTransformCommon);
    FUNC_PTR_ASSIGN(gInstance->MNNMatrixSub, MNNMatrixSubFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNMatrixAdd, MNNMatrixAddFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNStrassenMergeCFunction, ARM82StrassenMerge);
#ifdef MNN_LOW_MEMORY
    FUNC_PTR_ASSIGN(gInstance->MNNDynamicUpdateConvBiasScale, origin->MNNDynamicUpdateConvBiasScale);
#endif
    gInstance->penalty = 2.0f;
    FUNC_PTR_ASSIGN(gInstance->MNNScaleAndAddBias, MNNScaleAndAddBiasFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNGridSampleComputeCord, MNNGridSampleComputeCordFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNGridSampleInterp, MNNGridSampleInterp);
    FUNC_PTR_ASSIGN(gInstance->MNNGridSampleInterpGrad, MNNGridSampleInterpGrad);
    FUNC_PTR_ASSIGN(gInstance->MNNGridSampleComputeCord3D, MNNGridSampleComputeCord3DFp16);
    FUNC_PTR_ASSIGN(gInstance->MNNGridSampleInterp3D, MNNGridSampleInterp3D);
    FUNC_PTR_ASSIGN(gInstance->MNNRoiPoolingMax, MNNRoiPoolingMaxFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNRoiAlignMax, MNNRoiAlignMaxFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNRoiAlignAvg, MNNRoiAlignAvgFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNCopyC4WithStride, MNNCopyC8WithStrideFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNAddC4WithStride, MNNAddC8WithStrideFP16);

    // MatMul
    FUNC_PTR_ASSIGN(gInstance->MNNPackedMatMul, MNNPackedMatMulFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNPackedMatMulRemain, MNNPackedMatMulRemainFP16);
#if defined(__aarch64__)
#ifdef MNN_LOW_MEMORY
    // Weight Dequant Gemm Kernels
    FUNC_PTR_ASSIGN(gInstance->MNNPackedMatMul_int4, MNNPackedMatMulFP16_int4);
    FUNC_PTR_ASSIGN(gInstance->MNNPackedMatMulRemain_int4, MNNPackedMatMulRemainFP16_int4);
    FUNC_PTR_ASSIGN(gInstance->MNNPackedMatMul_int8, MNNPackedMatMulFP16_int8);
    FUNC_PTR_ASSIGN(gInstance->MNNPackedMatMulRemain_int8, MNNPackedMatMulRemainFP16_int8);
    // Dynamic Qaunt Helper Functions
    FUNC_PTR_ASSIGN(gInstance->MNNAbsMax, MNNAbsMaxFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNQuantScale, MNNQuantScaleFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNDynamicQuant, MNNDynamicQuantFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNQuantSum, MNNQuantSumFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNCountMaxMinValue, ARM82CountMinMaxValue);
    // Dynamic Quant Gemm Kernels.
    gInstance->supportFp16arith = origin->supportFp16arith;
    gInstance->supportSDot = origin->supportSDot;
    gInstance->supportI8mm = origin->supportI8mm;
#endif
    if (gInstance->supportSDot) {
        FUNC_PTR_ASSIGN(gInstance->MNNSumByAxisLForMatmul_A, MNNSumByAxisLForMatmul_A_ARM82);
    }
    if (gInstance->supportI8mm) {
        FUNC_PTR_ASSIGN(gInstance->MNNSumByAxisLForMatmul_A, MNNSumByAxisLForMatmul_A_ARM86);
    }
#endif
    FUNC_PTR_ASSIGN(gInstance->MNNPackC4ForMatMul_A, Arm82MNNPackForMatMul_A);
    FUNC_PTR_ASSIGN(gInstance->MNNGetMatMulPackMode, Arm82MNNGetMatMulPackMode);
    FUNC_PTR_ASSIGN(gInstance->MNNPackForMatMul_B, Arm82MNNPackForMatMul_B);
    gInstance->MNNComputeMatMulForH_1 = _MNNComputeMatMulForH_1_FP16;
    gInstance->MNNComputeMatMulForE_1 = _MNNComputeMatMulForE_1_FP16;

    FUNC_PTR_ASSIGN(gInstance->chooseWinoSourceTransformPack, Arm82WinogradFunction::chooseWinoSourceTransformPack);
    FUNC_PTR_ASSIGN(gInstance->chooseWinoSourceUnrollTransform, Arm82WinogradFunction::chooseSourceUnrollTransform);
    FUNC_PTR_ASSIGN(gInstance->chooseWinoDestUnrollTransform, Arm82WinogradFunction::chooseWinoDestUnrollTransform);

    gInstance->MNNDeconvRunForLineDepthwise = (decltype(gInstance->MNNDeconvRunForLineDepthwise))_MNNDeconvRunForLineDepthwise;
    gInstance->MNNDeconvRunForUnitDepthWise = (decltype(gInstance->MNNDeconvRunForUnitDepthWise))_MNNDeconvRunForUnitDepthWise;

    // Binary and Unary
    gInstance->MNNSelectBinaryFunctionForFloat = Arm82BinaryFloat::select;
    gInstance->MNNSelectUnaryFunctionForFloat = Arm82Unary::select;

    // Relu with slope
    gInstance->MNNReluWithSlopeChannel = Arm82Relu::reluWithSlopeChannel;

    gInstance->MNNPoolingMax = (decltype(gInstance->MNNPoolingMax))(poolingMax<float16_t, Vec, 8, -65535>);
    gInstance->MNNPoolingAvg = (decltype(gInstance->MNNPoolingAvg))(poolingAvg<float16_t, Vec, 8>);
    return true;
}

CoreFunctions* Arm82Functions::get() {
    return gInstance;
}
};
#endif
