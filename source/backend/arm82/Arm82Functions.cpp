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

#ifdef MNN_CPU_WEIGHT_DEQUANT_GEMM
void MNNPackedMatMulFP16_int4(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
void MNNPackedMatMulRemainFP16_int4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
void MNNPackedMatMulFP16_int8(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
void MNNPackedMatMulRemainFP16_int8(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
#endif

#ifdef MNN_LOW_MEMORY
void MNNAbsMaxFP16_Pack8(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack);
void MNNAbsMaxFP16_Pack4(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack);
void MNNQuantScaleFP16(float* sum, float* absmax, float* quant_scale, float* dequant_scale, size_t thread, size_t batch);
void MNNDynamicQuantFP16_Pack8(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack);
void MNNDynamicQuantFP16_Pack4(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack);
void MNNQuantSumFP16(float* sum, const float* dequant_scale, size_t thread, size_t batch);
void MNNGeneralIm2col_Arm82(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el, int32_t LP, int32_t pack);
void MNNGeneralIm2col_Arm86(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el, int32_t LP, int32_t pack);
#endif
#if defined(__aarch64__)
void CountMinMaxValue_FP16(float* source, float* minVal, float* maxVal, size_t sizeQuad);
void MNNDepthwiseConvFastKernelFP16(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                    size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                    size_t srcHStep, size_t dstHStep, const float* bias, const float* parameters);
#endif

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
                sumValue = Vec::fma(sumValue, Vec::load(A + x * 8), Vec::load(by + x * 8));
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
        auto hC16 = hC4 / 4;
        auto hC4R = hC4 % 4;
        for (int y=tId; y<hC16; y+=numberThread) {
            auto biasP = biasPtr + 8 * 4 * y;
            auto bs = B + 8 * 4 * y;
            Vec s0 = Vec(0.0f);
            Vec s1 = Vec(0.0f);
            Vec s2 = Vec(0.0f);
            Vec s3 = Vec(0.0f);
            if (biasPtr != nullptr) {
                s0 = Vec::load(biasP + 8 * 0);
                s1 = Vec::load(biasP + 8 * 1);
                s2 = Vec::load(biasP + 8 * 2);
                s3 = Vec::load(biasP + 8 * 3);
            }
            auto srcY = A + y * l * 8 * 4;
            for (int x=0; x<l; ++x) {
                auto a = Vec(A[x]);
                s0 = Vec::fma(s0, a, Vec::load(bs + h * x + 0 * 8));
                s1 = Vec::fma(s1, a, Vec::load(bs + h * x + 1 * 8));
                s2 = Vec::fma(s2, a, Vec::load(bs + h * x + 2 * 8));
                s3 = Vec::fma(s3, a, Vec::load(bs + h * x + 3 * 8));
            }
            Vec::save(C + 4 * 8 * y + 8 * 0, s0);
            Vec::save(C + 4 * 8 * y + 8 * 1, s1);
            Vec::save(C + 4 * 8 * y + 8 * 2, s2);
            Vec::save(C + 4 * 8 * y + 8 * 3, s3);
        }

        for (int y=hC16*4+tId; y<hC4; y+=numberThread) {
            auto bs = B + 8 * y;
            Vec sumValue = Vec(0.0f);
            if (biasPtr != nullptr) {
                sumValue = Vec::load(biasPtr + 8 * y);
            }
            auto srcY = A + y * l * 8;
            for (int x=0; x<l; ++x) {
                sumValue = Vec::fma(sumValue, Vec(A[x]), Vec::load(bs + h * x));
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

template<int EP, int LP>
static void _Arm82MNNPackC4ForMatMul_A(int8_t* destOrigin, int8_t const** sourceGroup, const int32_t* info, const int32_t* el) {
    const int pack = 8;
    int number = info[0];
    int eReal = info[1];
    int xStride = info[3];
    int xS4 = xStride * pack / sizeof(int32_t);
    int PUNIT = pack / LP;
    int FLOATPACK = pack / sizeof(int32_t);
    int eOutsideStride = info[2] / sizeof(int32_t);
    int eDest = EP;
    int realDstCount = info[4];
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        int eC = eOffset / EP;
        int eR = eOffset % EP;
        int eS = eDest - eR;
        bool lastBag = false;
        int eOutsideStride4LastBag = eOutsideStride;
        if (realDstCount % EP > 0) {
            int jobsE = realDstCount - eOffset - e;
            if (jobsE == 0 || (jobsE < (realDstCount % EP))) {
                lastBag = true;
            }
        }
        auto source = (int32_t*)sourceGroup[n];
        auto dest = (int32_t*)(destOrigin + eC * info[2] + eR * LP + lOffset * EP);
        //printf("e=%d, l=%d, eOffset=%d, lOffset=%d, eDest=%d\n", e, l, eOffset, lOffset, eDest);
        l = l / 4; // Use float instead of int8 * 4
        if (lastBag && e + eR < EP) {
            int elast = ALIMAX(eR + e, realDstCount % EP);
            dest = (int32_t*)(destOrigin + lOffset * elast + eC * info[2] + eR * LP);
        }
        int offsetLC = lOffset / 4;
        for (int x = 0; x < l; ++x) {
            int eRemain = e;
            auto xR                  = x % PUNIT;
            auto xC                  = x / PUNIT;
            auto d = dest;
            auto s = source + xC * eReal * FLOATPACK + xR;
            if (eR > 0) {
                int eStep = ALIMIN(eRemain, eS);
                for (int yi=0; yi<eStep; ++yi) {
                    d[yi] = s[yi * xS4];
                }
                eRemain-=eStep;
                if (!lastBag ||eRemain >= EP) {
                    d += (eOutsideStride - eR);
                } else {
                    int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                    eOutsideStride4LastBag = eOutsideStride - (EP * 4 * offsetLC / sizeof(float));
                    d += (eOutsideStride4LastBag - eR + offsetLC * eFill);
                }
                s += eS * xS4;
            }
            while (eRemain > 0) {
                int eStep = ALIMIN(eDest, eRemain);
                for (int yi=0; yi<eStep; ++yi) {
                    d[yi] = s[yi * xS4];
                }
                eRemain-=eStep;
                if (!lastBag || eRemain >= EP) {
                    d+= eOutsideStride;
                } else {
                    int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                    eOutsideStride4LastBag = eOutsideStride - (EP * 4 * offsetLC / sizeof(float));
                    d+= (eOutsideStride4LastBag + offsetLC * eFill);
                }
                s+= eStep * xS4;
            }
            if (lastBag && e + eR < EP) {
                int efill = ALIMAX(e + eR, realDstCount % EP);
                dest += efill;
            } else {
                dest += eDest;
            }
            offsetLC++;
        }
    }
}

template<int EP, int HP>
static void _ArmBasicMNNPackC4ForMatMul_A_L8(int8_t* destOrigin, int8_t const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eDest = EP;
    int offset = info[3];
    const int LP = 8;
    int eOutsideStride = info[2] / sizeof(int64_t);
    int realDstCount = info[4];
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        int eC = eOffset / EP;
        int eR = eOffset % EP;
        int eS = eDest - eR;
        bool lastBag = false;
        int eOutsideStride4LastBag = eOutsideStride;
        int eres = realDstCount - eOffset;
        if (realDstCount % EP > 0) {
            int jobsE = realDstCount - eOffset - e;
            if (jobsE == 0 || (jobsE < (realDstCount % EP))) {
                lastBag = true;
            }
        }
        auto dest = (int64_t*)(destOrigin + lOffset * eDest + eC * info[2] + eR * LP);
        auto source = (int64_t*)sourceGroup[n];
        int lRemain = l / LP;
        if (lastBag && e + eR < EP) {
            int elast = ALIMIN(ALIMAX(eR + e, realDstCount % EP), EP);
            dest = (int64_t*)(destOrigin + lOffset * elast + eC * info[2] + eR * LP);
        }
        int offsetLC = lOffset / LP;
        for (int x = 0; x < lRemain; ++x) {
            int eRemain = e;
            auto d = dest;
            auto s = source;
            if (1 == offset) {
                if (eR > 0) {
                    int eStep = ALIMIN(eRemain, eS);
                    ::memcpy(d, s, eStep * sizeof(int64_t));
                    eRemain-=eStep;
                    if (!lastBag ||eRemain >= EP) {
                        d += (eOutsideStride - eR);
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - (EP * offsetLC);
                        d += (eOutsideStride4LastBag - eR + offsetLC * eFill);
                    }
                    s += (eS * offset);
                }
                while (eRemain > 0) {
                    int eStep = ALIMIN(eDest, eRemain);
                    ::memcpy(d, s, eStep * sizeof(int64_t));
                    eRemain-=eStep;
                    if (!lastBag || eRemain >= EP) {
                        d+= eOutsideStride;
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - (EP * offsetLC);
                        d+= (eOutsideStride4LastBag + offsetLC * eFill);
                    }
                    s+= (eStep * offset);
                }
            } else {
                if (eR > 0) {
                    int eStep = ALIMIN(eRemain, eS);
                    for (int yi=0; yi<eStep; ++yi) {
                        d[yi] = s[yi * offset];
                    }
                    eRemain-=eStep;
                    if (!lastBag ||eRemain >= EP) {
                        d += (eOutsideStride - eR);
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - (EP * offsetLC);
                        d += (eOutsideStride4LastBag - eR + offsetLC * eFill);
                    }
                    s += eS * offset;
                }
                while (eRemain > 0) {
                    int eStep = ALIMIN(eDest, eRemain);
                    for (int yi=0; yi<eStep; ++yi) {
                        d[yi] = s[yi * offset];
                    }
                    eRemain-=eStep;
                    if (!lastBag || eRemain >= EP) {
                        d+= eOutsideStride;
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - (EP * offsetLC);
                        d+= (eOutsideStride4LastBag + offsetLC * eFill);
                    }
                    s+= eStep * offset;
                }
            }
            source += eReal;
            if (lastBag && e + eR < EP ) { // eR=0;eR>0
                int efill = ALIMAX(e + eR, realDstCount % EP);
                dest += efill;
            } else {
                dest += eDest;
            }
            offsetLC++;
        }
    }
}

#ifdef MNN_LOW_MEMORY
void MNNAbsMaxFP16(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack) {
    if (pack == 4) {
        MNNAbsMaxFP16_Pack4(source, absmax, src_depth_quad, realSize, pack);
        return;
    }
    if (pack == 8) {
        MNNAbsMaxFP16_Pack8(source, absmax, src_depth_quad, realSize, pack);
        return;
    }
    // source: (src_depth_quad, realSize, pack)
    auto srcStep = pack * realSize;
    auto srcPtr = (FLOAT16*)source;
    auto dstPtr = (FLOAT16*)absmax;
    for (int i = 0; i < realSize; ++i) {
        FLOAT16 absmaxVal = 0; // absmaxVal>=0
        for (int c = 0; c < src_depth_quad; ++c) {
            auto src = srcPtr + c * srcStep + i * pack;
            for (int k = 0; k < pack; ++k) {
                if (std::abs(src[k]) > absmaxVal) {
                    absmaxVal = std::abs(src[k]);
                }
            }
        }
        dstPtr[i] = absmaxVal;
    }
    return;
}

static void MNNDynamicQuantFP16(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack) {
    if (pack == 8) {
        MNNDynamicQuantFP16_Pack8(src, dst, scale, src_depth_quad,realSize, pack);
        return;
    }
    if (pack == 4) {
        MNNDynamicQuantFP16_Pack4(src, dst, scale, src_depth_quad,realSize, pack);
        return;
    }
    int8_t* dstPtr = dst;
    auto srcPtr = (FLOAT16*)src;

    for (int i = 0; i < realSize; ++i) {
        auto scaleVal = static_cast<FLOAT16>(scale[i]);
        for (int c = 0; c < src_depth_quad; ++c) {
            auto srcZ = srcPtr + c * pack * realSize + i * pack;
            auto dstZ = dstPtr + c * pack * realSize + i * pack;
            for (int k = 0; k < pack; ++k) {
                int val = (int)roundf(srcZ[k] * scaleVal);
                dstZ[k] = val;
            }
        }
    }
    return;
}
#endif

static CoreFunctions* gInstance = nullptr;
static CoreInt8Functions* gArm82CoreInt8Functions = nullptr;

bool Arm82Functions::init() {
    using Vec = MNN::Math::Vec<FLOAT16, 8>;
    auto origin = MNNGetCoreFunctions();
#define FUNC_PTR_ASSIGN(dst, src) dst = (decltype(dst))(src)
    gInstance = new CoreFunctions;
    gArm82CoreInt8Functions = new CoreInt8Functions;
    *gArm82CoreInt8Functions = *MNNGetInt8CoreFunctions();
    {
        if (origin->supportSDot) {
            gArm82CoreInt8Functions->MNNPackC4Int8ForMatMul_A = _Arm82MNNPackC4ForMatMul_A<12, 4>;
        }
        if (origin->supportI8mm) {
            gArm82CoreInt8Functions->MNNPackC4Int8ForMatMul_A = _ArmBasicMNNPackC4ForMatMul_A_L8<10, 8>;
        }
    }

    FUNC_PTR_ASSIGN(gInstance->MNNFp32ToFp8, MNNFp32ToFp8);
    FUNC_PTR_ASSIGN(gInstance->MNNFp16ToFp8, MNNFp16ToFp8);
    FUNC_PTR_ASSIGN(gInstance->MNNFp8ToFp32, MNNFp8ToFp32);
    FUNC_PTR_ASSIGN(gInstance->MNNFp8ToFp16, MNNFp8ToFp16);

    FUNC_PTR_ASSIGN(gInstance->MNNFp32ToLowp, MNNQuantizeFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNLowpToFp32, MNNDequantizeFP16);
    gInstance->bytes = 2;

    // Packed
    gInstance->pack = 8;
    FUNC_PTR_ASSIGN(gInstance->MNNPackCUnit, MNNPackC8FP16);
    FUNC_PTR_ASSIGN(gInstance->MNNUnpackCUnit, MNNUnPackC8FP16);
    FUNC_PTR_ASSIGN(gInstance->MNNPackCUnitTranspose, MNNPackTransposeInt16C8);
    FUNC_PTR_ASSIGN(gInstance->MNNUnpackCUnitTranspose, MNNUnpackTransposeInt16C8);
    FUNC_PTR_ASSIGN(gInstance->MNNConvRunForLineDepthwise, MNNConvRunForLineDepthwiseFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNAxByClampBroadcastUnit, MNNAxByClampBroadcastC8FP16);
    FUNC_PTR_ASSIGN(gInstance->MNNMatrixSub, MNNMatrixSubFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNMatrixAdd, MNNMatrixAddFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNStrassenMergeCFunction, ARM82StrassenMerge);
#ifdef MNN_LOW_MEMORY
    FUNC_PTR_ASSIGN(gInstance->MNNDynamicUpdateConvBiasScale, origin->MNNDynamicUpdateConvBiasScale);
    if (origin->supportSDot) {
        FUNC_PTR_ASSIGN(gInstance->MNNGeneralIm2Col, MNNGeneralIm2col_Arm82);
    }
    if (origin->supportI8mm) {
        FUNC_PTR_ASSIGN(gInstance->MNNGeneralIm2Col, MNNGeneralIm2col_Arm86);
    }
    
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
    gInstance->supportFp16arith = origin->supportFp16arith;
    gInstance->supportSDot = origin->supportSDot;
    gInstance->supportI8mm = origin->supportI8mm;
#ifdef MNN_CPU_WEIGHT_DEQUANT_GEMM
    // Weight Dequant Gemm Kernels
    FUNC_PTR_ASSIGN(gInstance->MNNPackedMatMul_int8, MNNPackedMatMulFP16_int8);
    FUNC_PTR_ASSIGN(gInstance->MNNPackedMatMulRemain_int8, MNNPackedMatMulRemainFP16_int8);
#endif
#ifdef MNN_LOW_MEMORY
    // Dynamic Qaunt Helper Functions
    FUNC_PTR_ASSIGN(gInstance->MNNAbsMax, MNNAbsMaxFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNQuantScale, MNNQuantScaleFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNDynamicQuant, MNNDynamicQuantFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNQuantSum, MNNQuantSumFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNCountMaxMinValue, ARM82CountMinMaxValue);
#endif
    FUNC_PTR_ASSIGN(gInstance->MNNSumByAxisLForMatmul_A, origin->MNNSumByAxisLForMatmul_A);
    FUNC_PTR_ASSIGN(gInstance->MNNDepthwiseConvFastKernel, MNNDepthwiseConvFastKernelFP16);
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
CoreInt8Functions* Arm82Functions::getInt8() {
    return gArm82CoreInt8Functions;
}
};
#endif
