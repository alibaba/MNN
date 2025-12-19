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
// void MNNPackTransposeInt16C8(int16_t* dst, const int16_t* src, size_t area, size_t depth, int32_t* areaOffset);

// C(UP_DIV(h,8), e, h8) = B(UP_DIV(h,hP), l, hP) * A(l, eP), hP = 24
// parameter: [aStride, l, h, cStride, bExtraStride]
// aStride in parameter is deprecated (useless), but for code clean, just retain it
void MNNPackedMatMulFP16(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);

// C(UP_DIV(h,8), e, h8) = B(UP_DIV(h,hP), l, hP) * A(l, e), hP = 24, e >= 1
// parameter: [aStride, l, h, cStride, bExtraStride]
void MNNPackedMatMulRemainFP16(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);

#ifdef MNN_CPU_WEIGHT_DEQUANT_GEMM
void MNNPackedMatMulFP16_int4(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
void MNNPackedMatMulRemainFP16_int4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
void MNNPackedMatMulFP16_int8(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
void MNNPackedMatMulRemainFP16_int8(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
#endif

#ifdef __aarch64__
#ifdef MNN_LOW_MEMORY
void MNNAbsMaxFP16_Pack8(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack);
void MNNAbsMaxFP16_Pack4(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack);
void MNNQuantScaleFP16(float* sum, float* absmax, float* quant_scale, float* dequant_scale, size_t thread, size_t batch);
void MNNDynamicQuantFP16_Pack8(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, const float* bias, size_t pack);
void MNNDynamicQuantFP16_Pack4(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, const float* bias, size_t pack);
void MNNGeneralIm2col_Arm82(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el, int32_t LP, int32_t pack);
void MNNGeneralIm2col_Arm86(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el, int32_t LP, int32_t pack);
#ifdef MNN_SME2
void MNNGeneralIm2col_Fp16Sme2(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el, int32_t LP, int32_t pack);
#endif
void MNNLocalMinMaxFP16_Pack4(float* dstMin, float* dstMax, const float* source, size_t blockNum, size_t blockLU, size_t EP, size_t LP, size_t loadDstBuffer);
void MNNLocalMinMaxFP16_Pack8(float* dstMin, float* dstMax, const float* source, size_t blockNum, size_t blockLU, size_t EP, size_t LP, size_t loadDstBuffer);
#endif // MNN_LOW_MEMORY
void CountMinMaxValue_FP16(float* source, float* minVal, float* maxVal, size_t sizeQuad);
#ifdef MNN_SME2
void MNNPackedMatMulRemainFP16_SME2(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
#endif
#endif

#if defined(__aarch64__)

void MNNDepthwiseConvFastKernelFP16(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                    size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                    size_t srcHStep, size_t dstHStep, const float* bias, const float* parameters);
#endif

void MNNConvRunForLineDepthwiseFP16(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height, size_t srcHStep, size_t dstHStep);
}


namespace MNN {

#define FP16_SME2_MATMUL_EP 16
#define FP16_SME2_MATMUL_LP 2
#define FP16_SME2_MATMUL_HP 64

static void Sme2MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *hP = FP16_SME2_MATMUL_HP;
    *eP = FP16_SME2_MATMUL_EP;
    *lP = FP16_SME2_MATMUL_LP;
}

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
        auto srcPtr = reinterpret_cast<__fp16*>(source);
        while (remain) {
            max_ = ALIMAX(srcPtr[0], max_);
            min_ = ALIMIN(srcPtr[0], min_);
            srcPtr += 1;
            remain--;
        }
        reinterpret_cast<__fp16*>(minVal)[0] = min_;
        reinterpret_cast<__fp16*>(maxVal)[0] = max_;
    }
}
#ifdef MNN_SME2
//(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias)
static void MNNPackedMatMulFP16_SME2(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    MNNPackedMatMulRemainFP16_SME2(C, A, B, 16, parameter, postParameters, bias, k, b);
}
#endif
#else
static void ARM82CountMinMaxValue(float* source, float* minVal, float* maxVal, size_t size) {
    auto srcPtr = (FLOAT16*)source;
    auto minPtr = (FLOAT16*)minVal;
    auto maxPtr = (FLOAT16*)maxVal;
    auto max_ = srcPtr[0], min_ = srcPtr[0];
    for (int i = 1; i < size; ++i) {
        if (max_ < srcPtr[i]) {
            max_ = srcPtr[i];
        }
        if (min_ > srcPtr[i]) {
            min_ = srcPtr[i];
        }
    }
    minPtr[0] = min_;
    maxPtr[0] = max_;
}
#endif

static void Arm82MNNPackForMatMul_B(float* destC, const float* sourceC, size_t h, size_t kernelsize, size_t ic, bool transpose) {
    auto l = kernelsize * ic;
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

static void Sme2MNNPackForMatMul_B(float* destC, const float* sourceC, size_t h, size_t kernelsize, size_t ic ,bool transpose) {
    auto dest = (int16_t*)destC;
    auto source = (int16_t*)sourceC;
    int LP = FP16_SME2_MATMUL_LP;
    int HP = FP16_SME2_MATMUL_HP;
    auto l = kernelsize * ic;
    memset(dest, 0, ROUND_UP(h, HP) * ROUND_UP(ic, LP) * kernelsize * sizeof(FLOAT16));
    auto stride0 = ROUND_UP(ic, LP) * kernelsize * HP;
    auto stride1 = HP *  ROUND_UP(ic, LP);
    auto stride2 = HP * LP;

    size_t srcStride0 = l; // [h,k2,ic]->[hu,k2,ic/lp,hp,lp]
    size_t srcStride1 = 1;
    if (!transpose) { // [k2,ic,h]->[hu,k2,ic/lp,hp,lp]
        srcStride0 = 1;
        srcStride1 = h;
    }
    for (int y = 0; y < h; ++y) {
        auto yHu = y / HP;
        auto yHp = y % HP;
        for (int k = 0; k < kernelsize; ++k) {
            for (int x = 0; x < ic; ++x) {
                auto xLu = x / LP;
                auto xLp = x % LP;
                dest[yHu * stride0 + k * stride1 + xLu * stride2 + yHp * LP + xLp] = source[y * srcStride0 + (x + k * ic) * srcStride1];
            }
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

static void MNNGridSampleComputeCord3DFp16(FLOAT* dst, const FLOAT* src, size_t inD, size_t inH, size_t inW, size_t outD, size_t outH, size_t outW, bool alignCorners) {
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
    // [depth/8, srcAreaOffset, 8] -> [area, dstAreaOffset]
    int srcAreaOffset = areaOffset[0];
    int dstAreaOffset = areaOffset[1];
    int c      = (int)depth;
    int cDiv8  = c / 8;
    int cAlign = cDiv8 * 8;
    int areaDiv4 = area / 4;
    int areaAlign = areaDiv4 * 4;

    if (areaAlign > 0) {
        for (int ci = 0; ci < cDiv8; ++ci) {
            auto srcH = src + ci * 8 * srcAreaOffset;
            auto dstH = dst + ci * 8;
            for (int hi = 0; hi < areaAlign; hi+=4) {
                auto src0 = srcH + hi * 8;
                auto src1 = srcH + hi * 8 + 8;
                auto src2 = srcH + hi * 8 + 16;
                auto src3 = srcH + hi * 8 + 24;

                auto dst0 = dstH + hi * dstAreaOffset;
                auto dst1 = dstH + hi * dstAreaOffset + dstAreaOffset;
                auto dst2 = dstH + hi * dstAreaOffset + 2 * dstAreaOffset;
                auto dst3 = dstH + hi * dstAreaOffset + 3 * dstAreaOffset;
                vst1q_s16(dst0, vld1q_s16(src0));
                vst1q_s16(dst1, vld1q_s16(src1));
                vst1q_s16(dst2, vld1q_s16(src2));
                vst1q_s16(dst3, vld1q_s16(src3));
            }
        }
    }
    if (areaAlign < area) {
        for (int ci = 0; ci < cDiv8; ++ci) {
            auto srcH = src + 8 * ci * srcAreaOffset;
            auto dstH = dst + ci * 8;
            for (int hi = areaAlign; hi < area; ++hi) {
                auto src0 = srcH + hi * 8;
                auto dst0 = dstH + hi * dstAreaOffset;
                vst1q_s16(dst0, vld1q_s16(src0));
            }
        }
    }
    if (c == cAlign) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + srcAreaOffset * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * 8;
        auto dstHeight = dstAlign + hi * dstAreaOffset;

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
    int areaDiv4 = area / 4;
    int areaAlign = areaDiv4 * 4;
    if (areaAlign > 0) {
        for (int ci = 0; ci < cDiv4; ++ci) {
            auto srcH = src + ci * 8;
            auto dstH = dst + ci * dstAreaOffset * 8;
            for (int hi = 0; hi < areaAlign; hi+=4) {
                auto src0 = srcH + hi * c;
                auto src1 = srcH + hi * c + c;
                auto src2 = srcH + hi * c + 2 * c;
                auto src3 = srcH + hi * c + 3 * c;
                auto dst0 = dstH + hi * 8;
                auto dst1 = dstH + hi * 8 + 8;
                auto dst2 = dstH + hi * 8 + 16;
                auto dst3 = dstH + hi * 8 + 24;
                vst1q_s16(dst0, vld1q_s16(src0));
                vst1q_s16(dst1, vld1q_s16(src1));
                vst1q_s16(dst2, vld1q_s16(src2));
                vst1q_s16(dst3, vld1q_s16(src3));
            }
        }
    }
    if (areaAlign < area) {
        for (int ci = 0; ci < cDiv4; ++ci) {
            auto srcH = src + ci * 8;
            auto dstH = dst + ci * dstAreaOffset * 8;
            for (int hi = areaAlign; hi < area; ++hi) {
                auto src0 = srcH + hi * c;
                auto dst0 = dstH + hi * 8;
                vst1q_s16(dst0, vld1q_s16(src0));
            }
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

inline void transpose_4x4_f32(float32x4_t& r0, float32x4_t& r1, float32x4_t& r2, float32x4_t& r3) {
    // Stage 1: Transpose 2x2 blocks of float32 elements between pairs of adjacent rows.
    float32x4x2_t temp0 = vtrnq_f32(r0, r1);
    float32x4x2_t temp1 = vtrnq_f32(r2, r3);

    // Intermediate state:
    // temp0.val[0] = [A0, B0, A2, B2]
    // temp0.val[1] = [A1, B1, A3, B3]
    // temp1.val[0] = [C0, D0, C2, D2]
    // temp1.val[1] = [C1, D1, C3, D3]

    // Stage 2: Manually swap the 64-bit blocks to finalize the transpose.
    // This correctly simulates the non-existent 64-bit transpose/zip.
    float64x2_t i0_f64 = vreinterpretq_f64_f32(temp0.val[0]);
    float64x2_t i1_f64 = vreinterpretq_f64_f32(temp0.val[1]);
    float64x2_t i2_f64 = vreinterpretq_f64_f32(temp1.val[0]);
    float64x2_t i3_f64 = vreinterpretq_f64_f32(temp1.val[1]);

    // Combine the low 64 bits of i0 and i2 to form the first part of the result.
    float32x4_t t0 = vreinterpretq_f32_f64(vcombine_f64(vget_low_f64(i0_f64), vget_low_f64(i2_f64)));
    // Combine the low 64 bits of i1 and i3 for the second part.
    float32x4_t t1 = vreinterpretq_f32_f64(vcombine_f64(vget_low_f64(i1_f64), vget_low_f64(i3_f64)));
    // Combine the high 64 bits of i0 and i2 for the third part.
    float32x4_t t2 = vreinterpretq_f32_f64(vcombine_f64(vget_high_f64(i0_f64), vget_high_f64(i2_f64)));
    // Combine the high 64 bits of i1 and i3 for the final part.
    float32x4_t t3 = vreinterpretq_f32_f64(vcombine_f64(vget_high_f64(i1_f64), vget_high_f64(i3_f64)));

    r0 = t0;
    r1 = t1;
    r2 = t2;
    r3 = t3;
}

static void Sme2MNNPackC4ForMatMul_A_FP16(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    const int lP = FP16_SME2_MATMUL_LP;
    const int pack = 8;
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];

    float32x4_t v0, v1, v2, v3, v4, v5, v6, v7;

    for (int n = 0; n < number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];

        auto destBase = (FLOAT16*)destOrigin + lOffset * eDest + eOffset * lP;
        auto sourceBase = (const FLOAT16*)(sourceGroup[n]);

        const int eTile = 8;
        const int lTile = 8;

        const int eMain = e / eTile;
        const int lMain = l / lTile;

        const size_t srcRowStride = (size_t)pack * offset;
        const size_t srcColBlockStride = (size_t)eReal * pack;
        const size_t dstColBlockStride = (size_t)eDest * lP;

        for (int y0 = 0; y0 < eMain; ++y0) {
            const int yBase = y0 * eTile;
            for (int x0 = 0; x0 < lMain; ++x0) {
                const int xBase = x0 * lTile;

                const auto srcBlockBase = sourceBase + yBase * srcRowStride + x0 * srcColBlockStride;

                v0 = vld1q_f32((const float*)(srcBlockBase + 0 * srcRowStride));
                v1 = vld1q_f32((const float*)(srcBlockBase + 1 * srcRowStride));
                v2 = vld1q_f32((const float*)(srcBlockBase + 2 * srcRowStride));
                v3 = vld1q_f32((const float*)(srcBlockBase + 3 * srcRowStride));
                v4 = vld1q_f32((const float*)(srcBlockBase + 4 * srcRowStride));
                v5 = vld1q_f32((const float*)(srcBlockBase + 5 * srcRowStride));
                v6 = vld1q_f32((const float *)(srcBlockBase + 6 * srcRowStride));
                v7 = vld1q_f32((const float *)(srcBlockBase + 7 * srcRowStride));

                transpose_4x4_f32(v0, v1, v2, v3);
                transpose_4x4_f32(v4, v5, v6, v7);

                float* addr0 = (float*)(destBase + yBase * lP + (xBase / lP) * dstColBlockStride);
                float* addr1= (float*)(destBase + yBase * lP + (xBase / lP + 1) * dstColBlockStride);
                float* addr2= (float*)(destBase + yBase * lP + (xBase / lP + 2) * dstColBlockStride);
                float* addr3= (float*)(destBase + yBase * lP + (xBase / lP + 3) * dstColBlockStride);

                vst1q_f32(addr0, v0);
                vst1q_f32(addr0 + 4, v4);
                vst1q_f32(addr1, v1);
                vst1q_f32(addr1 + 4, v5);
                vst1q_f32(addr2, v2);
                vst1q_f32(addr2 + 4, v6);
                vst1q_f32(addr3, v3);
                vst1q_f32(addr3 + 4, v7);
            }
        }

        const int eHandled = eMain * eTile;
        const int lHandled = lMain * lTile;

        // Process remaining rows
        for (int y = eHandled; y < e; ++y) {
            int yR = y % eDest;
            for (int x = 0; x < l; ++x) {
                int xR = x % pack;
                int xC = x / pack;
                destBase[(x / lP) * dstColBlockStride + yR * lP + (x % lP)] = sourceBase[xC * srcColBlockStride + y * srcRowStride + xR];
            }
        }

        // Process remaining columns for the already handled rows
        for (int y = 0; y < eHandled; ++y) {
            int yR = y % eDest;
            for (int x = lHandled; x < l; ++x) {
                int xR = x % pack;
                int xC = x / pack;
                destBase[(x / lP) * dstColBlockStride + yR * lP + (x % lP)] = sourceBase[xC * srcColBlockStride + y * srcRowStride + xR];
            }
        }
    }
}

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
static void MNNAttenPackAndScaleSingleHead(float* dst, const float* srcHeadBase, size_t srcRowStride, const float* scale, const int32_t* units, size_t seqLen, size_t headDim) {
    const int32_t eP = units[0];
    const int32_t lP = units[1];

    if (lP != 1 && lP != 2) {
        MNN_ERROR("This function only supports lP=1 or 2\n");
        return;
    }

    const float scaleVal = scale[0];
    const float16x8_t vScale = vdupq_n_f16(scaleVal);

    const size_t packedHeadDim = UP_DIV(headDim, lP);
    const size_t dstStrideDOuter = (size_t)eP * lP;
    const size_t dstStrideSOuter = packedHeadDim * dstStrideDOuter;

    for (int s = 0; s < seqLen; ++s) {
        const int sOuter = s / eP;
        const int sInner = s % eP;
        const FLOAT16* srcRowPtr = (FLOAT16*)srcHeadBase + s * srcRowStride;
        FLOAT16* dstBasePtr = (FLOAT16*)dst + sOuter * dstStrideSOuter + sInner * lP;

        if (lP == 1) {
            size_t d = 0;
            for (; d + 7 < headDim; d += 8) {
                float16x8_t sVec = vld1q_f16(srcRowPtr + d);
                sVec = vmulq_f16(sVec, vScale);

                dstBasePtr[(d + 0) * dstStrideDOuter] = sVec[0];
                dstBasePtr[(d + 1) * dstStrideDOuter] = sVec[1];
                dstBasePtr[(d + 2) * dstStrideDOuter] = sVec[2];
                dstBasePtr[(d + 3) * dstStrideDOuter] = sVec[3];
                dstBasePtr[(d + 4) * dstStrideDOuter] = sVec[4];
                dstBasePtr[(d + 5) * dstStrideDOuter] = sVec[5];
                dstBasePtr[(d + 6) * dstStrideDOuter] = sVec[6];
                dstBasePtr[(d + 7) * dstStrideDOuter] = sVec[7];
            }
            for (; d < headDim; ++d) {
                dstBasePtr[d * dstStrideDOuter] = srcRowPtr[d] * scaleVal;
            }
        } else { // lP == 2
            const FLOAT16* srcDPtr = srcRowPtr;
            FLOAT16* dstDPtr = dstBasePtr;
            size_t dRealSize = headDim;

            while (dRealSize >= 16) {
                float16x8_t s0 = vld1q_f16(srcDPtr);
                float16x8_t s1 = vld1q_f16(srcDPtr + 8);
                s0 = vmulq_f16(s0, vScale);
                s1 = vmulq_f16(s1, vScale);

                float16x4_t lowS0_f16 = vget_low_f16(s0);   // {s0, s1, s2, s3}
                float16x4_t highS0_f16 = vget_high_f16(s0); // {s4, s5, s6, s7}
                uint32x2_t lowS0_u32 = vreinterpret_u32_f16(lowS0_f16);
                uint32x2_t highS0_u32 = vreinterpret_u32_f16(highS0_f16);

                *((uint32_t*)(dstDPtr + 0 * dstStrideDOuter)) = vget_lane_u32(lowS0_u32, 0); // Store pair {s0, s1}
                *((uint32_t*)(dstDPtr + 1 * dstStrideDOuter)) = vget_lane_u32(lowS0_u32, 1); // Store pair {s2, s3}
                *((uint32_t*)(dstDPtr + 2 * dstStrideDOuter)) = vget_lane_u32(highS0_u32, 0); // Store pair {s4, s5}
                *((uint32_t*)(dstDPtr + 3 * dstStrideDOuter)) = vget_lane_u32(highS0_u32, 1); // Store pair {s6, s7}

                float16x4_t lowS1_f16 = vget_low_f16(s1);   // {s8, s9, s10, s11}
                float16x4_t highS1_f16 = vget_high_f16(s1); // {s12, s13, s14, s15}
                uint32x2_t lowS1_u32 = vreinterpret_u32_f16(lowS1_f16);
                uint32x2_t highS1_u32 = vreinterpret_u32_f16(highS1_f16);

                *((uint32_t*)(dstDPtr + 4 * dstStrideDOuter)) = vget_lane_u32(lowS1_u32, 0);
                *((uint32_t*)(dstDPtr + 5 * dstStrideDOuter)) = vget_lane_u32(lowS1_u32, 1);
                *((uint32_t*)(dstDPtr + 6 * dstStrideDOuter)) = vget_lane_u32(highS1_u32, 0);
                *((uint32_t*)(dstDPtr + 7 * dstStrideDOuter)) = vget_lane_u32(highS1_u32, 1);

                dRealSize -= 16;
                srcDPtr += 16;
                dstDPtr += 8 * dstStrideDOuter;
            }
            // Remainder loop with padding
            while (dRealSize > 0) {
                if (dRealSize >= 2) {
                    dstDPtr[0] = srcDPtr[0] * scaleVal;
                    dstDPtr[1] = srcDPtr[1] * scaleVal;

                    dRealSize -= 2;
                    srcDPtr += 2;
                    dstDPtr += dstStrideDOuter;
                } else { // dRealSize == 1
                    dstDPtr[0] = srcDPtr[0] * scaleVal;
                    dstDPtr[1] = (FLOAT16)0.0f; // Pad with zero
                    dRealSize = 0;
                }
            }
        }
    }
}

static void MNNFlashAttentionUpdateBlockOutput(float* dst, float* src, float* scale, float* normalizeScale, int depthQuad, int plane, int pack, int idx, int kvBlocks, int size, int bytes, int seqStart) {
    auto dstPtr = (float16_t*)dst;
    auto srcPtr = (float16_t*)src;
    const auto stride0 = plane * pack;

    if (idx == 0) {
        memcpy(dst, src, size * bytes);
    } else {
        for (int j = 0; j < depthQuad; ++j) {
            const auto baseOffset = j * stride0;
            int i = seqStart;

            for (; i + 4 < plane; i += 4) {

                auto pdst0 = dstPtr + baseOffset + (i + 0) * pack;
                auto psrc0 = srcPtr + baseOffset + (i + 0) * pack;
                auto pdst1 = dstPtr + baseOffset + (i + 1) * pack;
                auto psrc1 = srcPtr + baseOffset + (i + 1) * pack;
                auto pdst2 = dstPtr + baseOffset + (i + 2) * pack;
                auto psrc2 = srcPtr + baseOffset + (i + 2) * pack;
                auto pdst3 = dstPtr + baseOffset + (i + 3) * pack;
                auto psrc3 = srcPtr + baseOffset + (i + 3) * pack;

                float16x8_t src0 = vld1q_f16(psrc0);
                float16x8_t dst0 = vld1q_f16(pdst0);
                float16x8_t src1 = vld1q_f16(psrc1);
                float16x8_t dst1 = vld1q_f16(pdst1);
                float16x8_t src2 = vld1q_f16(psrc2);
                float16x8_t dst2 = vld1q_f16(pdst2);
                float16x8_t src3 = vld1q_f16(psrc3);
                float16x8_t dst3 = vld1q_f16(pdst3);

                float32x4_t svec0 = vdupq_n_f32(scale[i + 0]);
                float32x4_t svec1 = vdupq_n_f32(scale[i + 1]);
                float32x4_t svec2 = vdupq_n_f32(scale[i + 2]);
                float32x4_t svec3 = vdupq_n_f32(scale[i + 3]);


                float32x4_t res00 = vfmaq_f32(vcvt_f32_f16(vget_low_f16(src0)),  vcvt_f32_f16(vget_low_f16(dst0)),  svec0);
                float32x4_t res10 = vfmaq_f32(vcvt_f32_f16(vget_high_f16(src0)), vcvt_f32_f16(vget_high_f16(dst0)), svec0);

                float32x4_t res01 = vfmaq_f32(vcvt_f32_f16(vget_low_f16(src1)),  vcvt_f32_f16(vget_low_f16(dst1)),  svec1);
                float32x4_t res11 = vfmaq_f32(vcvt_f32_f16(vget_high_f16(src1)), vcvt_f32_f16(vget_high_f16(dst1)), svec1);

                float32x4_t res02 = vfmaq_f32(vcvt_f32_f16(vget_low_f16(src2)),  vcvt_f32_f16(vget_low_f16(dst2)),  svec2);
                float32x4_t res12 = vfmaq_f32(vcvt_f32_f16(vget_high_f16(src2)), vcvt_f32_f16(vget_high_f16(dst2)), svec2);

                float32x4_t res03 = vfmaq_f32(vcvt_f32_f16(vget_low_f16(src3)),  vcvt_f32_f16(vget_low_f16(dst3)),  svec3);
                float32x4_t res13 = vfmaq_f32(vcvt_f32_f16(vget_high_f16(src3)), vcvt_f32_f16(vget_high_f16(dst3)), svec3);

                vst1q_f16(pdst0, vcombine_f16(vcvt_f16_f32(res00), vcvt_f16_f32(res10)));
                vst1q_f16(pdst1, vcombine_f16(vcvt_f16_f32(res01), vcvt_f16_f32(res11)));
                vst1q_f16(pdst2, vcombine_f16(vcvt_f16_f32(res02), vcvt_f16_f32(res12)));
                vst1q_f16(pdst3, vcombine_f16(vcvt_f16_f32(res03), vcvt_f16_f32(res13)));
            }

            for (; i < plane; ++i) {
                auto pdst = dstPtr + baseOffset + i * pack;
                auto psrc = srcPtr + baseOffset + i * pack;

                float16x8_t srcF16 = vld1q_f16(psrc);
                float16x8_t dstF16 = vld1q_f16(pdst);
                float32x4_t svec = vdupq_n_f32(scale[i]);

                float32x4_t s0 = vcvt_f32_f16(vget_low_f16(srcF16));
                float32x4_t s1 = vcvt_f32_f16(vget_high_f16(srcF16));
                float32x4_t d0 = vcvt_f32_f16(vget_low_f16(dstF16));
                float32x4_t d1 = vcvt_f32_f16(vget_high_f16(dstF16));

                float32x4_t res0 = vfmaq_f32(s0, d0, svec);
                float32x4_t res1 = vfmaq_f32(s1, d1, svec);

                vst1q_f16(pdst, vcombine_f16(vcvt_f16_f32(res0), vcvt_f16_f32(res1)));
            }
        }
    }

    if (idx == kvBlocks - 1) {
        for (int j = 0; j < depthQuad; ++j) {
            const auto baseOffset = j * stride0;
            int i = 0;
            const int plane4 = plane - (plane % 4);
            for (; i < plane4; i += 4) {
                auto pdst0 = dstPtr + baseOffset + (i + 0) * pack;
                auto pdst1 = dstPtr + baseOffset + (i + 1) * pack;
                auto pdst2 = dstPtr + baseOffset + (i + 2) * pack;
                auto pdst3 = dstPtr + baseOffset + (i + 3) * pack;

                float16x8_t dst0 = vld1q_f16(pdst0);
                float16x8_t dst1 = vld1q_f16(pdst1);
                float16x8_t dst2 = vld1q_f16(pdst2);
                float16x8_t dst3 = vld1q_f16(pdst3);

                float32x4_t ns0 = vdupq_n_f32(1.0f / normalizeScale[i + 0]);
                float32x4_t ns1 = vdupq_n_f32(1.0f / normalizeScale[i + 1]);
                float32x4_t ns2 = vdupq_n_f32(1.0f / normalizeScale[i + 2]);
                float32x4_t ns3 = vdupq_n_f32(1.0f / normalizeScale[i + 3]);

                float32x4_t d00 = vmulq_f32(vcvt_f32_f16(vget_low_f16(dst0)),  ns0);
                float32x4_t d10 = vmulq_f32(vcvt_f32_f16(vget_high_f16(dst0)), ns0);
                float32x4_t d01 = vmulq_f32(vcvt_f32_f16(vget_low_f16(dst1)),  ns1);
                float32x4_t d11 = vmulq_f32(vcvt_f32_f16(vget_high_f16(dst1)), ns1);
                float32x4_t d02 = vmulq_f32(vcvt_f32_f16(vget_low_f16(dst2)),  ns2);
                float32x4_t d12 = vmulq_f32(vcvt_f32_f16(vget_high_f16(dst2)), ns2);
                float32x4_t d03 = vmulq_f32(vcvt_f32_f16(vget_low_f16(dst3)),  ns3);
                float32x4_t d13 = vmulq_f32(vcvt_f32_f16(vget_high_f16(dst3)), ns3);

                vst1q_f16(pdst0, vcombine_f16(vcvt_f16_f32(d00), vcvt_f16_f32(d10)));
                vst1q_f16(pdst1, vcombine_f16(vcvt_f16_f32(d01), vcvt_f16_f32(d11)));
                vst1q_f16(pdst2, vcombine_f16(vcvt_f16_f32(d02), vcvt_f16_f32(d12)));
                vst1q_f16(pdst3, vcombine_f16(vcvt_f16_f32(d03), vcvt_f16_f32(d13)));
            }

            for (; i < plane; ++i) {
                auto pdst = dstPtr + baseOffset + i * pack;
                float32x4_t nsvec = vdupq_n_f32(1.0f / normalizeScale[i]);

                float16x8_t dstF16 = vld1q_f16(pdst);
                float32x4_t d0 = vcvt_f32_f16(vget_low_f16(dstF16));
                float32x4_t d1 = vcvt_f32_f16(vget_high_f16(dstF16));

                d0 = vmulq_f32(d0, nsvec);
                d1 = vmulq_f32(d1, nsvec);

                vst1q_f16(pdst, vcombine_f16(vcvt_f16_f32(d0), vcvt_f16_f32(d1)));
            }
        }
    }
}

static void MNNAttenUnpackAndConvertFp16(float* dst, float* src, size_t depth, size_t planesize, int pack) {
    // src: (UP_DIV(depth, pack), planesize, pack), float16
    // dst: (planesize, depth), float32
    // pack=8

    if (planesize == 1) {
        MNNDequantizeFP16((int16_t*)src, dst, depth);
        return; // no need to convert
    }
    const auto depthDiv8 = UP_DIV(depth, pack);
    const auto srcStep = pack * planesize;
    const auto dstStep = depth;

    auto remainDepth = depth % pack;
    auto depthQuad = depthDiv8;
    if (remainDepth > 0) {
        depthQuad -= 1; // last quad is not full
    }

    for (int i = 0; i < depthQuad; ++i) {
        auto realsize = planesize;
        auto srcPtr = (FLOAT16*)src + i * srcStep;
        auto dstPtr = (float*)dst + i * pack;
        while (realsize >= 8) {
            float16x8_t s0_f16 = vld1q_f16(srcPtr + 0 * pack);
            float16x8_t s1_f16 = vld1q_f16(srcPtr + 1 * pack);
            float16x8_t s2_f16 = vld1q_f16(srcPtr + 2 * pack);
            float16x8_t s3_f16 = vld1q_f16(srcPtr + 3 * pack);
            float16x8_t s4_f16 = vld1q_f16(srcPtr + 4 * pack);
            float16x8_t s5_f16 = vld1q_f16(srcPtr + 5 * pack);
            float16x8_t s6_f16 = vld1q_f16(srcPtr + 6 * pack);
            float16x8_t s7_f16 = vld1q_f16(srcPtr + 7 * pack);

            float32x4_t d00_f32 = vcvt_f32_f16(vget_low_f16(s0_f16));
            float32x4_t d01_f32 = vcvt_f32_f16(vget_high_f16(s0_f16));
            float32x4_t d10_f32 = vcvt_f32_f16(vget_low_f16(s1_f16));
            float32x4_t d11_f32 = vcvt_f32_f16(vget_high_f16(s1_f16));
            float32x4_t d20_f32 = vcvt_f32_f16(vget_low_f16(s2_f16));
            float32x4_t d21_f32 = vcvt_f32_f16(vget_high_f16(s2_f16));
            float32x4_t d30_f32 = vcvt_f32_f16(vget_low_f16(s3_f16));
            float32x4_t d31_f32 = vcvt_f32_f16(vget_high_f16(s3_f16));
            float32x4_t d40_f32 = vcvt_f32_f16(vget_low_f16(s4_f16));
            float32x4_t d41_f32 = vcvt_f32_f16(vget_high_f16(s4_f16));
            float32x4_t d50_f32 = vcvt_f32_f16(vget_low_f16(s5_f16));
            float32x4_t d51_f32 = vcvt_f32_f16(vget_high_f16(s5_f16));
            float32x4_t d60_f32 = vcvt_f32_f16(vget_low_f16(s6_f16));
            float32x4_t d61_f32 = vcvt_f32_f16(vget_high_f16(s6_f16));
            float32x4_t d70_f32 = vcvt_f32_f16(vget_low_f16(s7_f16));
            float32x4_t d71_f32 = vcvt_f32_f16(vget_high_f16(s7_f16));

            vst1q_f32(dstPtr + 0 * dstStep, d00_f32); vst1q_f32(dstPtr + 0 * dstStep + 4, d01_f32);
            vst1q_f32(dstPtr + 1 * dstStep, d10_f32); vst1q_f32(dstPtr + 1 * dstStep + 4, d11_f32);
            vst1q_f32(dstPtr + 2 * dstStep, d20_f32); vst1q_f32(dstPtr + 2 * dstStep + 4, d21_f32);
            vst1q_f32(dstPtr + 3 * dstStep, d30_f32); vst1q_f32(dstPtr + 3 * dstStep + 4, d31_f32);
            vst1q_f32(dstPtr + 4 * dstStep, d40_f32); vst1q_f32(dstPtr + 4 * dstStep + 4, d41_f32);
            vst1q_f32(dstPtr + 5 * dstStep, d50_f32); vst1q_f32(dstPtr + 5 * dstStep + 4, d51_f32);
            vst1q_f32(dstPtr + 6 * dstStep, d60_f32); vst1q_f32(dstPtr + 6 * dstStep + 4, d61_f32);
            vst1q_f32(dstPtr + 7 * dstStep, d70_f32); vst1q_f32(dstPtr + 7 * dstStep + 4, d71_f32);

            srcPtr += 8 * pack;
            dstPtr += 8 * dstStep;
            realsize -= 8;
        }
        if (realsize >= 4) {
            float16x8_t s0_f16 = vld1q_f16(srcPtr + 0 * pack);
            float16x8_t s1_f16 = vld1q_f16(srcPtr + 1 * pack);
            float16x8_t s2_f16 = vld1q_f16(srcPtr + 2 * pack);
            float16x8_t s3_f16 = vld1q_f16(srcPtr + 3 * pack);

            float32x4_t d00_f32 = vcvt_f32_f16(vget_low_f16(s0_f16));
            float32x4_t d01_f32 = vcvt_f32_f16(vget_high_f16(s0_f16));
            float32x4_t d10_f32 = vcvt_f32_f16(vget_low_f16(s1_f16));
            float32x4_t d11_f32 = vcvt_f32_f16(vget_high_f16(s1_f16));
            float32x4_t d20_f32 = vcvt_f32_f16(vget_low_f16(s2_f16));
            float32x4_t d21_f32 = vcvt_f32_f16(vget_high_f16(s2_f16));
            float32x4_t d30_f32 = vcvt_f32_f16(vget_low_f16(s3_f16));
            float32x4_t d31_f32 = vcvt_f32_f16(vget_high_f16(s3_f16));

            vst1q_f32(dstPtr + 0 * dstStep, d00_f32); vst1q_f32(dstPtr + 0 * dstStep + 4, d01_f32);
            vst1q_f32(dstPtr + 1 * dstStep, d10_f32); vst1q_f32(dstPtr + 1 * dstStep + 4, d11_f32);
            vst1q_f32(dstPtr + 2 * dstStep, d20_f32); vst1q_f32(dstPtr + 2 * dstStep + 4, d21_f32);
            vst1q_f32(dstPtr + 3 * dstStep, d30_f32); vst1q_f32(dstPtr + 3 * dstStep + 4, d31_f32);

            srcPtr += 4 * pack;
            dstPtr += 4 * dstStep;
            realsize -= 4;
        }
        while (realsize > 0) {
            auto s0_fp16 = vld1q_f16(srcPtr);
            auto s00_fp32 = vcvt_f32_f16(vget_low_f16(s0_fp16));
            auto s01_fp32 = vcvt_f32_f16(vget_high_f16(s0_fp16));
            vst1q_f32(dstPtr, s00_fp32);
            vst1q_f32(dstPtr + 4, s01_fp32);
            srcPtr += pack;
            dstPtr += dstStep;
            realsize--;
        }
    }

    // process remain depth < 8
    if (remainDepth >= 4) {
        auto realsize = planesize;
        auto srcPtr = (FLOAT16*)src + (depthDiv8 - 1) * srcStep;
        auto dstPtr = (float*)dst + (depthDiv8 - 1) * pack;
        auto extraDepth = remainDepth - 4;

        float tmp0[4];
        float tmp1[4];
        float tmp2[4];
        float tmp3[4];
        float tmp4[4];
        float tmp5[4];
        float tmp6[4];
        float tmp7[4];

        while (realsize >= 8) {
            float16x8_t s0_f16 = vld1q_f16(srcPtr + 0 * pack);
            float16x8_t s1_f16 = vld1q_f16(srcPtr + 1 * pack);
            float16x8_t s2_f16 = vld1q_f16(srcPtr + 2 * pack);
            float16x8_t s3_f16 = vld1q_f16(srcPtr + 3 * pack);
            float16x8_t s4_f16 = vld1q_f16(srcPtr + 4 * pack);
            float16x8_t s5_f16 = vld1q_f16(srcPtr + 5 * pack);
            float16x8_t s6_f16 = vld1q_f16(srcPtr + 6 * pack);
            float16x8_t s7_f16 = vld1q_f16(srcPtr + 7 * pack);


            float32x4_t d00_f32 = vcvt_f32_f16(vget_low_f16(s0_f16));
            float32x4_t d01_f32 = vcvt_f32_f16(vget_high_f16(s0_f16));
            float32x4_t d10_f32 = vcvt_f32_f16(vget_low_f16(s1_f16));
            float32x4_t d11_f32 = vcvt_f32_f16(vget_high_f16(s1_f16));
            float32x4_t d20_f32 = vcvt_f32_f16(vget_low_f16(s2_f16));
            float32x4_t d21_f32 = vcvt_f32_f16(vget_high_f16(s2_f16));
            float32x4_t d30_f32 = vcvt_f32_f16(vget_low_f16(s3_f16));
            float32x4_t d31_f32 = vcvt_f32_f16(vget_high_f16(s3_f16));
            float32x4_t d40_f32 = vcvt_f32_f16(vget_low_f16(s4_f16));
            float32x4_t d41_f32 = vcvt_f32_f16(vget_high_f16(s4_f16));
            float32x4_t d50_f32 = vcvt_f32_f16(vget_low_f16(s5_f16));
            float32x4_t d51_f32 = vcvt_f32_f16(vget_high_f16(s5_f16));
            float32x4_t d60_f32 = vcvt_f32_f16(vget_low_f16(s6_f16));
            float32x4_t d61_f32 = vcvt_f32_f16(vget_high_f16(s6_f16));
            float32x4_t d70_f32 = vcvt_f32_f16(vget_low_f16(s7_f16));
            float32x4_t d71_f32 = vcvt_f32_f16(vget_high_f16(s7_f16));

            vst1q_f32(dstPtr + 0 * dstStep, d00_f32); vst1q_f32(tmp0, d01_f32);
            vst1q_f32(dstPtr + 1 * dstStep, d10_f32); vst1q_f32(tmp1, d11_f32);
            vst1q_f32(dstPtr + 2 * dstStep, d20_f32); vst1q_f32(tmp2, d21_f32);
            vst1q_f32(dstPtr + 3 * dstStep, d30_f32); vst1q_f32(tmp3, d31_f32);
            vst1q_f32(dstPtr + 4 * dstStep, d40_f32); vst1q_f32(tmp4, d41_f32);
            vst1q_f32(dstPtr + 5 * dstStep, d50_f32); vst1q_f32(tmp5, d51_f32);
            vst1q_f32(dstPtr + 6 * dstStep, d60_f32); vst1q_f32(tmp6, d61_f32);
            vst1q_f32(dstPtr + 7 * dstStep, d70_f32); vst1q_f32(tmp7, d71_f32);

            memcpy(dstPtr + 0 * dstStep + 4, tmp0, sizeof(float) * extraDepth);
            memcpy(dstPtr + 1 * dstStep + 4, tmp1, sizeof(float) * extraDepth);
            memcpy(dstPtr + 2 * dstStep + 4, tmp2, sizeof(float) * extraDepth);
            memcpy(dstPtr + 3 * dstStep + 4, tmp3, sizeof(float) * extraDepth);
            memcpy(dstPtr + 4 * dstStep + 4, tmp4, sizeof(float) * extraDepth);
            memcpy(dstPtr + 5 * dstStep + 4, tmp5, sizeof(float) * extraDepth);
            memcpy(dstPtr + 6 * dstStep + 4, tmp6, sizeof(float) * extraDepth);
            memcpy(dstPtr + 7 * dstStep + 4, tmp7, sizeof(float) * extraDepth);

            srcPtr += 8 * pack;
            dstPtr += 8 * dstStep;
            realsize -= 8;
        }
        if (realsize >= 4) {
            float16x8_t s0_f16 = vld1q_f16(srcPtr + 0 * pack);
            float16x8_t s1_f16 = vld1q_f16(srcPtr + 1 * pack);
            float16x8_t s2_f16 = vld1q_f16(srcPtr + 2 * pack);
            float16x8_t s3_f16 = vld1q_f16(srcPtr + 3 * pack);

            float32x4_t d00_f32 = vcvt_f32_f16(vget_low_f16(s0_f16));
            float32x4_t d01_f32 = vcvt_f32_f16(vget_high_f16(s0_f16));
            float32x4_t d10_f32 = vcvt_f32_f16(vget_low_f16(s1_f16));
            float32x4_t d11_f32 = vcvt_f32_f16(vget_high_f16(s1_f16));
            float32x4_t d20_f32 = vcvt_f32_f16(vget_low_f16(s2_f16));
            float32x4_t d21_f32 = vcvt_f32_f16(vget_high_f16(s2_f16));
            float32x4_t d30_f32 = vcvt_f32_f16(vget_low_f16(s3_f16));
            float32x4_t d31_f32 = vcvt_f32_f16(vget_high_f16(s3_f16));

            vst1q_f32(dstPtr + 0 * dstStep, d00_f32); vst1q_f32(tmp0, d01_f32);
            vst1q_f32(dstPtr + 1 * dstStep, d10_f32); vst1q_f32(tmp1, d11_f32);
            vst1q_f32(dstPtr + 2 * dstStep, d20_f32); vst1q_f32(tmp2, d21_f32);
            vst1q_f32(dstPtr + 3 * dstStep, d30_f32); vst1q_f32(tmp3, d31_f32);

            memcpy(dstPtr + 0 * dstStep + 4, tmp0, sizeof(float) * extraDepth);
            memcpy(dstPtr + 1 * dstStep + 4, tmp1, sizeof(float) * extraDepth);
            memcpy(dstPtr + 2 * dstStep + 4, tmp2, sizeof(float) * extraDepth);
            memcpy(dstPtr + 3 * dstStep + 4, tmp3, sizeof(float) * extraDepth);

            srcPtr += 4 * pack;
            dstPtr += 4 * dstStep;
            realsize -= 4;
        }
        while (realsize > 0) {
            auto s0_fp16 = vld1q_f16(srcPtr);
            auto d00_fp32 = vcvt_f32_f16(vget_low_f16(s0_fp16));
            auto d01_fp32 = vcvt_f32_f16(vget_high_f16(s0_fp16));
            vst1q_f32(dstPtr, d00_fp32);
            vst1q_f32(tmp0, d01_fp32);
            memcpy(dstPtr + 4, tmp0, sizeof(float) * extraDepth);
            srcPtr += pack;
            dstPtr += dstStep;
            realsize--;
        }
    }

    if (remainDepth > 0 && remainDepth < 4) {
        auto realsize = planesize;
        auto srcPtr = (FLOAT16*)src + (depthDiv8 - 1) * srcStep;
        auto dstPtr = (float*)dst + (depthDiv8 - 1) * pack;

        float tmp0[4];
        float tmp1[4];
        float tmp2[4];
        float tmp3[4];
        float tmp4[4];
        float tmp5[4];
        float tmp6[4];
        float tmp7[4];

        while (realsize >= 8) {
            float16x8_t s0_f16 = vld1q_f16(srcPtr + 0 * pack);
            float16x8_t s1_f16 = vld1q_f16(srcPtr + 1 * pack);
            float16x8_t s2_f16 = vld1q_f16(srcPtr + 2 * pack);
            float16x8_t s3_f16 = vld1q_f16(srcPtr + 3 * pack);
            float16x8_t s4_f16 = vld1q_f16(srcPtr + 4 * pack);
            float16x8_t s5_f16 = vld1q_f16(srcPtr + 5 * pack);
            float16x8_t s6_f16 = vld1q_f16(srcPtr + 6 * pack);
            float16x8_t s7_f16 = vld1q_f16(srcPtr + 7 * pack);

            float32x4_t d00_f32 = vcvt_f32_f16(vget_low_f16(s0_f16));
            float32x4_t d10_f32 = vcvt_f32_f16(vget_low_f16(s1_f16));
            float32x4_t d20_f32 = vcvt_f32_f16(vget_low_f16(s2_f16));
            float32x4_t d30_f32 = vcvt_f32_f16(vget_low_f16(s3_f16));
            float32x4_t d40_f32 = vcvt_f32_f16(vget_low_f16(s4_f16));
            float32x4_t d50_f32 = vcvt_f32_f16(vget_low_f16(s5_f16));
            float32x4_t d60_f32 = vcvt_f32_f16(vget_low_f16(s6_f16));
            float32x4_t d70_f32 = vcvt_f32_f16(vget_low_f16(s7_f16));

            vst1q_f32(tmp0, d00_f32);
            vst1q_f32(tmp1, d10_f32);
            vst1q_f32(tmp2, d20_f32);
            vst1q_f32(tmp3, d30_f32);
            vst1q_f32(tmp4, d40_f32);
            vst1q_f32(tmp5, d50_f32);
            vst1q_f32(tmp6, d60_f32);
            vst1q_f32(tmp7, d70_f32);

            memcpy(dstPtr + 0 * dstStep, tmp0, sizeof(float) * remainDepth);
            memcpy(dstPtr + 1 * dstStep, tmp1, sizeof(float) * remainDepth);
            memcpy(dstPtr + 2 * dstStep, tmp2, sizeof(float) * remainDepth);
            memcpy(dstPtr + 3 * dstStep, tmp3, sizeof(float) * remainDepth);
            memcpy(dstPtr + 4 * dstStep, tmp4, sizeof(float) * remainDepth);
            memcpy(dstPtr + 5 * dstStep, tmp5, sizeof(float) * remainDepth);
            memcpy(dstPtr + 6 * dstStep, tmp6, sizeof(float) * remainDepth);
            memcpy(dstPtr + 7 * dstStep, tmp7, sizeof(float) * remainDepth);

            srcPtr += 8 * pack;
            dstPtr += 8 * dstStep;
            realsize -= 8;
        }
        if (realsize >= 4) {
            float16x8_t s0_f16 = vld1q_f16(srcPtr + 0 * pack);
            float16x8_t s1_f16 = vld1q_f16(srcPtr + 1 * pack);
            float16x8_t s2_f16 = vld1q_f16(srcPtr + 2 * pack);
            float16x8_t s3_f16 = vld1q_f16(srcPtr + 3 * pack);

            float32x4_t d00_f32 = vcvt_f32_f16(vget_low_f16(s0_f16));
            float32x4_t d10_f32 = vcvt_f32_f16(vget_low_f16(s1_f16));
            float32x4_t d20_f32 = vcvt_f32_f16(vget_low_f16(s2_f16));
            float32x4_t d30_f32 = vcvt_f32_f16(vget_low_f16(s3_f16));

            vst1q_f32(tmp0, d00_f32);
            vst1q_f32(tmp1, d10_f32);
            vst1q_f32(tmp2, d20_f32);
            vst1q_f32(tmp3, d30_f32);

            memcpy(dstPtr + 0 * dstStep, tmp0, sizeof(float) * remainDepth);
            memcpy(dstPtr + 1 * dstStep, tmp1, sizeof(float) * remainDepth);
            memcpy(dstPtr + 2 * dstStep, tmp2, sizeof(float) * remainDepth);
            memcpy(dstPtr + 3 * dstStep, tmp3, sizeof(float) * remainDepth);

            srcPtr += 4 * pack;
            dstPtr += 4 * dstStep;
            realsize -= 4;
        }
        while (realsize > 0) {
            auto s0_f16 = vld1q_f16(srcPtr);
            float32x4_t d00_f32 = vcvt_f32_f16(vget_low_f16(s0_f16));
            vst1q_f32(tmp0, d00_f32);
            memcpy(dstPtr + 0 * dstStep, tmp0, sizeof(float) * remainDepth);
            srcPtr += pack;
            dstPtr += dstStep;
            realsize--;
        }
    }
}

static void MNNAttenPackAndConvertFp32LP1(float* dst, const float* src, const int32_t* units, size_t depth, size_t planesize) {
    int32_t eP = units[0];
    int32_t lP = units[1];

    if (lP != 1) {
        MNN_ERROR("This function only supports lP=1\n");
        return;
    }

    auto dstStride1 = eP;
    auto dstStride0 = planesize * dstStride1;

    for (int i = 0; i < depth; ++i) {
        size_t realsize = planesize;
        const float* srcPtr = src + i * planesize;
        FLOAT16* dstPtr = (FLOAT16*)dst + (i % eP) + (i / eP) * dstStride0;

        while (realsize >= 16) {
            float32x4_t s0_f32 = vld1q_f32(srcPtr);
            float32x4_t s1_f32 = vld1q_f32(srcPtr + 4);
            float32x4_t s2_f32 = vld1q_f32(srcPtr + 8);
            float32x4_t s3_f32 = vld1q_f32(srcPtr + 12);

            float16x4_t d0_f16 = vcvt_f16_f32(s0_f32);
            float16x4_t d1_f16 = vcvt_f16_f32(s1_f32);
            float16x4_t d2_f16 = vcvt_f16_f32(s2_f32);
            float16x4_t d3_f16 = vcvt_f16_f32(s3_f32);

            vst1_lane_f16(dstPtr,                  d0_f16, 0);
            vst1_lane_f16(dstPtr + dstStride1,     d0_f16, 1);
            vst1_lane_f16(dstPtr + 2 * dstStride1, d0_f16, 2);
            vst1_lane_f16(dstPtr + 3 * dstStride1, d0_f16, 3);

            vst1_lane_f16(dstPtr + 4 * dstStride1, d1_f16, 0);
            vst1_lane_f16(dstPtr + 5 * dstStride1, d1_f16, 1);
            vst1_lane_f16(dstPtr + 6 * dstStride1, d1_f16, 2);
            vst1_lane_f16(dstPtr + 7 * dstStride1, d1_f16, 3);

            vst1_lane_f16(dstPtr + 8 * dstStride1,  d2_f16, 0);
            vst1_lane_f16(dstPtr + 9 * dstStride1,  d2_f16, 1);
            vst1_lane_f16(dstPtr + 10 * dstStride1, d2_f16, 2);
            vst1_lane_f16(dstPtr + 11 * dstStride1, d2_f16, 3);

            vst1_lane_f16(dstPtr + 12 * dstStride1, d3_f16, 0);
            vst1_lane_f16(dstPtr + 13 * dstStride1, d3_f16, 1);
            vst1_lane_f16(dstPtr + 14 * dstStride1, d3_f16, 2);
            vst1_lane_f16(dstPtr + 15 * dstStride1, d3_f16, 3);

            srcPtr += 16;
            dstPtr += 16 * dstStride1;
            realsize -= 16;
        }

        if (realsize >= 8) {
            float32x4_t s0_f32 = vld1q_f32(srcPtr);
            float32x4_t s1_f32 = vld1q_f32(srcPtr + 4);

            float16x4_t d0_f16 = vcvt_f16_f32(s0_f32);
            float16x4_t d1_f16 = vcvt_f16_f32(s1_f32);

            vst1_lane_f16(dstPtr,              d0_f16, 0);
            vst1_lane_f16(dstPtr + dstStride1, d0_f16, 1);
            vst1_lane_f16(dstPtr + 2 * dstStride1, d0_f16, 2);
            vst1_lane_f16(dstPtr + 3 * dstStride1, d0_f16, 3);

            vst1_lane_f16(dstPtr + 4 * dstStride1, d1_f16, 0);
            vst1_lane_f16(dstPtr + 5 * dstStride1, d1_f16, 1);
            vst1_lane_f16(dstPtr + 6 * dstStride1, d1_f16, 2);
            vst1_lane_f16(dstPtr + 7 * dstStride1, d1_f16, 3);

            srcPtr += 8;
            dstPtr += 8 * dstStride1;
            realsize -= 8;
        }

        if (realsize >= 4) {
            float32x4_t s0_f32 = vld1q_f32(srcPtr);
            float16x4_t d0_f16 = vcvt_f16_f32(s0_f32);

            vst1_lane_f16(dstPtr,              d0_f16, 0);
            vst1_lane_f16(dstPtr + dstStride1, d0_f16, 1);
            vst1_lane_f16(dstPtr + 2 * dstStride1, d0_f16, 2);
            vst1_lane_f16(dstPtr + 3 * dstStride1, d0_f16, 3);

            srcPtr += 4;
            dstPtr += 4 * dstStride1;
            realsize -= 4;
        }

        for (; realsize > 0; --realsize) {
            *dstPtr = (FLOAT16)(*srcPtr);
            srcPtr++;
            dstPtr += dstStride1;
        }
    }
}

static void MNNAttenPackAndConvertFp32(float* dst, float* src, const int32_t* units, size_t depth, size_t planesize) {
    int32_t eP = units[0];
    int32_t lP = units[1]; // Now lP=1 or 2

    if (lP != 1 && lP != 2) {
        MNN_ERROR("This function only supports lP=1 or 2\n");
        return;
    }

    // src [depth, planesize] (float32)
    // dst [depth/eP, planesize/lP, eP, lP] (float16)

    if (lP == 1) {
        MNNAttenPackAndConvertFp32LP1(dst, src, units, depth, planesize);
        return;
    }

    auto dstStride1 = eP * lP;
    auto dstStride0 = UP_DIV(planesize, lP) * dstStride1;

    for (int i = 0; i < depth; ++i) {
        size_t realsize = planesize;
        const float* srcPtr = src + i * planesize;
        FLOAT16* dstPtr = (FLOAT16*)dst + (i % eP) * lP + (i / eP) * dstStride0;

        while (realsize >= 16) {
            float32x4_t s0 = vld1q_f32(srcPtr);
            float32x4_t s1 = vld1q_f32(srcPtr + 4);
            float32x4_t s2 = vld1q_f32(srcPtr + 8);
            float32x4_t s3 = vld1q_f32(srcPtr + 12);

            float16x4_t h0 = vcvt_f16_f32(s0);
            float16x4_t h1 = vcvt_f16_f32(s1);
            float16x4_t h2 = vcvt_f16_f32(s2);
            float16x4_t h3 = vcvt_f16_f32(s3);

            vst1_lane_u32((uint32_t*)dstPtr, vreinterpret_u32_f16(h0), 0);
            vst1_lane_u32((uint32_t*)(dstPtr + dstStride1), vreinterpret_u32_f16(h0), 1);

            vst1_lane_u32((uint32_t*)(dstPtr + 2 * dstStride1), vreinterpret_u32_f16(h1), 0);
            vst1_lane_u32((uint32_t*)(dstPtr + 3 * dstStride1), vreinterpret_u32_f16(h1), 1);

            vst1_lane_u32((uint32_t*)(dstPtr + 4 * dstStride1), vreinterpret_u32_f16(h2), 0);
            vst1_lane_u32((uint32_t*)(dstPtr + 5 * dstStride1), vreinterpret_u32_f16(h2), 1);

            vst1_lane_u32((uint32_t*)(dstPtr + 6 * dstStride1), vreinterpret_u32_f16(h3), 0);
            vst1_lane_u32((uint32_t*)(dstPtr + 7 * dstStride1), vreinterpret_u32_f16(h3), 1);

            realsize -= 16;
            srcPtr += 16;
            dstPtr += 8 * dstStride1;
        }

        if (realsize >= 8) {
            float32x4_t s0 = vld1q_f32(srcPtr);
            float32x4_t s1 = vld1q_f32(srcPtr + 4);

            float16x4_t h0 = vcvt_f16_f32(s0);
            float16x4_t h1 = vcvt_f16_f32(s1);

            vst1_lane_u32((uint32_t*)dstPtr, vreinterpret_u32_f16(h0), 0);
            vst1_lane_u32((uint32_t*)(dstPtr + dstStride1), vreinterpret_u32_f16(h0), 1);

            vst1_lane_u32((uint32_t*)(dstPtr + 2 * dstStride1), vreinterpret_u32_f16(h1), 0);
            vst1_lane_u32((uint32_t*)(dstPtr + 3 * dstStride1), vreinterpret_u32_f16(h1), 1);

            realsize -= 8;
            srcPtr += 8;
            dstPtr += 4 * dstStride1;
        }

        if (realsize >= 4) {
            float32x4_t s0 = vld1q_f32(srcPtr);
            float16x4_t h0 = vcvt_f16_f32(s0);

            vst1_lane_u32((uint32_t*)dstPtr, vreinterpret_u32_f16(h0), 0);
            vst1_lane_u32((uint32_t*)(dstPtr + dstStride1), vreinterpret_u32_f16(h0), 1);

            realsize -= 4;
            srcPtr += 4;
            dstPtr += 2 * dstStride1;
        }

        if (realsize >= 2) {
            float32x2_t s0 = vld1_f32(srcPtr);
            float16x4_t h0 = vcvt_f16_f32(vcombine_f32(s0, s0));

            vst1_lane_u32((uint32_t*)dstPtr, vreinterpret_u32_f16(h0), 0);

            realsize -= 2;
            srcPtr += 2;
            dstPtr += dstStride1;
        }

        if (realsize > 0) {
            dstPtr[0] = (FLOAT16)srcPtr[0];
            dstPtr[1] = (FLOAT16)0.0f;
        }
    }
}

static void MNNQuantAttentionKeyFP16(int8_t* dst, const float* source, float* sumKeyPtr, float* maxKeyPtr, int32_t* params) {
    int32_t kvNumHead = params[0];
    int32_t seqLen = params[1];
    int32_t headDim = params[2];
    int32_t blockNum = params[3];

    int32_t lP = params[5];
    int32_t hP = params[6];
    int32_t pastLength = params[7];
    int32_t kvHeadIdx = params[8];

    auto blockHeadDim = UP_DIV(headDim, blockNum);
    auto weightStride1 = ROUND_UP(blockHeadDim, lP) * hP;
    auto weightStride2 = lP * hP;
    auto packedWeightStride1 = weightStride1 + 2 * sizeof(float) * hP;

    auto sourceFp16 = (FLOAT16*)source;
    auto maxKeyFp16 = (FLOAT16*)maxKeyPtr;
    int8_t tempBuffer[8];
    float32x4_t neg128Vec = vdupq_n_f32(-128.0f);

    // Get max: [1, headDim]
    if (seqLen > 1) {
        for (int s = 0; s < seqLen; ++s) {
            const FLOAT16* keySrc = sourceFp16 + s * kvNumHead * headDim + kvHeadIdx * headDim;
            int d = 0;
            for (; d <= headDim - 16; d += 16) {
                float16x8_t maxVec0 = vld1q_f16(maxKeyFp16 + d);
                float16x8_t maxVec1 = vld1q_f16(maxKeyFp16 + d + 8);
                float16x8_t srcVec0 = vld1q_f16(keySrc + d);
                float16x8_t srcVec1 = vld1q_f16(keySrc + d + 8);
                maxVec0 = vmaxq_f16(maxVec0, srcVec0);
                maxVec1 = vmaxq_f16(maxVec1, srcVec1);
                vst1q_f16(maxKeyFp16 + d, maxVec0);
                vst1q_f16(maxKeyFp16 + d + 8, maxVec1);
            }
            for (; d <= headDim - 8; d += 8) {
                float16x8_t maxVec = vld1q_f16(maxKeyFp16 + d);
                float16x8_t srcVec = vld1q_f16(keySrc + d);
                maxVec = vmaxq_f16(maxVec, srcVec);
                vst1q_f16(maxKeyFp16 + d, maxVec);
            }
            for (; d < headDim; ++d) {
                maxKeyFp16[d] = ALIMAX(maxKeyFp16[d], keySrc[d]);
            }
        }
    }

    // Quant fp16
    for (int s = 0; s < seqLen; s++) {
        const FLOAT16* keySrc = sourceFp16 + s * kvNumHead * headDim + kvHeadIdx * headDim;

        float16x8_t minVec = vdupq_n_f16(keySrc[0]);
        float16x8_t maxVec = vdupq_n_f16(keySrc[0]);

        int d = 0;
        for (; d <= headDim - 8; d += 8) {
            float16x8_t srcVec = vld1q_f16(keySrc + d);
            float16x8_t maxKeyVec = vld1q_f16(maxKeyFp16 + d);
            float16x8_t keyDataF16 = vsubq_f16(srcVec, maxKeyVec);

            minVec = vminq_f16(minVec, keyDataF16);
            maxVec = vmaxq_f16(maxVec, keyDataF16);

            float32x4_t keyDataF32Low = vcvt_f32_f16(vget_low_f16(keyDataF16));
            float32x4_t keyDataF32High = vcvt_f32_f16(vget_high_f16(keyDataF16));
        }

        FLOAT16 minKey = vminvq_f16(minVec);
        FLOAT16 maxKey = vmaxvq_f16(maxVec);

        for (; d < headDim; ++d) {
            auto keydata = keySrc[d] - maxKeyFp16[d];
            minKey = ALIMIN(minKey, keydata);
            maxKey = ALIMAX(maxKey, keydata);
        }

        int outIndex = (pastLength + s) / hP;
        int inIndex  = (pastLength + s) % hP;

        float range = (float)maxKey - (float)minKey;
        float quantScaleVal = 0;
        float biasVal = minKey + 128.0f * range / 255.0;
        if (range <= 1e-6f) {
            quantScaleVal = 0.f;
        } else {
            quantScaleVal = 255.0f / range;
        }

        for (int k = 0; k < blockNum; ++k) {
            int8_t* weightDstBase = dst + outIndex * blockNum * packedWeightStride1 + k * packedWeightStride1;
            float* scaleDst = (float*)(weightDstBase + weightStride1);
            float* biasDst = scaleDst + hP;

            scaleDst[inIndex] = range / 255.f;
            biasDst[inIndex] = biasVal;

            float32x4_t scaleVecFp32 = vdupq_n_f32(quantScaleVal);
            float32x4_t negMinKeyVecF32 = vdupq_n_f32(-(float)minKey);

            const FLOAT16* currentKeyBlock = keySrc + k * blockHeadDim;
            const FLOAT16* currentMaxBlock = maxKeyFp16 + k * blockHeadDim;

            int32x4_t sumInt32_0 = vdupq_n_s32(0);
            int32x4_t sumInt32_1 = vdupq_n_s32(0);
            int headDimIdx = 0;
            for (; headDimIdx <= blockHeadDim - 8; headDimIdx += 8) {
                float16x8_t srcVecFp16 = vld1q_f16(currentKeyBlock + headDimIdx);
                float16x8_t maxVecFp16 = vld1q_f16(currentMaxBlock + headDimIdx);

                float16x8_t keyDataF16 = vsubq_f16(srcVecFp16, maxVecFp16);

                float32x4_t keyDataLowFp32 = vcvt_f32_f16(vget_low_f16(keyDataF16));
                float32x4_t keyDataHighFp32 = vcvt_f32_f16(vget_high_f16(keyDataF16));

                keyDataLowFp32 = vaddq_f32(keyDataLowFp32, negMinKeyVecF32);
                keyDataHighFp32 = vaddq_f32(keyDataHighFp32, negMinKeyVecF32);

                keyDataLowFp32 = vmulq_f32(keyDataLowFp32, scaleVecFp32);
                keyDataHighFp32 = vmulq_f32(keyDataHighFp32, scaleVecFp32);

                keyDataLowFp32 = vaddq_f32(keyDataLowFp32, neg128Vec);
                keyDataHighFp32 = vaddq_f32(keyDataHighFp32, neg128Vec);

                int32x4_t keyDataLowInt32 = vcvtaq_s32_f32(keyDataLowFp32);
                int32x4_t keyDataHighInt32 = vcvtaq_s32_f32(keyDataHighFp32);

                int16x4_t s16Low = vmovn_s32(keyDataLowInt32);
                int16x4_t s16High = vmovn_s32(keyDataHighInt32);

                int16x8_t s16Combined = vcombine_s16(s16Low, s16High);

                // sum
                sumInt32_0 = vaddq_s32(sumInt32_0, keyDataLowInt32);
                sumInt32_1 = vaddq_s32(sumInt32_1, keyDataHighInt32);

                int8x8_t s8Vec = vqmovn_s16(s16Combined);

                if (lP == 8) {
                    int i = headDimIdx / lP;
                    int8_t* dstPtr = weightDstBase + i * weightStride2 + inIndex * lP;
                    vst1_s8(dstPtr, s8Vec);
                } else if (lP == 4) {
                    vst1_s8(tempBuffer, s8Vec);
                    int iLow = headDimIdx / lP;
                    int iHigh = (headDimIdx + 4) / lP;

                    int8_t* dstPtrLow = weightDstBase + iLow * weightStride2 + inIndex * lP;
                    int8_t* dstPtrHigh = weightDstBase + iHigh * weightStride2 + inIndex * lP;

                    std::memcpy(dstPtrLow, tempBuffer, 4);
                    std::memcpy(dstPtrHigh, tempBuffer + 4, 4);
                } else {
                    vst1_s8(tempBuffer, s8Vec);
                    for (int nk = 0; nk < 8; ++nk) {
                        int headDimCurr = headDimIdx + nk;
                        int i = headDimCurr / lP;
                        int j = headDimCurr % lP;
                        weightDstBase[i * weightStride2 + inIndex * lP + j] = tempBuffer[nk];
                    }
                }

            }

            int32_t sumInt32 = vaddvq_s32(sumInt32_0) + vaddvq_s32(sumInt32_1);

            for (; headDimIdx < blockHeadDim; ++headDimIdx) {
                int i = headDimIdx / lP;
                int j = headDimIdx % lP;
                float keyVal = (float)currentKeyBlock[headDimIdx] - (float)currentMaxBlock[headDimIdx];
                float quantVal = (keyVal - minKey) * quantScaleVal - 128.0f;
                int32_t roundedVal = static_cast<int32_t>(roundf(quantVal));
                int8_t finalVal = static_cast<int8_t>(std::max(-128, std::min(127, roundedVal)));
                weightDstBase[i * weightStride2 + inIndex * lP + j] = finalVal;
                sumInt32 += finalVal;
            }

            // store sum
            sumKeyPtr[outIndex * hP + inIndex] = sumInt32 * range / 255.f + (minKey * (float)(blockHeadDim) + 128.0f * range * (float)(blockHeadDim) / 255.0);
        }
    }
}

static void MNNQuantAttentionValueFP16(int8_t* dst, const float* source, float* valueSum, int32_t* params) {
    // float   value src : [kvSeq,kvNumHead,headDim]
    // int8_t  value dest: [updiv(maxLength,flashAttentionBlockKv), updiv(headDim,hp),updiv(flashAttentionBlockKv,lp),hp,lp]
    // float   value sum: [updiv(maxLength,flashAttentionBlockKv), roundup(headDim,hp)]
    int32_t kvNumHead = params[0];
    int32_t seqLen = params[1];
    int32_t headDim = params[2];
    int32_t blockNum = params[3];
    int32_t maxLength = params[4];

    int32_t lP = params[5];
    int32_t hP = params[6];
    int32_t pastLength = params[7];
    int32_t kvHeadIdx = params[8];

    int32_t flashAttentionBlockKv = params[9];

    auto blockKvseq = UP_DIV(seqLen + pastLength, blockNum);
    auto weightStride2 = lP * hP;
    auto weightStride1 = UP_DIV(flashAttentionBlockKv, lP) * weightStride2;

    auto packedStride1 = (int)(weightStride1 + 2 * hP * sizeof(float));
    auto packedStride0 = UP_DIV(headDim, hP) * packedStride1;

    auto srcStride0 = kvNumHead * headDim;

    auto sourceFp16 = (FLOAT16*)source;

    // quant scale & bias
    if (pastLength == 0) {
        for (int d = 0; d < headDim; ++d) {
            float* scalePtr = (float*)(dst + (d / hP) * packedStride1 + weightStride1) + (d % hP);
            float* biasPtr = scalePtr + hP;

            // find min,max
            float dMax = sourceFp16[d + kvHeadIdx * headDim];
            float dMin = dMax;
            for (int s = 0; s < seqLen; ++s) {
                float data = sourceFp16[s * srcStride0 + d + kvHeadIdx * headDim];
                dMax = ALIMAX(dMax, data);
                dMin = ALIMIN(dMin, data);
            }

            // scale & bias
            float range = dMax - dMin;
            if (range < 1e-6) {
                scalePtr[0] = 0.f;
                biasPtr[0] = dMax;
            } else {
                float scale = range / 255.f;
                float bias  = range / 255.f * 128.f + dMin;
                scalePtr[0] = scale;
                biasPtr[0] = bias;
            }
        }
    }

    // copy the scale&bias to each blockKv
    //                                    pastLength == 0: First time prefill
    // (seqLen + pastLength) % flashAttentionBlockKv == 0: Open a new blockKv
    if (pastLength == 0 || (pastLength % flashAttentionBlockKv) == 0) {
        int32_t d0 = UP_DIV(maxLength, flashAttentionBlockKv);
        int32_t d1 = UP_DIV(headDim, hP);
        for (int k = 0; k < d0; ++k) {
            for (int r = 0; r < d1; ++r) {
                float* scalePtr = (float*)(dst + k * packedStride0 + r * packedStride1 + weightStride1);
                float* biasPtr  = scalePtr + hP;
                memcpy(scalePtr, dst + r * packedStride1 + weightStride1, hP * sizeof(float));
                memcpy(biasPtr, dst + r * packedStride1 + weightStride1 + hP * sizeof(float), hP * sizeof(float));
            }
        }
    }

    std::vector<float> qScales(headDim);
    std::vector<float> qBiases(headDim);
    std::vector<float> deqScales(headDim);
    std::vector<float> deqBiases(headDim);
    int8_t tmpQ[8];

    for (int d = 0; d < headDim; ++d) {
        float* scaleBase = (float*)(dst + (d / hP) * packedStride1 + weightStride1) + (d % hP);
        float* biasBase = scaleBase + hP;

        float s_val = scaleBase[0];
        float b_val = biasBase[0];

        deqScales[d] = s_val;
        deqBiases[d] = b_val;

        bool is_small = s_val < 1e-6f;
        qScales[d] = is_small ? 0.0f : (1.0f / s_val);
        qBiases[d] = is_small ? 0.0f : (-b_val / s_val);
    }

    const __fp16* srcBasePtr = sourceFp16 + kvHeadIdx * headDim;

    const int32_t sumStride = ROUND_UP(headDim, hP);

    for (int s = 0; s < seqLen; ++s) {
        int kvSeqIndx = s + pastLength;

        int blkIdx = kvSeqIndx / flashAttentionBlockKv;
        int blkRem = kvSeqIndx % flashAttentionBlockKv;

        int idxInnerCommon = blkIdx * packedStride0 + (blkRem / lP) * weightStride2 + (blkRem % lP);

        float* curSumRow = valueSum + blkIdx * sumStride;

        const __fp16* srcRow = srcBasePtr + s * srcStride0;

        int d = 0;
        for (; d <= headDim - 8; d += 8) {
            // --- Load Source ---
            float16x8_t vSrc16 = vld1q_f16(srcRow + d);
            float32x4_t vSrc0 = vcvt_f32_f16(vget_low_f16(vSrc16));
            float32x4_t vSrc1 = vcvt_high_f32_f16(vSrc16);

            // --- Load Quant Params ---
            float32x4_t vQs0 = vld1q_f32(&qScales[d]);
            float32x4_t vQb0 = vld1q_f32(&qBiases[d]);
            float32x4_t vQs1 = vld1q_f32(&qScales[d + 4]);
            float32x4_t vQb1 = vld1q_f32(&qBiases[d + 4]);

            // --- Quantize: x * qs + qb ---
            float32x4_t vRes0 = vaddq_f32(vmulq_f32(vSrc0, vQs0), vQb0);
            float32x4_t vRes1 = vaddq_f32(vmulq_f32(vSrc1, vQs1), vQb1);

            // --- Round & Saturate ---
            int32x4_t vInt32_0 = vcvtaq_s32_f32(vRes0);
            int32x4_t vInt32_1 = vcvtaq_s32_f32(vRes1);

            int16x8_t vInt16 = vcombine_s16(vqmovn_s32(vInt32_0), vqmovn_s32(vInt32_1));
            int8x8_t vInt8 = vqmovn_s16(vInt16); // Clamp to [-128, 127]

            vst1_s8(tmpQ, vInt8);
            for (int k = 0; k < 8; ++k) {
                int cur_d = d + k;
                int dstOffset = (cur_d / hP) * packedStride1 + idxInnerCommon + (cur_d % hP) * lP;
                dst[dstOffset] = tmpQ[k];
            }

            int16x8_t vXq16 = vmovl_s8(vInt8);
            float32x4_t vXqF0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vXq16)));
            float32x4_t vXqF1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vXq16)));

            float32x4_t vDs0 = vld1q_f32(&deqScales[d]);
            float32x4_t vDb0 = vld1q_f32(&deqBiases[d]);
            float32x4_t vDs1 = vld1q_f32(&deqScales[d + 4]);
            float32x4_t vDb1 = vld1q_f32(&deqBiases[d + 4]);

            // Dequant
            float32x4_t vDeq0 = vaddq_f32(vmulq_f32(vXqF0, vDs0), vDb0);
            float32x4_t vDeq1 = vaddq_f32(vmulq_f32(vXqF1, vDs1), vDb1);

            float* sumPtr = curSumRow + d;
            vst1q_f32(sumPtr, vaddq_f32(vld1q_f32(sumPtr), vDeq0));
            vst1q_f32(sumPtr + 4, vaddq_f32(vld1q_f32(sumPtr + 4), vDeq1));
        }

        for (; d < headDim; ++d) {
            float xf = (float)srcRow[d];

            float val_f = xf * qScales[d] + qBiases[d];
            int32_t val_i = (int32_t)roundf(val_f);
            if (val_i > 127) val_i = 127;
            if (val_i < -128) val_i = -128;
            int8_t xq = (int8_t)val_i;

            int dstOffset = (d / hP) * packedStride1 + idxInnerCommon + (d % hP) * lP;
            dst[dstOffset] = xq;

            curSumRow[d] += ((float)xq * deqScales[d] + deqBiases[d]);
        }
    }

/*
    // Quant fp16
    for (int d = 0; d < headDim; ++d) {
        // dst address
        int idxBase = (d / hP) * packedStride1 + (d % hP) * lP;
        int8_t*   dstBase = dst + idxBase;
        float*  scaleBase = (float*)(dst + (d / hP) * packedStride1 + weightStride1) + (d % hP);
        float*   biasBase = scaleBase + hP;
        float*   sumBase = valueSum + (d / hP) * hP + (d % hP);

        float qscale = scaleBase[0] < 1e-6 ? 0 : 1.0f / scaleBase[0];
        float qbias = scaleBase[0] < 1e-6 ? 0 : (-biasBase[0] / scaleBase[0]);
        // quant
        for (int s = 0; s < seqLen; ++s) {
            int kvSeqIndx = s + pastLength;
            int idxInner = (kvSeqIndx / flashAttentionBlockKv) * packedStride0 + (kvSeqIndx % flashAttentionBlockKv) / lP * weightStride2 + (kvSeqIndx % flashAttentionBlockKv) % lP;
            float xf = sourceFp16[s * srcStride0 + d + kvHeadIdx * headDim];
            int8_t xq = ALIMAX(ALIMIN(127, static_cast<int32_t>(roundf(xf * qscale + qbias))), -128);
            dstBase[idxInner] = xq;

            // sum
            int idxSum = (kvSeqIndx / flashAttentionBlockKv) * ROUND_UP(headDim, hP);
            sumBase[idxSum] += ((float)xq * scaleBase[0] + biasBase[0]);
        }
    }
        */
}

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

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

static void MNNDynamicQuantFP16(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack, const float* bias = nullptr) {
    if (pack == 8) {
        MNNDynamicQuantFP16_Pack8(src, dst, scale, src_depth_quad,realSize, nullptr, pack);
        return;
    }
    if (pack == 4) {
        MNNDynamicQuantFP16_Pack4(src, dst, scale, src_depth_quad,realSize, nullptr, pack);
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

static void MNNAsyQuantFunc_Arm82(int8_t* dst, const float* src, float* qscale, float* qbias, const size_t* info) {
    // input shape: [kernelsize, blockNum, blockLU, EP, LP]
    // qscale&qbias [blockNum, EP]
    auto blockNum = info[0];
    auto EP = info[1];        // real area for data
    auto LP = info[2];        // Innermost data layout, may come from backend's pack or gemmint8 units' SRC_UNIT
    auto DST_XUNIT = info[3]; // backend gemmint8 units
    auto SRC_UNIT = info[4];
    auto kernelsize = info[5];
    auto blockLU = info[6];
    auto stride0 = blockNum * blockLU * EP * LP;
    auto stride1 = blockLU * EP * LP;
    auto srcPtr = (FLOAT16*)src;
#ifdef __aarch64__
    if (LP == 4 || LP == 8) {
        for (int k = 0; k < kernelsize; ++k) {
            for (int i = 0; i < blockNum; ++i) {
                if (LP == 4) {
                    MNNDynamicQuantFP16_Pack4((float*)(srcPtr + k * stride0 + i * stride1), dst + k * stride0 + i * stride1, qscale + i * EP, blockLU, EP, qbias + i * EP, LP);
                }
                if (LP == 8) {
                    MNNDynamicQuantFP16_Pack8((float*)(srcPtr + k * stride0 + i * stride1), dst + k * stride0 + i * stride1, qscale + i * EP, blockLU, EP, qbias + i * EP, LP);
                }
            }
        }
        return;
    }
#endif
    for (int i = 0; i < EP; ++i) {
        for (int bk = 0; bk < blockNum; ++bk) {
            float quant_scale = qscale[i + bk * EP];
            float quant_bias  = qbias[i + bk * EP];
            for (int n = 0; n < kernelsize; ++n) {
                for (int k = 0; k < blockLU; ++k) {
                    for (int j = 0; j < LP; ++j) {
                        int dataIndx = n * stride0 + bk * stride1 + k * EP * LP + i * LP + j;
                        auto data_ = static_cast<float>(srcPtr[dataIndx]);
                        int qval = static_cast<int32_t>(roundf(data_ * quant_scale + quant_bias));
                        dst[dataIndx] = qval;
                    }
                }
            }
        }
    }
}

static void MNNAsyQuantInfo_FP16(float* scale, float* bias, float* qscale, float* qbias, float* dstMin, float* dstMax, float* src, const size_t* info) {
    auto blockNum = info[0];
    auto plane = info[1];        // real area for data
    auto innerSide = info[2];    // Innermost data layout, may come from backend's pack or gemmint8 units' SRC_UNIT
    auto DST_XUNIT = info[3];
    auto kernelsize = info[5];
    auto blockLU = info[6];
    auto stride0 = blockNum * blockLU * plane * innerSide;
    auto stride1 = blockLU * plane * innerSide;
    auto srcPtr = (FLOAT16*)src;

    // input shape: [kernelsize,blocknum,blocklu,DST_XUNIT,SRC_UNIT] or [ic/core->pack, plane, core->pack]
    // dequant scale/bias : [EU, blockNum, step]
    // quant scale/bias: [blockNum, plane]
    if (info[7] == 1) { // scale&bias:[1]
        FLOAT16 maxval, minval;
        ARM82CountMinMaxValue(src, (float*)(&minval), (float*)(&maxval) , kernelsize * stride0);
        if (info[8] == 1 && (maxval - minval) > 1e-7) {
            if (minval > 0.f) {
                minval = 0.f;
            } else if (maxval < 0.f){
                maxval = 0.f;
            }
        }
        auto range = maxval - minval;
        if (range <= 1e-7) {
            scale[0] = 0.f;
            qscale[0] = 0.f;
            qbias[0] = 0.f;
            bias[0] = maxval;
        } else {
            qscale[0] = 255.f / range;
            scale[0] = range / 255.f;
            qbias[0] = roundf(-minval * 255.f / range)- 128.f;
            bias[0] = -qbias[0] * scale[0];
        }
        return;
    }

#ifdef __aarch64__
    if (DST_XUNIT == 12 || DST_XUNIT == 16) { // Arm82/SME2, fp16: core->pack=8, SRC_UNIT=4
        // max,min shape: [blockNum, EP]
        if (innerSide == 4) {
            for (int i = 0; i < kernelsize; ++i) {
                MNNLocalMinMaxFP16_Pack4(dstMin, dstMax, (float*)(srcPtr + i * stride0), blockNum, blockLU, plane, innerSide, i);
            }
        }
        if (innerSide == 8) {
            for (int i = 0; i < kernelsize; ++i) {
                MNNLocalMinMaxFP16_Pack8(dstMin, dstMax, (float*)(srcPtr + i * stride0), blockNum, blockLU, plane, innerSide, i);
            }
        }
        // scale, bias
        if (DST_XUNIT == 12) {
            auto success = MNNAsyLocalQuantInfo_EP12_FP16(scale, bias, qscale, qbias, dstMin, dstMax, info);
            if (!success) {
                MNN_ERROR("Call error: MNNAsyLocalQuantInfo_EP12_FP16\n");
                return;
            }
            return;
        }
        if (DST_XUNIT == 16) {
            auto success = MNNAsyLocalQuantInfo_EP16_FP16(scale, bias, qscale, qbias, dstMin, dstMax, info);
            if (!success) {
                MNN_ERROR("Call error: MNNAsyLocalQuantInfo_EP16_FP16\n");
                return;
            }
            return;
        }
    }
    if (DST_XUNIT == 10 && innerSide == 8) { // Arm86, fp16: core->pack=8, SRC_UNIT=8
        // max,min shape: [blockNum, plane]
        for (int i = 0; i < kernelsize; ++i) {
            MNNLocalMinMaxFP16_Pack8(dstMin, dstMax, (float*)(srcPtr + i * stride0), blockNum, blockLU, plane, innerSide, i);
        }
        // scale, bias
        auto success = MNNAsyLocalQuantInfo_EP10_FP16(scale, bias, qscale, qbias, dstMin, dstMax, info);
        if (!success) {
            MNN_ERROR("Call error: MNNAsyLocalQuantInfo_EP10_FP16\n");
            return;
        }
        return;
    }
#else
    // aarch32
    // max,min shape: [blockNum, plane]
    auto minPtr = (FLOAT16*)dstMin;
    auto maxPtr = (FLOAT16*)dstMax;
    for (int i = 0; i < plane; ++i) {
        for (int bk = 0; bk < blockNum; ++bk) {
            auto idx0 = i * innerSide + bk * stride1;
            auto max_ = srcPtr[idx0];
            auto min_ = max_;
            for (int n = 0; n < kernelsize; ++n) {
                for (int k = 0; k < blockLU; ++k) {
                    for (int j = 0; j < innerSide; ++j) {
                        auto dataIndx = idx0 + n * stride0 + k * (plane * innerSide) + j;
                        auto data_ = srcPtr[dataIndx];
                        max_ = ALIMAX(max_, data_);
                        min_ = ALIMIN(min_, data_);
                    }
                }
            }
            auto sindx = i + bk * plane;
            minPtr[sindx] = min_;
            maxPtr[sindx] = max_;
        }
    }
    // scale, bias
    for (int i = 0; i < plane; ++i) {
        auto step = ALIMIN(DST_XUNIT, plane - (i / DST_XUNIT) * DST_XUNIT);
        auto sind0 = (i / DST_XUNIT) * DST_XUNIT * blockNum + (i % DST_XUNIT);
        for (int k = 0; k < blockNum; ++k) {
            auto sind = sind0 + k * step;
            auto qind = i + k * plane;
            auto max_ = (float)maxPtr[qind];
            auto min_ = (float)minPtr[qind];
            auto range = max_ - min_;
            if (fabs(range) < 1e-7) {
                qscale[qind] = 0.f;
                qbias[qind] = 0.f;
                scale[sind] = 0.f;
                bias[sind] = max_;
            } else {
                qscale[qind] = 255.f / range;
                qbias[qind] = -min_ * 255.f / range - 128.0f;
                scale[sind] = range / 255.f;
                bias[sind] = min_ + (128.f / 255.f) * range;

            }
        }
    }
#endif
}

#endif // MNN_LOW_MEMORY

#define EXP_APPROX_MIN_INPUT vdupq_n_f32(-88.0f)
#define EXP_APPROX_MAX_INPUT vdupq_n_f32(88.0f)
#define EXP_APPROX_LN2         vdupq_n_f32(0.69314718056f)  // ln(2)
#define EXP_APPROX_LN2_INV     vdupq_n_f32(1.44269504089f)   // 1/ln(2)
// Fourth-order polynomial approximation coefficients of exp(r):
// P(x) = c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
#define EXP_APPROX_C4          vdupq_n_f32(0.0416624f)
#define EXP_APPROX_C3          vdupq_n_f32(0.166665f)
#define EXP_APPROX_C2          vdupq_n_f32(0.500000f)
#define EXP_APPROX_C1          vdupq_n_f32(1.0f)
#define EXP_APPROX_C0          vdupq_n_f32(1.0f)

#ifndef __aarch64__
static inline float32x4_t vrndaq_f32_compat(float32x4_t x) {
    float32x4_t sign = vbslq_f32(vdupq_n_u32(0x80000000), x, vdupq_n_f32(0.0f));
    return vcvtq_f32_s32(vcvtq_s32_f32(vaddq_f32(x, vbslq_f32(vcltq_f32(x, vdupq_n_f32(0.0f)), vdupq_n_f32(-0.5f), vdupq_n_f32(0.5f)))));
}
#endif

static inline float32x4_t expApprox(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, EXP_APPROX_MIN_INPUT), EXP_APPROX_MAX_INPUT);

    float32x4_t k_float;
    float32x4_t r;
    float32x4_t exp_r;
#if defined(__aarch64__)
    k_float = vrndaq_f32(vmulq_f32(x, EXP_APPROX_LN2_INV));

    // r = x - k * ln(2)
    r = vfmsq_f32(x, k_float, EXP_APPROX_LN2);

    // P(r) = (c0 + c2*r^2 + c4*r^4) + r*(c1 + c3*r^2)
    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t p_odd = vfmaq_f32(EXP_APPROX_C1, EXP_APPROX_C3, r2);

    float32x4_t p_even = vfmaq_f32(EXP_APPROX_C0, EXP_APPROX_C2, r2);
    p_even = vfmaq_f32(p_even, EXP_APPROX_C4, vmulq_f32(r2, r2));
    exp_r = vfmaq_f32(p_even, p_odd, r);
#else

    k_float = vrndaq_f32_compat(vmulq_f32(x, EXP_APPROX_LN2_INV));


    r = vsubq_f32(x, vmulq_f32(k_float, EXP_APPROX_LN2));

    // 2. c0 + r*(c1 + r*(c2 + r*(c3 + r*c4)))
    exp_r = vmlaq_f32(EXP_APPROX_C3, EXP_APPROX_C4, r); // c3 + c4*r
    exp_r = vmlaq_f32(EXP_APPROX_C2, exp_r, r);         // c2 + r*(...)
    exp_r = vmlaq_f32(EXP_APPROX_C1, exp_r, r);         // c1 + r*(...)
    exp_r = vmlaq_f32(EXP_APPROX_C0, exp_r, r);         // c0 + r*(...)

#endif

    int32x4_t k_int = vcvtq_s32_f32(k_float);
    int32x4_t k_shifted = vshlq_n_s32(k_int, 23);
    return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(exp_r), k_shifted));
}
static void MNNSoftmaxFp16(float* dest, const float* source, float* runningMax, float* runningSum, float* updateScale, int outside, int reduceSize, int kvSeqOffset, int validOffset, int pack, bool mask) {
    const int reduceSize_8 = UP_DIV(reduceSize, 8);
    auto softmaxDst = (FLOAT16*)dest;
    auto softmaxSrc = (FLOAT16*)source;

    // source shape: [reduceSizeOuter, outside, reduceSizeInner]
    // for C4, [up_div(reduceSize,4), outside,4] => reduceSizeOuter=up_div(reduceSize,4), reduceSizeInner=4
    // for C,  [outside, reduceSize]             => reduceSizeOuter=1, reduceSizeInner=reduceSize

    const int packUnit = 8;
    int reduceSizeOuter = 1;
    int reduceSizeInner = reduceSize;
    int stride0         = packUnit;
    if (pack > 1) {
        reduceSizeOuter = UP_DIV(reduceSize, pack);
        reduceSizeInner = pack;
        stride0         = outside * reduceSizeInner;
    }


    for (int k = 0; k < outside; ++k) {
        if (mask && kvSeqOffset > k + validOffset) {
            if (updateScale){
                updateScale[k] = 1;
            }
            for (int j = 0; j < reduceSizeOuter; ++j) {
                int i = 0;
                for (; i < reduceSizeInner; i += packUnit) {
                    auto destPtr = softmaxDst + j * stride0 + k * reduceSizeInner + i;
                    vst1q_f16(destPtr, vdupq_n_f16(0.0f));
                }
                if (i < reduceSizeInner) {
                    memset(softmaxDst + j * stride0 + k * reduceSizeInner + i, 0, (reduceSizeInner - i) * sizeof(__fp16));
                }
            }
            continue;
        }

        const int validReduceSize = mask ? ALIMIN(reduceSize, k + (validOffset + 1) - kvSeqOffset) : reduceSize;
        const int remain = validReduceSize % packUnit;
        const int sizeDiv = validReduceSize / packUnit;

        // 1. newMax
        float oldMax = -65504.0f;
        if (runningMax) {
            oldMax = runningMax[k];
        }

        auto newMaxVec = vdupq_n_f16(-65504.0f);

        for (int j = 0; j < sizeDiv; ++j) {
            auto srcPtr = softmaxSrc + j * stride0 + k * reduceSizeInner;
            float16x8_t srcVec = vld1q_f16(srcPtr);
            newMaxVec = vmaxq_f16(newMaxVec, srcVec);
        }
        float newMax = vmaxvq_f16(newMaxVec);

        if (remain > 0) {
            auto srcPtr = softmaxSrc + sizeDiv * stride0  + k * reduceSizeInner;
            for (int i = 0; i < remain; ++i) {
                newMax = ALIMAX(newMax, (float)srcPtr[i]);
            }
        }

        const float finalMax = ALIMAX(oldMax, newMax);
        const float32x4_t finalMaxVec = vdupq_n_f32(finalMax);

        // 2. exp(x - finalMax)
        float sum = 0.0f;
        float32x4_t sumVec0 = vdupq_n_f32(0.0f);
        float32x4_t sumVec1 = vdupq_n_f32(0.0f);

        for (int j = 0; j < sizeDiv; ++j) {
            auto idx = j * stride0 + k * reduceSizeInner;
            auto srcPtr = softmaxSrc + idx;
            auto dstPtr = softmaxDst + idx;

            float16x8_t srcVec = vld1q_f16(srcPtr);

            // F16 -> F32
            float32x4_t srcLo = vcvt_f32_f16(vget_low_f16(srcVec));
            float32x4_t srcHi = vcvt_f32_f16(vget_high_f16(srcVec));

            // sub max
            srcLo = vsubq_f32(srcLo, finalMaxVec);
            srcHi = vsubq_f32(srcHi, finalMaxVec);

            // exp
            srcLo = expApprox(srcLo);
            srcHi = expApprox(srcHi);

            // sum
            sumVec0 = vaddq_f32(sumVec0, srcLo);
            sumVec1 = vaddq_f32(sumVec1, srcHi);

            // F32 -> F16 and store
            vst1q_f16(dstPtr, vcombine_f16(vcvt_f16_f32(srcLo), vcvt_f16_f32(srcHi)));
        }

        if (remain > 0) {
            auto idx = sizeDiv * stride0  + k * reduceSizeInner;
            auto srcPtr = softmaxSrc + idx;
            auto dstPtr = softmaxDst + idx;

            __fp16 tempDst[8] = {0, 0, 0, 0, 0, 0, 0, 0};

            for(int i = 0; i < remain; ++i) {
                float val = expf((float)srcPtr[i] - finalMax);
                sum += val;
                tempDst[i] = (__fp16)val;
            }
            if (pack > 1) {
                memcpy(dstPtr, tempDst, packUnit * sizeof(__fp16));
            } else {
                memcpy(dstPtr, tempDst, remain * sizeof(__fp16));
            }
        }

        sum += vaddvq_f32(sumVec0) + vaddvq_f32(sumVec1);

        // 3. update runningMax, runningSum, scale or normalize softmax results
        if (runningMax != nullptr && runningSum != nullptr && updateScale != nullptr) {
            // update runningSum, runningMax, scale
            float scaleForSum = expf(oldMax - finalMax);
            runningSum[k] = runningSum[k] * scaleForSum + sum;
            runningMax[k] = finalMax;
            updateScale[k] = scaleForSum;
        } else {
            // Normalize softmax results
            if (runningMax != nullptr && runningSum != nullptr) {
                sum += runningSum[k] * expf(oldMax - finalMax);
            }
            float scale = 1.0f / (sum + 1e-20f);
            float16x8_t scale_vec = vdupq_n_f16((__fp16)scale);

            for (int j = 0; j < sizeDiv; ++j) {
                auto pDest = softmaxDst + j * stride0 + k * reduceSizeInner;
                float16x8_t data = vld1q_f16(pDest);
                data = vmulq_f16(data, scale_vec);
                vst1q_f16(pDest, data);
            }

            if (remain > 0) {
                auto pDest = softmaxDst + sizeDiv * stride0 + k * reduceSizeInner;
                for (int i = 0; i < remain; ++i) {
                    pDest[i] = (__fp16)((float)pDest[i] * scale);
                }
            }
        }

        // 4. memset invalid positions to zero
        if (pack > 1) {
            if (validReduceSize % packUnit > 0) {
                memset(softmaxDst + sizeDiv * stride0 + k * reduceSizeInner + (validReduceSize % packUnit), 0, (packUnit - (validReduceSize % packUnit)) * sizeof(__fp16));
            }
            auto validDiv8 = UP_DIV(validReduceSize, packUnit);
            auto allDiv8 = UP_DIV(reduceSize, packUnit);
            for (int j = validDiv8; j < allDiv8; ++j) {
                auto destPtr = softmaxDst + j * stride0 + k * reduceSizeInner;
                memset(destPtr, 0, packUnit * sizeof(__fp16));
            }
        } else {
            memset(softmaxDst + k * reduceSizeInner + validReduceSize, 0, (reduceSize - validReduceSize) * sizeof(__fp16));
        }
    }
}

static CoreFunctions* gInstance = nullptr;
static CoreInt8Functions* gArm82CoreInt8Functions = nullptr;

bool Arm82Functions::init() {
    using Vec = MNN::Math::Vec<FLOAT16, 8>;
    auto origin = MNNGetCoreFunctions();
#define FUNC_PTR_ASSIGN(dst, src) dst = (decltype(dst))(src)
    gInstance = new CoreFunctions;
    gArm82CoreInt8Functions = new CoreInt8Functions;
    *gArm82CoreInt8Functions = *MNNGetInt8CoreFunctions();
    gInstance->int8MatmulRelatedFunctions = origin->int8MatmulRelatedFunctions;
    {
        if (origin->supportSDot) {
            gArm82CoreInt8Functions->MNNPackC4Int8ForMatMul_A = _Arm82MNNPackC4ForMatMul_A<12, 4>;
            gInstance->arm82MatmulRelatedFunctions = origin->arm82MatmulRelatedFunctions;
            gInstance->arm82MatmulRelatedFunctions.MNNPackC4Int8ForMatMul_A = _Arm82MNNPackC4ForMatMul_A<12, 4>;
        }
        if (origin->supportI8mm) {
            gArm82CoreInt8Functions->MNNPackC4Int8ForMatMul_A = _ArmBasicMNNPackC4ForMatMul_A_L8<10, 8>;
            gInstance->supportI8mm = true;
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
    gInstance->MNNReorderWeightInt4 = origin->MNNReorderWeightInt4;
    gInstance->MNNSumWeightInt8 = origin->MNNSumWeightInt8;
    gInstance->MNNSumWeightInt8SmeHp128 = origin->MNNSumWeightInt8SmeHp128;
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
    FUNC_PTR_ASSIGN(gInstance->MNNGetMatMulPackMode, Arm82MNNGetMatMulPackMode);
    FUNC_PTR_ASSIGN(gInstance->MNNPackedMatMul, MNNPackedMatMulFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNPackedMatMulRemain, MNNPackedMatMulRemainFP16);
    FUNC_PTR_ASSIGN(gInstance->MNNPackC4ForMatMul_A, Arm82MNNPackForMatMul_A);
    FUNC_PTR_ASSIGN(gInstance->MNNPackForMatMul_B, Arm82MNNPackForMatMul_B);

    FUNC_PTR_ASSIGN(gInstance->MNNSoftmax, MNNSoftmaxFp16);
#if defined(__aarch64__)
    gInstance->supportFp16arith = origin->supportFp16arith;
    gInstance->supportSDot = origin->supportSDot;
    gInstance->supportI8mm = origin->supportI8mm;
    gInstance->supportSME2 = origin->supportSME2;
    gInstance->smeCoreNumber = origin->smeCoreNumber;
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
    FUNC_PTR_ASSIGN(gInstance->MNNAsyQuantFunc, MNNAsyQuantFunc_Arm82);
    FUNC_PTR_ASSIGN(gInstance->MNNAsyQuantInfo, MNNAsyQuantInfo_FP16); // return 'plane' min&max
    FUNC_PTR_ASSIGN(gInstance->MNNDynamicUpdateConvBiasScale, origin->MNNDynamicUpdateConvBiasScale);

    if (origin->supportSDot) {
        FUNC_PTR_ASSIGN(gInstance->MNNGeneralIm2Col, MNNGeneralIm2col_Arm82);
        gInstance->arm82MatmulRelatedFunctions.MNNGeneralIm2Col = MNNGeneralIm2col_Arm82;
    }
    if (origin->supportI8mm) {
        FUNC_PTR_ASSIGN(gInstance->MNNGeneralIm2Col, MNNGeneralIm2col_Arm86);
    }
#endif // MNN_LOW_MEMORY
    FUNC_PTR_ASSIGN(gInstance->MNNCountMaxMinValue, ARM82CountMinMaxValue); // return one min&max
    FUNC_PTR_ASSIGN(gInstance->MNNSumByAxisLForMatmul_A, origin->MNNSumByAxisLForMatmul_A);
    FUNC_PTR_ASSIGN(gInstance->MNNDepthwiseConvFastKernel, MNNDepthwiseConvFastKernelFP16);
#endif // __aarch64__

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
    // Attention
    FUNC_PTR_ASSIGN(gInstance->MNNAttenUnpackAndConvertFp16, MNNAttenUnpackAndConvertFp16);
    FUNC_PTR_ASSIGN(gInstance->MNNAttenPackAndConvertFp32, MNNAttenPackAndConvertFp32);
    FUNC_PTR_ASSIGN(gInstance->MNNAttenPackAndScaleSingleHead, MNNAttenPackAndScaleSingleHead);
    FUNC_PTR_ASSIGN(gInstance->MNNFlashAttentionUpdateBlockOutput, MNNFlashAttentionUpdateBlockOutput);
    gInstance->MNNQuantAttentionKey = MNNQuantAttentionKeyFP16;
    gInstance->MNNQuantAttentionValue = MNNQuantAttentionValueFP16;
#endif // MNN_SUPPORT_TRANSFORMER_FUSE

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

    {
        gInstance->int8MatmulRelatedFunctions.MNNPackC4Int8ForMatMul_A = gArm82CoreInt8Functions->MNNPackC4Int8ForMatMul_A;
        gInstance->int8MatmulRelatedFunctions.MNNGeneralIm2Col = gInstance->MNNGeneralIm2Col;
    }

#ifdef __aarch64__
#ifdef MNN_SME2
        if (origin->supportSME2) {
            gArm82CoreInt8Functions->MNNPackC4Int8ForMatMul_A = _Arm82MNNPackC4ForMatMul_A<16, 4>;

            FUNC_PTR_ASSIGN(gInstance->MNNPackedMatMul, MNNPackedMatMulFP16_SME2);
            FUNC_PTR_ASSIGN(gInstance->MNNPackedMatMulRemain, MNNPackedMatMulRemainFP16_SME2);
            FUNC_PTR_ASSIGN(gInstance->MNNGetMatMulPackMode, Sme2MNNGetMatMulPackMode);
            FUNC_PTR_ASSIGN(gInstance->MNNPackC4ForMatMul_A, Sme2MNNPackC4ForMatMul_A_FP16);
            FUNC_PTR_ASSIGN(gInstance->MNNPackForMatMul_B, Sme2MNNPackForMatMul_B);

#ifdef MNN_LOW_MEMORY
            FUNC_PTR_ASSIGN(gInstance->MNNGeneralIm2Col, MNNGeneralIm2col_Fp16Sme2);
#endif
        }
#endif // MNN_SME2
#endif // __aarch64__

    // Update the function pointers in the int8MatmulRelatedFunctions struct.
    gInstance->int8MatmulRelatedFunctions.MNNPackC4Int8ForMatMul_A = gArm82CoreInt8Functions->MNNPackC4Int8ForMatMul_A;
    gInstance->int8MatmulRelatedFunctions.MNNGeneralIm2Col = gInstance->MNNGeneralIm2Col;


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
