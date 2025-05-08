//
//  Arm82OptFunc.hpp
//  MNN
//
//  Created by MNN on 2019/02/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#include "Arm82OptFunc.hpp"
#include "Arm82Vec.hpp"
#include "core/Macro.h"
#include "half.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

extern "C" {
void MNNExpFP16(void* dst, const void* src, const FLOAT16* params, size_t blockCount);

void MNNQuantizeFP16_UNIT4(int16_t* dst, const float* src, size_t size, const float* minMax);

}

void Arm82MNNExp(FLOAT16* dst, const FLOAT16* src, size_t dataSize) {
    int blockCount = (int)dataSize / 16;
    static FLOAT16 params[] = {
        (FLOAT16)log(2.0f), (FLOAT16)(1.0f / log(2.0f)), 1.0f, 1.0f, 0.5f, 1.0f / 6.0f, 1.0f / 24.0f, 1.0f / 120.0f};
    if (blockCount > 0) {
        MNNExpFP16(dst, src, params, blockCount);
    }
    int remainSize = dataSize % 16;
    if (remainSize > 0) {
        int16_t srcTemp[16];
        int16_t dstTemp[16];
        ::memcpy(srcTemp, src + blockCount * 16, remainSize * sizeof(int16_t));
        MNNExpFP16(dstTemp, srcTemp, params, 1);
        ::memcpy(dst + blockCount * 16, dstTemp, remainSize * sizeof(int16_t));
    }
}

void Arm82MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
#ifdef __aarch64__
    *hP = 16;
#else
    *hP = 8;
#endif
    *eP = 12;
    *lP = 1;
}

void MNNQuantizeFP16(const float* src, int16_t* dst, size_t size) {
    int sizeDiv4 = size / 4;
    int remain   = size - sizeDiv4 * 4;
    float minMax[] = {
        -65504.0f,
        65504.0f
    };
    if (sizeDiv4 > 0) {
        MNNQuantizeFP16_UNIT4(dst, src, sizeDiv4, minMax);
        src += sizeDiv4 * 4;
        dst += sizeDiv4 * 4;
    }
    if (remain > 0) {
        float tempSrc[4];
        int16_t tempDst[4];
        ::memcpy(tempSrc, src, remain * sizeof(float));
        MNNQuantizeFP16_UNIT4(tempDst, tempSrc, 1, minMax);
        ::memcpy(dst, tempDst, remain * sizeof(int16_t));
    }
}

void MNNDequantizeFP16(const int16_t* srcint, float* dst, size_t size) {
    auto src = (const FLOAT16*)srcint;
    int sizeDiv4 = size / 4;
    int remain   = size - sizeDiv4 * 4;
    for (int i = 0; i < sizeDiv4; ++i) {
        auto S = vld1_f16(src);
        auto D = vcvt_f32_f16(S);
        vst1q_f32(dst, D);
        dst += 4;
        src += 4;
    }
    if (remain > 0) {
        FLOAT16 tempSrc[4];
        float tempDst[4];
        ::memcpy(tempSrc, src, remain * sizeof(int16_t));
        auto S = vld1_f16(tempSrc);
        auto D = vcvt_f32_f16(S);
        vst1q_f32(tempDst, D);
        ::memcpy(dst, tempDst, remain * sizeof(float));
    }
}

extern "C" {
void MNNPackC8FP16_C8(int16_t* dest, const int16_t* source, size_t area, size_t depth, int32_t* areaOffset);
void MNNUnpackC8FP16_C8(int16_t* dest, const int16_t* source, size_t area, size_t depth, int32_t* areaOffset);
};
void MNNPackC8FP16(int16_t* dest, const int16_t* source, size_t area, size_t depth, int32_t* areaOffset) {
    const int UNIT = 8;
    int srcAreaOffset = areaOffset[0];
    int dstAreaOffset = areaOffset[1];
    int depthC = depth / UNIT;
    int depthR = depth % UNIT;
    if (depthC > 0) {
        MNNPackC8FP16_C8(dest, source, area, depth, areaOffset);
    }
#ifdef MNN_ARM82_REFCODE
    for (int p=0; p<depthC; ++p) {
        auto dst = dest + p * UNIT * dstAreaOffset;
        auto src = source + p * UNIT * srcAreaOffset;
        for (int z = 0; z < UNIT; ++z) {
            auto dstPlane = dst + z;
            auto srcPlane = src + srcAreaOffset * z;
            for (int x = 0; x < area; ++x) {
                dstPlane[x * UNIT] = srcPlane[x];
            }
        }
    }
#endif
    // TODO: Optimize it
    if (depthR > 0) {
        auto dst = dest + depthC * UNIT * dstAreaOffset;
        auto src = source + depthC * UNIT * srcAreaOffset;
        ::memset(dst, 0, area * UNIT * sizeof(int16_t));
        for (int z = 0; z < depthR; ++z) {
            auto srcPlane = z * srcAreaOffset + src;
            auto dstPlane = dst + z;
            for (int x = 0; x < area; ++x) {
                dstPlane[x * UNIT] = srcPlane[x];
            }
        }
    }
}


void MNNUnPackC8FP16(int16_t* dest, const int16_t* source, size_t area, size_t depth, int32_t* areaOffset) {
    const int UNIT = 8;
    int depthC = depth / UNIT;
    int depthR = depth % UNIT;
    int srcAreaOffset = areaOffset[0];
    int dstAreaOffset = areaOffset[1];
    if (depthC > 0) {
        MNNUnpackC8FP16_C8(dest, source, area, depth, areaOffset);
    }
#ifdef MNN_ARM82_REFCODE
    for (int p=0; p<depthC; ++p) {
        auto dst = dest + p * UNIT * dstAreaOffset;
        auto src = source + p * UNIT * srcAreaOffset;
        for (int z = 0; z < UNIT; ++z) {
            auto srcPlane = src + z;
            auto dstPlane = dst + dstAreaOffset * z;
            for (int x = 0; x < area; ++x) {
                dstPlane[x] = srcPlane[x * UNIT];
            }
        }
    }
#endif
    // TODO: Optimize it
    if (depthR > 0) {
        auto dst = dest + depthC * UNIT * dstAreaOffset;
        auto src = source + depthC * UNIT * srcAreaOffset;
        for (int z = 0; z < depthR; ++z) {
            auto srcPlane = src + z;
            auto dstPlane = dst + dstAreaOffset * z;
            for (int x = 0; x < area; ++x) {
                dstPlane[x] = srcPlane[x * UNIT];
            }
        }
    }
}

void MNNNC4HW4TONC8HW8(FLOAT16* dst, const float* source, size_t plane, size_t channel) {
    const int c4 = UP_DIV(channel, 4);
    const int c8 = UP_DIV(channel, 8);
    memset(dst, 0, plane * c8 * 8 * sizeof(FLOAT16));
    auto dest = (float16_t*)dst;
    for (int c = 0; c < c4; ++c) {
        int ci          = c / 2;
        int cj          = c % 2;
        auto dstChannel = dest + ci * 8 * plane + cj * 4;
        auto srcChannle = source + c * plane * 4;

        for (int i = 0; i < plane; ++i) {
            float32x4_t a = vld1q_f32(srcChannle + i * 4);
            vst1_f16(dstChannel + i * 8, vcvt_f16_f32(a));
        }
    }
}

void MNNNC8HW8TONC4HW4(float* dest, const FLOAT16* src, size_t plane, size_t channel) {
    const int c4 = UP_DIV(channel, 4);
    auto source = (float16_t*)src;
    for (int c = 0; c < c4; ++c) {
        int ci          = c / 2;
        int cj          = c % 2;
        auto srcChannel = source + ci * 8 * plane + cj * 4;
        auto dstChannel = dest + c * plane * 4;

        for (int i = 0; i < plane; ++i) {
            float16x4_t a = vld1_f16(srcChannel + i * 8);
            vst1q_f32(dstChannel + i * 4, vcvt_f32_f16(a));
        }
    }
}

void MNNNC8HW8TONHWC(float* dest, const FLOAT16* src, size_t plane, size_t channel) {
    int c      = (int)channel;
    int cDiv8  = c / 8;
    int cAlign = cDiv8 * 8;
#if defined(MNN_USE_NEON) && defined(__aarch64__)
    auto source = (float16_t*)src;
#else
    auto source = src;
#endif
    for (int hi = 0; hi < plane; ++hi) {
        const auto srcHeight = source + hi * 8;
        float* dstHeight     = dest + hi * c;
        for (int ci = 0; ci < cDiv8; ++ci) {
#if defined(MNN_USE_NEON) && defined(__aarch64__)
            float16x8_t a = vld1q_f16(srcHeight + 8 * ci * plane);
            vst1q_f32(dstHeight + 8 * ci, vcvt_high_f32_f16(a));
#else
            half_float::half dataHalf[8];
            memcpy(dataHalf, srcHeight + 8 * ci * plane, 8 * sizeof(FLOAT16));
            for (int i = 0; i < 8; ++i) {
                dstHeight[ci * 8 + i] = float(dataHalf[i]);
            }
#endif
        }
    }
    if (cAlign == c) {
        return;
    }
    int cReamin         = c - cAlign;
    const auto srcAlign = reinterpret_cast<const half_float::half*>(source + plane * cAlign);
    auto dstAlign       = dest + cAlign;
    for (int hi = 0; hi < plane; ++hi) {
        const auto srcHeight = srcAlign + hi * 8;
        float* dstHeight     = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = float(srcHeight[ci]);
        }
    }
}

#ifdef __aarch64__
#ifdef MNN_LOW_MEMORY

bool MNNAsyLocalQuantInfo_EP12_FP16(float* scale, float* bias, float* qscale, float* qbias, const float* srcMin, const float* srcMax, const size_t* info) {
    // dequant scale/bias : [EU, blockNum, step]
    // quant scale/bias: [blockNum, EP]

    auto blockNum = info[0];
    auto EP = info[1];
    auto DST_XUNIT = info[3];
    if (DST_XUNIT != 12) {
        MNN_ERROR("Call error: MNNAsyLocalQuantInfo_EP12_Arm82\n");
        return false;
    }
    auto stride = EP * blockNum;

    auto minfloat = vdupq_n_f32(1e-6);
    auto _255f = vdupq_n_f32(255.f);
    auto _128f = vdupq_n_f32(128.f);
    auto _0f = vdupq_n_f32(0.f);

    auto minPtr = (FLOAT16*)srcMin;
    auto maxPtr = (FLOAT16*)srcMax;
    for (int k = 0; k < blockNum; ++k) {
        auto qind = k * EP;
        auto realDstCount = EP;
        auto scalePtr = scale + k * ALIMIN(EP, DST_XUNIT);
        auto biasPtr = bias + k * ALIMIN(EP, DST_XUNIT);
        while (realDstCount > DST_XUNIT - 1) {
            auto max0_fp16 = vld1_f16(maxPtr + qind);
            auto max1_fp16 = vld1_f16(maxPtr + qind + 4);
            auto max2_fp16 = vld1_f16(maxPtr + qind + 8);
            auto min0_fp16 = vld1_f16(minPtr + qind);
            auto min1_fp16 = vld1_f16(minPtr + qind + 4);
            auto min2_fp16 = vld1_f16(minPtr + qind + 8);

            // float16 -> float32
            auto max0 = vcvt_f32_f16(max0_fp16);
            auto max1 = vcvt_f32_f16(max1_fp16);
            auto max2 = vcvt_f32_f16(max2_fp16);

            auto min0 = vcvt_f32_f16(min0_fp16);
            auto min1 = vcvt_f32_f16(min1_fp16);
            auto min2 = vcvt_f32_f16(min2_fp16);
            // diff
            auto diff0 = vsubq_f32(max0, min0);
            auto diff1 = vsubq_f32(max1, min1);
            auto diff2 = vsubq_f32(max2, min2);

            auto qscaleV0 = vdivq_f32(_255f, diff0);
            auto qscaleV1 = vdivq_f32(_255f, diff1);
            auto qscaleV2 = vdivq_f32(_255f, diff2);
            auto scaleV0 = vdivq_f32(diff0, _255f);
            auto scaleV1 = vdivq_f32(diff1, _255f);
            auto scaleV2 = vdivq_f32(diff2, _255f);
            
            auto qbiasV0 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min0), diff0), _128f));
            auto qbiasV1 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min1), diff1), _128f));
            auto qbiasV2 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min2), diff2), _128f));
            auto biasV0 = vaddq_f32(vdivq_f32(vmulq_f32(diff0, _128f), _255f), min0);
            auto biasV1 = vaddq_f32(vdivq_f32(vmulq_f32(diff1, _128f), _255f), min1);
            auto biasV2 = vaddq_f32(vdivq_f32(vmulq_f32(diff2, _128f), _255f), min2);

            auto _0bic = vclezq_f32(diff0);
            auto _1bic = vclezq_f32(diff1);
            auto _2bic = vclezq_f32(diff2);

            qscaleV0 = vbslq_f32(_0bic, _0f, qscaleV0);
            qscaleV1 = vbslq_f32(_1bic, _0f, qscaleV1);
            qscaleV2 = vbslq_f32(_2bic, _0f, qscaleV2);

            qbiasV0 = vrndaq_f32(vbslq_f32(_0bic, _0f, qbiasV0));
            qbiasV1 = vrndaq_f32(vbslq_f32(_1bic, _0f, qbiasV1));
            qbiasV2 = vrndaq_f32(vbslq_f32(_2bic, _0f, qbiasV2));

            scaleV0 = vbslq_f32(_0bic, _0f, scaleV0);
            scaleV1 = vbslq_f32(_1bic, _0f, scaleV1);
            scaleV2 = vbslq_f32(_2bic, _0f, scaleV2);

            biasV0 = vbslq_f32(_0bic, max0, biasV0);
            biasV1 = vbslq_f32(_1bic, max1, biasV1);
            biasV2 = vbslq_f32(_2bic, max2, biasV2);

            vst1q_f32(qscale + qind, qscaleV0);
            vst1q_f32(qscale + qind + 4, qscaleV1);
            vst1q_f32(qscale + qind + 8, qscaleV2);

            vst1q_f32(qbias + qind, qbiasV0);
            vst1q_f32(qbias + qind + 4, qbiasV1);
            vst1q_f32(qbias + qind + 8, qbiasV2);

            vst1q_f32(scalePtr, scaleV0);
            vst1q_f32(scalePtr + 4, scaleV1);
            vst1q_f32(scalePtr + 8, scaleV2);

            vst1q_f32(biasPtr, biasV0);
            vst1q_f32(biasPtr + 4, biasV1);
            vst1q_f32(biasPtr + 8, biasV2);

            realDstCount -= DST_XUNIT;
            qind += DST_XUNIT;
            scalePtr += (blockNum * DST_XUNIT);
            biasPtr += (blockNum * DST_XUNIT);
        }
        if (realDstCount == 0) {
            continue;
        }

        auto remainE = realDstCount;
        auto stride0 = remainE * blockNum;
        scalePtr = scale + (EP / DST_XUNIT) * blockNum * DST_XUNIT + k * remainE;
        biasPtr = bias + (EP / DST_XUNIT) * blockNum * DST_XUNIT + k * remainE;
        if (realDstCount > 7) {
            auto max0_fp16 = vld1_f16(maxPtr + qind);
            auto max1_fp16 = vld1_f16(maxPtr + qind + 4);
            auto min0_fp16 = vld1_f16(minPtr + qind);
            auto min1_fp16 = vld1_f16(minPtr + qind + 4);

            // float16 -> float32
            auto max0 = vcvt_f32_f16(max0_fp16);
            auto max1 = vcvt_f32_f16(max1_fp16);

            auto min0 = vcvt_f32_f16(min0_fp16);
            auto min1 = vcvt_f32_f16(min1_fp16);
            // diff
            auto diff0 = vsubq_f32(max0, min0);
            auto diff1 = vsubq_f32(max1, min1);

            auto qscaleV0 = vdivq_f32(_255f, diff0);
            auto qscaleV1 = vdivq_f32(_255f, diff1);
            auto scaleV0 = vdivq_f32(diff0, _255f);
            auto scaleV1 = vdivq_f32(diff1, _255f);
            
            auto qbiasV0 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min0), diff0), _128f));
            auto qbiasV1 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min1), diff1), _128f));
            auto biasV0 = vaddq_f32(vdivq_f32(vmulq_f32(diff0, _128f), _255f), min0);
            auto biasV1 = vaddq_f32(vdivq_f32(vmulq_f32(diff1, _128f), _255f), min1);

            auto _0bic = vclezq_f32(diff0);
            auto _1bic = vclezq_f32(diff1);

            qscaleV0 = vbslq_f32(_0bic, _0f, qscaleV0);
            qscaleV1 = vbslq_f32(_1bic, _0f, qscaleV1);

            qbiasV0 = vrndaq_f32(vbslq_f32(_0bic, _0f, qbiasV0));
            qbiasV1 = vrndaq_f32(vbslq_f32(_1bic, _0f, qbiasV1));

            scaleV0 = vbslq_f32(_0bic, _0f, scaleV0);
            scaleV1 = vbslq_f32(_1bic, _0f, scaleV1);

            biasV0 = vbslq_f32(_0bic, max0, biasV0);
            biasV1 = vbslq_f32(_1bic, max1, biasV1);

            vst1q_f32(qscale + qind, qscaleV0);
            vst1q_f32(qscale + qind + 4, qscaleV1);

            vst1q_f32(qbias + qind, qbiasV0);
            vst1q_f32(qbias + qind + 4, qbiasV1);

            vst1q_f32(scalePtr, scaleV0);
            vst1q_f32(scalePtr + 4, scaleV1);

            vst1q_f32(biasPtr, biasV0);
            vst1q_f32(biasPtr + 4, biasV1);
            realDstCount -= 8;
            qind += 8;
            scalePtr += 8;
            biasPtr += 8;
        }
        if (realDstCount > 3) {
            auto max0_fp16 = vld1_f16(maxPtr + qind);
            auto min0_fp16 = vld1_f16(minPtr + qind);

            // float16 -> float32
            auto max0 = vcvt_f32_f16(max0_fp16);
            auto min0 = vcvt_f32_f16(min0_fp16);
            // diff
            auto diff0 = vsubq_f32(max0, min0);

            auto qscaleV0 = vdivq_f32(_255f, diff0);
            auto scaleV0 = vdivq_f32(diff0, _255f);
            
            auto qbiasV0 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min0), diff0), _128f));
            auto biasV0 = vaddq_f32(vdivq_f32(vmulq_f32(diff0, _128f), _255f), min0);

            auto _0bic = vclezq_f32(diff0);

            qscaleV0 = vbslq_f32(_0bic, _0f, qscaleV0);
            qbiasV0 = vrndaq_f32(vbslq_f32(_0bic, _0f, qbiasV0));
            scaleV0 = vbslq_f32(_0bic, _0f, scaleV0);
            biasV0 = vbslq_f32(_0bic, max0, biasV0);

            vst1q_f32(qscale + qind, qscaleV0);

            vst1q_f32(qbias + qind, qbiasV0);

            vst1q_f32(scalePtr, scaleV0);

            vst1q_f32(biasPtr, biasV0);

            realDstCount -= 4;
            qind += 4;
            scalePtr += 4;
            biasPtr += 4;
        }
        while (realDstCount > 0) {
            auto max0_fp16 = vld1_dup_f16(maxPtr + qind);
            auto min0_fp16 = vld1_dup_f16(minPtr + qind);

            // float16->float32
            auto max0 = vcvt_f32_f16(max0_fp16);
            auto min0 = vcvt_f32_f16(min0_fp16);
            auto diff0 = vsubq_f32(max0, min0);

            auto qscaleV0 = vdivq_f32(_255f, diff0);
            auto scaleV0 = vdivq_f32(diff0, _255f);
            
            auto qbiasV0 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min0), diff0), _128f));
            auto biasV0 = vaddq_f32(vdivq_f32(vmulq_f32(diff0, _128f), _255f), min0);

            auto _0bic = vclezq_f32(diff0);

            qscaleV0 = vbslq_f32(_0bic, _0f, qscaleV0);

            qbiasV0 = vrndaq_f32(vbslq_f32(_0bic, _0f, qbiasV0));

            scaleV0 = vbslq_f32(_0bic, _0f, scaleV0);

            biasV0 = vbslq_f32(_0bic, max0, biasV0);

            vst1q_lane_f32(qscale + qind, qscaleV0, 0);

            vst1q_lane_f32(qbias + qind, qbiasV0, 0);

            vst1q_lane_f32(scalePtr, scaleV0, 0);

            vst1q_lane_f32(biasPtr, biasV0, 0);

            realDstCount -= 1;
            qind += 1;
            scalePtr += 1;
            biasPtr += 1;
        }
    }
    return true;
}

bool MNNAsyLocalQuantInfo_EP10_FP16(float* scale, float* bias, float* qscale, float* qbias, const float* srcMin, const float* srcMax, const size_t* info) {
    // dequant scale/bias : [EU, blockNum, step]
    // quant scale/bias: [blockNum, EP]

    auto blockNum = info[0];
    auto EP = info[1];
    auto DST_XUNIT = info[3];
    if (DST_XUNIT != 10) {
        MNN_ERROR("Call error: MNNAsyLocalQuantInfo_EP12_Arm82\n");
        return false;
    }
    auto stride = EP * blockNum;
    auto minfloat = vdupq_n_f32(1e-6);
    auto _255f = vdupq_n_f32(255.f);
    auto _128f = vdupq_n_f32(128.f);
    auto _0f = vdupq_n_f32(0.f);
    auto minPtr = (FLOAT16*)srcMin;
    auto maxPtr = (FLOAT16*)srcMax;
    float16x4_t max2_fp16;
    float16x4_t min2_fp16;
    for (int k = 0; k < blockNum; ++k) {
        auto qind = k * EP;
        auto realDstCount = EP;
        auto scalePtr = scale + k * ALIMIN(EP, DST_XUNIT);
        auto biasPtr = bias + k * ALIMIN(EP, DST_XUNIT);
        while (realDstCount > DST_XUNIT - 1) {
            auto max0_fp16 = vld1_f16(maxPtr + qind);
            auto max1_fp16 = vld1_f16(maxPtr + qind + 4);
            max2_fp16[0] = maxPtr[qind + 8];
            max2_fp16[1] = maxPtr[qind + 9];
            auto min0_fp16 = vld1_f16(minPtr + qind);
            auto min1_fp16 = vld1_f16(minPtr + qind + 4);
            min2_fp16[0] = minPtr[qind + 8];
            min2_fp16[1] = minPtr[qind + 9];

            // float16 -> float32
            auto max0 = vcvt_f32_f16(max0_fp16);
            auto max1 = vcvt_f32_f16(max1_fp16);
            auto max2 = vcvt_f32_f16(max2_fp16);

            auto min0 = vcvt_f32_f16(min0_fp16);
            auto min1 = vcvt_f32_f16(min1_fp16);
            auto min2 = vcvt_f32_f16(min2_fp16);
            // diff
            auto diff0 = vsubq_f32(max0, min0);
            auto diff1 = vsubq_f32(max1, min1);
            auto diff2 = vsubq_f32(max2, min2);

            auto qscaleV0 = vdivq_f32(_255f, diff0);
            auto qscaleV1 = vdivq_f32(_255f, diff1);
            auto qscaleV2 = vdivq_f32(_255f, diff2);
            auto scaleV0 = vdivq_f32(diff0, _255f);
            auto scaleV1 = vdivq_f32(diff1, _255f);
            auto scaleV2 = vdivq_f32(diff2, _255f);
            
            auto qbiasV0 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min0), diff0), _128f));
            auto qbiasV1 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min1), diff1), _128f));
            auto qbiasV2 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min2), diff2), _128f));
            auto biasV0 = vaddq_f32(vdivq_f32(vmulq_f32(diff0, _128f), _255f), min0);
            auto biasV1 = vaddq_f32(vdivq_f32(vmulq_f32(diff1, _128f), _255f), min1);
            auto biasV2 = vaddq_f32(vdivq_f32(vmulq_f32(diff2, _128f), _255f), min2);

            auto _0bic = vclezq_f32(diff0);
            auto _1bic = vclezq_f32(diff1);
            auto _2bic = vclezq_f32(diff2);

            qscaleV0 = vbslq_f32(_0bic, _0f, qscaleV0);
            qscaleV1 = vbslq_f32(_1bic, _0f, qscaleV1);
            qscaleV2 = vbslq_f32(_2bic, _0f, qscaleV2);

            qbiasV0 = vrndaq_f32(vbslq_f32(_0bic, _0f, qbiasV0));
            qbiasV1 = vrndaq_f32(vbslq_f32(_1bic, _0f, qbiasV1));
            qbiasV2 = vrndaq_f32(vbslq_f32(_2bic, _0f, qbiasV2));

            scaleV0 = vbslq_f32(_0bic, _0f, scaleV0);
            scaleV1 = vbslq_f32(_1bic, _0f, scaleV1);
            scaleV2 = vbslq_f32(_2bic, _0f, scaleV2);

            biasV0 = vbslq_f32(_0bic, max0, biasV0);
            biasV1 = vbslq_f32(_1bic, max1, biasV1);
            biasV2 = vbslq_f32(_2bic, max2, biasV2);

            vst1q_f32(qscale + qind, qscaleV0);
            vst1q_f32(qscale + qind + 4, qscaleV1);
            vst1_f32(qscale + qind + 8, vget_low_f32(qscaleV2));

            vst1q_f32(qbias + qind, qbiasV0);
            vst1q_f32(qbias + qind + 4, qbiasV1);
            vst1_f32(qbias + qind + 8, vget_low_f32(qbiasV2));

            vst1q_f32(scalePtr, scaleV0);
            vst1q_f32(scalePtr + 4, scaleV1);
            vst1_f32(scalePtr + 8, vget_low_f32(scaleV2));

            vst1q_f32(biasPtr, biasV0);
            vst1q_f32(biasPtr + 4, biasV1);
            vst1_f32(biasPtr + 8, vget_low_f32(biasV2));

            realDstCount -= DST_XUNIT;
            qind += DST_XUNIT;
            scalePtr += (blockNum * DST_XUNIT);
            biasPtr += (blockNum * DST_XUNIT);
        }
        if (realDstCount == 0) {
            continue;
        }

        auto remainE = realDstCount;
        auto stride0 = remainE * blockNum;
        scalePtr = scale + (EP / DST_XUNIT) * blockNum * DST_XUNIT + k * remainE;
        biasPtr = bias + (EP / DST_XUNIT) * blockNum * DST_XUNIT + k * remainE;
        if (realDstCount > 7) {
            auto max0_fp16 = vld1_f16(maxPtr + qind);
            auto max1_fp16 = vld1_f16(maxPtr + qind + 4);
            auto min0_fp16 = vld1_f16(minPtr + qind);
            auto min1_fp16 = vld1_f16(minPtr + qind + 4);

            // float16 -> float32
            auto max0 = vcvt_f32_f16(max0_fp16);
            auto max1 = vcvt_f32_f16(max1_fp16);

            auto min0 = vcvt_f32_f16(min0_fp16);
            auto min1 = vcvt_f32_f16(min1_fp16);
            // diff
            auto diff0 = vsubq_f32(max0, min0);
            auto diff1 = vsubq_f32(max1, min1);

            auto qscaleV0 = vdivq_f32(_255f, diff0);
            auto qscaleV1 = vdivq_f32(_255f, diff1);
            auto scaleV0 = vdivq_f32(diff0, _255f);
            auto scaleV1 = vdivq_f32(diff1, _255f);
            
            auto qbiasV0 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min0), diff0), _128f));
            auto qbiasV1 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min1), diff1), _128f));
            auto biasV0 = vaddq_f32(vdivq_f32(vmulq_f32(diff0, _128f), _255f), min0);
            auto biasV1 = vaddq_f32(vdivq_f32(vmulq_f32(diff1, _128f), _255f), min1);

            auto _0bic = vclezq_f32(diff0);
            auto _1bic = vclezq_f32(diff1);

            qscaleV0 = vbslq_f32(_0bic, _0f, qscaleV0);
            qscaleV1 = vbslq_f32(_1bic, _0f, qscaleV1);

            qbiasV0 = vrndaq_f32(vbslq_f32(_0bic, _0f, qbiasV0));
            qbiasV1 = vrndaq_f32(vbslq_f32(_1bic, _0f, qbiasV1));

            scaleV0 = vbslq_f32(_0bic, _0f, scaleV0);
            scaleV1 = vbslq_f32(_1bic, _0f, scaleV1);

            biasV0 = vbslq_f32(_0bic, max0, biasV0);
            biasV1 = vbslq_f32(_1bic, max1, biasV1);

            vst1q_f32(qscale + qind, qscaleV0);
            vst1q_f32(qscale + qind + 4, qscaleV1);

            vst1q_f32(qbias + qind, qbiasV0);
            vst1q_f32(qbias + qind + 4, qbiasV1);

            vst1q_f32(scalePtr, scaleV0);
            vst1q_f32(scalePtr + 4, scaleV1);

            vst1q_f32(biasPtr, biasV0);
            vst1q_f32(biasPtr + 4, biasV1);
            realDstCount -= 8;
            qind += 8;
            scalePtr += 8;
            biasPtr += 8;
        }
        if (realDstCount > 3) {
            auto max0_fp16 = vld1_f16(maxPtr + qind);
            auto min0_fp16 = vld1_f16(minPtr + qind);

            // float16 -> float32
            auto max0 = vcvt_f32_f16(max0_fp16);
            auto min0 = vcvt_f32_f16(min0_fp16);
            // diff
            auto diff0 = vsubq_f32(max0, min0);

            auto qscaleV0 = vdivq_f32(_255f, diff0);
            auto scaleV0 = vdivq_f32(diff0, _255f);
            
            auto qbiasV0 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min0), diff0), _128f));
            auto biasV0 = vaddq_f32(vdivq_f32(vmulq_f32(diff0, _128f), _255f), min0);

            auto _0bic = vclezq_f32(diff0);

            qscaleV0 = vbslq_f32(_0bic, _0f, qscaleV0);
            qbiasV0 = vrndaq_f32(vbslq_f32(_0bic, _0f, qbiasV0));
            scaleV0 = vbslq_f32(_0bic, _0f, scaleV0);
            biasV0 = vbslq_f32(_0bic, max0, biasV0);

            vst1q_f32(qscale + qind, qscaleV0);

            vst1q_f32(qbias + qind, qbiasV0);

            vst1q_f32(scalePtr, scaleV0);

            vst1q_f32(biasPtr, biasV0);

            realDstCount -= 4;
            qind += 4;
            scalePtr += 4;
            biasPtr += 4;
        }
        while (realDstCount > 0) {
            auto max0_fp16 = vld1_dup_f16(maxPtr + qind);
            auto min0_fp16 = vld1_dup_f16(minPtr + qind);

            // float16->float32
            auto max0 = vcvt_f32_f16(max0_fp16);
            auto min0 = vcvt_f32_f16(min0_fp16);
            auto diff0 = vsubq_f32(max0, min0);

            auto qscaleV0 = vdivq_f32(_255f, diff0);
            auto scaleV0 = vdivq_f32(diff0, _255f);
            
            auto qbiasV0 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min0), diff0), _128f));
            auto biasV0 = vaddq_f32(vdivq_f32(vmulq_f32(diff0, _128f), _255f), min0);

            auto _0bic = vclezq_f32(diff0);

            qscaleV0 = vbslq_f32(_0bic, _0f, qscaleV0);

            qbiasV0 = vrndaq_f32(vbslq_f32(_0bic, _0f, qbiasV0));

            scaleV0 = vbslq_f32(_0bic, _0f, scaleV0);

            biasV0 = vbslq_f32(_0bic, max0, biasV0);

            vst1q_lane_f32(qscale + qind, qscaleV0, 0);

            vst1q_lane_f32(qbias + qind, qbiasV0, 0);

            vst1q_lane_f32(scalePtr, scaleV0, 0);

            vst1q_lane_f32(biasPtr, biasV0, 0);

            realDstCount -= 1;
            qind += 1;
            scalePtr += 1;
            biasPtr += 1;
        }
    }
    return true;
}

#endif // MNN_LOW_MEMORY
#endif // __aarch64__

#endif
