//
//  MNNDynamicQuantFunctions.hpp
//  MNN
//
//  Created by MNN on 2021/02/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDYNAMICQUANTFUNCTIOINS_HPP
#define MNNDYNAMICQUANTFUNCTIOINS_HPP

#ifdef __aarch64__
#include <arm_neon.h>
using namespace MNN;
static bool MNNAsyLocalQuantInfo_EP10_FP32(float* scale, float* bias, float* qscale, float* qbias, const float* srcMin, const float* srcMax, const size_t* info) {
    auto blockNum = info[0];
    auto EP = info[1];
    auto DST_XUNIT = info[3];
    if (DST_XUNIT != 10) {
        MNN_ERROR("Function called error\n");
        return false;
    }
    auto stride = EP * blockNum;
    // dequant scale/bias : [EU, blockNum, step]
    // quant scale/bias: [blockNum, EP]
    auto minfloat = vdupq_n_f32(1e-6);
    auto _255f = vdupq_n_f32(255.f);
    auto _128f = vdupq_n_f32(128.f);
    auto _0f = vdupq_n_f32(0.f);
    auto _255f_float32x2 = vdup_n_f32(255.f);
    auto _128f_float32x2 = vdup_n_f32(128.f);
    auto _0f_float32x2 = vdup_n_f32(0.f);
    for (int k = 0; k < blockNum; ++k) {
        auto qind = k * EP;
        auto realDstCount = EP;
        auto scalePtr = scale + k * ALIMIN(EP, DST_XUNIT);
        auto biasPtr = bias + k * ALIMIN(EP, DST_XUNIT);
        while (realDstCount > DST_XUNIT - 1) {
            auto max0 = vld1q_f32(srcMax + qind);
            auto max1 = vld1q_f32(srcMax + qind + 4);
            auto max2 = vld1_f32(srcMax + qind + 8); // float32x2_t
            auto min0 = vld1q_f32(srcMin + qind);    
            auto min1 = vld1q_f32(srcMin + qind + 4);
            auto min2 = vld1_f32(srcMin + qind + 8); // float32x2_t

            auto diff0 = vsubq_f32(max0, min0);
            auto diff1 = vsubq_f32(max1, min1);
            auto diff2 = vsub_f32(max2, min2);       // float32x2_t

            auto qscaleV0 = vdivq_f32(_255f, diff0);
            auto qscaleV1 = vdivq_f32(_255f, diff1);
            auto qscaleV2 = vdiv_f32(_255f_float32x2, diff2);  // float32x2_t
            auto scaleV0 = vdivq_f32(diff0, _255f);
            auto scaleV1 = vdivq_f32(diff1, _255f);
            auto scaleV2 = vdiv_f32(diff2, _255f_float32x2);   // float32x2_t
            
            auto qbiasV0 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min0), diff0), _128f));
            auto qbiasV1 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min1), diff1), _128f));
            auto qbiasV2 = vneg_f32(vadd_f32(vdiv_f32(vmul_f32(_255f_float32x2, min2), diff2), _128f_float32x2));
            auto biasV0 = vaddq_f32(vdivq_f32(vmulq_f32(diff0, _128f), _255f), min0);
            auto biasV1 = vaddq_f32(vdivq_f32(vmulq_f32(diff1, _128f), _255f), min1);
            auto biasV2 = vadd_f32(vdiv_f32(vmul_f32(diff2, _128f_float32x2), _255f_float32x2), min2);

            auto _0bic = vclezq_f32(diff0);
            auto _1bic = vclezq_f32(diff1);
            auto _2bic = vclez_f32(diff2);

            qscaleV0 = vbslq_f32(_0bic, _0f, qscaleV0);
            qscaleV1 = vbslq_f32(_1bic, _0f, qscaleV1);
            qscaleV2 = vbsl_f32(_2bic, _0f_float32x2, qscaleV2);

            qbiasV0 = vrndaq_f32(vbslq_f32(_0bic, _0f, qbiasV0));
            qbiasV1 = vrndaq_f32(vbslq_f32(_1bic, _0f, qbiasV1));
            qbiasV2 = vrnda_f32(vbsl_f32(_2bic, _0f_float32x2, qbiasV2));

            scaleV0 = vbslq_f32(_0bic, _0f, scaleV0);
            scaleV1 = vbslq_f32(_1bic, _0f, scaleV1);
            scaleV2 = vbsl_f32(_2bic, _0f_float32x2, scaleV2);

            biasV0 = vbslq_f32(_0bic, max0, biasV0);
            biasV1 = vbslq_f32(_1bic, max1, biasV1);
            biasV2 = vbsl_f32(_2bic, max2, biasV2);

            vst1q_f32(qscale + qind, qscaleV0);
            vst1q_f32(qscale + qind + 4, qscaleV1);
            vst1_f32(qscale + qind + 8, qscaleV2);

            vst1q_f32(qbias + qind, qbiasV0);
            vst1q_f32(qbias + qind + 4, qbiasV1);
            vst1_f32(qbias + qind + 8, qbiasV2);

            vst1q_f32(scalePtr, scaleV0);
            vst1q_f32(scalePtr + 4, scaleV1);
            vst1_f32(scalePtr + 8, scaleV2);

            vst1q_f32(biasPtr, biasV0);
            vst1q_f32(biasPtr + 4, biasV1);
            vst1_f32(biasPtr + 8, biasV2);

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
            auto max0 = vld1q_f32(srcMax + qind);
            auto max1 = vld1q_f32(srcMax + qind + 4);
            auto min0 = vld1q_f32(srcMin + qind);
            auto min1 = vld1q_f32(srcMin + qind + 4);
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
            auto max0 = vld1q_f32(srcMax + qind);
            auto min0 = vld1q_f32(srcMin + qind);
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
            auto max0 = vld1q_dup_f32(srcMax + qind);
            auto min0 = vld1q_dup_f32(srcMin + qind);
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

static bool MNNAsyLocalQuantInfo_EP12_FP32(float* scale, float* bias, float* qscale, float* qbias, const float* srcMin, const float* srcMax, const size_t* info) {
    auto blockNum = info[0];
    auto EP = info[1];
    auto DST_XUNIT = info[3];
    if (DST_XUNIT != 12) {
        MNN_ERROR("Function called error\n");
        return false;
    }
    auto stride = EP * blockNum;
    // dequant scale/bias : [EU, blockNum, step]
    // quant scale/bias: [blockNum, EP]
    auto minfloat = vdupq_n_f32(1e-6);
    auto _255f = vdupq_n_f32(255.f);
    auto _128f = vdupq_n_f32(128.f);
    auto _0f = vdupq_n_f32(0.f);
    for (int k = 0; k < blockNum; ++k) {
        auto qind = k * EP;
        auto realDstCount = EP;
        auto scalePtr = scale + k * ALIMIN(EP, DST_XUNIT);
        auto biasPtr = bias + k * ALIMIN(EP, DST_XUNIT);
        while (realDstCount > DST_XUNIT - 1) {
            auto max0 = vld1q_f32(srcMax + qind);
            auto max1 = vld1q_f32(srcMax + qind + 4);
            auto max2 = vld1q_f32(srcMax + qind + 8);
            auto min0 = vld1q_f32(srcMin + qind);
            auto min1 = vld1q_f32(srcMin + qind + 4);
            auto min2 = vld1q_f32(srcMin + qind + 8);

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
            auto max0 = vld1q_f32(srcMax + qind);
            auto max1 = vld1q_f32(srcMax + qind + 4);
            auto min0 = vld1q_f32(srcMin + qind);
            auto min1 = vld1q_f32(srcMin + qind + 4);
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
            auto max0 = vld1q_f32(srcMax + qind);
            auto min0 = vld1q_f32(srcMin + qind);
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
            auto max0 = vld1q_dup_f32(srcMax + qind);
            auto min0 = vld1q_dup_f32(srcMin + qind);
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

static bool MNNAsyLocalQuantInfo_EP16_FP32(float* scale, float* bias, float* qscale, float* qbias, const float* srcMin, const float* srcMax, const size_t* info) {
    auto blockNum = info[0];
    auto EP = info[1];
    auto DST_XUNIT = info[3];
    if (DST_XUNIT != 16) {
        MNN_ERROR("Function called error\n");
        return false;
    }
    auto stride = EP * blockNum;
    // dequant scale/bias : [EU, blockNum, step]
    // quant scale/bias: [blockNum, EP]
    auto minfloat = vdupq_n_f32(1e-6);
    auto _255f = vdupq_n_f32(255.f);
    auto _128f = vdupq_n_f32(128.f);
    auto _0f = vdupq_n_f32(0.f);
    for (int k = 0; k < blockNum; ++k) {
        auto qind = k * EP;
        auto realDstCount = EP;
        auto scalePtr = scale + k * ALIMIN(EP, DST_XUNIT);
        auto biasPtr = bias + k * ALIMIN(EP, DST_XUNIT);
        while (realDstCount > DST_XUNIT - 1) {
            auto max0 = vld1q_f32(srcMax + qind);
            auto max1 = vld1q_f32(srcMax + qind + 4);
            auto max2 = vld1q_f32(srcMax + qind + 8);
            auto max3 = vld1q_f32(srcMax + qind + 12);
            auto min0 = vld1q_f32(srcMin + qind);
            auto min1 = vld1q_f32(srcMin + qind + 4);
            auto min2 = vld1q_f32(srcMin + qind + 8);
            auto min3 = vld1q_f32(srcMin + qind + 12);

            auto diff0 = vsubq_f32(max0, min0);
            auto diff1 = vsubq_f32(max1, min1);
            auto diff2 = vsubq_f32(max2, min2);
            auto diff3 = vsubq_f32(max3, min3);

            auto qscaleV0 = vdivq_f32(_255f, diff0);
            auto qscaleV1 = vdivq_f32(_255f, diff1);
            auto qscaleV2 = vdivq_f32(_255f, diff2);
            auto qscaleV3 = vdivq_f32(_255f, diff3);
            auto scaleV0 = vdivq_f32(diff0, _255f);
            auto scaleV1 = vdivq_f32(diff1, _255f);
            auto scaleV2 = vdivq_f32(diff2, _255f);
            auto scaleV3 = vdivq_f32(diff3, _255f);
            
            auto qbiasV0 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min0), diff0), _128f));
            auto qbiasV1 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min1), diff1), _128f));
            auto qbiasV2 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min2), diff2), _128f));
            auto qbiasV3 = vnegq_f32(vaddq_f32(vdivq_f32(vmulq_f32(_255f, min3), diff3), _128f));
            auto biasV0 = vaddq_f32(vdivq_f32(vmulq_f32(diff0, _128f), _255f), min0);
            auto biasV1 = vaddq_f32(vdivq_f32(vmulq_f32(diff1, _128f), _255f), min1);
            auto biasV2 = vaddq_f32(vdivq_f32(vmulq_f32(diff2, _128f), _255f), min2);
            auto biasV3 = vaddq_f32(vdivq_f32(vmulq_f32(diff3, _128f), _255f), min3);

            auto _0bic = vclezq_f32(diff0);
            auto _1bic = vclezq_f32(diff1);
            auto _2bic = vclezq_f32(diff2);
            auto _3bic = vclezq_f32(diff3);

            qscaleV0 = vbslq_f32(_0bic, _0f, qscaleV0);
            qscaleV1 = vbslq_f32(_1bic, _0f, qscaleV1);
            qscaleV2 = vbslq_f32(_2bic, _0f, qscaleV2);
            qscaleV3 = vbslq_f32(_3bic, _0f, qscaleV3);

            qbiasV0 = vrndaq_f32(vbslq_f32(_0bic, _0f, qbiasV0));
            qbiasV1 = vrndaq_f32(vbslq_f32(_1bic, _0f, qbiasV1));
            qbiasV2 = vrndaq_f32(vbslq_f32(_2bic, _0f, qbiasV2));
            qbiasV3 = vrndaq_f32(vbslq_f32(_3bic, _0f, qbiasV3));

            scaleV0 = vbslq_f32(_0bic, _0f, scaleV0);
            scaleV1 = vbslq_f32(_1bic, _0f, scaleV1);
            scaleV2 = vbslq_f32(_2bic, _0f, scaleV2);
            scaleV3 = vbslq_f32(_3bic, _0f, scaleV3);

            biasV0 = vbslq_f32(_0bic, max0, biasV0);
            biasV1 = vbslq_f32(_1bic, max1, biasV1);
            biasV2 = vbslq_f32(_2bic, max2, biasV2);
            biasV3 = vbslq_f32(_3bic, max3, biasV3);

            vst1q_f32(qscale + qind, qscaleV0);
            vst1q_f32(qscale + qind + 4, qscaleV1);
            vst1q_f32(qscale + qind + 8, qscaleV2);
            vst1q_f32(qscale + qind + 12, qscaleV3);

            vst1q_f32(qbias + qind, qbiasV0);
            vst1q_f32(qbias + qind + 4, qbiasV1);
            vst1q_f32(qbias + qind + 8, qbiasV2);
            vst1q_f32(qbias + qind + 12, qbiasV3);

            vst1q_f32(scalePtr, scaleV0);
            vst1q_f32(scalePtr + 4, scaleV1);
            vst1q_f32(scalePtr + 8, scaleV2);
            vst1q_f32(scalePtr + 12, scaleV3);

            vst1q_f32(biasPtr, biasV0);
            vst1q_f32(biasPtr + 4, biasV1);
            vst1q_f32(biasPtr + 8, biasV2);
            vst1q_f32(biasPtr + 12, biasV3);

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
            auto max0 = vld1q_f32(srcMax + qind);
            auto max1 = vld1q_f32(srcMax + qind + 4);
            auto min0 = vld1q_f32(srcMin + qind);
            auto min1 = vld1q_f32(srcMin + qind + 4);
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
            auto max0 = vld1q_f32(srcMax + qind);
            auto min0 = vld1q_f32(srcMin + qind);
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
            auto max0 = vld1q_dup_f32(srcMax + qind);
            auto min0 = vld1q_dup_f32(srcMin + qind);
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
#endif



#endif // MNNDYNAMICQUANTFUNCTIOINS_HPP
