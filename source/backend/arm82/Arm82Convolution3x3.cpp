//
//  Arm82Convolution3x3.cpp
//  MNN
//
//  Created by MNN on 2020/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "backend/arm82/Arm82Convolution3x3.hpp"
#include "backend/arm82/Arm82OptFunc.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

constexpr int CONV3X3_WINO_OUT    = 4;
constexpr int CONV3X3_WINO_KER    = 3;
constexpr int CONV3X3_WINO_IN     = CONV3X3_WINO_OUT + CONV3X3_WINO_KER - 1;
constexpr int CONV3X3_WEIGHT_UNIT = CONV3X3_WINO_IN * CONV3X3_WINO_IN * ARMV82_CHANNEL_UNIT;

constexpr int CONV3X3_WINO_TILE    = 8;
constexpr int CONV3X3_WINO_SRC_NUM = CONV3X3_WINO_IN * CONV3X3_WINO_IN * ARMV82_CHANNEL_UNIT;

namespace MNN {

// winograd F(4,3)
#ifdef MNN_USE_NEON
static void kernelTransform_wino_4x4_3x3(const FLOAT16* src, FLOAT16* dst, int step) {
    FLOAT16 midResult6X3[6][3];

    for (int i = 0; i < CONV3X3_WINO_KER; ++i) {
        FLOAT16 a0i = src[i];
        FLOAT16 a1i = src[1 * CONV3X3_WINO_KER + i];
        FLOAT16 a2i = src[2 * CONV3X3_WINO_KER + i];

        midResult6X3[0][i] = 0.25f * a0i;
        midResult6X3[1][i] = (a0i + a1i + a2i) * -0.1666666666666667f;
        midResult6X3[2][i] = (a0i - a1i + a2i) * -0.1666666666666667f;
        midResult6X3[3][i] = a0i * 0.04166667f + a1i * 0.08333333f + a2i * 0.1666666666666667f;
        midResult6X3[4][i] = a0i * 0.04166667f - a1i * 0.08333333f + a2i * 0.1666666666666667f;
        midResult6X3[5][i] = a2i;
    }

    for (int i = 0; i < CONV3X3_WINO_IN; ++i) {
        auto curRowDst      = dst;
        curRowDst[0 * step] = 0.25f * midResult6X3[i][0];
        curRowDst[1 * step] = (midResult6X3[i][0] + midResult6X3[i][1] + midResult6X3[i][2]) * -0.1666666666666667f;
        curRowDst[2 * step] = (midResult6X3[i][0] - midResult6X3[i][1] + midResult6X3[i][2]) * -0.1666666666666667f;
        curRowDst[3 * step] = midResult6X3[i][0] * 0.04166667f + midResult6X3[i][1] * 0.08333333f +
                              midResult6X3[i][2] * 0.1666666666666667f;
        curRowDst[4 * step] = midResult6X3[i][0] * 0.04166667f - midResult6X3[i][1] * 0.08333333f +
                              midResult6X3[i][2] * 0.1666666666666667f;
        curRowDst[5 * step] = midResult6X3[i][2];
        dst += CONV3X3_WINO_IN * step;
    }
}

static void sourceTransform_wino_4x4_3x3(const FLOAT16* src, FLOAT16* dst, int step) {
    FLOAT16 midResult[6][6][ARMV82_CHANNEL_UNIT];

    float16x8_t value_4     = vmovq_n_f16(4);
    float16x8_t value_neg_5 = vmovq_n_f16(-5);
    float16x8_t value_neg_4 = vmovq_n_f16(-4);
    float16x8_t value_2     = vmovq_n_f16(2);

    for (int i = 0; i < CONV3X3_WINO_IN; ++i) {
        float16x8_t a0i = vld1q_f16(src + (0 * CONV3X3_WINO_IN + i) * ARMV82_CHANNEL_UNIT);
        float16x8_t a1i = vld1q_f16(src + (1 * CONV3X3_WINO_IN + i) * ARMV82_CHANNEL_UNIT);
        float16x8_t a2i = vld1q_f16(src + (2 * CONV3X3_WINO_IN + i) * ARMV82_CHANNEL_UNIT);
        float16x8_t a3i = vld1q_f16(src + (3 * CONV3X3_WINO_IN + i) * ARMV82_CHANNEL_UNIT);
        float16x8_t a4i = vld1q_f16(src + (4 * CONV3X3_WINO_IN + i) * ARMV82_CHANNEL_UNIT);
        float16x8_t a5i = vld1q_f16(src + (5 * CONV3X3_WINO_IN + i) * ARMV82_CHANNEL_UNIT);

        float16x8_t b0 = vfmaq_f16(a4i, a2i, value_neg_4);
        float16x8_t b1 = vfmaq_f16(a3i, a1i, value_neg_4);
        float16x8_t b2 = vsubq_f16(a4i, a2i);
        float16x8_t b3 = vmulq_f16(vsubq_f16(a3i, a1i), value_2);
        float16x8_t b4 = vfmaq_f16(a4i, a0i, value_4);
        float16x8_t b5 = vfmaq_f16(a5i, a1i, value_4);

        float16x8_t r0 = vfmaq_f16(b4, value_neg_5, a2i);
        float16x8_t r1 = vaddq_f16(b0, b1);
        float16x8_t r2 = vsubq_f16(b0, b1);
        float16x8_t r3 = vaddq_f16(b2, b3);
        float16x8_t r4 = vsubq_f16(b2, b3);
        float16x8_t r5 = vfmaq_f16(b5, value_neg_5, a3i);

        vst1q_f16(midResult[0][i], r0);
        vst1q_f16(midResult[1][i], r1);
        vst1q_f16(midResult[2][i], r2);
        vst1q_f16(midResult[3][i], r3);
        vst1q_f16(midResult[4][i], r4);
        vst1q_f16(midResult[5][i], r5);
    }

    for (int i = 0; i < CONV3X3_WINO_IN; ++i) {
        float16x8_t a0i = vld1q_f16(midResult[i][0]);
        float16x8_t a1i = vld1q_f16(midResult[i][1]);
        float16x8_t a2i = vld1q_f16(midResult[i][2]);
        float16x8_t a3i = vld1q_f16(midResult[i][3]);
        float16x8_t a4i = vld1q_f16(midResult[i][4]);
        float16x8_t a5i = vld1q_f16(midResult[i][5]);

        float16x8_t b0 = vfmaq_f16(a4i, a2i, value_neg_4);
        float16x8_t b1 = vfmaq_f16(a3i, a1i, value_neg_4);
        float16x8_t b2 = vsubq_f16(a4i, a2i);
        float16x8_t b3 = vmulq_f16(vsubq_f16(a3i, a1i), value_2);
        float16x8_t b4 = vfmaq_f16(a4i, a0i, value_4);
        float16x8_t b5 = vfmaq_f16(a5i, a1i, value_4);

        float16x8_t r0 = vfmaq_f16(b4, value_neg_5, a2i);
        float16x8_t r1 = vaddq_f16(b0, b1);
        float16x8_t r2 = vsubq_f16(b0, b1);
        float16x8_t r3 = vaddq_f16(b2, b3);
        float16x8_t r4 = vsubq_f16(b2, b3);
        float16x8_t r5 = vfmaq_f16(b5, value_neg_5, a3i);

        vst1q_f16(dst + 0 * step, r0);
        vst1q_f16(dst + 1 * step, r1);
        vst1q_f16(dst + 2 * step, r2);
        vst1q_f16(dst + 3 * step, r3);
        vst1q_f16(dst + 4 * step, r4);
        vst1q_f16(dst + 5 * step, r5);
        dst += CONV3X3_WINO_IN * step;
    }
}

static void dstTransform_wino_4x4_3x3(const FLOAT16* src, const FLOAT16* bias, bool relu, bool relu6, FLOAT16* dst,
                                      int step) {
    FLOAT16 midResult[4][6][ARMV82_CHANNEL_UNIT];

    float16x8_t value_0 = vmovq_n_f16(0);
    float16x8_t value_6 = vmovq_n_f16(6);
    float16x8_t value_2 = vmovq_n_f16(2);
    float16x8_t value_4 = vmovq_n_f16(4);
    float16x8_t value_8 = vmovq_n_f16(8);

    float16x8_t value_bias = vld1q_f16(bias);

    for (int i = 0; i < CONV3X3_WINO_IN; ++i) {
        float16x8_t a0i = vld1q_f16(src + (CONV3X3_WINO_IN * 0 + i) * step);
        float16x8_t a1i = vld1q_f16(src + (CONV3X3_WINO_IN * 1 + i) * step);
        float16x8_t a2i = vld1q_f16(src + (CONV3X3_WINO_IN * 2 + i) * step);
        float16x8_t a3i = vld1q_f16(src + (CONV3X3_WINO_IN * 3 + i) * step);
        float16x8_t a4i = vld1q_f16(src + (CONV3X3_WINO_IN * 4 + i) * step);
        float16x8_t a5i = vld1q_f16(src + (CONV3X3_WINO_IN * 5 + i) * step);

        float16x8_t b0 = vaddq_f16(a1i, a2i);
        float16x8_t b1 = vaddq_f16(a3i, a4i);
        float16x8_t b2 = vsubq_f16(a1i, a2i);
        float16x8_t b3 = vsubq_f16(a3i, a4i);

        float16x8_t r0 = vaddq_f16(vaddq_f16(b0, b1), a0i);
        float16x8_t r1 = vfmaq_f16(b2, b3, value_2);
        float16x8_t r2 = vfmaq_f16(b0, b1, value_4);
        float16x8_t r3 = vaddq_f16(a5i, vfmaq_f16(b2, b3, value_8));

        vst1q_f16(midResult[0][i], r0);
        vst1q_f16(midResult[1][i], r1);
        vst1q_f16(midResult[2][i], r2);
        vst1q_f16(midResult[3][i], r3);
    }

    for (int i = 0; i < CONV3X3_WINO_OUT; ++i) {
        float16x8_t a0i = vld1q_f16(midResult[i][0]);
        float16x8_t a1i = vld1q_f16(midResult[i][1]);
        float16x8_t a2i = vld1q_f16(midResult[i][2]);
        float16x8_t a3i = vld1q_f16(midResult[i][3]);
        float16x8_t a4i = vld1q_f16(midResult[i][4]);
        float16x8_t a5i = vld1q_f16(midResult[i][5]);

        float16x8_t b0 = vaddq_f16(a1i, a2i);
        float16x8_t b1 = vaddq_f16(a3i, a4i);
        float16x8_t b2 = vsubq_f16(a1i, a2i);
        float16x8_t b3 = vsubq_f16(a3i, a4i);

        float16x8_t r0 = vaddq_f16(vaddq_f16(b0, b1), a0i);
        float16x8_t r1 = vfmaq_f16(b2, b3, value_2);
        float16x8_t r2 = vfmaq_f16(b0, b1, value_4);
        float16x8_t r3 = vaddq_f16(a5i, vfmaq_f16(b2, b3, value_8));

        r0 = vaddq_f16(r0, value_bias);
        r1 = vaddq_f16(r1, value_bias);
        r2 = vaddq_f16(r2, value_bias);
        r3 = vaddq_f16(r3, value_bias);

        if (relu) {
            r0 = vmaxq_f16(r0, value_0);
            r1 = vmaxq_f16(r1, value_0);
            r2 = vmaxq_f16(r2, value_0);
            r3 = vmaxq_f16(r3, value_0);
        }
        if (relu6) {
            r0 = vmaxq_f16(r0, value_0);
            r1 = vmaxq_f16(r1, value_0);
            r2 = vmaxq_f16(r2, value_0);
            r3 = vmaxq_f16(r3, value_0);
            r0 = vminq_f16(r0, value_6);
            r1 = vminq_f16(r1, value_6);
            r2 = vminq_f16(r2, value_6);
            r3 = vminq_f16(r3, value_6);
        }

        vst1q_f16(dst + 0 * ARMV82_CHANNEL_UNIT, r0);
        vst1q_f16(dst + 1 * ARMV82_CHANNEL_UNIT, r1);
        vst1q_f16(dst + 2 * ARMV82_CHANNEL_UNIT, r2);
        vst1q_f16(dst + 3 * ARMV82_CHANNEL_UNIT, r3);
        dst += CONV3X3_WINO_OUT * ARMV82_CHANNEL_UNIT;
    }
}

#endif

Arm82Convolution3x3::Arm82Convolution3x3(const MNN::Convolution2D* convParam, Backend* bn) : Execution(bn) {
    const auto commonParam  = convParam->common();
    mCommon                 = commonParam;
    int inputChannel        = commonParam->inputCount();
    const int outputChannel = commonParam->outputCount();

    if (inputChannel == 0) {
        if (convParam->quanParameter()) {
            inputChannel = convParam->quanParameter()->buffer()->size() / (2 * 9 * outputChannel);
        } else {
            inputChannel = convParam->weight()->size() / (9 * outputChannel);
        }
    }

    const int icDiv8 = UP_DIV(inputChannel, ARMV82_CHANNEL_UNIT);
    const int ocDiv8 = UP_DIV(outputChannel, ARMV82_CHANNEL_UNIT);
    mRelu            = mCommon->relu();
    mRelu6           = mCommon->relu6();
    // transform weight
    {
        mWeightFp16.reset(
            Tensor::createDevice<uint16_t>({icDiv8 * ocDiv8 * CONV3X3_WEIGHT_UNIT * ARMV82_CHANNEL_UNIT}));
        mValid = bn->onAcquireBuffer(mWeightFp16.get(), Backend::STATIC);
        if (!mValid) {
            return;
        }

        memset(mWeightFp16->host<uint16_t>(), 0, mWeightFp16->size());

        const FLOAT16* fp16WeightPtr = nullptr;
        // Set source size align avoid of heap error
        std::vector<FLOAT16> weightFp16(ocDiv8 * ARMV82_CHANNEL_UNIT * inputChannel * CONV3X3_WINO_KER * CONV3X3_WINO_KER, 0);
        if (convParam->quanParameter()) {
            // the data type of weight is fp16
            fp16WeightPtr = weightFp16.data();
            ::memcpy(weightFp16.data(), (convParam->quanParameter()->buffer()->data()), convParam->quanParameter()->buffer()->size());
        } else {
            // the data type of weight is fp32, then quantize weight to be fp16 data type
            int size = convParam->weight()->size();
            MNNQuantizeFP16(weightFp16.data(), convParam->weight()->data(), size);
            fp16WeightPtr = weightFp16.data();
        }

        const auto srcWeightPtr = fp16WeightPtr;
        auto dstWeightPtr       = mWeightFp16->host<FLOAT16>();

        auto transformWeight = [&](int ocUnit, int ocStart, int ocEnd, FLOAT16* weight) {
            for (int oc = ocStart; oc < ocEnd; ++oc) {
                const int oci             = oc / ocUnit;
                const int ocj             = oc % ocUnit;
                const auto srcWeightOcPtr = srcWeightPtr + oc * inputChannel * CONV3X3_WINO_KER * CONV3X3_WINO_KER;
                auto dstWeightOcPtr       = weight + oci * icDiv8 * ARMV82_CHANNEL_UNIT * ocUnit + ocj;
                for (int ic = 0; ic < inputChannel; ++ic) {
                    const auto srcWeightIcPtr = srcWeightOcPtr + ic * CONV3X3_WINO_KER * CONV3X3_WINO_KER;
                    auto dstWeightIcPtr       = dstWeightOcPtr + ic * ocUnit;

                    kernelTransform_wino_4x4_3x3(srcWeightIcPtr, dstWeightIcPtr,
                                                 icDiv8 * ocDiv8 * ARMV82_CHANNEL_UNIT * ARMV82_CHANNEL_UNIT);
                }
            }
        };

        const int ocDivDoubleUnit = ocDiv8 / 2;
        if (ocDivDoubleUnit > 0) {
            transformWeight((ARMV82_CHANNEL_UNIT * 2), 0, ocDivDoubleUnit * (ARMV82_CHANNEL_UNIT * 2), dstWeightPtr);
        }
        if (ocDiv8 % 2 == 1) {
            transformWeight(ARMV82_CHANNEL_UNIT, ocDivDoubleUnit * (ARMV82_CHANNEL_UNIT * 2), outputChannel,
                            dstWeightPtr);
        }
    }

    mBiasFp16.reset(Tensor::createDevice<uint16_t>({ocDiv8 * ARMV82_CHANNEL_UNIT}));
    mValid = bn->onAcquireBuffer(mBiasFp16.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }

    // TODO, bias is fp32, save bias also in fp16?
    auto biasDstPtr = mBiasFp16->host<FLOAT16>();
    memset(biasDstPtr, 0, mBiasFp16->size());
    MNNQuantizeFP16(biasDstPtr, convParam->bias()->data(), outputChannel);
}

Arm82Convolution3x3::~Arm82Convolution3x3() {
    if (nullptr != mWeightFp16) {
        backend()->onReleaseBuffer(mWeightFp16.get(), Backend::STATIC);
    }
    if (nullptr != mBiasFp16) {
        backend()->onReleaseBuffer(mBiasFp16.get(), Backend::STATIC);
    }
}

ErrorCode Arm82Convolution3x3::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    mPadX = mCommon->padX();
    mPadY = mCommon->padY();
    if (mCommon->padMode() == PadMode_SAME) {
        int kernelWidthSize  = (mCommon->kernelX() - 1) * mCommon->dilateX() + 1;
        int kernelHeightSize = (mCommon->kernelY() - 1) * mCommon->dilateY() + 1;

        int padNeededWidth  = (output->width() - 1) * mCommon->strideX() + kernelWidthSize - input->width();
        int padNeededHeight = (output->height() - 1) * mCommon->strideY() + kernelHeightSize - input->height();
        mPadX               = padNeededWidth / 2;
        mPadY               = padNeededHeight / 2;
    }

    mThreadNums                          = std::max(static_cast<Arm82Backend*>(backend())->numberThread(), 1);
    mTransformBuffer.buffer().dimensions = 4;
    mTransformBuffer.setType(DataType_DT_BFLOAT16);
    mTransformBuffer.setLength(0, mThreadNums);
    mTransformBuffer.setLength(1, CONV3X3_WINO_TILE);
    mTransformBuffer.setLength(
        2, UP_DIV(input->channel(), ARMV82_CHANNEL_UNIT) + UP_DIV(output->channel(), ARMV82_CHANNEL_UNIT) + 1);
    mTransformBuffer.setLength(3, CONV3X3_WINO_SRC_NUM);
    TensorUtils::setLinearLayout(&mTransformBuffer);

    bool allocSuccess = backend()->onAcquireBuffer(&mTransformBuffer, Backend::DYNAMIC);
    if (!allocSuccess) {
        return OUT_OF_MEMORY;
    }

    mDummyBias.buffer().dimensions = 1;
    mDummyBias.setType(DataType_DT_BFLOAT16);
    mDummyBias.setLength(0, UP_DIV(output->channel(), ARMV82_CHANNEL_UNIT) * ARMV82_CHANNEL_UNIT);
    allocSuccess = backend()->onAcquireBuffer(&mDummyBias, Backend::DYNAMIC);
    if (!allocSuccess) {
        return OUT_OF_MEMORY;
    }

    backend()->onReleaseBuffer(&mTransformBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mDummyBias, Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode Arm82Convolution3x3::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input       = inputs[0];
    auto output      = outputs[0];
    const int batch  = input->batch();
    const int ih     = input->height();
    const int iw     = input->width();
    const int ihw    = ih * iw;
    const int icDiv8 = UP_DIV(input->channel(), ARMV82_CHANNEL_UNIT);
    const int oh     = output->height();
    const int ow     = output->width();
    const int ohw    = oh * ow;
    const int ocDiv8 = UP_DIV(output->channel(), ARMV82_CHANNEL_UNIT);

    const int hUnit = UP_DIV(oh, CONV3X3_WINO_OUT);
    const int wUnit = UP_DIV(ow, CONV3X3_WINO_OUT);

    const int hPadded = hUnit * CONV3X3_WINO_OUT - oh;
    const int wPadded = wUnit * CONV3X3_WINO_OUT - ow;

    const int outUnitCount = hUnit * wUnit;
    const int tileCount    = UP_DIV(outUnitCount, CONV3X3_WINO_TILE);

    const auto weightPtr    = mWeightFp16->host<FLOAT16>();
    const auto biasDummyPtr = mDummyBias.host<FLOAT16>();
    const auto biasPtr      = mBiasFp16->host<FLOAT16>();

    memset(mDummyBias.host<FLOAT16>(), 0, mDummyBias.size());

    auto srcGetAndTransformFunc = [=](int xIndex, int realTile, const FLOAT16* srcOrigin, FLOAT16* transformedBuffer,
                                      FLOAT16* tempBuffer) {
        memset(tempBuffer, 0, CONV3X3_WINO_TILE * CONV3X3_WINO_SRC_NUM * sizeof(FLOAT16));
        for (int tindex = 0; tindex < realTile; ++tindex) {
            int index  = xIndex + tindex;
            int hindex = index / wUnit;
            int windex = index % wUnit;

            int srcX = windex * CONV3X3_WINO_OUT - mPadX;
            int srcY = hindex * CONV3X3_WINO_OUT - mPadY;
            int sy   = ALIMAX(0, srcY) - srcY;
            int ey   = ALIMIN(srcY + CONV3X3_WINO_IN, ih) - srcY;
            int sx   = ALIMAX(0, srcX) - srcX;
            int ex   = ALIMIN(srcX + CONV3X3_WINO_IN, iw) - srcX;

            const auto srcStart = srcOrigin + (srcX + srcY * iw) * ARMV82_CHANNEL_UNIT;
            auto curTransPtr    = transformedBuffer + tindex * ARMV82_CHANNEL_UNIT;
            auto curTempBuffer  = tempBuffer + tindex * CONV3X3_WINO_SRC_NUM;

            for (int c = 0; c < icDiv8; ++c) {
                const auto curChannelSrcPtr = srcStart + c * ihw * ARMV82_CHANNEL_UNIT;
                auto curChannelTransPtr     = curTransPtr + c * CONV3X3_WINO_TILE * ARMV82_CHANNEL_UNIT;
                if (ex > sx) {
                    for (int yy = sy; yy < ey; ++yy) {
                        const auto srcPtr = curChannelSrcPtr + yy * iw * ARMV82_CHANNEL_UNIT;
                        auto dstPtr       = curTempBuffer + yy * CONV3X3_WINO_IN * ARMV82_CHANNEL_UNIT;

                        memcpy(dstPtr + ARMV82_CHANNEL_UNIT * sx, srcPtr + ARMV82_CHANNEL_UNIT * sx,
                               (ex - sx) * sizeof(FLOAT16) * ARMV82_CHANNEL_UNIT);
                    }
                }

                sourceTransform_wino_4x4_3x3(curTempBuffer, curChannelTransPtr,
                                             ARMV82_CHANNEL_UNIT * CONV3X3_WINO_TILE * icDiv8);
            }
        }

        // shuffel channel
        if (realTile > (CONV3X3_WINO_TILE / 2)) {
            MNNShuffleChannelC8(transformedBuffer, transformedBuffer,
                                (size_t)(icDiv8 * CONV3X3_WINO_IN * CONV3X3_WINO_IN), 0);
        } else {
            for (int i = 0; i < CONV3X3_WINO_IN * CONV3X3_WINO_IN; ++i) {
                auto dst = transformedBuffer + i * ARMV82_CHANNEL_UNIT * CONV3X3_WINO_TILE * icDiv8;
                MNNShuffleChannelC8(dst, dst, (size_t)(icDiv8), 1);
            }
        }
    };

    auto dstTransformAndSave = [=](int xIndex, int realTile, const FLOAT16* transformedBuffer, const FLOAT16* bias,
                                   bool relu, bool relu6, FLOAT16* dstOrigin, FLOAT16* tempBuffer) {
        for (int tindex = 0; tindex < realTile; ++tindex) {
            int index  = xIndex + tindex;
            int hindex = index / wUnit;
            int windex = index % wUnit;
            int dstX   = windex * CONV3X3_WINO_OUT;
            int dstY   = hindex * CONV3X3_WINO_OUT;

            const auto curTransPtr = transformedBuffer + tindex * ARMV82_CHANNEL_UNIT;
            auto dstStartPtr       = dstOrigin + (dstX + dstY * ow) * ARMV82_CHANNEL_UNIT;
            auto curTempBuffer     = tempBuffer + tindex * CONV3X3_WINO_SRC_NUM;

            int hReamin = CONV3X3_WINO_OUT;
            int wReamin = CONV3X3_WINO_OUT;

            if (hindex == (hUnit - 1)) {
                hReamin = CONV3X3_WINO_OUT - hPadded;
            }
            if (windex == (wUnit - 1)) {
                wReamin = CONV3X3_WINO_OUT - wPadded;
            }

            for (int z = 0; z < ocDiv8; ++z) {
                const auto curChannelTransPtr = curTransPtr + z * CONV3X3_WINO_TILE * ARMV82_CHANNEL_UNIT;
                auto dstZ                     = dstStartPtr + z * ohw * ARMV82_CHANNEL_UNIT;

                dstTransform_wino_4x4_3x3(curChannelTransPtr, bias + z * ARMV82_CHANNEL_UNIT, relu, relu6,
                                          curTempBuffer, ocDiv8 * CONV3X3_WINO_TILE * ARMV82_CHANNEL_UNIT);

                // save 4x4 outputs from tempBuffer
                for (int i = 0; i < hReamin; ++i) {
                    memcpy(dstZ + i * ow * ARMV82_CHANNEL_UNIT,
                           curTempBuffer + i * CONV3X3_WINO_OUT * ARMV82_CHANNEL_UNIT,
                           sizeof(FLOAT16) * ARMV82_CHANNEL_UNIT * wReamin);
                }
            }
        }
    };

    auto threadFunction = [&](size_t tId, size_t tileStart, int tileStep, int tileEnd, const FLOAT16* srcOrigin,
                              FLOAT16* dstOrigin) {
        auto curThreadTransformPtr = mTransformBuffer.host<FLOAT16>() + tId * mTransformBuffer.stride(0);
        auto srcTransformedPtr     = curThreadTransformPtr;
        auto dstTransformedPtr     = curThreadTransformPtr + CONV3X3_WINO_TILE * CONV3X3_WINO_SRC_NUM * icDiv8;
        auto tempBufferPtr = curThreadTransformPtr + CONV3X3_WINO_TILE * CONV3X3_WINO_SRC_NUM * (icDiv8 + ocDiv8);

        for (size_t tindex = tileStart; tindex < tileEnd; tindex += tileStep) {
            int xIndex      = (int)tindex * CONV3X3_WINO_TILE;
            int xRemain     = outUnitCount - xIndex;
            int realTileNum = xRemain > CONV3X3_WINO_TILE ? CONV3X3_WINO_TILE : xRemain;

            srcGetAndTransformFunc(xIndex, realTileNum, srcOrigin, srcTransformedPtr, tempBufferPtr);

            // matmul
            for (int i = 0; i < CONV3X3_WINO_IN * CONV3X3_WINO_IN; ++i) {
                MNNGemmFP16C8_UNIT(dstTransformedPtr + i * ocDiv8 * CONV3X3_WINO_TILE * ARMV82_CHANNEL_UNIT,
                                   srcTransformedPtr + i * ARMV82_CHANNEL_UNIT * CONV3X3_WINO_TILE * icDiv8,
                                   weightPtr + i * icDiv8 * ocDiv8 * ARMV82_CHANNEL_UNIT * ARMV82_CHANNEL_UNIT,
                                   biasDummyPtr, icDiv8, ARMV82_CHANNEL_UNIT * CONV3X3_WINO_TILE * sizeof(FLOAT16),
                                   ocDiv8, 0, 0, realTileNum);
            }

            dstTransformAndSave(xIndex, realTileNum, dstTransformedPtr, biasPtr, mRelu, mRelu6, dstOrigin,
                                tempBufferPtr);
        }
    };

    const auto srcOriginPtr  = input->host<FLOAT16>();
    auto dstOriginPtr        = output->host<FLOAT16>();
    const int inBatchStride  = icDiv8 * ihw * ARMV82_CHANNEL_UNIT;
    const int outBatchStride = ocDiv8 * ohw * ARMV82_CHANNEL_UNIT;
    for (int bIndex = 0; bIndex < batch; ++bIndex) {
        const auto curSrcBatchPtr = srcOriginPtr + bIndex * inBatchStride;
        auto curDstBatchPtr       = dstOriginPtr + bIndex * outBatchStride;

        if (tileCount >= mThreadNums) {
            MNN_CONCURRENCY_BEGIN(tId, mThreadNums)
            threadFunction((int)tId, (int)tId, mThreadNums, (tileCount / mThreadNums) * mThreadNums, curSrcBatchPtr,
                           curDstBatchPtr);
#ifdef MNN_USE_THREAD_POOL
            MNN_CONCURRENCY_ARM82_END();
#else
            MNN_CONCURRENCY_END();
#endif
        }
        if (tileCount % mThreadNums != 0) {
            threadFunction(0, (tileCount / mThreadNums) * mThreadNums, 1, tileCount, curSrcBatchPtr, curDstBatchPtr);
        }
    }

    return NO_ERROR;
}

} // namespace MNN

#endif
