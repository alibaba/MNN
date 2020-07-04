//
//  Arm82Interp.cpp
//  MNN
//
//  Created by MNN on 2020/04/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__
#include "backend/arm82/Arm82Interp.hpp"
#include <math.h>
#include "core/Concurrency.h"
#include "core/Macro.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

static void Arm82BilinearSampleCUnit(const FLOAT16* src, FLOAT16* dst, const int* position, const FLOAT16* factor,
                                     int width) {
    for (int i = 0; i < width; ++i) {
        FLOAT16 f = factor[i];
#ifdef MNN_USE_NEON
        float16x8_t vdf = vdupq_n_f16(f);
        float16x8_t vsf = vdupq_n_f16(1.0f - f);
        float16x8_t A   = vld1q_f16(src + ARMV82_CHANNEL_UNIT * position[2 * i + 0]);
        float16x8_t B   = vld1q_f16(src + ARMV82_CHANNEL_UNIT * position[2 * i + 1]);
        vst1q_f16(dst + ARMV82_CHANNEL_UNIT * i, A * vsf + B * vdf);
#else
        for (int k = 0; k < ARMV82_CHANNEL_UNIT; ++k) {
            FLOAT16 A                        = src[ARMV82_CHANNEL_UNIT * position[2 * i + 0] + k];
            FLOAT16 B                        = src[ARMV82_CHANNEL_UNIT * position[2 * i + 1] + k];
            dst[ARMV82_CHANNEL_UNIT * i + k] = A * (1 - f) + B * f;
        }
#endif
    }
}

static void Arm82BilinearLineCUnit(FLOAT16* dst, const FLOAT16* A, const FLOAT16* B, const FLOAT16* factor, int width) {
#ifdef MNN_USE_NEON
    float16x8_t vdf = vdupq_n_f16(*factor);
    float16x8_t vsf = vdupq_n_f16(1.0f) - vdf;
    for (int i = 0; i < width; ++i) {
        float16x8_t value_a = vld1q_f16(A + ARMV82_CHANNEL_UNIT * i);
        float16x8_t value_b = vld1q_f16(B + ARMV82_CHANNEL_UNIT * i);
        vst1q_f16(dst + ARMV82_CHANNEL_UNIT * i, value_a * vsf + value_b * vdf);
    }

#else
    FLOAT16 f = *factor;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < ARMV82_CHANNEL_UNIT; ++j) {
            int index = i * ARMV82_CHANNEL_UNIT + j;
            dst[index] = A[index] * (1 - f) + B[index] * f;
        }
    }
#endif
}

static inline int CLAMP(int a, int min, int max) {
    if (a < min) {
        a = min;
    } else if (a > max) {
        a = max;
    }
    return a;
}

Arm82Interp::Arm82Interp(Backend* backend, float widthScale, float heightScale, int resizeType, bool AlignCorners,
                         bool halfPixelCenters)
    : Execution(backend),
      mWidthScale(widthScale),
      mHeightScale(heightScale),
      mResizeType(resizeType),
      mAlignCorners(AlignCorners),
      mHalfPixelCenters(halfPixelCenters) {
}

Arm82Interp::~Arm82Interp() {
}

ErrorCode Arm82Interp::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];

    const int iw = input->width();
    const int ih = input->height();
    const int ow = output->width();
    const int oh = output->height();

    if (mAlignCorners) {
        if (oh == 1) {
            mHeightScale = 0.0f;
        } else {
            mHeightScale = (float)(ih - 1) / (float)(oh - 1);
        }

        if (ow == 1) {
            mWidthScale = 0.0f;
        } else {
            mWidthScale = (float)(iw - 1) / (float)(ow - 1);
        }
    } else {
        mHeightScale = (float)(ih) / (float)(oh);
        mWidthScale  = (float)(iw) / (float)(ow);
    }

    const float xScaling                  = mWidthScale;
    const float yScaling                  = mHeightScale;
    mWidthPosition.buffer().dim[0].extent = 2 * ow;
    mWidthPosition.buffer().dimensions    = 1;
    mWidthPosition.setType(DataType_DT_INT32);
    backend()->onAcquireBuffer(&mWidthPosition, Backend::DYNAMIC_SEPERATE);

    mWidthFactor.buffer().dim[0].extent = ow;
    mWidthFactor.buffer().dimensions    = 1;
    mWidthFactor.setType(DataType_DT_INT16);
    backend()->onAcquireBuffer(&mWidthFactor, Backend::DYNAMIC_SEPERATE);

    auto _wPositionPtr = mWidthPosition.host<int>();
    auto _wFactorPtr   = mWidthFactor.host<FLOAT16>();

    for (int x = 0; x < ow; ++x) {
        FLOAT16 srcX;
        if (mHalfPixelCenters) {
            srcX = (x + 0.5) * xScaling - 0.5;
        } else {
            srcX = x * xScaling;
        }
        int x1                   = floor(srcX);
        FLOAT16 x2Factor         = srcX - x1;
        _wFactorPtr[x]           = x2Factor;
        _wPositionPtr[2 * x + 0] = CLAMP(x1, 0, iw - 1);
        _wPositionPtr[2 * x + 1] = CLAMP(x1 + 1, 0, iw - 1);
    }

    mHeightPosition.buffer().dim[0].extent = 2 * oh;
    mHeightPosition.buffer().dimensions    = 1;
    mHeightPosition.setType(DataType_DT_INT32);
    backend()->onAcquireBuffer(&mHeightPosition, Backend::DYNAMIC_SEPERATE);

    mHeightFactor.buffer().dim[0].extent = oh;
    mHeightFactor.buffer().dimensions    = 1;
    mHeightFactor.setType(DataType_DT_INT16);
    backend()->onAcquireBuffer(&mHeightFactor, Backend::DYNAMIC_SEPERATE);

    auto _hPositionPtr = mHeightPosition.host<int>();
    auto _hFactorPtr   = mHeightFactor.host<FLOAT16>();

    for (int y = 0; y < oh; ++y) {
        FLOAT16 srcY;
        if (mHalfPixelCenters) {
            srcY = (y + 0.5) * yScaling - 0.5;
        } else {
            srcY = y * yScaling;
        }

        int y1                   = floor(srcY);
        FLOAT16 y2Factor         = srcY - y1;
        _hFactorPtr[y]           = y2Factor;
        _hPositionPtr[2 * y + 0] = CLAMP(y1, 0, ih - 1);
        _hPositionPtr[2 * y + 1] = CLAMP(y1 + 1, 0, ih - 1);
    }

    mTheadNumbers = static_cast<Arm82Backend*>(backend())->numberThread();

    mLineBuffer.buffer().dimensions    = 1;
    mLineBuffer.buffer().dim[0].extent = 2 * ARMV82_CHANNEL_UNIT * ow * mTheadNumbers;
    mLineBuffer.setType(DataType_DT_INT16);
    backend()->onAcquireBuffer(&mLineBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mLineBuffer, Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode Arm82Interp::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input              = inputs[0];
    auto output                   = outputs[0];
    const int batches             = input->batch();
    const int iw                  = input->width();
    const int ih                  = input->height();
    const int ow                  = output->width();
    const int oh                  = output->height();
    const int inputBatchStride    = iw * ih * ARMV82_CHANNEL_UNIT;
    const int outputBatchStride   = ow * oh * ARMV82_CHANNEL_UNIT;
    const int inputChannelStride  = iw * ih;
    const int outputChannelStride = ow * oh;
    const int channelDivUnit      = UP_DIV(input->channel(), ARMV82_CHANNEL_UNIT);

    if (mResizeType == 2) {
        const auto widthPositionPtr  = mWidthPosition.host<int>();
        const auto widthFactorPtr    = mWidthFactor.host<FLOAT16>();
        const auto heightPositionPtr = mHeightPosition.host<int>();
        const auto heightFactorPtr   = mHeightFactor.host<FLOAT16>();
        auto lineBuffer              = mLineBuffer.host<FLOAT16>();

        for (int b = 0; b < batches; ++b) {
            const auto curInputBatchPtr = input->host<FLOAT16>() + b * inputBatchStride;
            auto curOutputBatchPtr      = output->host<FLOAT16>() + b * outputBatchStride;

            auto threadFucntion = [&](size_t tId, const FLOAT16* src, FLOAT16* dst) {
                for (int n = (int)tId; n < channelDivUnit; n += mTheadNumbers) {
                    auto _lineBuffer                = lineBuffer + 2 * ARMV82_CHANNEL_UNIT * ow * tId;
                    auto _line0                     = _lineBuffer;
                    auto _line1                     = _lineBuffer + ARMV82_CHANNEL_UNIT * ow;
                    int yUsed[2]                    = {0, 0};
                    int yCache[2]                   = {-1, -1};
                    FLOAT16* yCacheLine[2]          = {_line0, _line1};
                    FLOAT16* const yCacheStorage[2] = {_line0, _line1};

                    const auto curSrc = src + n * ARMV82_CHANNEL_UNIT * inputChannelStride;
                    auto curDst       = dst + n * ARMV82_CHANNEL_UNIT * outputChannelStride;

                    for (int h = 0; h < oh; ++h) {
                        int yPosition[2];
                        yPosition[0] = heightPositionPtr[2 * h + 0];
                        yPosition[1] = heightPositionPtr[2 * h + 1];

                        for (int j = 0; j < 2; ++j) {
                            yUsed[j] = 0;
                        }
                        for (int j = 0; j < 2; ++j) {
                            bool find = false;
                            for (int k = 0; k < 2; ++k) {
                                if (yPosition[j] == yCache[k]) {
                                    yUsed[k]      = 1;
                                    yCacheLine[j] = yCacheStorage[k];
                                    find          = true;
                                }
                            }

                            if (!find) {
                                const auto curLine = curSrc + yPosition[j] * iw * ARMV82_CHANNEL_UNIT;
                                for (int k = 0; k < 2; ++k) {
                                    if (!yUsed[k]) {
                                        yCache[k]     = yPosition[j];
                                        yUsed[k]      = 1;
                                        yCacheLine[j] = yCacheStorage[k];
                                        Arm82BilinearSampleCUnit(curLine, yCacheLine[j], widthPositionPtr,
                                                                 widthFactorPtr, ow);
                                        break;
                                    }
                                }
                            }
                        }
                        Arm82BilinearLineCUnit(curDst + ow * h * ARMV82_CHANNEL_UNIT, yCacheLine[0], yCacheLine[1],
                                               heightFactorPtr + h, ow);
                    }
                }
            };

            MNN_CONCURRENCY_BEGIN(tId, mTheadNumbers)
            threadFucntion(tId, curInputBatchPtr, curOutputBatchPtr);
#ifdef MNN_USE_THREAD_POOL
            MNN_CONCURRENCY_ARM82_END();
#else
            MNN_CONCURRENCY_END();
#endif
        }
    } else {
        return NOT_SUPPORT;
    }

    return NO_ERROR;
}

class Arm82InterpCreator : public Arm82Backend::Arm82Creator {
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto param = op->main_as_Interp();
        // TODO, support other resize types
        if(param->resizeType() != 2){
            return nullptr;
        }
        return new Arm82Interp(backend, param->widthScale(), param->heightScale(), param->resizeType(),
                               param->alignCorners(), param->halfPixelCenters());
    }
};

REGISTER_ARM82_OP_CREATOR(OpType_Interp, Arm82InterpCreator);

} // namespace MNN

#endif
