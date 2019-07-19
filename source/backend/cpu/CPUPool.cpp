//
//  CPUPool.cpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUPool.hpp"
#include <float.h>
#include <math.h>
#include "Macro.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#include "Concurrency.h"

static void pooling_max_pad(const float *channelInput, float *offsetOutput, int inputWidth, int inputHeight,
                            int inputStep4, int inputSize4, int kernelWidth, int kernelHeight, int iw, int ih) {
#ifdef MNN_USE_NEON
    float32x4_t max = vdupq_n_f32(-FLT_MAX);
#else
    float max0 = -FLT_MAX;
    float max1 = -FLT_MAX;
    float max2 = -FLT_MAX;
    float max3 = -FLT_MAX;
#endif

    const float *bottomLine = channelInput + inputSize4 - inputStep4;
    for (int kh = 0; kh < kernelHeight; kh++) {
        const int h                  = ih + kh;
        const float *paddedLineInput = nullptr;
        if (h < 0) { // top replicate
            paddedLineInput = channelInput;
        } else if (h >= inputHeight) { // bottom replicate
            paddedLineInput = bottomLine;
        } else {
            paddedLineInput = channelInput + h * inputStep4;
        }

        const float *rightEdge = paddedLineInput + inputStep4 - 4;
        for (int kw = 0; kw < kernelWidth; kw++) {
            const int w              = iw + kw;
            const float *cursorInput = nullptr;
            if (w < 0) { // left replicate
                cursorInput = paddedLineInput;
            } else if (w >= inputWidth) { // right replicate
                cursorInput = rightEdge;
            } else {
                cursorInput = paddedLineInput + 4 * w;
            }
#ifdef MNN_USE_NEON
            max = vmaxq_f32(max, vld1q_f32(cursorInput));
#else
            max0 = std::max(max0, cursorInput[0]);
            max1 = std::max(max1, cursorInput[1]);
            max2 = std::max(max2, cursorInput[2]);
            max3 = std::max(max3, cursorInput[3]);
#endif
        }
    }

#ifdef MNN_USE_NEON
    vst1q_f32(offsetOutput, max);
#else
    offsetOutput[0] = max0;
    offsetOutput[1] = max1;
    offsetOutput[2] = max2;
    offsetOutput[3] = max3;
#endif
}

static void poolingMax(const float *channelInput, int inputWidth, int inputHeight, float *channelOutput,
                       int outputWidth, int outputHeight, int kernelWidth, int kernelHeight, int strideWidth,
                       int strideHeight, int padWidth, int padHeight, MNN::PoolPadType padType) {
    int padTop    = padHeight <= 0 ? 0 : (padHeight + strideHeight - 1) / strideHeight;
    int padBottom = (padHeight + inputHeight - kernelHeight) / strideHeight + 1;
    int padLeft   = padWidth <= 0 ? 0 : (padWidth + strideWidth - 1) / strideWidth;
    int padRight  = (padWidth + inputWidth - kernelWidth) / strideWidth + 1;

    const int inputStep4       = 4 * inputWidth;
    const int inputSize4       = inputStep4 * inputHeight;
    const int strideInputStep4 = strideHeight * inputStep4;
    const int outputStep4      = 4 * outputWidth;
    const int strideWidth4     = 4 * strideWidth;

    { // handle paddings top
        float *lineOutput = channelOutput;
        for (int oh = 0, ih = -padHeight; oh < padTop; oh++, ih += strideHeight, lineOutput += outputStep4) {
            float *offsetOutput = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < outputWidth; ow++, iw += strideWidth, offsetOutput += 4) {
                pooling_max_pad(channelInput, offsetOutput, inputWidth, inputHeight, inputStep4, inputSize4,
                                kernelWidth, kernelHeight, iw, ih);
            }
        }
        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4) {
            float *offsetOutput = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < padLeft; ow++, iw += strideWidth, offsetOutput += 4) {
                pooling_max_pad(channelInput, offsetOutput, inputWidth, inputHeight, inputStep4, inputSize4,
                                kernelWidth, kernelHeight, iw, ih);
            }
            offsetOutput = lineOutput + padRight * 4;
            for (int ow = padRight, iw = -padWidth + ow * strideWidth; ow < outputWidth;
                 ow++, iw += strideWidth, offsetOutput += 4) {
                pooling_max_pad(channelInput, offsetOutput, inputWidth, inputHeight, inputStep4, inputSize4,
                                kernelWidth, kernelHeight, iw, ih);
            }
        }
        for (int oh = padBottom, ih = -padHeight + oh * strideHeight; oh < outputHeight;
             oh++, ih += strideHeight, lineOutput += outputStep4) {
            float *offsetOutput = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < outputWidth; ow++, iw += strideWidth, offsetOutput += 4) {
                pooling_max_pad(channelInput, offsetOutput, inputWidth, inputHeight, inputStep4, inputSize4,
                                kernelWidth, kernelHeight, iw, ih);
            }
        }
    }

    { // handle no paddings
        const float *lineInput =
            channelInput + (padTop * strideHeight - padHeight) * inputStep4 + (padLeft * strideWidth - padWidth) * 4;
        float *lineOutput = channelOutput + padTop * outputStep4 + padLeft * 4;

        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const float *offsetInput = lineInput;
            float *offsetOutput      = lineOutput;
            for (int ow = padLeft, iw = -padWidth + ow * strideWidth; ow < padRight;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
#ifdef MNN_USE_NEON
                float32x4_t max = vdupq_n_f32(-FLT_MAX);
#else
                float max0 = -FLT_MAX;
                float max1 = -FLT_MAX;
                float max2 = -FLT_MAX;
                float max3 = -FLT_MAX;
#endif
                const float *kernelInput = offsetInput;
                for (int kh = 0; kh < kernelHeight; kh++, kernelInput += inputStep4) {
                    const float *cursorInput = kernelInput;
                    for (int kw = 0; kw < kernelWidth; kw++, cursorInput += 4) {
#ifdef MNN_USE_NEON
                        max = vmaxq_f32(max, vld1q_f32(cursorInput));
#else
                        max0 = std::max(max0, cursorInput[0]);
                        max1 = std::max(max1, cursorInput[1]);
                        max2 = std::max(max2, cursorInput[2]);
                        max3 = std::max(max3, cursorInput[3]);
#endif
                    }
                }

#ifdef MNN_USE_NEON
                vst1q_f32(offsetOutput, max);
#else
                offsetOutput[0] = max0;
                offsetOutput[1] = max1;
                offsetOutput[2] = max2;
                offsetOutput[3] = max3;
#endif
            }
        }
    }
}

static void poolingAvgPad(const float *offsetInput, float *offsetOutput, int inputWidth, int inputHeight,
                          int kernelWidth, int kernelHeight, int inputStep4, int iw, int ih, int padWidth,
                          int padHeight, MNN::PoolPadType padType) {
#ifdef MNN_USE_NEON
    float32x4_t sum = vdupq_n_f32(0);
#else
    float sum0 = 0;
    float sum1 = 0;
    float sum2 = 0;
    float sum3 = 0;
#endif

    const int khs = 0 < -ih ? -ih : 0;                                                 // max
    const int khe = kernelHeight < inputHeight - ih ? kernelHeight : inputHeight - ih; // min
    const int kws = 0 < -iw ? -iw : 0;                                                 // max
    const int kwe = kernelWidth < inputWidth - iw ? kernelWidth : inputWidth - iw;     // min

    // sum
    int count = 0;
    if (padType == MNN::PoolPadType_CAFFE) {
        count = (ALIMIN(ih + kernelHeight, inputHeight + padHeight) - ih) *
                (ALIMIN(iw + kernelWidth, inputWidth + padWidth) - iw);
    } else {
        count = (khe - khs) * (kwe - kws);
    }

    const float *kernelInput = offsetInput + khs * inputStep4;
    for (int kh = khs; kh < khe; kh++, kernelInput += inputStep4) {
        const float *cursorInput = kernelInput + kws * 4;
        for (int kw = kws; kw < kwe; kw++, cursorInput += 4) {
#ifdef MNN_USE_NEON
            sum += vld1q_f32(cursorInput);
#else
            sum0 += cursorInput[0];
            sum1 += cursorInput[1];
            sum2 += cursorInput[2];
            sum3 += cursorInput[3];
#endif
        }
    }

    // avg
    if (count > 0) {
#ifdef MNN_USE_NEON
        vst1q_f32(offsetOutput, sum / vdupq_n_f32(count));
#else
        offsetOutput[0] = sum0 / (float)count;
        offsetOutput[1] = sum1 / (float)count;
        offsetOutput[2] = sum2 / (float)count;
        offsetOutput[3] = sum3 / (float)count;
#endif
    } else {
#ifdef MNN_USE_NEON
        vst1q_f32(offsetOutput, vdupq_n_f32(0));
#else
        offsetOutput[0] = 0;
        offsetOutput[1] = 0;
        offsetOutput[2] = 0;
        offsetOutput[3] = 0;
#endif
    }
}

static void poolingAvg(const float *channelInput, int inputWidth, int inputHeight, float *channelOutput,
                       int outputWidth, int outputHeight, int kernelWidth, int kernelHeight, int strideWidth,
                       int strideHeight, int padWidth, int padHeight, MNN::PoolPadType padType) {
    int padTop    = padHeight <= 0 ? 0 : (padHeight + strideHeight - 1) / strideHeight;
    int padBottom = (padHeight + inputHeight - kernelHeight) / strideHeight + 1;
    int padLeft   = padWidth <= 0 ? 0 : (padWidth + strideWidth - 1) / strideWidth;
    int padRight  = (padWidth + inputWidth - kernelWidth) / strideWidth + 1;

    const int inputStep4       = 4 * inputWidth;
    const int strideInputStep4 = strideHeight * inputStep4;
    const int outputStep4      = 4 * outputWidth;
    const int strideWidth4     = 4 * strideWidth;

    { // handle paddings
        const float *lineInput = channelInput - padHeight * inputStep4 - padWidth * 4;
        float *lineOutput      = channelOutput;
        for (int oh = 0, ih = -padHeight; oh < padTop;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const float *offsetInput = lineInput;
            float *offsetOutput      = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < outputWidth;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
                poolingAvgPad(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType);
            }
        }
        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const float *offsetInput = lineInput;
            float *offsetOutput      = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < padLeft;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
                poolingAvgPad(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType);
            }
            offsetInput  = lineInput + padRight * strideWidth * 4;
            offsetOutput = lineOutput + padRight * 4;
            for (int ow = padRight, iw = -padWidth + ow * strideWidth; ow < outputWidth;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
                poolingAvgPad(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType);
            }
        }
        for (int oh = padBottom, ih = -padHeight + oh * strideHeight; oh < outputHeight;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const float *offsetInput = lineInput;
            float *offsetOutput      = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < outputWidth;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
                poolingAvgPad(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType);
            }
        }
    }

    { // handle no paddings
        const float *lineInput =
            channelInput + (padTop * strideHeight - padHeight) * inputStep4 + (padLeft * strideWidth - padWidth) * 4;
        float *lineOutput = channelOutput + padTop * outputStep4 + padLeft * 4;

        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const float *offsetInput = lineInput;
            float *offsetOutput      = lineOutput;
            for (int ow = padLeft, iw = -padWidth + ow * strideWidth; ow < padRight;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
#ifdef MNN_USE_NEON
                float32x4_t sum = vdupq_n_f32(0);
#else
                float sum0 = 0;
                float sum1 = 0;
                float sum2 = 0;
                float sum3 = 0;
#endif
                // sum
                int count                = 0;
                const float *kernelInput = offsetInput;
                for (int kh = 0; kh < kernelHeight; kh++, kernelInput += inputStep4) {
                    const float *cursorInput = kernelInput;
                    for (int kw = 0; kw < kernelWidth; kw++, cursorInput += 4) {
#ifdef MNN_USE_NEON
                        sum += vld1q_f32(cursorInput);
#else
                        sum0 += cursorInput[0];
                        sum1 += cursorInput[1];
                        sum2 += cursorInput[2];
                        sum3 += cursorInput[3];
#endif
                        count++;
                    }
                }

                // avg
                if (count > 0) {
#ifdef MNN_USE_NEON
                    vst1q_f32(offsetOutput, sum / vdupq_n_f32(count));
#else
                    offsetOutput[0] = sum0 / (float)count;
                    offsetOutput[1] = sum1 / (float)count;
                    offsetOutput[2] = sum2 / (float)count;
                    offsetOutput[3] = sum3 / (float)count;
#endif
                } else {
#ifdef MNN_USE_NEON
                    vst1q_f32(offsetOutput, vdupq_n_f32(0));
#else
                    offsetOutput[0] = 0;
                    offsetOutput[1] = 0;
                    offsetOutput[2] = 0;
                    offsetOutput[3] = 0;
#endif
                }
            }
        }
    }
}

namespace MNN {

CPUPool::CPUPool(Backend *b, const Pool *parameter) : MNN::Execution(b), mParameter(parameter) {
    // nothing to do
}

ErrorCode CPUPool::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto layer       = mParameter;
    int strideWidth  = layer->strideX();
    int strideHeight = layer->strideY();
    int padWidth     = layer->padX();
    int padHeight    = layer->padY();

    // edit const if global
    auto input       = inputs[0];
    auto output      = outputs[0];
    int kernelWidth  = std::min(layer->kernelX(), input->width());
    int kernelHeight = std::min(layer->kernelY(), input->height());
    if (layer->isGlobal()) {
        kernelWidth  = input->width();
        kernelHeight = input->height();
        strideWidth  = input->width();
        strideHeight = input->height();
        padWidth     = 0;
        padHeight    = 0;
    }
    if (layer->padType() == PoolPadType_SAME) {
        int padNeededWidth  = (output->width() - 1) * strideWidth + kernelWidth - input->width();
        int padNeededHeight = (output->height() - 1) * strideHeight + kernelHeight - input->height();
        padWidth            = padNeededWidth > 0 ? padNeededWidth / 2 : 0;
        padHeight           = padNeededHeight > 0 ? padNeededHeight / 2 : 0;
    } else if (layer->padType() == PoolPadType_VALID) {
        padWidth = padHeight = 0;
    }
    auto poolType      = layer->type();
    auto planeFunction = poolingMax;
    if (poolType == PoolType_AVEPOOL) {
        planeFunction = poolingAvg;
    }
    auto totalDepth        = input->batch() * UP_DIV(input->channel(), 4);
    auto inputData         = input->host<float>();
    auto outputData        = output->host<float>();
    auto inputPlaneStride  = 4 * input->width() * input->height();
    auto outputPlaneStride = 4 * output->width() * output->height();
    int threadNumber       = ((CPUBackend *)backend())->threadNumber();
    auto padType           = layer->padType();
    mFunction              = std::make_pair(threadNumber, [=](int tId) {
        for (int channel = (int)tId; channel < totalDepth; channel += threadNumber) {
            // run
            planeFunction(inputData + channel * inputPlaneStride, input->width(), input->height(),
                          outputData + outputPlaneStride * channel, output->width(), output->height(), kernelWidth,
                          kernelHeight, strideWidth, strideHeight, padWidth, padHeight, padType);
        }
    });
    return NO_ERROR;
}

ErrorCode CPUPool::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_CONCURRENCY_BEGIN(tId, mFunction.first) {
        mFunction.second((int)tId);
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

class CPUPoolCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUPool(backend, op->main_as_Pool());
    }
};

REGISTER_CPU_OP_CREATOR(CPUPoolCreator, OpType_Pooling);

} // namespace MNN
