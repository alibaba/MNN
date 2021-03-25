//
//  CPUPool.cpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUPool.hpp"
#include <float.h>
#include <math.h>
#include "core/Macro.h"

#include "core/Concurrency.h"
#include "math/Vec.hpp"

using Vec4 = MNN::Math::Vec<float, 4>;

static void pooling_max_pad(const float *channelInput, float *offsetOutput, int inputWidth, int inputHeight,
                            int inputStep4, int inputSize4, int kernelWidth, int kernelHeight, int iw, int ih) {
    Vec4 max = Vec4(-FLT_MAX);

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
            max = Vec4::max(max, Vec4::load(cursorInput));
        }
    }
    Vec4::save(offsetOutput, max);
}

static void poolingMax(const float *channelInput, int inputWidth, int inputHeight, float *channelOutput,
                       int outputWidth, int outputHeight, int kernelWidth, int kernelHeight, int strideWidth,
                       int strideHeight, int padWidth, int padHeight, MNN::PoolPadType padType, MNN::AvgPoolCountType countType) {
    // Compute Mid Rect
    int l = 0, t = 0, r = outputWidth, b = outputHeight;
    for (; l * strideWidth - padWidth < 0 && l < outputWidth; l++) {
        // do nothing
    }
    for (; t * strideHeight - padHeight < 0 && t < outputHeight; t++) {
        // do nothing
    }
    for (; (r - 1) * strideWidth - padWidth + (kernelWidth - 1) >= inputWidth && r > l; r--) {
        // do nothing
    }
    for (; (b - 1) * strideHeight - padHeight + (kernelHeight - 1) >= inputHeight && b > t; b--) {
        // do nothing
    }
    int padTop = t, padBottom = b, padLeft = l, padRight = r;

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
        int wCount = padRight - padLeft;
        int wCountC4 = wCount / 4;
        int wCountRemain = wCount - wCountC4 * 4;
        int strideWidthFuse = strideWidth4 * 4;

        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const float *offsetInput = lineInput;
            float *offsetOutput      = lineOutput;
            for (int owf = 0; owf < wCountC4; ++owf, offsetOutput += 16, offsetInput += strideWidthFuse) {
                Vec4 max0 = Vec4(-FLT_MAX);
                Vec4 max1 = Vec4(-FLT_MAX);
                Vec4 max2 = Vec4(-FLT_MAX);
                Vec4 max3 = Vec4(-FLT_MAX);
                const float *kernelInput = offsetInput;
                for (int kh = 0; kh < kernelHeight; kh++, kernelInput += inputStep4) {
                    const float *cursorInput = kernelInput;
                    for (int kw = 0; kw < kernelWidth; kw++, cursorInput += 4) {
                        max0 = Vec4::max(max0, Vec4::load(cursorInput + 0 * strideWidth4));
                        max1 = Vec4::max(max1, Vec4::load(cursorInput + 1 * strideWidth4));
                        max2 = Vec4::max(max2, Vec4::load(cursorInput + 2 * strideWidth4));
                        max3 = Vec4::max(max3, Vec4::load(cursorInput + 3 * strideWidth4));
                    }
                }
                Vec4::save(offsetOutput + 4 * 0, max0);
                Vec4::save(offsetOutput + 4 * 1, max1);
                Vec4::save(offsetOutput + 4 * 2, max2);
                Vec4::save(offsetOutput + 4 * 3, max3);
            }
            for (int ow = 0; ow < wCountRemain;
                 ow++, offsetOutput += 4, offsetInput += strideWidth4) {
                const float *kernelInput = offsetInput;
                Vec4 max = Vec4(-FLT_MAX);
                for (int kh = 0; kh < kernelHeight; kh++, kernelInput += inputStep4) {
                    const float *cursorInput = kernelInput;
                    for (int kw = 0; kw < kernelWidth; kw++, cursorInput += 4) {
                        max = Vec4::max(max, Vec4::load(cursorInput));
                    }
                }

                Vec4::save(offsetOutput, max);
            }
        }
    }
}

static void poolingAvgPad(const float *offsetInput, float *offsetOutput, int inputWidth, int inputHeight,
                          int kernelWidth, int kernelHeight, int inputStep4, int iw, int ih, int padWidth,
                          int padHeight, MNN::PoolPadType padType, MNN::AvgPoolCountType countType) {
    Vec4 sum = Vec4(0.0f);

    const int khs = 0 < -ih ? -ih : 0;                                                 // max
    const int khe = kernelHeight < inputHeight - ih ? kernelHeight : inputHeight - ih; // min
    const int kws = 0 < -iw ? -iw : 0;                                                 // max
    const int kwe = kernelWidth < inputWidth - iw ? kernelWidth : inputWidth - iw;     // min

    // sum
    int count = 0;
    if (countType == MNN::AvgPoolCountType_DEFAULT) {
        if (padType == MNN::PoolPadType_CAFFE) {
            countType = MNN::AvgPoolCountType_INCLUDE_PADDING;
        } else {
            countType = MNN::AvgPoolCountType_EXCLUDE_PADDING;
        }
    }
    if (countType == MNN::AvgPoolCountType_INCLUDE_PADDING) {
        count = (ALIMIN(ih + kernelHeight, inputHeight + padHeight) - ih) *
                (ALIMIN(iw + kernelWidth, inputWidth + padWidth) - iw);
    } else {
        count = (khe - khs) * (kwe - kws);
    }

    const float *kernelInput = offsetInput + khs * inputStep4;
    for (int kh = khs; kh < khe; kh++, kernelInput += inputStep4) {
        const float *cursorInput = kernelInput + kws * 4;
        for (int kw = kws; kw < kwe; kw++, cursorInput += 4) {
            sum = sum + Vec4::load(cursorInput);
        }
    }

    // avg
    if (count > 0) {
        Vec4 divs = Vec4(1.0f / count);
        Vec4::save(offsetOutput, sum * divs);
    } else {
        Vec4::save(offsetOutput, Vec4(0.0f));
    }
}

static void poolingAvg(const float *channelInput, int inputWidth, int inputHeight, float *channelOutput,
                       int outputWidth, int outputHeight, int kernelWidth, int kernelHeight, int strideWidth,
                       int strideHeight, int padWidth, int padHeight, MNN::PoolPadType padType, MNN::AvgPoolCountType countType) {
    // Compute Mid Rect
    int l = 0, t = 0, r = outputWidth, b = outputHeight;
    for (; l * strideWidth - padWidth < 0 && l < outputWidth; l++) {
        // do nothing
    }
    for (; t * strideHeight - padHeight < 0 && t < outputHeight; t++) {
        // do nothing
    }
    for (; (r - 1) * strideWidth - padWidth + (kernelWidth - 1) >= inputWidth && r > l; r--) {
        // do nothing
    }
    for (; (b - 1) * strideHeight - padHeight + (kernelHeight - 1) >= inputHeight && b > t; b--) {
        // do nothing
    }
    int padTop = t, padBottom = b, padLeft = l, padRight = r;


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
                              iw, ih, padWidth, padHeight, padType, countType);
            }
        }
        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const float *offsetInput = lineInput;
            float *offsetOutput      = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < padLeft;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
                poolingAvgPad(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType, countType);
            }
            offsetInput  = lineInput + padRight * strideWidth * 4;
            offsetOutput = lineOutput + padRight * 4;
            for (int ow = padRight, iw = -padWidth + ow * strideWidth; ow < outputWidth;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
                poolingAvgPad(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType, countType);
            }
        }
        for (int oh = padBottom, ih = -padHeight + oh * strideHeight; oh < outputHeight;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const float *offsetInput = lineInput;
            float *offsetOutput      = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < outputWidth;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
                poolingAvgPad(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType, countType);
            }
        }
    }

    { // handle no paddings
        const float *lineInput =
            channelInput + (padTop * strideHeight - padHeight) * inputStep4 + (padLeft * strideWidth - padWidth) * 4;
        float *lineOutput = channelOutput + padTop * outputStep4 + padLeft * 4;

        int count = kernelHeight * kernelWidth;
        Vec4 divs = Vec4(1.0f / count);
        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const float *offsetInput = lineInput;
            float *offsetOutput      = lineOutput;
            for (int ow = padLeft, iw = -padWidth + ow * strideWidth; ow < padRight;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
                Vec4 sum = Vec4(0.0f);
                // sum
                const float *kernelInput = offsetInput;
                for (int kh = 0; kh < kernelHeight; kh++, kernelInput += inputStep4) {
                    const float *cursorInput = kernelInput;
                    for (int kw = 0; kw < kernelWidth; kw++, cursorInput += 4) {
                        sum = sum + Vec4::load(cursorInput);
                    }
                }
                Vec4::save(offsetOutput, sum * divs);
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
    int kernelWidth  = layer->kernelX();
    int kernelHeight = layer->kernelY();
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
    auto countType         = layer->countType();
    if (layer->pads() != nullptr && padType == PoolPadType_CAFFE) {
        padType = PoolPadType_VALID;
    }
    mFunction              = std::make_pair(threadNumber, [=](int tId) {
        for (int channel = (int)tId; channel < totalDepth; channel += threadNumber) {
            // run
            planeFunction(inputData + channel * inputPlaneStride, input->width(), input->height(),
                          outputData + outputPlaneStride * channel, output->width(), output->height(), kernelWidth,
                          kernelHeight, strideWidth, strideHeight, padWidth, padHeight, padType, countType);
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
