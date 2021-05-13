//
//  CPUPool.hpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUPool_hpp
#define CPUPool_hpp

#include "backend/cpu/CPUBackend.hpp"
#include <float.h>
#include <math.h>
#include "core/Macro.h"

#include "core/Concurrency.h"

namespace MNN {

template<typename T, typename VEC>
static void pooling_max_pad(const T* channelInput, T* offsetOutput, int inputWidth, int inputHeight,
                            int inputStep4, int inputSize4, int kernelWidth, int kernelHeight, int iw, int ih) {
    VEC max = VEC(-std::numeric_limits<T>::max());

    const T *bottomLine = channelInput + inputSize4 - inputStep4;
    for (int kh = 0; kh < kernelHeight; kh++) {
        const int h                  = ih + kh;
        const T *paddedLineInput = nullptr;
        if (h < 0) { // top replicate
            paddedLineInput = channelInput;
        } else if (h >= inputHeight) { // bottom replicate
            paddedLineInput = bottomLine;
        } else {
            paddedLineInput = channelInput + h * inputStep4;
        }

        const T *rightEdge = paddedLineInput + inputStep4 - 4;
        for (int kw = 0; kw < kernelWidth; kw++) {
            const int w              = iw + kw;
            const T *cursorInput = nullptr;
            if (w < 0) { // left replicate
                cursorInput = paddedLineInput;
            } else if (w >= inputWidth) { // right replicate
                cursorInput = rightEdge;
            } else {
                cursorInput = paddedLineInput + 4 * w;
            }
            max = VEC::max(max, VEC::load(cursorInput));
        }
    }
    VEC::save(offsetOutput, max);
}

template<typename T, typename VEC>
static void poolingMax(const T *channelInput, int inputWidth, int inputHeight, T *channelOutput,
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
        T *lineOutput = channelOutput;
        for (int oh = 0, ih = -padHeight; oh < padTop; oh++, ih += strideHeight, lineOutput += outputStep4) {
            T *offsetOutput = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < outputWidth; ow++, iw += strideWidth, offsetOutput += 4) {
                pooling_max_pad<T, VEC>(channelInput, offsetOutput, inputWidth, inputHeight, inputStep4, inputSize4,
                                kernelWidth, kernelHeight, iw, ih);
            }
        }
        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4) {
            T *offsetOutput = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < padLeft; ow++, iw += strideWidth, offsetOutput += 4) {
                pooling_max_pad<T, VEC>(channelInput, offsetOutput, inputWidth, inputHeight, inputStep4, inputSize4,
                                kernelWidth, kernelHeight, iw, ih);
            }
            offsetOutput = lineOutput + padRight * 4;
            for (int ow = padRight, iw = -padWidth + ow * strideWidth; ow < outputWidth;
                 ow++, iw += strideWidth, offsetOutput += 4) {
                pooling_max_pad<T, VEC>(channelInput, offsetOutput, inputWidth, inputHeight, inputStep4, inputSize4,
                                kernelWidth, kernelHeight, iw, ih);
            }
        }
        for (int oh = padBottom, ih = -padHeight + oh * strideHeight; oh < outputHeight;
             oh++, ih += strideHeight, lineOutput += outputStep4) {
            T *offsetOutput = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < outputWidth; ow++, iw += strideWidth, offsetOutput += 4) {
                pooling_max_pad<T, VEC>(channelInput, offsetOutput, inputWidth, inputHeight, inputStep4, inputSize4,
                                kernelWidth, kernelHeight, iw, ih);
            }
        }
    }

    { // handle no paddings
        const T *lineInput =
            channelInput + (padTop * strideHeight - padHeight) * inputStep4 + (padLeft * strideWidth - padWidth) * 4;
        T *lineOutput = channelOutput + padTop * outputStep4 + padLeft * 4;
        int wCount = padRight - padLeft;
        int wCountC4 = wCount / 4;
        int wCountRemain = wCount - wCountC4 * 4;
        int strideWidthFuse = strideWidth4 * 4;

        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const T *offsetInput = lineInput;
            T *offsetOutput      = lineOutput;
            for (int owf = 0; owf < wCountC4; ++owf, offsetOutput += 16, offsetInput += strideWidthFuse) {
                VEC max0 = VEC(-std::numeric_limits<T>::max());
                VEC max1 = VEC(-std::numeric_limits<T>::max());
                VEC max2 = VEC(-std::numeric_limits<T>::max());
                VEC max3 = VEC(-std::numeric_limits<T>::max());
                const T *kernelInput = offsetInput;
                for (int kh = 0; kh < kernelHeight; kh++, kernelInput += inputStep4) {
                    const T *cursorInput = kernelInput;
                    for (int kw = 0; kw < kernelWidth; kw++, cursorInput += 4) {
                        max0 = VEC::max(max0, VEC::load(cursorInput + 0 * strideWidth4));
                        max1 = VEC::max(max1, VEC::load(cursorInput + 1 * strideWidth4));
                        max2 = VEC::max(max2, VEC::load(cursorInput + 2 * strideWidth4));
                        max3 = VEC::max(max3, VEC::load(cursorInput + 3 * strideWidth4));
                    }
                }
                VEC::save(offsetOutput + 4 * 0, max0);
                VEC::save(offsetOutput + 4 * 1, max1);
                VEC::save(offsetOutput + 4 * 2, max2);
                VEC::save(offsetOutput + 4 * 3, max3);
            }
            for (int ow = 0; ow < wCountRemain;
                 ow++, offsetOutput += 4, offsetInput += strideWidth4) {
                const T *kernelInput = offsetInput;
                VEC max = VEC(-std::numeric_limits<T>::max());
                for (int kh = 0; kh < kernelHeight; kh++, kernelInput += inputStep4) {
                    const T *cursorInput = kernelInput;
                    for (int kw = 0; kw < kernelWidth; kw++, cursorInput += 4) {
                        max = VEC::max(max, VEC::load(cursorInput));
                    }
                }

                VEC::save(offsetOutput, max);
            }
        }
    }
}

template<typename T, typename VEC>
static void poolingAvgPad(const T *offsetInput, T *offsetOutput, int inputWidth, int inputHeight,
                          int kernelWidth, int kernelHeight, int inputStep4, int iw, int ih, int padWidth,
                          int padHeight, MNN::PoolPadType padType, MNN::AvgPoolCountType countType) {
    VEC sum = VEC(0.0f);

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

    const T *kernelInput = offsetInput + khs * inputStep4;
    for (int kh = khs; kh < khe; kh++, kernelInput += inputStep4) {
        const T *cursorInput = kernelInput + kws * 4;
        for (int kw = kws; kw < kwe; kw++, cursorInput += 4) {
            sum = sum + VEC::load(cursorInput);
        }
    }

    // avg
    if (count > 0) {
        VEC divs = VEC(1.0f / count);
        VEC::save(offsetOutput, sum * divs);
    } else {
        VEC::save(offsetOutput, VEC(0.0f));
    }
}

template<typename T, typename VEC>
static void poolingAvg(const T* channelInput, int inputWidth, int inputHeight, T *channelOutput,
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
        const T *lineInput = channelInput - padHeight * inputStep4 - padWidth * 4;
        T *lineOutput      = channelOutput;
        for (int oh = 0, ih = -padHeight; oh < padTop;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const T *offsetInput = lineInput;
            T *offsetOutput      = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < outputWidth;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
                poolingAvgPad<T, VEC>(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType, countType);
            }
        }
        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const T *offsetInput = lineInput;
            T *offsetOutput      = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < padLeft;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
                poolingAvgPad<T, VEC>(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType, countType);
            }
            offsetInput  = lineInput + padRight * strideWidth * 4;
            offsetOutput = lineOutput + padRight * 4;
            for (int ow = padRight, iw = -padWidth + ow * strideWidth; ow < outputWidth;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
                poolingAvgPad<T, VEC>(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType, countType);
            }
        }
        for (int oh = padBottom, ih = -padHeight + oh * strideHeight; oh < outputHeight;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const T *offsetInput = lineInput;
            T *offsetOutput      = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < outputWidth;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
                poolingAvgPad<T, VEC>(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType, countType);
            }
        }
    }

    { // handle no paddings
        const T *lineInput =
            channelInput + (padTop * strideHeight - padHeight) * inputStep4 + (padLeft * strideWidth - padWidth) * 4;
        T *lineOutput = channelOutput + padTop * outputStep4 + padLeft * 4;

        int count = kernelHeight * kernelWidth;
        VEC divs = VEC(1.0f / count);
        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const T *offsetInput = lineInput;
            T *offsetOutput      = lineOutput;
            for (int ow = padLeft, iw = -padWidth + ow * strideWidth; ow < padRight;
                 ow++, iw += strideWidth, offsetOutput += 4, offsetInput += strideWidth4) {
                VEC sum = VEC(0);
                // sum
                const T *kernelInput = offsetInput;
                for (int kh = 0; kh < kernelHeight; kh++, kernelInput += inputStep4) {
                    const T *cursorInput = kernelInput;
                    for (int kw = 0; kw < kernelWidth; kw++, cursorInput += 4) {
                        sum = sum + VEC::load(cursorInput);
                    }
                }
                VEC::save(offsetOutput, sum * divs);
            }
        }
    }
}


template<typename T, typename VEC>
class CPUPool : public Execution {
public:
    CPUPool(Backend *b, const Pool *parameter) : MNN::Execution(b), mParameter(parameter) {
        // Do nothing
    }
    virtual ~CPUPool() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
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
        auto totalDepth        = input->batch() * UP_DIV(input->channel(), 4);
        auto inputData         = input->host<T>();
        auto outputData        = output->host<T>();
        auto inputPlaneStride  = 4 * input->width() * input->height();
        auto outputPlaneStride = 4 * output->width() * output->height();
        int threadNumber       = ((CPUBackend *)backend())->threadNumber();
        auto padType           = layer->padType();
        auto countType         = layer->countType();
        if (layer->pads() != nullptr && padType == PoolPadType_CAFFE) {
            padType = PoolPadType_VALID;
        }
        if (poolType == PoolType_AVEPOOL) {
            mFunction              = std::make_pair(threadNumber, [=](int tId) {
                for (int channel = (int)tId; channel < totalDepth; channel += threadNumber) {
                    // run
                    poolingAvg<T, VEC>(inputData + channel * inputPlaneStride, input->width(), input->height(),
                                  outputData + outputPlaneStride * channel, output->width(), output->height(), kernelWidth,
                                  kernelHeight, strideWidth, strideHeight, padWidth, padHeight, padType, countType);
                }
            });
        } else {
            mFunction              = std::make_pair(threadNumber, [=](int tId) {
                for (int channel = (int)tId; channel < totalDepth; channel += threadNumber) {
                    // run
                    poolingMax<T, VEC>(inputData + channel * inputPlaneStride, input->width(), input->height(),
                                  outputData + outputPlaneStride * channel, output->width(), output->height(), kernelWidth,
                                  kernelHeight, strideWidth, strideHeight, padWidth, padHeight, padType, countType);
                }
            });
        }

        return NO_ERROR;
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        MNN_CONCURRENCY_BEGIN(tId, mFunction.first) {
            mFunction.second((int)tId);
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }

private:
    const Pool *mParameter;
    std::pair<int, std::function<void(int)> > mFunction;
};

} // namespace MNN

#endif /* CPUPool_hpp */
