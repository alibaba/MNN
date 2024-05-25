//
//  CPUPool.hpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUPool_hpp
#define CPUPool_hpp

#include <float.h>
#include <math.h>
#include "core/Macro.h"
#include "CaffeOp_generated.h"

namespace MNN {

template<typename T, typename VEC, int PACK, int MAXVALUE>
static void pooling_max_pad(const T* channelInput, T* offsetOutput, int inputWidth, int inputHeight,
                            int inputStep4, int inputSize4, int kernelWidth, int kernelHeight, int iw, int ih) {
    VEC max = VEC(MAXVALUE);

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

        const T *rightEdge = paddedLineInput + inputStep4 - PACK;
        for (int kw = 0; kw < kernelWidth; kw++) {
            const int w              = iw + kw;
            const T *cursorInput = nullptr;
            if (w < 0) { // left replicate
                cursorInput = paddedLineInput;
            } else if (w >= inputWidth) { // right replicate
                cursorInput = rightEdge;
            } else {
                cursorInput = paddedLineInput + PACK * w;
            }
            max = VEC::max(max, VEC::load(cursorInput));
        }
    }
    VEC::save(offsetOutput, max);
}

template<typename T, typename VEC, int PACK, int MAXVALUE>
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

    const int inputStep4       = PACK * inputWidth;
    const int inputSize4       = inputStep4 * inputHeight;
    const int strideInputStep4 = strideHeight * inputStep4;
    const int outputStep4      = PACK * outputWidth;
    const int strideWidth4     = PACK * strideWidth;

    { // handle paddings top
        T *lineOutput = channelOutput;
        for (int oh = 0, ih = -padHeight; oh < padTop; oh++, ih += strideHeight, lineOutput += outputStep4) {
            T *offsetOutput = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < outputWidth; ow++, iw += strideWidth, offsetOutput += PACK) {
                pooling_max_pad<T, VEC, PACK, MAXVALUE>(channelInput, offsetOutput, inputWidth, inputHeight, inputStep4, inputSize4,
                                kernelWidth, kernelHeight, iw, ih);
            }
        }
        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4) {
            T *offsetOutput = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < padLeft; ow++, iw += strideWidth, offsetOutput += PACK) {
                pooling_max_pad<T, VEC, PACK, MAXVALUE>(channelInput, offsetOutput, inputWidth, inputHeight, inputStep4, inputSize4,
                                kernelWidth, kernelHeight, iw, ih);
            }
            offsetOutput = lineOutput + padRight * PACK;
            for (int ow = padRight, iw = -padWidth + ow * strideWidth; ow < outputWidth;
                 ow++, iw += strideWidth, offsetOutput += PACK) {
                pooling_max_pad<T, VEC, PACK, MAXVALUE>(channelInput, offsetOutput, inputWidth, inputHeight, inputStep4, inputSize4,
                                kernelWidth, kernelHeight, iw, ih);
            }
        }
        for (int oh = padBottom, ih = -padHeight + oh * strideHeight; oh < outputHeight;
             oh++, ih += strideHeight, lineOutput += outputStep4) {
            T *offsetOutput = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < outputWidth; ow++, iw += strideWidth, offsetOutput += PACK) {
                pooling_max_pad<T, VEC, PACK, MAXVALUE>(channelInput, offsetOutput, inputWidth, inputHeight, inputStep4, inputSize4,
                                kernelWidth, kernelHeight, iw, ih);
            }
        }
    }

    { // handle no paddings
        const T *lineInput =
            channelInput + (padTop * strideHeight - padHeight) * inputStep4 + (padLeft * strideWidth - padWidth) * PACK;
        T *lineOutput = channelOutput + padTop * outputStep4 + padLeft * PACK;
        int wCount = padRight - padLeft;
        int wCountC4 = wCount / 4;
        int wCountRemain = wCount - wCountC4 * 4;
        int strideWidthFuse = strideWidth4 * 4;

        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const T *offsetInput = lineInput;
            T *offsetOutput      = lineOutput;
            for (int owf = 0; owf < wCountC4; ++owf, offsetOutput += 4 * PACK, offsetInput += strideWidthFuse) {
                VEC max0 = VEC(MAXVALUE);
                VEC max1 = VEC(MAXVALUE);
                VEC max2 = VEC(MAXVALUE);
                VEC max3 = VEC(MAXVALUE);
                const T *kernelInput = offsetInput;
                for (int kh = 0; kh < kernelHeight; kh++, kernelInput += inputStep4) {
                    const T *cursorInput = kernelInput;
                    for (int kw = 0; kw < kernelWidth; kw++, cursorInput += PACK) {
                        max0 = VEC::max(max0, VEC::load(cursorInput + 0 * strideWidth4));
                        max1 = VEC::max(max1, VEC::load(cursorInput + 1 * strideWidth4));
                        max2 = VEC::max(max2, VEC::load(cursorInput + 2 * strideWidth4));
                        max3 = VEC::max(max3, VEC::load(cursorInput + 3 * strideWidth4));
                    }
                }
                VEC::save(offsetOutput + PACK * 0, max0);
                VEC::save(offsetOutput + PACK * 1, max1);
                VEC::save(offsetOutput + PACK * 2, max2);
                VEC::save(offsetOutput + PACK * 3, max3);
            }
            for (int ow = 0; ow < wCountRemain;
                 ow++, offsetOutput += PACK, offsetInput += strideWidth4) {
                const T *kernelInput = offsetInput;
                VEC max = VEC(MAXVALUE);
                for (int kh = 0; kh < kernelHeight; kh++, kernelInput += inputStep4) {
                    const T *cursorInput = kernelInput;
                    for (int kw = 0; kw < kernelWidth; kw++, cursorInput += PACK) {
                        max = VEC::max(max, VEC::load(cursorInput));
                    }
                }

                VEC::save(offsetOutput, max);
            }
        }
    }
}
template<typename T, int MAXVALUE>
static void poolingMaxWithRedice(const T *channelInput, int inputWidth, int inputHeight, T *channelOutput,
                       int outputWidth, int outputHeight, int kernelWidth, int kernelHeight, int strideWidth,
                       int strideHeight, int padWidth, int padHeight, MNN::PoolPadType padType, MNN::AvgPoolCountType countType, int *rediceOutput) {

    const int inputStep4       = 4 * inputWidth;
    const int inputSize4       = inputStep4 * inputHeight;
    const int strideInputStep4 = strideHeight * inputStep4;
    const int outputStep4      = 4 * outputWidth;
    const int strideWidth4     = 4 * strideWidth;

    const T *lineInput = channelInput + (-padHeight) * inputStep4 + (-padWidth) * 4;
    T *lineOutput = channelOutput;
    int *lineRediceOutput = rediceOutput;

    for (int oh = 0, ih = -padHeight; oh < outputHeight;
        oh++, ih += strideHeight, lineOutput += outputStep4, lineRediceOutput += outputStep4, lineInput += strideInputStep4) {
        const T *offsetInput = lineInput;
        T *offsetOutput      = lineOutput;
        int *offsetRediceOutput  = lineRediceOutput;
        for (int ow = 0, iw = -padWidth; ow < outputWidth; ++ow, iw += strideWidth, offsetOutput += 4, offsetRediceOutput += 4, offsetInput += strideWidth4) {
            T max0 = float(MAXVALUE);
            T max1 = float(MAXVALUE);
            T max2 = float(MAXVALUE);
            T max3 = float(MAXVALUE);
            int indice0 = 0, indice1 = 0, indice2 = 0, indice3 = 0;
            const T *kernelInput = offsetInput;
            for (int kh = 0; kh < kernelHeight && (kh + ih) >= 0 && (kh + ih) < inputHeight; kh++, kernelInput += inputStep4) {
                const T *cursorInput = kernelInput;
                for (int kw = 0; kw < kernelWidth && (kw + iw) >= 0 && (kw + iw) < inputWidth; kw++, cursorInput += 4) {
                    T in0 = cursorInput[0];
                    T in1 = cursorInput[1];
                    T in2 = cursorInput[2];
                    T in3 = cursorInput[3];
                    int indice = (kh + ih) * inputWidth + kw + iw;
                    if(in0 > max0){
                        max0 = in0;
                        indice0 = indice;
                    }
                    if(in1 > max1){
                        max1 = in1;
                        indice1 = indice;
                    }
                    if(in2 > max2){
                        max2 = in2;
                        indice2 = indice;
                    }
                    if(in3 > max3){
                        max3 = in3;
                        indice3 = indice;
                    }
                }
            }
            offsetOutput[0] = max0;
            offsetOutput[1] = max1;
            offsetOutput[2] = max2;
            offsetOutput[3] = max3;
            offsetRediceOutput[0] = indice0;
            offsetRediceOutput[1] = indice1;
            offsetRediceOutput[2] = indice2;
            offsetRediceOutput[3] = indice3;
        }
    }
}

template<typename T, typename VEC, int PACK>
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
        const T *cursorInput = kernelInput + kws * PACK;
        for (int kw = kws; kw < kwe; kw++, cursorInput += PACK) {
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

template<typename T, typename VEC, int PACK>
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

    const int inputStep4       = PACK * inputWidth;
    const int strideInputStep4 = strideHeight * inputStep4;
    const int outputStep4      = PACK * outputWidth;
    const int strideWidth4     = PACK * strideWidth;

    { // handle paddings
        const T *lineInput = channelInput - padHeight * inputStep4 - padWidth * PACK;
        T *lineOutput      = channelOutput;
        for (int oh = 0, ih = -padHeight; oh < padTop;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const T *offsetInput = lineInput;
            T *offsetOutput      = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < outputWidth;
                 ow++, iw += strideWidth, offsetOutput += PACK, offsetInput += strideWidth4) {
                poolingAvgPad<T, VEC, PACK>(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType, countType);
            }
        }
        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const T *offsetInput = lineInput;
            T *offsetOutput      = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < padLeft;
                 ow++, iw += strideWidth, offsetOutput += PACK, offsetInput += strideWidth4) {
                poolingAvgPad<T, VEC, PACK>(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType, countType);
            }
            offsetInput  = lineInput + padRight * strideWidth * PACK;
            offsetOutput = lineOutput + padRight * PACK;
            for (int ow = padRight, iw = -padWidth + ow * strideWidth; ow < outputWidth;
                 ow++, iw += strideWidth, offsetOutput += PACK, offsetInput += strideWidth4) {
                poolingAvgPad<T, VEC, PACK>(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType, countType);
            }
        }
        for (int oh = padBottom, ih = -padHeight + oh * strideHeight; oh < outputHeight;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const T *offsetInput = lineInput;
            T *offsetOutput      = lineOutput;
            for (int ow = 0, iw = -padWidth; ow < outputWidth;
                 ow++, iw += strideWidth, offsetOutput += PACK, offsetInput += strideWidth4) {
                poolingAvgPad<T, VEC, PACK>(offsetInput, offsetOutput, inputWidth, inputHeight, kernelWidth, kernelHeight, inputStep4,
                              iw, ih, padWidth, padHeight, padType, countType);
            }
        }
    }

    { // handle no paddings
        const T *lineInput =
            channelInput + (padTop * strideHeight - padHeight) * inputStep4 + (padLeft * strideWidth - padWidth) * PACK;
        T *lineOutput = channelOutput + padTop * outputStep4 + padLeft * PACK;

        int count = kernelHeight * kernelWidth;
        VEC divs = VEC(1.0f / count);
        for (int oh = padTop, ih = -padHeight + oh * strideHeight; oh < padBottom;
             oh++, ih += strideHeight, lineOutput += outputStep4, lineInput += strideInputStep4) {
            const T *offsetInput = lineInput;
            T *offsetOutput      = lineOutput;
            for (int ow = padLeft, iw = -padWidth + ow * strideWidth; ow < padRight;
                 ow++, iw += strideWidth, offsetOutput += PACK, offsetInput += strideWidth4) {
                VEC sum = VEC(0);
                // sum
                const T *kernelInput = offsetInput;
                for (int kh = 0; kh < kernelHeight; kh++, kernelInput += inputStep4) {
                    const T *cursorInput = kernelInput;
                    for (int kw = 0; kw < kernelWidth; kw++, cursorInput += PACK) {
                        sum = sum + VEC::load(cursorInput) * divs;
                    }
                }
                VEC::save(offsetOutput, sum);
            }
        }
    }
}

} // namespace MNN

#endif /* CPUPool_hpp */
