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

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#include "core/Concurrency.h"
#include "math/Vec4.hpp"

using Vec4 = MNN::Math::Vec4;

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
    if (layer->pads() != nullptr && padType == PoolPadType_CAFFE) {
        padType = PoolPadType_VALID;
    }
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

CPUPool3D::CPUPool3D(Backend *b, const Pool3D *param) : MNN::Execution(b) {
    mType = param->type();
    mPadType = param->padType();
    for (auto kernel: *param->kernels()) {
        mKernels.push_back(kernel);
    }
    for (auto stride: *param->strides()) {
        mStrides.push_back(stride);
    }
    if (mPadType != PoolPadType_SAME) {
        for (auto pad: *param->pads()) {
            mPads.push_back(pad);
        }
    }
}

ErrorCode CPUPool3D::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    if (mPadType == PoolPadType_SAME) {
        mPads.clear();
        for (unsigned int i = 0; i < output->dimensions() - 2; ++i) {
            const int inputLength = input->length(i + 2), outputLength = output->length(i + 2);
            const int inputLengthNeed = (outputLength - 1) * mStrides[i] + mKernels[i];
            mPads.push_back((inputLengthNeed - inputLength) / 2);
        }
    }

    if (mKernels[0] != 1 || mStrides[0] != 1) {
        const int batch = input->length(0), channel = input->length(1), inputDepth = input->length(2);
        const int outputHeight = output->length(3), outputWidth = output->length(4);
        mTempStorage.reset(Tensor::createDevice<float>({batch, channel, inputDepth, outputHeight, outputWidth}, Tensor::CAFFE_C4));
        backend()->onAcquireBuffer(mTempStorage.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mTempStorage.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

ErrorCode CPUPool3D::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    MNN_ASSERT(input->dimensions() == 5);

    const int kernelDepth = mKernels[0], kernelHeight = mKernels[1], kernelWidth = mKernels[2];
    const int strideDepth = mStrides[0], strideHeight = mStrides[1], strideWidth = mStrides[2];
    const int outputDepth = output->length(2), outputHeight = output->length(3), outputWidth = output->length(4);
    const int inputDepth = input->length(2), inputHeight = input->length(3), inputWidth = input->length(4);
    const int channel = input->length(1), batch = input->length(0);
    const int padDepth = mPads[0], padHeight = mPads[1], padWidth = mPads[2];
    const int threadNumber = ((CPUBackend*)backend())->threadNumber();

    {
        auto planeFunction = poolingMax;
        if (mType == PoolType_AVEPOOL) {
            planeFunction = poolingAvg;
        }
        auto srcData           = input->host<float>();
        auto dstData           = mTempStorage.get() != nullptr ? mTempStorage->host<float>() : output->host<float>();
        auto inputPlaneStride  = 4 * inputHeight * inputWidth;
        auto outputPlaneStride = 4 * outputHeight * outputWidth;
        auto padType           = mPadType;

        auto planeFunc = [=](int tId) {
            for (int o = tId; o < batch * UP_DIV(channel, 4) * inputDepth; o += threadNumber) {
                planeFunction(srcData + o * inputPlaneStride, inputWidth, inputHeight,
                              dstData + o * outputPlaneStride, outputWidth, outputHeight, kernelWidth,
                              kernelHeight, strideWidth, strideHeight, padWidth, padHeight, padType);
            }
        };
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            planeFunc((int)tId);
        }
        MNN_CONCURRENCY_END();
    }

    if (mTempStorage.get() != nullptr) {
        using InnerFuncType = std::function<void(float*, const float*, int, int)>;
        InnerFuncType innerFunc = [=](float* dst, const float* src, int step, int kernel) {
            Vec4 max = Vec4::load(src);
            for (int i = 1; i < kernel; ++i) {
                max = Vec4::max(max, Vec4::load(src + i * step));
            }
            Vec4::save(dst, max);
        };

        if (mType == PoolType_AVEPOOL) {
            innerFunc = [=](float* dst, const float* src, int step, int kernel) {
                Vec4 sum = Vec4::load(src);
                for (int i = 1; i < kernel; ++i) {
                    sum = sum + Vec4::load(src + i * step);
                }
                Vec4::save(dst, sum * ((float)1 / kernel));
            };
        }

        const float* srcData = mTempStorage->host<float>();
        float* dstData = output->host<float>();

        auto reduceDepthFunc = [=, &innerFunc](int tId) {
            const int outputPlaneStride = outputHeight * outputWidth * 4;
            for (int o = tId; o < batch * UP_DIV(channel, 4); o += threadNumber) {
                auto srcZData = srcData + o * inputDepth * outputPlaneStride;
                auto dstZData = dstData + o * outputDepth * outputPlaneStride;
                for (int i = 0; i < outputHeight * outputWidth; ++i) {
                    for (int d = 0; d < outputDepth; ++d) {
                        int dRawSrc = d * strideDepth - padDepth;
                        int dSrc = ALIMAX(dRawSrc, 0);
                        int kernel = ALIMIN(dRawSrc + kernelDepth, inputDepth) - dSrc;
                        if (kernel == 0) {
                            Vec4::save(dstZData + d * outputPlaneStride + i * 4, Vec4((float)0));
                            continue;
                        }
                        innerFunc(dstZData + d * outputPlaneStride + i * 4, srcZData + dSrc * outputPlaneStride + i * 4,
                                  outputPlaneStride, kernel);
                    }
                }
            }
        };
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            reduceDepthFunc((int)tId);
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

class CPUPool3DCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUPool3D(backend, op->main_as_Pool3D());
    }
};

REGISTER_CPU_OP_CREATOR(CPUPool3DCreator, OpType_Pooling3D);

} // namespace MNN
