//
//  ConvolutionDepthwise3x3.cpp
//  MNN
//
//  Created by MNN on 2019/4/3.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvolutionDepthwise3x3.hpp"
#include "CPUBackend.hpp"
#include "Concurrency.h"
#include "Macro.h"
#include "Vec4.hpp"

using namespace MNN::Math;
extern "C" {
void MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow);
void MNNConvDwF23SourceTransUnit(const float *source, float *dest, size_t unit);
}
static void _multiAndDestTransformCommon(float **cacheLine, const float *weigth, float *dest, int cacheLineSize,
                                         int ow) {
    int unit = ow / 2;
    for (int x = 0; x < unit; ++x) {
        auto offset = 4 * 4 * x;
        Vec4 m0     = 0.0f;
        Vec4 m1     = 0.0f;
        Vec4 m2     = 0.0f;
        Vec4 m3     = 0.0f;

        for (int i = 0; i < cacheLineSize; ++i) {
            m0 = m0 + Vec4::load(weigth + i * 16 + 4 * 0) * Vec4::load(cacheLine[i] + offset + 4 * 0);
            m1 = m1 + Vec4::load(weigth + i * 16 + 4 * 1) * Vec4::load(cacheLine[i] + offset + 4 * 1);
            m2 = m2 + Vec4::load(weigth + i * 16 + 4 * 2) * Vec4::load(cacheLine[i] + offset + 4 * 2);
            m3 = m3 + Vec4::load(weigth + i * 16 + 4 * 3) * Vec4::load(cacheLine[i] + offset + 4 * 3);
        }

        auto o0 = m0 + m1 + m2;
        auto o1 = m1 - m2 + m3;
        Vec4::save(dest + 8 * x + 0 * 4, o0);
        Vec4::save(dest + 8 * x + 1 * 4, o1);
    }
    if (unit * 2 < ow) {
        auto offset = 4 * 4 * unit;
        Vec4 m0     = 0.0f;
        Vec4 m1     = 0.0f;
        Vec4 m2     = 0.0f;

        for (int i = 0; i < cacheLineSize; ++i) {
            m0 = m0 + Vec4::load(weigth + i * 16 + 4 * 0) * Vec4::load(cacheLine[i] + offset + 4 * 0);
            m1 = m1 + Vec4::load(weigth + i * 16 + 4 * 1) * Vec4::load(cacheLine[i] + offset + 4 * 1);
            m2 = m2 + Vec4::load(weigth + i * 16 + 4 * 2) * Vec4::load(cacheLine[i] + offset + 4 * 2);
        }

        auto o0 = m0 + m1 + m2;
        Vec4::save(dest + 8 * unit + 0 * 4, o0);
    }
}

static void _sourceTransformCommon(const float *source, float *dest, int unit, int iw, int pad, int su, int eu) {
    for (int x = 0; x < su; ++x) {
        auto dstX = dest + 4 * 4 * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec4 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec4::load(source + 4 * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec4::save(dstX + 4 * 0, m0);
        Vec4::save(dstX + 4 * 1, m1);
        Vec4::save(dstX + 4 * 2, m2);
        Vec4::save(dstX + 4 * 3, m3);
    }
    MNNConvDwF23SourceTransUnit(source + 4 * (su * 2 - pad), dest + 4 * 4 * su, eu - su);

    for (int x = eu; x < unit; ++x) {
        auto dstX = dest + 4 * 4 * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec4 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec4::load(source + 4 * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec4::save(dstX + 4 * 0, m0);
        Vec4::save(dstX + 4 * 1, m1);
        Vec4::save(dstX + 4 * 2, m2);
        Vec4::save(dstX + 4 * 3, m3);
    }
}

#ifndef MNN_USE_NEON
void MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow) {
    _multiAndDestTransformCommon(cacheLine, weigth, dest, 3, ow);
}
void MNNConvDwF23SourceTransUnit(const float *source, float *dest, size_t unit) {
    for (int x = 0; x < unit; ++x) {
        auto dstX = dest + 4 * 4 * x;
        auto sx   = x * 2;
        Vec4 v[4];
        for (int i = 0; i < 4; ++i) {
            v[i] = Vec4::load(source + 4 * sx + 4 * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec4::save(dstX + 4 * 0, m0);
        Vec4::save(dstX + 4 * 1, m1);
        Vec4::save(dstX + 4 * 2, m2);
        Vec4::save(dstX + 4 * 3, m3);
    }
}
#endif

namespace MNN {
ConvolutionDepthwise3x3::ConvolutionDepthwise3x3(const Convolution2DCommon *common, Backend *b,
                                                 const float *originWeight, size_t originWeightSize, const float *bias,
                                                 size_t biasSize)
    : CPUConvolution(common, b) {
    MNN_ASSERT(3 == common->kernelX() && 3 == common->kernelY());
    MNN_ASSERT(1 == common->strideX() && 1 == common->strideY());
    MNN_ASSERT(1 == common->dilateX() && 1 == common->dilateY());
    mBias.reset(Tensor::createDevice<float>({(int)ALIGN_UP4(biasSize)}));
    mValid = backend()->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Error for alloc memory in ConvolutionDepthwise3x3\n");
        return;
    }
    ::memset(mBias->host<float>(), 0, mBias->size());
    ::memcpy(mBias->host<float>(), bias, biasSize * sizeof(float));
    auto channel   = common->outputCount();
    auto channelC4 = UP_DIV(channel, 4);
    mWeight.reset(Tensor::createDevice<float>({channelC4, 3, 4, 4}));
    mValid = backend()->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Error for alloc memory in ConvolutionDepthwise3x3\n");
        return;
    }
    auto weightHost = mWeight->host<float>();

    /* 1D-Winograd F(2,3) and tiling */
    for (int c = 0; c < channel; ++c) {
        auto cIndex     = c / 4;
        auto cRemain    = c % 4;
        auto weightDstZ = weightHost + cIndex * 4 * 4 * 3 + cRemain;
        auto weightSrcZ = originWeight + c * 9;
        for (int y = 0; y < 3; ++y) {
            auto k0 = weightSrcZ[3 * y + 0];
            auto k1 = weightSrcZ[3 * y + 1];
            auto k2 = weightSrcZ[3 * y + 2];

            auto m0 = k0;
            auto m1 = 0.5f * (k0 + k1 + k2);
            auto m2 = 0.5f * (k0 - k1 + k2);
            auto m3 = k2;

            weightDstZ[y * 16 + 4 * 0] = m0;
            weightDstZ[y * 16 + 4 * 1] = m1;
            weightDstZ[y * 16 + 4 * 2] = m2;
            weightDstZ[y * 16 + 4 * 3] = m3;
        }
    }
}

ConvolutionDepthwise3x3::~ConvolutionDepthwise3x3() {
    if (nullptr != mBias) {
        backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
    }
    if (nullptr != mWeight) {
        backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    }
}

ErrorCode ConvolutionDepthwise3x3::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    int numberThread = ((CPUBackend *)backend())->threadNumber();
    auto output      = outputs[0];
    auto owUnit      = UP_DIV(output->width(), 2);
    // 3 cacheline, 4 is the unit of transform
    mCacheLine.reset(Tensor::createDevice<float>({numberThread, 3, owUnit * 4, 4}));
    auto valid = backend()->onAcquireBuffer(mCacheLine.get(), Backend::DYNAMIC);
    if (!valid) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mCacheLine.get(), Backend::DYNAMIC);
    auto iw       = inputs[0]->width();
    mSourceStartX = UP_DIV(mPadX, 2);
    mSourceEndX   = std::max((iw + mPadX - 4) / 2, mSourceStartX);

    // auto rate = (float)(mSourceEndX-mSourceStartX) / (float)owUnit;
    // FUNC_PRINT_ALL(rate, f);
    return NO_ERROR;
}

ErrorCode ConvolutionDepthwise3x3::onExecute(const std::vector<Tensor *> &inputs,
                                             const std::vector<Tensor *> &outputs) {
    auto input    = inputs[0];
    auto output   = outputs[0];
    int channelC4 = UP_DIV(input->channel(), 4);
    int initSize  = std::min(input->height(), 2);
    int batch     = input->batch();
    int ow        = output->width();
    int oh        = output->height();
    int owUnit    = UP_DIV(ow, 2);

    auto iw           = input->width();
    auto ih           = input->height();
    auto kernelOrigin = mWeight->host<float>();

    /*oy-mPadY>=0*/
    int middelYStart = mPadY;

    /*oy-mPadY+3-1 < ih*/
    int middelYEnd = std::max(ih - 2 + mPadY, middelYStart);

    int threadNumber = ((CPUBackend *)backend())->threadNumber();
    auto maxKernelH  = std::min(mPadY + ih, 3);

    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        auto inputOrigin  = input->host<float>() + batchIndex * input->stride(0);
        auto outputOrigin = output->host<float>() + batchIndex * output->stride(0);
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            auto cacheLineStart = mCacheLine->host<float>() + tId * mCacheLine->stride(0);
            for (int z = (int)tId; z < channelC4; z += threadNumber) {
                auto inputZ     = inputOrigin + 4 * z * iw * ih;
                auto outputZ    = outputOrigin + 4 * z * ow * oh;
                auto kernelZ    = kernelOrigin + z * mWeight->stride(0);
                auto cacheLine0 = cacheLineStart + 16 * owUnit * 0;
                auto cacheLine1 = cacheLineStart + 16 * owUnit * 1;
                auto cacheLine2 = cacheLineStart + 16 * owUnit * 2;

                float *cacheLine[3] = {cacheLine0, cacheLine1, cacheLine2};

                // Init
                for (int i = 0; i < initSize; ++i) {
                    _sourceTransformCommon(inputZ + i * iw * 4, cacheLine[i], owUnit, iw, mPadX, mSourceStartX,
                                           mSourceEndX);
                }

                // Compute Top
                for (int y = 0; y < middelYStart; ++y) {
                    auto outputY      = outputZ + y * 4 * ow;
                    int cacheLineSize = y - mPadY + maxKernelH;
                    if (cacheLineSize <= 0) {
                        ::memset(outputY, 0, 4 * ow * sizeof(float));
                        continue;
                    }
                    auto kernelPtr = kernelZ + (maxKernelH - cacheLineSize) * 16;
                    _multiAndDestTransformCommon(cacheLine, kernelPtr, outputY, cacheLineSize, ow);
                }

                // Compute Mid
                for (int y = middelYStart; y < middelYEnd; ++y) {
                    auto outputY = outputZ + y * 4 * ow;
                    auto iy      = y - mPadY + 2;
                    _sourceTransformCommon(inputZ + 4 * iy * iw, cacheLine[2], owUnit, iw, mPadX, mSourceStartX,
                                           mSourceEndX);
                    // FUNC_PRINT(ow);
                    MNNConvDwF23MulTransUnit(cacheLine, kernelZ, outputY, ow);

                    auto temp    = cacheLine[0];
                    cacheLine[0] = cacheLine[1];
                    cacheLine[1] = cacheLine[2];
                    cacheLine[2] = temp;
                }

                // Compute Bottom
                for (int y = middelYEnd; y < oh; ++y) {
                    auto outputY      = outputZ + y * 4 * ow;
                    int cacheLineSize = (ih - y + mPadY);
                    if (cacheLineSize <= 0) {
                        ::memset(outputY, 0, 4 * ow * sizeof(float));
                        continue;
                    }
                    _multiAndDestTransformCommon(cacheLine, kernelZ, outputY, cacheLineSize, ow);
                    cacheLine[0] = cacheLine[1];
                    cacheLine[1] = cacheLine[2];
                }
                mPostFunction(outputZ, mBias->host<float>() + 4 * z, ow * oh, 1);
            }
        }
        MNN_CONCURRENCY_END();
    }
    return NO_ERROR;
}
} // namespace MNN
