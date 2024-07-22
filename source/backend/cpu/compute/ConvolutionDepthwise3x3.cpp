//
//  ConvolutionDepthwise3x3.cpp
//  MNN
//
//  Created by MNN on 2019/4/3.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionDepthwise3x3.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/Macro.h"

namespace MNN {
ConvolutionDepthwise3x3::ConvolutionDepthwise3x3(std::shared_ptr<CPUConvolution::Resource> resource, const Convolution2DCommon* common, Backend* b) : CPUConvolution(common, b) {
    mResource = resource;
}

ConvolutionDepthwise3x3::ConvolutionDepthwise3x3(const Convolution2DCommon *common, Backend *b,
                                                 const float *originWeight, size_t originWeightSize, const float *bias,
                                                 size_t biasSize)
    : CPUConvolution(common, b) {
    MNN_ASSERT(3 == common->kernelX() && 3 == common->kernelY());
    MNN_ASSERT(1 == common->strideX() && 1 == common->strideY());
    MNN_ASSERT(1 == common->dilateX() && 1 == common->dilateY());
    mResource.reset(new Resource);
    mResource->backend = b;
    auto core = static_cast<CPUBackend*>(b)->functions();
    auto pack = core->pack;
    auto bytes = core->bytes;
    auto success = mResource->copyBiasAlign(bias, biasSize);
    if (!success) {
        mValid = false;
        return;
    }
    auto channel   = common->outputCount();
    auto channelC4 = UP_DIV(channel, pack);
    auto unitSize = channelC4 * pack * 3 * 4;
    mResource->mWeight.reset(Tensor::createDevice<uint8_t>({unitSize * bytes}));
    mValid = backend()->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    AutoStorage<float> tempWeightStorge;
    auto weightHost = mResource->mWeight->host<float>();
    if (bytes < 4) {
        // Lowp need extra float storage for transform
        tempWeightStorge.reset(unitSize);
        if (nullptr == tempWeightStorge.get()) {
            mValid = false;
            return;
        }
        weightHost = tempWeightStorge.get();
    }
    ::memset(weightHost, 0,  unitSize * sizeof(float));
    /* 1D-Winograd F(2,3) and tiling */
    for (int c = 0; c < channel; ++c) {
        auto cIndex     = c / pack;
        auto cRemain    = c % pack;
        auto weightDstZ = weightHost + cIndex * pack * 4 * 3 + cRemain;
        auto weightSrcZ = originWeight + c * 9;
        for (int y = 0; y < 3; ++y) {
            auto k0 = weightSrcZ[3 * y + 0];
            auto k1 = weightSrcZ[3 * y + 1];
            auto k2 = weightSrcZ[3 * y + 2];

            auto m0 = k0;
            auto m1 = 0.5f * (k0 + k1 + k2);
            auto m2 = 0.5f * (k0 - k1 + k2);
            auto m3 = k2;

            weightDstZ[(y * 4 + 0) * pack] = m0;
            weightDstZ[(y * 4 + 1) * pack] = m1;
            weightDstZ[(y * 4 + 2) * pack] = m2;
            weightDstZ[(y * 4 + 3) * pack] = m3;
        }
    }
    if (bytes < 4) {
        core->MNNFp32ToLowp(weightHost, mResource->mWeight->host<int16_t>(), unitSize);
    }
}

ConvolutionDepthwise3x3::~ConvolutionDepthwise3x3() {
    // Do nothing
}

bool ConvolutionDepthwise3x3::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvolutionDepthwise3x3(mResource, op->main_as_Convolution2D()->common(), bn);
    *dst = dstExe;
    return true;
}

ErrorCode ConvolutionDepthwise3x3::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    const int numberThread = ((CPUBackend *)backend())->threadNumber();
    auto output      = outputs[0];
    auto owUnit      = UP_DIV(output->width(), 2);
    auto core        = static_cast<CPUBackend*>(backend())->functions();
    // 3 cacheline
    mCacheLine.reset(Tensor::createDevice<uint8_t>({numberThread, 3 * 4 * owUnit * core->pack * core->bytes}));
    auto valid = backend()->onAcquireBuffer(mCacheLine.get(), Backend::DYNAMIC);
    if (!valid) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mCacheLine.get(), Backend::DYNAMIC);
    auto iw       = inputs[0]->width();
    mSourceStartX = UP_DIV(mPadX, 2);
    mSourceEndX   = std::max((iw + mPadX - 4) / 2, mSourceStartX);
    mPostParameters = getPostParameters();
    // auto rate = (float)(mSourceEndX-mSourceStartX) / (float)owUnit;
    // FUNC_PRINT_ALL(rate, f);

    int channelC4 = UP_DIV(inputs[0]->channel(), core->pack);
    int batch     = inputs[0]->batch();
    auto total = channelC4 * batch;

    mDivides.resize(numberThread+1);
    mDivides[0] = 0;
    static_cast<const CPURuntime*>(backend()->getRuntime())->computeDivideSizes(total, mDivides.data() + 1);
    
    return NO_ERROR;
}

ErrorCode ConvolutionDepthwise3x3::onExecute(const std::vector<Tensor *> &inputs,
                                             const std::vector<Tensor *> &outputs) {
    auto input    = inputs[0];
    auto output   = outputs[0];
    auto core        = static_cast<CPUBackend*>(backend())->functions();

    int channelC4 = UP_DIV(input->channel(), core->pack);
    int initSize  = std::min(input->height(), 2);
    int batch     = input->batch();
    int ow        = output->width();
    int oh        = output->height();
    int owUnit    = UP_DIV(ow, 2);

    auto iw           = input->width();
    auto ih           = input->height();
    auto kernelOrigin = mResource->mWeight->host<uint8_t>();

    /*oy-mPadY>=0*/
    int middelYStart = mPadY;

    /*oy-mPadY+3-1 < ih*/
    int middelYEnd = std::max(ih - 2 + mPadY, middelYStart);

    int threadNumber = ((CPUBackend *)backend())->threadNumber();
    auto maxKernelH  = std::min(mPadY + ih, 3);
    auto inputOrigin  = input->host<uint8_t>();
    auto outputOrigin = output->host<uint8_t>();
    MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
        auto cacheLineStart = mCacheLine->host<uint8_t>() + tId * mCacheLine->stride(0);
        for (int index = mDivides[tId]; index < mDivides[tId+1]; ++index) {
            int z = index / batch;
            auto biasPtr = (const float*)(mResource->mBias->host<uint8_t>() + core->bytes * core->pack * z);
            auto inputZ     = inputOrigin + core->pack * index * iw * ih * core->bytes;
            auto outputZ    = outputOrigin + core->pack * index * ow * oh * core->bytes;
            auto kernelZ    = kernelOrigin + z * core->pack * core->bytes * 4 * 3;
            auto cacheLine0 = cacheLineStart + 4 * core->pack * core->bytes * owUnit * 0;
            auto cacheLine1 = cacheLineStart + 4 * core->pack * core->bytes * owUnit * 1;
            auto cacheLine2 = cacheLineStart + 4 * core->pack * core->bytes * owUnit * 2;

            float *cacheLine[3] = {(float*)cacheLine0, (float*)cacheLine1, (float*)cacheLine2};

            // Init
            for (int i = 0; i < initSize; ++i) {
                core->MNNSourceTransformCommonF23((const float*)(inputZ + i * iw * core->bytes * core->pack), cacheLine[i], owUnit, iw, mPadX, mSourceStartX,
                                       mSourceEndX);
            }

            // Compute Top
            for (int y = 0; y < middelYStart; ++y) {
                auto outputY      = outputZ + y * core->bytes * core->pack * ow;
                int cacheLineSize = y - mPadY + maxKernelH;
                if (cacheLineSize <= 0) {
                    ::memset(outputY, 0, core->bytes * ow * core->pack);
                    core->MNNAxByClampBroadcastUnit((float*)outputY, (float*)outputY, biasPtr, ow, 0, 0, 1,  mPostParameters.data());
                    continue;
                }
                auto kernelPtr = kernelZ + (maxKernelH - cacheLineSize) * 4 * core->pack * core->bytes;
                cacheLineSize = std::min(cacheLineSize, ih);
                core->MNNMultiAndDestTransformCommon23(cacheLine, (float*)kernelPtr, (float*)outputY, cacheLineSize, ow, biasPtr, mPostParameters.data());
            }

            // Compute Mid
            for (int y = middelYStart; y < middelYEnd; ++y) {
                auto outputY = outputZ + y * core->bytes * core->pack * ow;
                auto iy      = y - mPadY + 2;
                core->MNNSourceTransformCommonF23((float*)(inputZ + core->bytes * core->pack * iy * iw), cacheLine[2], owUnit, iw, mPadX, mSourceStartX,
                                       mSourceEndX);
                // FUNC_PRINT(ow);
                core->MNNConvDwF23MulTransUnit(cacheLine, (float*)kernelZ, (float*)outputY, ow, biasPtr, mPostParameters.data());

                auto temp    = cacheLine[0];
                cacheLine[0] = cacheLine[1];
                cacheLine[1] = cacheLine[2];
                cacheLine[2] = temp;
            }

            // Compute Bottom
            for (int y = middelYEnd; y < oh; ++y) {
                auto outputY      = outputZ + y * core->bytes * core->pack * ow;
                int cacheLineSize = (ih - y + mPadY);
                if (cacheLineSize <= 0) {
                    ::memset(outputY, 0, ow * core->bytes * core->pack);
                    core->MNNAxByClampBroadcastUnit((float*)outputY, (float*)outputY, biasPtr, ow, 0, 0, 1,  mPostParameters.data());
                    continue;
                }
                core->MNNMultiAndDestTransformCommon23(cacheLine, (float*)kernelZ, (float*)outputY, cacheLineSize, ow, biasPtr, mPostParameters.data());
                cacheLine[0] = cacheLine[1];
                cacheLine[1] = cacheLine[2];
            }
        }
    } MNN_CONCURRENCY_END();
    return NO_ERROR;
}
} // namespace MNN
