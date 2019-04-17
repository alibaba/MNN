//
//  ConvolutionTiledExecutor.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvolutionTiledExecutor.hpp"
#include "AutoTime.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Concurrency.h"
#include "ConvOpt.h"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {
ConvolutionTiledExecutor::ConvolutionTiledExecutor(const Convolution2DCommon* common, Backend* b,
                                                   const float* originWeight, size_t originWeightSize,
                                                   const float* bias, size_t biasSize)
    : MNN::CPUConvolution(common, b) {
    auto outputCount = (int)biasSize;
    mSrcCount        = (int)originWeightSize / outputCount / mCommon->kernelX() / mCommon->kernelY();
    auto alignSize =
        CPUConvolution::reorderWeightSize(mSrcCount, outputCount, mCommon->kernelX() * mCommon->kernelY(), 4);
    mWeight.reset(Tensor::createDevice<float>({alignSize}));
    mValid = backend()->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    if (mSrcCount % 4 != 0 || outputCount % 4 != 0) {
        ::memset(mWeight->host<float>(), 0, mWeight->size());
    }
    CPUConvolution::reorderWeight(mWeight->host<float>(), originWeight, mSrcCount, outputCount,
                                  mCommon->kernelX() * mCommon->kernelY(), 4);

    mBias.reset(Tensor::createDevice<float>({ALIGN_UP4((int)biasSize)}));
    mValid = backend()->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    ::memset(mBias->host<float>(), 0, mBias->size());
    ::memcpy(mBias->host<float>(), bias, biasSize * sizeof(float));
}
ConvolutionTiledExecutor::~ConvolutionTiledExecutor() {
    if (nullptr != mBias) {
        backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
    }
    if (nullptr != mWeight) {
        backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    }
}
ErrorCode ConvolutionTiledExecutor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto layer  = mCommon;
    auto input  = inputs[0];
    auto output = outputs[0];
    mFunctions.clear();
    CONV_SETUP_KERNELSIZE(4);
    int threadNumber  = ((CPUBackend*)backend())->threadNumber();
    auto postFunction = getPostFunction();
    auto biasPtr      = mBias->host<float>();
    auto weightPtr    = mWeight->host<float>();

    if (width <= CONVOLUTION_TILED_NUMBWR * 4 || dst_depth_quad < 4 || src_depth_quad < 4) {
        threadNumber                      = std::min(dst_depth_quad, threadNumber);
        std::function<void(int)> function = [=](int tId) {
            for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
                auto dstOrigin = output->host<float>() + batchIndex * output->stride(0);
                auto srcOrigin = input->host<float>() + batchIndex * input->stride(0);
                for (int dz = tId; dz < dst_depth_quad; dz += threadNumber) {
                    float* dst_z     = dstOrigin + dz * width * height * 4;
                    float* bias_z    = biasPtr + 4 * dz;
                    float* weight_dz = weightPtr + dz * weight_z_step;
                    int dx, dy;
                    // Compute Border
                    CONVOLUVTION_RUN_BASIC(0, 0, width, t, float, nullptr);
                    CONVOLUVTION_RUN_BASIC(0, b, width, height, float, nullptr);
                    CONVOLUVTION_RUN_BASIC(0, t, l, b, float, nullptr);
                    CONVOLUVTION_RUN_BASIC(r, t, width, b, float, nullptr);

                    if (r > l && b > t) {
                        // Compute Mid
                        for (dy = t; dy < b; ++dy) {
                            int srcStartY = dy * strideY - padY;
                            float* dst_y  = dst_z + width * 4 * dy;
                            float* src_dy = srcOrigin + srcStartY * src_width * 4;
                            MNNConvSlideWindowMiddle(dst_y + l * 4, src_dy + (l * strideX - padX) * 4, weight_dz, r - l,
                                                     strideX_step, src_depth_quad, src_z_step, kernel_width,
                                                     kernel_height, dilateX_step, dilateY_step, nullptr);
                        }
                    }
                    postFunction(dst_z, bias_z, width * height, 1);
                }
            }
        };
        mFunctions.emplace_back(std::make_pair(std::min(dst_depth_quad, threadNumber), std::move(function)));
        return NO_ERROR;
    }
    auto& tempBuffer = mTempBuffer.buffer();
    int srcXC = 1 + (CONVOLUTION_TILED_NUMBWR - 1) * mCommon->strideX() + mCommon->dilateX() * (mCommon->kernelX() - 1);

    tempBuffer.dim[0].extent = threadNumber;
    tempBuffer.dim[1].extent = srcXC * mCommon->kernelY();
    tempBuffer.dim[2].extent = UP_DIV(mSrcCount, 4);
    tempBuffer.dim[3].extent = 4;
    TensorUtils::setLinearLayout(&mTempBuffer);

    bool success = backend()->onAcquireBuffer(&mTempBuffer, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mTempBuffer, Backend::DYNAMIC);

    int xCount                             = UP_DIV(width, CONVOLUTION_TILED_NUMBWR);
    auto threadNumberFirst                 = std::min(threadNumber, xCount);
    std::function<void(int)> firstFunction = [=](int tId) {
        auto _xBuffer = mTempBuffer.host<float>() + tId * mTempBuffer.buffer().dim[0].stride;
        for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
            auto dstOrigin = output->host<float>() + batchIndex * output->stride(0);
            auto srcOrigin = input->host<float>() + batchIndex * input->stride(0);

            for (int x = (int)tId; x < xCount; x += threadNumberFirst) {
                int xIndex    = (int)x * CONVOLUTION_TILED_NUMBWR;
                int xReamin   = width - xIndex;
                int xC        = xReamin > CONVOLUTION_TILED_NUMBWR ? CONVOLUTION_TILED_NUMBWR : xReamin;
                int srcXC     = 1 + (xC - 1) * strideX + dilateX * (kernel_width - 1);
                int dx        = xIndex;
                int srcStartX = dx * strideX - padX;
                int srcEndX   = srcStartX + srcXC >= src_width ? src_width : srcStartX + srcXC;

                int dstOffset = 0;
                if (srcStartX < 0) {
                    dstOffset = -srcStartX;
                    srcStartX = 0;
                }
                int copyCount = srcEndX - srcStartX;

                auto src_x = srcOrigin + 4 * srcStartX;

                for (int dy = 0; dy < height; ++dy) {
                    // Expand
                    ::memset(_xBuffer, 0, mTempBuffer.buffer().dim[0].stride * sizeof(float));
                    int srcStartY = dy * strideY - padY;
                    int sfy       = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));
                    int efy       = ALIMIN(kernel_height, UP_DIV(src_height - srcStartY, dilateY));
                    for (int sz = 0; sz < src_depth_quad; ++sz) {
                        auto dst_z = _xBuffer + sz * srcXC * kernel_height * 4;
                        auto src_z = src_x + sz * src_z_step;
                        for (int ky = sfy; ky < efy; ++ky) {
                            int sy     = srcStartY + ky * dilateY;
                            auto src_y = src_z + 4 * sy * src_width;
                            auto dst_y = dst_z + (ky * srcXC + dstOffset) * 4;
                            ::memcpy(dst_y, src_y, copyCount * 4 * sizeof(float));
                        }
                    }

                    for (int dz = 0; dz < dst_depth_quad; ++dz) {
                        float* dst_z           = dstOrigin + dz * width * height * 4 + xIndex * 4 + width * 4 * dy;
                        const float* weight_dz = weightPtr + dz * weight_z_step;
                        MNNConvSlideWindowMiddle(dst_z, _xBuffer, weight_dz, xC, strideX_step, src_depth_quad,
                                                 srcXC * 4 * kernel_height, kernel_width, kernel_height, dilateX_step,
                                                 srcXC * 4, nullptr);
                    }
                }
            }
        }
    };
    mFunctions.emplace_back(std::make_pair(threadNumberFirst, firstFunction));
    int threadNumberSecond                  = std::min(threadNumber, dst_depth_quad);
    std::function<void(int)> secondFunction = [this, biasPtr, width, height, dst_depth_quad, output, postFunction,
                                               threadNumberSecond](int tId) {
        for (int batchIndex = 0; batchIndex < output->batch(); ++batchIndex) {
            auto dstOrigin = output->host<float>() + batchIndex * output->stride(0);
            for (int dz = tId; dz < dst_depth_quad; dz += threadNumberSecond) {
                float* dst_z  = dstOrigin + dz * width * height * 4;
                float* bias_z = biasPtr + 4 * dz;
                postFunction(dst_z, bias_z, width * height, 1);
            }
        }
    };
    mFunctions.emplace_back(std::make_pair(threadNumberSecond, secondFunction));
    return NO_ERROR;
}

ErrorCode ConvolutionTiledExecutor::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    for (auto& iter : mFunctions) {
        MNN_CONCURRENCY_BEGIN(tId, iter.first) {
            iter.second((int)tId);
        }
        MNN_CONCURRENCY_END();
    }
    return NO_ERROR;
}
} // namespace MNN
