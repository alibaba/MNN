//
//  ConvolutionTiledExecutor.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionTiledExecutor.hpp"
#include <MNN/AutoTime.hpp>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "math/Vec4.hpp"

namespace MNN {
ErrorCode ConvolutionTiledExecutorMultiInput::onExecute(const std::vector<Tensor*>& inputs,
                                                        const std::vector<Tensor*>& outputs) {
    int depth       = inputs[1]->channel();
    int outputCount = inputs[1]->batch();
    ::memset(mTempWeight->host<float>(), 0, mTempWeight->size());
    if (nullptr != mTempBias) {
        ::memset(mTempBias->host<float>(), 0, mTempBias->size());
        if (inputs.size() > 2) {
            ::memcpy(mTempBias->host<float>(), inputs[2]->host<float>(), inputs[2]->size());
        }
    }
    CPUConvolution::reorderWeight(mTempWeight->host<float>(), inputs[1]->host<float>(), depth, outputCount,
                                  inputs[1]->width() * inputs[1]->height(), mTempWeightCache->host<float>());
    return mProxy->onExecute(mInputs, outputs);
}
ErrorCode ConvolutionTiledExecutorMultiInput::onResize(const std::vector<Tensor*>& inputs,
                                                       const std::vector<Tensor*>& outputs) {
    int depth       = inputs[1]->channel();
    int outputCount = outputs[0]->channel();
    mTempWeight.reset(Tensor::createDevice<float>(
        {UP_DIV(outputCount, 4), UP_DIV(depth, 4), inputs[1]->width() * inputs[1]->height(), 16}));
    mTempWeightCache.reset(Tensor::createDevice<float>(
        {UP_DIV(outputCount, 4), UP_DIV(depth, 4), inputs[1]->width() * inputs[1]->height(), 16}));
    backend()->onAcquireBuffer(mTempWeight.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mTempWeightCache.get(), Backend::DYNAMIC);
    mTempBias.reset();
    if (inputs.size() > 2 && inputs[2]->elementSize() % 4 == 0) {
        mInputs = {inputs[0], mTempWeight.get(), inputs[2]};
    } else {
        mTempBias.reset(Tensor::createDevice<float>({ALIGN_UP4(outputCount)}));
        backend()->onAcquireBuffer(mTempBias.get(), Backend::DYNAMIC);
        mInputs = {inputs[0], mTempWeight.get(), mTempBias.get()};
    }
    backend()->onReleaseBuffer(mTempWeightCache.get(), Backend::DYNAMIC);
    auto errorCode = mProxy->onResize(mInputs, outputs);
    backend()->onReleaseBuffer(mTempWeight.get(), Backend::DYNAMIC);
    if (nullptr != mTempBias) {
        backend()->onReleaseBuffer(mTempBias.get(), Backend::DYNAMIC);
    }
    return errorCode;
}

ConvolutionTiledExecutor::ConvolutionTiledExecutor(const Convolution2DCommon* common, Backend* b,
                                                   const float* originWeight, size_t originWeightSize,
                                                   const float* bias, size_t biasSize)
    : MNN::Execution(b) {
    auto outputCount = (int)biasSize;
    // TODO, use common->inputCount to get srcCount
    auto srcCount    = (int)originWeightSize / outputCount / common->kernelX() / common->kernelY();
    mWeight.reset(Tensor::createDevice<float>(
        {UP_DIV(outputCount, 4), UP_DIV(srcCount, 4), (int)common->kernelX(), common->kernelY(), 16}));
    std::shared_ptr<Tensor> tempWeight(Tensor::createDevice<float>(
        {UP_DIV(outputCount, 4), UP_DIV(srcCount, 4), (int)common->kernelX(), common->kernelY(), 16}));
    mValid = backend()->onAcquireBuffer(mWeight.get(), Backend::STATIC) &&
             backend()->onAcquireBuffer(tempWeight.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }

    CPUConvolution::reorderWeight(mWeight->host<float>(), originWeight, srcCount, outputCount,
                                  common->kernelX() * common->kernelY(), tempWeight->host<float>());
    backend()->onReleaseBuffer(tempWeight.get(), Backend::STATIC);
    mBias.reset(Tensor::createDevice<float>({ALIGN_UP4((int)biasSize)}));
    mValid = backend()->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    ::memset(mBias->host<float>(), 0, mBias->size());
    ::memcpy(mBias->host<float>(), bias, biasSize * sizeof(float));
    mProxy.reset(new ConvolutionTiledExecutorBasic(common, b));
}
ConvolutionTiledExecutor::~ConvolutionTiledExecutor() {
    if (nullptr != mBias) {
        backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
    }
    if (nullptr != mWeight) {
        backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    }
}
ErrorCode ConvolutionTiledExecutorBasic::onResize(const std::vector<Tensor*>& inputs,
                                                  const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(3 == inputs.size());
    CPUConvolution::onResize(inputs, outputs);
    auto layer  = mCommon;
    auto input  = inputs[0];
    auto weight = inputs[1];
    auto bias   = inputs[2];
    auto output = outputs[0];
    mFunctions.clear();
    CONV_SETUP_KERNELSIZE(4);
    auto dst_depth_quad = UP_DIV(output->channel(), 4);
    int threadNumber    = ((CPUBackend*)backend())->threadNumber();
    auto postFunction   = getPostFunction();
    auto biasPtr        = bias->host<float>();
    auto weightPtr      = weight->host<float>();
    auto weight_z_step  = kernel_height * kernel_width * src_depth_quad * 16;
    auto weight_sy_step = kernel_width * 16;
    auto weight_sz_step = kernel_width * kernel_height * 16;
    int strideX_step    = strideX * 4;
    int src_z_step      = input->width() * input->height() * 4;

    if (width * height <= CONVOLUTION_TILED_NUMBER * 4 || dst_depth_quad < 4 || src_depth_quad < 4) {
        // Use Slice Window
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
    auto icC4 = UP_DIV(input->channel(), 4);
    auto ocC4 = UP_DIV(output->channel(), 4);

    tempBuffer.dim[0].extent = threadNumber;
    tempBuffer.dim[1].extent = CONVOLUTION_TILED_NUMBER;
    tempBuffer.dim[2].extent = icC4 * mCommon->kernelY() * mCommon->kernelX(); // srcCount/4 * kx*ky
    tempBuffer.dim[3].extent = 4;
    TensorUtils::setLinearLayout(&mTempBuffer);

    bool success = backend()->onAcquireBuffer(&mTempBuffer, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mTempBuffer, Backend::DYNAMIC);

    int count                             = UP_DIV(width*height, CONVOLUTION_TILED_NUMBER);
    int plane = width * height;
    auto threadNumberFirst                 = std::min(threadNumber, count);
    std::function<void(int)> firstFunction = [=](int tId) {
        auto colBuffer = mTempBuffer.host<float>() + mTempBuffer.stride(0) * tId;
        for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
            auto dstOrigin = output->host<float>() + batchIndex * output->stride(0);
            auto srcOrigin = input->host<float>() + batchIndex * input->stride(0);

            for (int x = (int)tId; x < count; x += threadNumberFirst) {
                int start    = (int)x * CONVOLUTION_TILED_NUMBER;
                int remain   = plane - start;
                int xC        = remain > CONVOLUTION_TILED_NUMBER ? CONVOLUTION_TILED_NUMBER : remain;
                // Im2Col
                ::memset(colBuffer, 0, mTempBuffer.stride(0) * sizeof(float));
                for (int i = 0; i<xC; ++i) {
                    int index = start + i;
                    int ox = index % width;
                    int oy = index / width;
                    int sxSta = ox * strideX - padX;
                    int sySta = oy * strideY - padY;
                    for (int ky=0; ky<kernel_height; ++ky) {
                        auto sy = sySta + ky * dilateY;
                        if (sy < 0 || sy >= src_height) {
                            continue;
                        }
                        for (int kx=0; kx<kernel_width; ++kx) {
                            auto sx = sxSta + kx * dilateX;
                            if (sx < 0 || sx >= src_width) {
                                continue;
                            }
                            auto src = srcOrigin + sx * 4 + sy * 4 * src_width;
                            auto dst = colBuffer + i * 4 + 4 * xC * (kx + ky*kernel_width);
                            for (int sz=0; sz<icC4; ++sz) {
                                Math::Vec4::save(dst + 4 * xC * kernel_height * kernel_width * sz, Math::Vec4::load(src + src_z_step * sz));
                            }
                        }
                    }
                }
                // GEMM
                if (xC == CONVOLUTION_TILED_NUMBER) {
                    MNNGemmFloatUnit_4(dstOrigin + start * 4, colBuffer,
                                        weightPtr, icC4 * kernel_width * kernel_height, width * height * 4, ocC4, 0);
                } else {
                    MNNGemmFloatCommon_4(dstOrigin + start * 4, colBuffer,
                                        weightPtr, icC4 * kernel_width * kernel_height, width * height * 4, ocC4, xC, 0);
                }
            }
        }
    };
    mFunctions.emplace_back(std::make_pair(threadNumberFirst, firstFunction));
    int threadNumberSecond                  = std::min(threadNumber, dst_depth_quad);
    std::function<void(int)> secondFunction = [biasPtr, width, height, dst_depth_quad, output, postFunction,
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

ErrorCode ConvolutionTiledExecutorBasic::onExecute(const std::vector<Tensor*>& inputs,
                                                   const std::vector<Tensor*>& outputs) {
    for (auto& iter : mFunctions) {
        MNN_CONCURRENCY_BEGIN(tId, iter.first) {
            iter.second((int)tId);
        }
        MNN_CONCURRENCY_END();
    }
    return NO_ERROR;
}
} // namespace MNN
