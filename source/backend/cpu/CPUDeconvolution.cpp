//
//  CPUDeconvolution.cpp
//  MNN
//
//  Created by MNN on 2018/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUDeconvolution.hpp"
#include "CPUBackend.hpp"
#include "Concurrency.h"
#include "Macro.h"
#include "Matrix.hpp"
#include "TensorUtils.hpp"
#include "compute/ConvOpt.h"
#include "compute/DeconvolutionWithStride.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {
CPUDeconvolutionCommon::CPUDeconvolutionCommon(const Op* convOp, Backend* b)
    : CPUConvolution(convOp->main_as_Convolution2D()->common(), b) {
    auto conv2D     = convOp->main_as_Convolution2D();
    int outputCount = mCommon->outputCount();
    mBias.reset(Tensor::createDevice<float>(std::vector<int>{ALIGN_UP4(outputCount)}));
    bool success = b->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!success) {
        mValid = false;
        return;
    }
    ::memset(mBias->host<float>(), 0, mBias->size());
    ::memcpy(mBias->host<float>(), conv2D->bias()->data(), conv2D->bias()->size() * sizeof(float));

    mSrcCount =
        conv2D->weight()->size() * mCommon->group() / mCommon->kernelX() / mCommon->kernelY() / mCommon->outputCount();
}
CPUDeconvolutionCommon::~CPUDeconvolutionCommon() {
    backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
}

ErrorCode CPUDeconvolutionCommon::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    if (mCommon->padMode() == PadMode_SAME) {
        const int outputWidth  = output->width();
        const int outputHeight = output->height();

        const int outputWidthPadded  = (input->width() - 1) * mCommon->strideX() + mCommon->kernelX();
        const int outputHeightPadded = (input->height() - 1) * mCommon->strideY() + mCommon->kernelY();

        const int padNeededWidth  = outputWidthPadded - outputWidth;
        const int padNeededHeight = outputHeightPadded - outputHeight;

        mPadX = padNeededWidth / 2;
        mPadY = padNeededHeight / 2;
        return NO_ERROR;
    }
    mPadX = mCommon->padX();
    mPadY = mCommon->padY();

    return NO_ERROR;
}

CPUDeconvolution::CPUDeconvolution(const Op* convOp, Backend* backend) : MNN::CPUDeconvolutionCommon(convOp, backend) {
    auto layer              = convOp->main_as_Convolution2D()->common();
    const float* tempWeight = convOp->main_as_Convolution2D()->weight()->data();
    int fw                  = layer->kernelX();
    int fh                  = layer->kernelY();
    int srcCount            = mSrcCount;
    int alignedWeightSize   = ALIGN_UP4(layer->outputCount()) * ALIGN_UP4(srcCount) * fw * fh;
    mWeight.reset(Tensor::createDevice<float>(std::vector<int>{alignedWeightSize}));
    bool success = backend->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    if (!success) {
        mValid = false;
        return;
    }
    float* dest = mWeight->host<float>();
    MNN_ASSERT(nullptr != dest);
    int outputCount = layer->outputCount();
    int srcCountD4  = UP_DIV(srcCount, 4);
    //        MNN_PRINT("ic:%d, oc:%d\n", mSrcCount, outputCount);
    for (int b = 0; b < outputCount; ++b) {
        int b_4      = b / 4;
        float* dst_b = dest + b_4 * 16 * fw * fh * srcCountD4;
        int mx       = b % 4;
        for (int d = 0; d < srcCount; ++d) {
            int my       = d % 4;
            int d_4      = d / 4;
            float* dst_d = dst_b + d_4 * 16;
            for (int y = 0; y < fh; ++y) {
                float* dst_y = dst_d + y * fw * 16 * srcCountD4;
                for (int x = 0; x < fw; ++x) {
                    float* dst_x       = dst_y + x * 16 * srcCountD4;
                    dst_x[4 * my + mx] = tempWeight[x + y * fw + b * fw * fh + d * fw * fh * outputCount];
                }
            }
        }
    }

    mTempColBuffer.reset(new Tensor(4));
    mTempSrcBuffer.reset(new Tensor(4));
}

CPUDeconvolution::~CPUDeconvolution() {
    backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
}

ErrorCode CPUDeconvolution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    CPUDeconvolutionCommon::onResize(inputs, outputs);
    auto input  = inputs[0];
    auto output = outputs[0];
    auto ic     = input->channel();
    auto oc     = output->channel();

    auto kernelCount = UP_DIV(oc, 4) * mCommon->kernelX() * mCommon->kernelY();
    int number       = std::max(1, ((CPUBackend*)backend())->threadNumber());
    mTempColBuffer->setLength(0, number);
    mTempColBuffer->setLength(1, kernelCount);
    mTempColBuffer->setLength(2, CONVOLUTION_TILED_NUMBWR1x1);
    mTempColBuffer->setLength(3, 4);

    mTempSrcBuffer->setLength(0, number);
    mTempSrcBuffer->setLength(1, UP_DIV(ic, 4));
    mTempSrcBuffer->setLength(2, CONVOLUTION_TILED_NUMBWR1x1);
    mTempSrcBuffer->setLength(3, 4);

    TensorUtils::setLinearLayout(mTempSrcBuffer.get());
    TensorUtils::setLinearLayout(mTempColBuffer.get());

    auto res = backend()->onAcquireBuffer(mTempSrcBuffer.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    res = backend()->onAcquireBuffer(mTempColBuffer.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mTempSrcBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempColBuffer.get(), Backend::DYNAMIC);
    auto layer = mCommon;

    // Revert Set input, output for use easier
    output = inputs[0];
    input  = outputs[0];

    CONV_SETUP_KERNELSIZE(4);
    auto dst_depth_quad = UP_DIV(output->channel(), 4);
    auto src_z_step = input->width() * input->height() * 4;
    int count        = width * height;
    int tileCount    = UP_DIV(count, CONVOLUTION_TILED_NUMBWR1x1);
    int threadNumber = std::max(1, ((CPUBackend*)backend())->threadNumber());
    threadNumber     = std::min(threadNumber, tileCount);
    auto weightAddr  = mWeight->host<float>();
    mThreadNumber    = threadNumber;
    mFunction = [this, width, height, count, tileCount, threadNumber, dst_depth_quad, kernelCount, strideX, strideY,
                 padX, padY, src_height, src_width, src_z_step, dilateX, dilateY, kernel_width, kernel_height,
                 src_depth_quad, dilateX_step, dilateY_step,
                 weightAddr](const float* dstOrigin, float* srcOrigin, int tId) {
        auto tempSource = mTempSrcBuffer->host<float>() + tId * mTempSrcBuffer->stride(0);
        auto tempDest   = mTempColBuffer->host<float>() + tId * mTempColBuffer->stride(0);
        for (int tIndex = (int)tId; tIndex < tileCount; tIndex += threadNumber) {
            int xStart = tIndex * CONVOLUTION_TILED_NUMBWR1x1;
            int xCount = std::min(count - xStart, CONVOLUTION_TILED_NUMBWR1x1);

            // Copy Dest
            {
                auto dstStart = dstOrigin + xStart * 4;
                for (int dz = 0; dz < dst_depth_quad; ++dz) {
                    auto source = dstStart + dz * width * height * 4;
                    auto dest   = tempSource + dz * CONVOLUTION_TILED_NUMBWR1x1 * 4;
                    ::memcpy(dest, source, xCount * 4 * sizeof(float));
                }
            }

            // Gemm
            MNNGemmFloatUnit_4(tempDest, tempSource, weightAddr, dst_depth_quad, CONVOLUTION_TILED_NUMBWR1x1 * 4,
                               kernelCount, 0);

            // Col2Image
            std::unique_lock<std::mutex> __l(mLock);
            for (int i = 0; i < xCount; ++i) {
                auto oIndex = i + xStart;
                int ox      = oIndex % width;
                int oy      = oIndex / width;

                int srcStartX = ox * strideX - padX;
                int srcStartY = oy * strideY - padY;

                int sfy = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));
                int efy = ALIMIN(kernel_height, UP_DIV(src_height - srcStartY, dilateY));

                int sfx = ALIMAX(0, (UP_DIV(-srcStartX, dilateX)));
                int efx = ALIMIN(kernel_width, UP_DIV(src_width - srcStartX, dilateX));

                auto dstStart = srcOrigin + srcStartX * 4 + srcStartY * src_width * 4;
                auto srcStart = tempDest + 4 * i;
                for (int z = 0; z < src_depth_quad; ++z) {
                    auto dstZ = dstStart + z * src_z_step;
                    auto srcZ = srcStart + kernel_width * kernel_height * 4 * CONVOLUTION_TILED_NUMBWR1x1 * z;

                    for (int fy = sfy; fy < efy; ++fy) {
                        auto dstY = dstZ + fy * dilateY_step;
                        auto srcY = srcZ + fy * kernel_width * CONVOLUTION_TILED_NUMBWR1x1 * 4;
                        for (int fx = sfx; fx < efx; ++fx) {
                            auto dstX = dstY + fx * dilateX_step;
                            auto srcX = srcY + fx * CONVOLUTION_TILED_NUMBWR1x1 * 4;
#ifdef MNN_USE_NEON
                            vst1q_f32(dstX, vld1q_f32(dstX) + vld1q_f32(srcX));
#else
                            for (int j = 0; j < 4; ++j) {
                                dstX[j] += srcX[j];
                            }
#endif
                        }
                    }
                }
            }
        }
    };
    return NO_ERROR;
}

ErrorCode CPUDeconvolution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto output   = inputs[0];
    auto input    = outputs[0];
    auto srcPlane = input->width() * input->height();
    auto icC4     = UP_DIV(input->channel(), 4);
    for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
        float* srcOrigin = input->host<float>() + batchIndex * input->stride(0);
        memset(srcOrigin, 0, input->stride(0) * sizeof(float));
        const float* dstOrigin = output->host<float>() + batchIndex * output->stride(0);
        MNN_CONCURRENCY_BEGIN(tId, mThreadNumber) {
            mFunction(dstOrigin, srcOrigin, (int)tId);
        }
        MNN_CONCURRENCY_END();
        mPostFunction(srcOrigin, mBias->host<float>(), srcPlane, icC4);
    }
    return NO_ERROR;
}
class CPUDeconvolutionCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto convOp = op->main_as_Convolution2D();
        auto common = convOp->common();
        if (common->strideY() > 1 || common->strideX() > 1) {
            if (common->dilateX() == 1 && common->dilateY() == 1) {
                return new DeconvolutionWithStride(op, backend);
            }
        }

        return new CPUDeconvolution(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDeconvolutionCreator, OpType_Deconvolution);
} // namespace MNN
