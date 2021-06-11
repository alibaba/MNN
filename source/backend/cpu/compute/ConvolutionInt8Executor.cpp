//
//  ConvolutionInt8Executor.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionInt8Executor.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "backend/cpu/compute/ConvolutionIntFactory.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

#ifdef MNN_USE_SSE
extern "C" {
void MNNInt8ToUInt8(void* ptr, int count);
}
#endif

namespace MNN {

ConvolutionInt8Executor::ConvolutionInt8Executor(const Convolution2DCommon* convOp, Backend* b,
                                                 const ConvolutionCommon::Int8Common* common, const float* bias,
                                                 size_t biasSize) : MNN::CPUConvolution(convOp, b) {
    auto core = static_cast<CPUBackend*>(b)->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    
    mBias.reset(ROUND_UP(biasSize, UNIT));
    mBias.clear();
    auto biasDest = mBias.get();
    mAMin         = common->quan->aMin();
    mAMax         = common->quan->aMax();
    mQuanScale    = common->quan->quantScale();

    // The postTreat will contain scale_bias and biasRelu, so the bias will be add twice
    for (int i = 0; i < biasSize; ++i) {
        biasDest[i] = bias[i] * 0.5f;
    }
    int outputCount = (int)biasSize;
    mQuan           = common->quan;
    MNN_ASSERT(nullptr != mQuan);
    mAlpha.reset(ROUND_UP(common->alpha.size(), UNIT));
    mAlpha.clear();
    ::memcpy(mAlpha.get(), common->alpha.get(), common->alpha.size() * sizeof(float));

    auto weightLength       = common->weight.size();
    mSrcCount               = (int)weightLength / mCommon->kernelX() / mCommon->kernelY() / outputCount;
    auto kx                 = mCommon->kernelX();
    auto ky                 = mCommon->kernelY();
    auto kernelCount        = kx * ky;
    auto srcCount           = mSrcCount;
    auto outputCountUnit    = UP_DIV(outputCount, UNIT);
    auto srcCountUnit       = UP_DIV(srcCount, UNIT);
    auto totalKernelCountD8 = UP_DIV(srcCountUnit * kx * ky, SRC_UNIT / UNIT);
    mWeight.reset(Tensor::createDevice<int8_t>(std::vector<int>{outputCountUnit, totalKernelCountD8, UNIT, SRC_UNIT}));
    mFakeBias.reset(Tensor::createDevice<int32_t>({(int)ROUND_UP(biasSize, UNIT)}));
    mValid = b->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    mValid &= b->onAcquireBuffer(mFakeBias.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Memory not enough\n");
        return;
    }
    ::memset(mWeight->host<int8_t>(), 0, mWeight->size());
    auto dst = mWeight->host<int8_t>();
    for (int k = 0; k < kernelCount; ++k) {
        auto srcK = common->weight.get() + k;
        for (int y = 0; y < srcCount; ++y) {
            int yOutSide    = y / UNIT;
            int yInside     = y % UNIT;
            int yIndex      = yOutSide + k * srcCountUnit;
            int ySubOutside = yIndex / (SRC_UNIT / UNIT);
            int ySubInside  = yIndex % (SRC_UNIT / UNIT);

            auto dstY = dst + ySubOutside * mWeight->stride(1) + ySubInside * UNIT + yInside;
            auto srcY = srcK + y * kernelCount;
            for (int x = 0; x < outputCount; ++x) {
                int xOutSide = x / UNIT;
                int xInside  = x % UNIT;

                auto dstX = dstY + xOutSide * mWeight->stride(0) + xInside * SRC_UNIT;
                auto srcX = srcY + x * kernelCount * srcCount;

                dstX[0] = srcX[0];
            }
        }
    }
    ::memset(mFakeBias->host<int32_t>(), 0, mFakeBias->size());
#ifdef MNN_USE_SSE
    for (int oz = 0; oz < outputCount; ++oz) {
        auto srcZ = common->weight.get() + oz * kernelCount * srcCount;
        int32_t offset = 0;
        for (int i = 0; i < kernelCount * srcCount; ++i) {
            offset += srcZ[i] * (-128);
        }
        mFakeBias->host<int32_t>()[oz] = offset;
    }
#endif
}

ConvolutionInt8Executor::~ConvolutionInt8Executor() {
    if (mWeight != nullptr) {
        backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    }
    if (mFakeBias != nullptr) {
        backend()->onReleaseBuffer(mFakeBias.get(), Backend::STATIC);
    }
}

ErrorCode ConvolutionInt8Executor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    
    CPUConvolution::onResize(inputs, outputs);
    int tileCount           = UP_DIV(outputs[0]->width() * outputs[0]->height(), DST_XUNIT);
    auto outputCountUnit    = UP_DIV(outputs[0]->channel(), UNIT);
    int number              = std::max(((CPUBackend*)backend())->threadNumber(), 1);
    number                  = std::min(number, tileCount);
    mIm2ColParamter.dilateX = mCommon->dilateX();
    mIm2ColParamter.dilateY = mCommon->dilateY();
    mIm2ColParamter.strideX = mCommon->strideX();
    mIm2ColParamter.strideY = mCommon->strideY();
    mIm2ColParamter.padX    = mPadX;
    mIm2ColParamter.padY    = mPadY;
    mIm2ColParamter.ih      = inputs[0]->height();
    mIm2ColParamter.iw      = inputs[0]->width();
    mIm2ColParamter.icDiv4  = UP_DIV(inputs[0]->channel(), UNIT);
    mIm2ColParamter.ow      = outputs[0]->width();
    mIm2ColParamter.oh      = outputs[0]->height();
    mIm2ColParamter.kernelX = mCommon->kernelX();
    mIm2ColParamter.kernelY = mCommon->kernelY();
    mIm2ColParamter.kernelCountUnit =
        UP_DIV(mIm2ColParamter.icDiv4 * mIm2ColParamter.kernelY * mIm2ColParamter.kernelX, (SRC_UNIT / UNIT));
    mIm2ColParamter.srcZStep = inputs[0]->stride(1) * UNIT;
    mIm2ColParamter.srcYStep = inputs[0]->stride(2) * UNIT;

    TensorUtils::copyShape(inputs[0], &mSrcCopyBuffer, true);
    mSrcCopyBuffer.buffer().dim[0].extent = 1;
    mSrcCopyBuffer.buffer().type          = halide_type_of<int8_t>();
    TensorUtils::setLinearLayout(&mSrcCopyBuffer);
    mTempBuffer.buffer().type          = halide_type_of<int8_t>();
    mTempBuffer.buffer().dimensions    = 3;
    mTempBuffer.buffer().dim[0].extent = number;
    mTempBuffer.buffer().dim[1].extent = DST_XUNIT;
    mTempBuffer.buffer().dim[2].extent = mWeight->length(1) * SRC_UNIT;
    TensorUtils::setLinearLayout(&mTempBuffer);

    mTempDstBuffer.buffer().type          = halide_type_of<float>();
    mTempDstBuffer.buffer().dimensions    = 3;
    mTempDstBuffer.buffer().dim[0].extent = number;
    mTempDstBuffer.buffer().dim[1].extent = DST_XUNIT;
    mTempDstBuffer.buffer().dim[2].extent = outputCountUnit * UNIT;
    TensorUtils::setLinearLayout(&mTempDstBuffer);

    bool success = backend()->onAcquireBuffer(&mSrcCopyBuffer, Backend::DYNAMIC);
    success &= backend()->onAcquireBuffer(&mTempBuffer, Backend::DYNAMIC);
    success &= backend()->onAcquireBuffer(&mTempDstBuffer, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mSrcCopyBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTempDstBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTempBuffer, Backend::DYNAMIC);

    mPostParameters = getPostParameters();
    return NO_ERROR;
}

ErrorCode ConvolutionInt8Executor::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto coreFloat = static_cast<CPUBackend*>(backend())->functions();
    auto coreInt = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    coreInt->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    
    auto gemmKernel = coreInt->Int8GemmKernel;
    
    //        AUTOTIME;
    auto input        = inputs[0];
    auto output       = outputs[0];
    auto weightOrigin = mWeight->host<int8_t>();
    auto dstZStep     = output->width() * output->height() * UNIT;
    int threadNumber  = 1;
    
    auto im2ColProc = coreInt->chooseIm2Col(&mIm2ColParamter, input->channel());
    int batch            = input->batch();
    int width            = output->width();
    int height           = output->height();
    auto ocC4            = UP_DIV(output->channel(), UNIT);
    auto kernelCountUnit = mIm2ColParamter.kernelCountUnit;
    int count            = width * height;
    float quantScale[] = {
        mQuanScale,
        mQuanScale,
        mQuanScale,
        mQuanScale
    };
    int8_t zeroPoint = 0;
    
    QuanPostTreatParameters quanParam;
    quanParam.bias = mFakeBias->host<int32_t>();
    quanParam.scale = nullptr;

    // MNN_PRINT("%s, %d, %d, %d,%d->%d,%d\n", layer->layer.layerId, layer->kernelSize[0], layer->kernelSize[1],
    // input->d1, input->d2, output->d1, output->d2);

    int inputTotalSize = mSrcCopyBuffer.elementSize();
    int8_t* srcCopy    = mSrcCopyBuffer.host<int8_t>();
    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        auto srcOrigin = input->host<float>() + input->stride(0) * batchIndex;
        auto dstOrigin = output->host<float>() + output->stride(0) * batchIndex;

        MNNFloat2Int8(srcOrigin, srcCopy, inputTotalSize / 4, quantScale, mAMin, mAMax, zeroPoint);
        int tileCount = UP_DIV(count, DST_XUNIT);

        threadNumber        = std::max(((CPUBackend*)backend())->threadNumber(), 1);
        threadNumber        = std::min(threadNumber, tileCount);
        auto outputOrigin   = output->host<float>() + batchIndex * output->stride(0);
        auto threadFunction = [&](int tId) {
            auto colAddr        = mTempBuffer.host<int8_t>() + tId * mTempBuffer.buffer().dim[0].stride;
            auto gemmOutputAddr = mTempDstBuffer.host<float>() + tId * mTempDstBuffer.buffer().dim[0].stride;

            for (int tIndex = (int)tId; tIndex < tileCount; tIndex += threadNumber) {
                int xIndexStart  = tIndex * DST_XUNIT;
                int realDstCount = ALIMIN(count - xIndexStart, DST_XUNIT);

                im2ColProc(colAddr, srcCopy, zeroPoint, &mIm2ColParamter, xIndexStart, realDstCount);
                
                auto outputInTile = outputOrigin + xIndexStart * UNIT;
                // GEMM
                
#ifdef MNN_USE_SSE
                const int col_buffer_size = mIm2ColParamter.kernelCountUnit * DST_XUNIT * SRC_UNIT;
                MNNInt8ToUInt8(colAddr, col_buffer_size);
#endif
                gemmKernel((int8_t*)outputInTile, colAddr, weightOrigin, kernelCountUnit, dstZStep * sizeof(float), ocC4, &quanParam, realDstCount);
            }
        };

        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            threadFunction((int)tId);
        }
        MNN_CONCURRENCY_END();

        threadNumber = std::max(((CPUBackend*)backend())->threadNumber(), 1);
        threadNumber = std::min(threadNumber, ocC4);
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            for (int z = (int)tId; z < ocC4; z += threadNumber) {
                coreFloat->MNNScaleAndAddBias(dstOrigin + z * dstZStep, dstOrigin + z * dstZStep, mBias.get() + UNIT * z,
                                   mAlpha.get() + UNIT * z, width * height, 1);
                coreFloat->MNNAxByClampBroadcastUnit(dstOrigin + z * dstZStep, dstOrigin + z * dstZStep, mBias.get() + UNIT * z, width * height, 0, 0, 1, mPostParameters.data());
            }
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

} // namespace MNN
