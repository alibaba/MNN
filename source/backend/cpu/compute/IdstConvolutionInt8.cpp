//
//  IdstConvolutionInt8.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "IdstConvolutionInt8.hpp"
#include "ConvInt8TiledExecutor.hpp"
#include "ConvolutionTiledExecutor.hpp"
#include "CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/BufferAllocator.hpp"
#include "ConvOpt.h"
#include "ConvolutionIntFactory.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "Int8FunctionsOpt.h"
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
#define QUANT_INFO_BYTES 4
namespace MNN {

IdstConvolutionInt8::IdstConvolutionInt8(const Convolution2DCommon* convOp, Backend* b,
                                                 const ConvolutionCommon::Int8Common* common, const float* bias,
                                                 size_t biasSize) : MNN::CPUConvolution(convOp, b) {
    auto core = static_cast<CPUBackend*>(b)->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int PackUnit = static_cast<CPUBackend*>(b)->functions()->pack;
    int ocUp4 = ROUND_UP(biasSize, PackUnit);
    mBias.reset(ocUp4);
    mBias.clear();
    auto biasDest = mBias.get();
    mAMin         = common->quan->aMin();
    mAMax         = common->quan->aMaxOrBits();
    mQuanScale    = common->quan->quantScale();

    // The postTreat will contain scale_bias and biasRelu, so the bias will be add twice
    for (int i = 0; i < biasSize; ++i) {
        biasDest[i] = bias[i] * 0.5f;
    }
    int outputCount = (int)biasSize;
    mQuan           = common->quan;
    MNN_ASSERT(nullptr != mQuan);
    mAlpha.reset(ROUND_UP(common->alpha.size(), PackUnit));
    mAlpha.clear();
    ::memcpy(mAlpha.get(), common->alpha.get(), common->alpha.size() * sizeof(float));

    auto weightLength       = common->weight.size();
    mSrcCount               = (int)weightLength / mCommon->kernelX() / mCommon->kernelY() / outputCount;
    auto kx                 = mCommon->kernelX();
    auto ky                 = mCommon->kernelY();
    auto kernelCount        = kx * ky;
    auto srcCount           = mSrcCount;
    std::vector<int> shape;
    shape = {1, UP_DIV(outputCount, UNIT), UP_DIV(srcCount, SRC_UNIT) * kernelCount, UNIT, SRC_UNIT};
    mFakeBias.reset(Tensor::createDevice<float>({ocUp4}));
    int weightlen = shape[0] * shape[1] * shape[2] * shape[3] * shape[4];
    int quantlen = 2 * ocUp4 * QUANT_INFO_BYTES;
    mWeight.reset(Tensor::createDevice<int8_t>({weightlen + quantlen}));
    mValid = b->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    mValid &= b->onAcquireBuffer(mFakeBias.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Memory not enough\n");
        return;
    }
    AutoStorage<uint8_t> weightReordered(weightlen);
    AutoStorage<float> fakeWeightScaleBias(2 * ocUp4);
    if (weightReordered.get() == nullptr || fakeWeightScaleBias.get() == nullptr) {
        MNN_ERROR("Memory not enough\n");
        return;
    }
    int32_t info[6] = {1, outputCount, srcCount, kernelCount, UNIT, SRC_UNIT};
    ConvInt8TiledExecutor::reorderWeight(weightReordered.get(), (uint8_t*)common->weight.get(), info);
    ::memset(mFakeBias->host<float>(), 0, mFakeBias->size());
    auto ptr = (float*)fakeWeightScaleBias.get();
    ::memset(ptr, 0, 2 * ocUp4 * 4);
    for (int i = 0; i < ocUp4; ++i) {
        ptr[i] = 1.f;
    }
#ifdef MNN_USE_SSE
    for (int oz = 0; oz < outputCount; ++oz) {
        auto srcZ = common->weight.get() + oz * kernelCount * srcCount;
        int32_t offset = 0;
        for (int i = 0; i < kernelCount * srcCount; ++i) {
            offset += srcZ[i] * (-128);
        }
        mFakeBias->host<float>()[oz] = static_cast<float>(offset) * 1.f;
    }
#endif
    int32_t params[6] = {shape[0], shape[1], shape[2], shape[3], shape[4], ocUp4};
    ConvInt8TiledExecutor::packWeightAndQuantInfo(mWeight->host<int8_t>(), (int8_t*)weightReordered.get(), (int8_t*)fakeWeightScaleBias.get(), params, QUANT_INFO_BYTES);
}

IdstConvolutionInt8::~IdstConvolutionInt8() {
    // Do nothing
}

ErrorCode IdstConvolutionInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int PackUnit = static_cast<CPUBackend*>(backend())->functions()->pack;

    CPUConvolution::onResize(inputs, outputs);
    ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY, static_cast<CPUBackend*>(backend())->functions(), core);
    auto ow = mIm2ColParamter.ow;
    auto oh = mIm2ColParamter.oh;
    int tileCount           = UP_DIV(ow * oh, DST_XUNIT);
    auto outputCountUnit    = UP_DIV(outputs[0]->channel(), PackUnit);
    int number              = std::max(((CPUBackend*)backend())->threadNumber(), 1);
    number                  = std::min(number, tileCount);
    TensorUtils::copyShape(inputs[0], &mSrcCopyBuffer, true);
    mSrcCopyBuffer.buffer().dim[0].extent = 1;
    mSrcCopyBuffer.buffer().type          = halide_type_of<int8_t>();
    TensorUtils::setLinearLayout(&mSrcCopyBuffer);
    mTempBuffer.buffer().type          = halide_type_of<int8_t>();
    mTempBuffer.buffer().dimensions    = 3;
    mTempBuffer.buffer().dim[0].extent = number;
    mTempBuffer.buffer().dim[1].extent = DST_XUNIT;
    mTempBuffer.buffer().dim[2].extent = mIm2ColParamter.kernelCountUnit * SRC_UNIT;
    TensorUtils::setLinearLayout(&mTempBuffer);

    bool success = backend()->onAcquireBuffer(&mSrcCopyBuffer, Backend::DYNAMIC);
    success &= backend()->onAcquireBuffer(&mTempBuffer, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    auto bufferAlloc = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    auto blitInfoSize = ConvolutionTiledExecutor::computeBlitInfoSize(DST_XUNIT, mIm2ColParamter.ow, mIm2ColParamter.kernelX * mIm2ColParamter.kernelY, number);
    mBlitInfo = bufferAlloc->alloc(blitInfoSize.first);
    if (mBlitInfo.invalid()) {
        return OUT_OF_MEMORY;
    }
    bufferAlloc->free(mBlitInfo);
    mBlitInfoStride = blitInfoSize.second;
    
    backend()->onReleaseBuffer(&mSrcCopyBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTempBuffer, Backend::DYNAMIC);

    mPostParameters = getPostParameters();
    return NO_ERROR;
}

ErrorCode IdstConvolutionInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto coreFloat = static_cast<CPUBackend*>(backend())->functions();
    auto coreInt = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT__, SRC_UNIT, DST_XUNIT;
    coreInt->MNNGetGemmUnit(&UNIT__, &SRC_UNIT, &DST_XUNIT);
    int PackUnit = static_cast<CPUBackend*>(backend())->functions()->pack;
    
    auto gemmKernel = coreInt->Int8GemmKernel;
    if (SRC_UNIT > PackUnit) {
        memset(mTempBuffer.host<int8_t>(), 0, mTempBuffer.size());
    }
    
    //        AUTOTIME;
    auto input        = inputs[0];
    auto output       = outputs[0];
    auto weightOrigin = mWeight->host<int8_t>();
    auto dstZStep     = mIm2ColParamter.ow * mIm2ColParamter.oh * PackUnit * input->batch();
    int threadNumber  = 1;
    
    auto blitProc = coreInt->MNNPackC4Int8ForMatMul_A;
    int batch            = input->batch();
    int width            = mIm2ColParamter.ow;
    int height           = mIm2ColParamter.oh;
    auto ocC4            = UP_DIV(output->channel(), PackUnit);
    auto kernelCountUnit = mIm2ColParamter.kernelCountUnit;
    int count            = width * height;
    float quantScale[] = {
        mQuanScale,
        mQuanScale,
        mQuanScale,
        mQuanScale
    };
    float zeroPoint = 0;
    
    std::vector<float> fakeScale(ocC4 * PackUnit, 1.0f);
    QuanPostTreatParameters quanParam;
    quanParam.biasFloat = mFakeBias->host<float>();
    quanParam.scale = fakeScale.data();
    quanParam.useInt8 = 0;
    float fp32minmax[2] = {-std::numeric_limits<float>().max(), std::numeric_limits<float>().max()};
    quanParam.fp32minmax = fp32minmax;
    std::vector<float> fakeSrcKernleSum(DST_XUNIT, 0.f);
    quanParam.srcKernelSum = fakeSrcKernleSum.data();

    // MNN_PRINT("%s, %d, %d, %d,%d->%d,%d\n", layer->layer.layerId, layer->kernelSize[0], layer->kernelSize[1],
    // input->d1, input->d2, output->d1, output->d2);

    auto bn = static_cast<CPUBackend*>(backend());
    int inputTotalSize = bn->getTensorSize(&mSrcCopyBuffer, true);
    int8_t* srcCopy    = mSrcCopyBuffer.host<int8_t>();
    const int col_buffer_size = mIm2ColParamter.kernelCountUnit * DST_XUNIT * SRC_UNIT * sizeof(int8_t);
    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        auto srcOrigin = input->host<float>() + input->stride(0) * batchIndex;
        auto dstOrigin = output->host<float>() + output->stride(0) * batchIndex;

        MNNFloat2Int8(srcOrigin, srcCopy, inputTotalSize / 4, &mQuanScale, mAMin, mAMax, &zeroPoint, 0);
        int tileCount = UP_DIV(count, DST_XUNIT);

        threadNumber        = std::max(((CPUBackend*)backend())->threadNumber(), 1);
        threadNumber        = std::min(threadNumber, tileCount);
        auto outputOrigin   = output->host<float>() + batchIndex * output->stride(0);
        auto threadFunction = [&](int tId) {
            auto colAddr        = mTempBuffer.host<int8_t>() + tId * mTempBuffer.buffer().dim[0].stride;
            auto srcPtr     = (int8_t const **)(mBlitInfo.ptr() + tId * mBlitInfoStride.first);
            auto el         = (int32_t *)(srcPtr + mBlitInfoStride.second);

            int32_t info[5];
            info[1] = mIm2ColParamter.iw * mIm2ColParamter.ih;
            info[2] = DST_XUNIT;
            info[3] = mIm2ColParamter.strideX;

            for (int tIndex = (int)tId; tIndex < tileCount; tIndex += threadNumber) {
                int xIndexStart  = tIndex * DST_XUNIT;
                int realDstCount = ALIMIN(count - xIndexStart, DST_XUNIT);
                auto res = ConvolutionTiledExecutor::turnIm2ColToBlitInfo((const float**)srcPtr, el, xIndexStart, realDstCount, mIm2ColParamter, (const uint8_t*)srcCopy, sizeof(int8_t));
                int number = res.first;
                bool needZero = res.second;
                if (needZero) {
                    ::memset(colAddr, zeroPoint, col_buffer_size);
                }
                info[0] = number;
                info[4] = realDstCount;
                if (number > 0) {
                    blitProc(colAddr, srcPtr, info, el);
                }
                auto outputInTile = outputOrigin + xIndexStart * PackUnit;
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
                coreFloat->MNNScaleAndAddBias(dstOrigin + z * dstZStep, dstOrigin + z * dstZStep, mBias.get() + PackUnit * z,
                                   mAlpha.get() + PackUnit * z, width * height, 1);
                coreFloat->MNNAxByClampBroadcastUnit(dstOrigin + z * dstZStep, dstOrigin + z * dstZStep, mBias.get() + PackUnit * z, width * height, 0, 0, 1, mPostParameters.data());
            }
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

} // namespace MNN
