//
//  ConvInt8TiledExecutor.cpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvInt8TiledExecutor.hpp"
#include "ConvolutionTiledExecutor.hpp"
#include "core/Macro.h"
#include "core/BufferAllocator.hpp"

#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
namespace MNN {

ConvInt8TiledExecutor::ConvInt8TiledExecutor(Backend* backend, const Convolution2DCommon* convOp, std::shared_ptr<ResourceInt8> res): CPUConvolution(convOp, backend), mResource(res), mMutableResource(res, backend) {
    mValid = mMutableResource.mValid;
}

ConvInt8TiledExecutor::~ConvInt8TiledExecutor() {
    // Do nothing
}

bool ConvInt8TiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    return false;
}

ErrorCode ConvInt8TiledExecutor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mMutableResource.updateInputOutputScale(TensorUtils::getQuantInfo(inputs[0]), TensorUtils::getQuantInfo(outputs[0]));
    CPUConvolution::onResize(inputs, outputs);
    ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY, static_cast<CPUBackend*>(backend())->functions(), static_cast<CPUBackend*>(backend())->int8Functions());
    return NO_ERROR;
}

void ConvInt8TiledExecutor::reorderWeight(Tensor* weight, const uint8_t* weightSrc, int SRC_UNIT, int UNIT, int ic, int oc, int kernelCount) {
    auto weightDst = weight->host<uint8_t>();
    memset(weightDst, 0, weight->size());
    if (SRC_UNIT > UNIT) {
        auto icDivU = UP_DIV(ic, UNIT);
        for (int k = 0; k < kernelCount; ++k) {
            const auto srcK = weightSrc + k;
            for (int y = 0; y < ic; ++y) {
                const int yOutSide    = y / UNIT;
                const int yInSide     = y % UNIT;
                const int yIndex      = yOutSide + k * icDivU;
                const int ySubOutSide = yIndex / (SRC_UNIT / UNIT);
                const int ySubInSide  = yIndex % (SRC_UNIT / UNIT);

                auto dstY       = weightDst + ySubOutSide * weight->stride(1) + ySubInSide * UNIT + yInSide;
                const auto srcY = srcK + y * kernelCount;
                for (int x = 0; x < oc; ++x) {
                    const int xOutSide = x / UNIT;
                    const int xInSide  = x % UNIT;
                    const int dstIndex = xOutSide * weight->stride(0) + xInSide * SRC_UNIT;
                    const int srcIndex = x * kernelCount * ic;
                    dstY[dstIndex]     = srcY[srcIndex];
                }
            }
        }
    } else {
        for (int k = 0; k < kernelCount; ++k) {
            auto icDivU = UP_DIV(ic, SRC_UNIT);
            const auto srcK = weightSrc + k;
            for (int y = 0; y < ic; ++y) {
                const int yOutSide    = y / SRC_UNIT;
                const int yInSide     = y % SRC_UNIT;

                auto dstY       = weightDst + (yOutSide + k * icDivU)  * weight->stride(1) + yInSide;
                const auto srcY = srcK + y * kernelCount;
                for (int x = 0; x < oc; ++x) {
                    const int xOutSide = x / UNIT;
                    const int xInSide  = x % UNIT;
                    const int dstIndex = xOutSide * weight->stride(0) + xInSide * SRC_UNIT;
                    const int srcIndex = x * kernelCount * ic;
                    dstY[dstIndex]     = srcY[srcIndex];
                }
            }
        }
    }
}

static bool _reorderWeightInside(Backend* bn, const Convolution2DCommon* common,
                          const std::shared_ptr<Tensor>& weightOrigin,
                          std::shared_ptr<Tensor>& weight) {
    auto core = static_cast<CPUBackend*>(bn)->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    // reorder weight, [oc, ic, k^2] => [oc/unit, ((ic/unit)*k^2)/(src_unit/unit), unit(oc), (src_unit/unit), unit(ic)]
    int oc = common->outputCount(), ic = common->inputCount(), kernelCount = common->kernelX() * common->kernelY();
    std::vector<int> shape;
    if (SRC_UNIT > UNIT) {
        MNN_ASSERT(SRC_UNIT % UNIT == 0);
        shape = {UP_DIV(oc, UNIT), UP_DIV(UP_DIV(ic, UNIT) * kernelCount, SRC_UNIT / UNIT), UNIT, SRC_UNIT};
    } else {
        shape = {UP_DIV(oc, UNIT), UP_DIV(ic, SRC_UNIT) * kernelCount, UNIT, SRC_UNIT};
    }

    weight.reset(Tensor::createDevice<int8_t>(shape));

    bool succ = bn->onAcquireBuffer(weight.get(), Backend::STATIC);
    if (!succ) {
        MNN_ERROR("Memory not enough");
        return false;
    }
    ConvInt8TiledExecutor::reorderWeight(weight.get(), weightOrigin->host<uint8_t>(), SRC_UNIT, UNIT, ic, oc, kernelCount);
    return true;
}

DenseConvInt8TiledExecutor::DenseConvInt8TiledExecutor(Backend* backend, const Convolution2D* convOp, std::shared_ptr<ResourceInt8> res) : ConvInt8TiledExecutor(backend, convOp->common(), res) {
    std::shared_ptr<Tensor> weightOrigin = mResource->mWeightInt8;
    mValid = _reorderWeightInside(backend, convOp->common(), weightOrigin, mResource->mWeightInt8);
    if(!mValid) {
        return;
    }
    // choose int8 gemm kernel
    auto core = static_cast<CPUBackend*>(backend)->int8Functions();
    mGemmKernel = core->Int8GemmKernel;
#ifdef MNN_USE_SSE
    int actBits = convOp->symmetricQuan()->nbits();
    if (actBits <= 7) {
        mGemmKernel = core->Int8GemmKernelFast;
    }
#else
    if(convOp->symmetricQuan()->method() == QuantizeAlgo_OVERFLOW_AWARE){
        mGemmKernel = core->Int8GemmKernelFast;
    }
#endif
}

DenseConvInt8TiledExecutor::DenseConvInt8TiledExecutor(Backend* backend, const Convolution2DCommon* common, const DenseConvInt8TiledExecutor& exe)
    : ConvInt8TiledExecutor(backend, common, exe.mResource), mGemmKernel(exe.mGemmKernel) {

}

DenseConvInt8TiledExecutor::~DenseConvInt8TiledExecutor() {
    // Do nothing
}

bool DenseConvInt8TiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new DenseConvInt8TiledExecutor(bn, op->main_as_Convolution2D()->common(), *this);
    if (!exe->valid()) {
        return false;
    }
    *dst = exe;
    return true;
}

void DenseConvInt8TiledExecutor::getPackParameter(int* Unit, int* srcUnit, int* DestUnit, const CoreInt8Functions* core) {
    core->MNNGetGemmUnit(Unit, srcUnit, DestUnit);
}


ErrorCode DenseConvInt8TiledExecutor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Timer kernelTimer;
    ConvInt8TiledExecutor::onResize(inputs, outputs);
    auto output = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();

    int UNIT, SRC_UNIT, DST_XUNIT;
    getPackParameter(&UNIT, &SRC_UNIT, &DST_XUNIT, core);
    const int threads = std::max(static_cast<CPUBackend*>(backend())->threadNumber(), 1);
    auto planeSize = output->width() * output->height() * output->batch();
    auto planeSizeInThread = UP_DIV(planeSize, threads);
    const int L2Size = 2048;
    const int tileLimitByC = UP_DIV(L2Size, mIm2ColParamter.kernelCountUnit * SRC_UNIT);
    int tileLimit = ALIMIN(tileLimitByC, planeSizeInThread);
    mIm2ColCount = UP_DIV(tileLimit, DST_XUNIT);
    auto DynamicDestUnit = DST_XUNIT * mIm2ColCount;
    mTileCount        = UP_DIV(planeSize, DynamicDestUnit);
    mThreadNums       = std::min(threads, mTileCount);

    auto input  = inputs[0];
    // set im2col tensor info
    mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({mThreadNums, DST_XUNIT * mIm2ColCount * mResource->mWeightInt8->length(1) * SRC_UNIT}));
    bool success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    auto bufferAlloc = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    auto blitInfoSize = ConvolutionTiledExecutor::computeBlitInfoSize(DST_XUNIT * mIm2ColCount, mIm2ColParamter.ow, mIm2ColParamter.kernelX * mIm2ColParamter.kernelY, mThreadNums);
    mBlitInfo = bufferAlloc->alloc(blitInfoSize.first);
    if (mBlitInfo.invalid()) {
        return OUT_OF_MEMORY;
    }
    bufferAlloc->free(mBlitInfo);
    mBlitInfoStride = blitInfoSize.second;

    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    // MNN_PRINT("dense conv2d int8 resize: cost time: %llu us\n", kernelTimer.durationInUs());
    return NO_ERROR;
}

ErrorCode DenseConvInt8TiledExecutor::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Timer kernelTimer;
    const auto input = inputs[0];
    auto output      = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();

    int UNIT__, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT__, &SRC_UNIT, &DST_XUNIT);
    auto blitProc = core->MNNPackC4Int8ForMatMul_A;
    const int plane = output->batch() * mIm2ColParamter.oh * mIm2ColParamter.ow;
    int PackUnit = static_cast<CPUBackend*>(backend())->functions()->pack;
    const int dstZStep = plane * PackUnit;

    const int batch = input->batch();
    const int ocDiv4 = UP_DIV(output->channel(), PackUnit);
    const auto kernelCountUnitDouble = mIm2ColParamter.kernelCountUnit;
    //auto remain = outputPlaneLen % GEMM_INT8_DST_XUNIT;
    //FUNC_PRINT(remain);

    const auto inputDataPtr = input->host<int8_t>();
    const auto weightDataPtr = mResource->mWeightInt8->host<int8_t>();

    auto im2colPtr           = mTempIm2ColBuffer->host<int8_t>();
    auto outputDataPtr       = output->host<int8_t>();
    QuanPostTreatParameters quanParam;
    quanParam.bias = mMutableResource.mBiasInt32->host<int32_t>();
    quanParam.scale = mMutableResource.mScaleFloat->host<float>();
    quanParam.maxValue = mMutableResource.mClampMax;
    if (mResource->mRelu) {
        quanParam.minValue = mMutableResource.mOutputZeroPoint;
    } else {
        quanParam.minValue = mMutableResource.mClampMin;
    }
    int dstBytes = static_cast<CPUBackend*>(backend())->getBytes(backend(), output);
    if (dstBytes != 1) {
        quanParam.useInt8 = 0;
    }
    //MNN_PRINT("max: %d, min: %d\n", quanParam.maxValue, quanParam.minValue);
    const int col_buffer_unit_size = mIm2ColParamter.kernelCountUnit * DST_XUNIT * SRC_UNIT * sizeof(int8_t);
    auto col_buffer_size = col_buffer_unit_size * mIm2ColCount;
    auto threadFunction = [&](int tId) {
        auto colAddr        = im2colPtr + tId * mTempIm2ColBuffer->stride(0);
        auto srcPtr     = (int8_t const **)(mBlitInfo.ptr() + tId * mBlitInfoStride.first);
        auto el         = (int32_t *)(srcPtr + mBlitInfoStride.second);

        int32_t info[4];
        info[1] = mIm2ColParamter.iw * mIm2ColParamter.ih * batch;
        info[2] = col_buffer_unit_size;
        info[3] = mIm2ColParamter.strideX;
        for (int tIndex = tId; tIndex < mTileCount; tIndex += mThreadNums) {
            const int xIndexStart  = tIndex * DST_XUNIT * mIm2ColCount;
            int realDstCount = ALIMIN(plane - xIndexStart, DST_XUNIT * mIm2ColCount);

            // im2col
            auto res = ConvolutionTiledExecutor::turnIm2ColToBlitInfo((const float**)srcPtr, el, xIndexStart, realDstCount, mIm2ColParamter, (const uint8_t*)inputDataPtr, 1);
            int number = res.first;
            bool needZero = res.second;
            if (needZero) {
#ifdef MNN_USE_SSE
                ::memset(colAddr, mMutableResource.mInputZeroPoint + 128, col_buffer_size);
#else
                ::memset(colAddr, mMutableResource.mInputZeroPoint, col_buffer_size);
#endif
            }
            info[0] = number;
            if (number > 0) {
                blitProc(colAddr, srcPtr, info, el);
            }
            auto outputInTilePtr = outputDataPtr + xIndexStart * PackUnit * dstBytes;
            auto colAddrTemp = colAddr;
            do {
                int step = ALIMIN(DST_XUNIT, realDstCount);
                mGemmKernel(outputInTilePtr, colAddrTemp, weightDataPtr, kernelCountUnitDouble, dstZStep * dstBytes, ocDiv4, &quanParam, step);
                realDstCount-=step;
                outputInTilePtr += DST_XUNIT * PackUnit * dstBytes;
                colAddrTemp += col_buffer_unit_size;
            } while(realDstCount > 0);
        }
    };
    MNN_CONCURRENCY_BEGIN(tId, mThreadNums) {
        threadFunction((int)tId);
    }
    MNN_CONCURRENCY_END();
    // MNN_PRINT("dense conv2d int8 execute: cost time: %llu us\n", kernelTimer.durationInUs());
    return NO_ERROR;
}





} // namespace MNN
