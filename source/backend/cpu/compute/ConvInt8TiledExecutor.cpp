//
//  ConvInt8TiledExecutor.cpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvInt8TiledExecutor.hpp"
#include "core/Macro.h"

#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include <math.h>
#ifdef MNN_USE_SSE
extern "C" {
void MNNInt8ToUInt8(void* ptr, int count);
}
#endif
namespace MNN {

static bool reorderWeight(Backend* bn, const Convolution2DCommon* common,
                          const std::shared_ptr<Tensor>& weightOrigin,
                          std::shared_ptr<Tensor>& weight) {
    auto core = static_cast<CPUBackend*>(bn)->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    // reorder weight, [oc, ic, k^2] => [oc/unit, ((ic/unit)*k^2)/(src_unit/unit), unit(oc), (src_unit/unit), unit(ic)]
    int oc = common->outputCount(), ic = common->inputCount(), kernelCount = common->kernelX() * common->kernelY();
    std::vector<int> shape = {UP_DIV(oc, UNIT), UP_DIV(UP_DIV(ic, UNIT) * kernelCount, SRC_UNIT / UNIT), UNIT, SRC_UNIT};
    
    weight.reset(Tensor::createDevice<int8_t>(shape));
    
    bool succ = bn->onAcquireBuffer(weight.get(), Backend::STATIC);
    if (!succ) {
        MNN_ERROR("Memory not enough");
        return false;
    }
    auto weightSrc = weightOrigin->host<int8_t>();
    auto weightDst = weight->host<int8_t>();
    memset(weightDst, 0, weight->size());
    for (int k = 0; k < kernelCount; ++k) {
        const auto srcK = weightSrc + k;
        for (int y = 0; y < ic; ++y) {
            const int yOutSide    = y / UNIT;
            const int yInSide     = y % UNIT;
            const int yIndex      = yOutSide + k * UP_DIV(ic, UNIT);
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
    return true;
}

ConvInt8TiledExecutor::ConvInt8TiledExecutor(Backend* backend, const Convolution2D* convOp, std::shared_ptr<ResourceInt8> res): CPUConvolution(convOp->common(), backend), mResource(res) {
    std::shared_ptr<Tensor> weightOrigin;
    weightOrigin.swap(mResource->mWeightInt8);
    mValid = reorderWeight(backend, convOp->common(), weightOrigin, mResource->mWeightInt8);
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

ConvInt8TiledExecutor::ConvInt8TiledExecutor(Backend* backend, const Convolution2DCommon* common, std::shared_ptr<Tensor> weight, bool fastgemm)
: CPUConvolution(common, backend) {
    auto core = static_cast<CPUBackend*>(backend)->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int oc = common->outputCount(), ic = common->inputCount(), kernel = common->kernelY() * common->kernelX();
    mResource.reset(new ResourceInt8);
    mResource->backend = backend;
    mResource->mBiasInt32.reset(Tensor::createDevice<int32_t>({ROUND_UP(oc, UNIT)}));
    mValid = backend->onAcquireBuffer(mResource->mBiasInt32.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("Memory not enough\n");
        return;
    }
    ::memset(mResource->mBiasInt32->host<int32_t>(), 0, mResource->mBiasInt32->size());
#ifdef MNN_USE_SSE
    for (int oz = 0; oz < oc; ++oz) {
        int32_t offset = 0;
        for (int i = 0; i < ic * kernel; ++i) {
            offset += (int32_t)(weight->host<int8_t>()[oz * ic * kernel + i]) * (-128);
        }
        mResource->mBiasInt32->host<int32_t>()[oz] = offset;
    }
#endif
    mValid = reorderWeight(backend, common, weight, mResource->mWeightInt8);
    if(!mValid) {
        MNN_ERROR("Memory not enough\n");
        return;
    }
    // choose int8 gemm kernel
    mGemmKernel = core->Int8GemmKernel;
    if (fastgemm) {
        mGemmKernel = core->Int8GemmKernelFast;
    }
    mDoPostProcess = false;
}

ConvInt8TiledExecutor::ConvInt8TiledExecutor(Backend* backend, const Convolution2DCommon* common, const ConvInt8TiledExecutor& exe)
    : CPUConvolution(common, backend), mGemmKernel(exe.mGemmKernel),
    mDoPostProcess(exe.mDoPostProcess), mResource(exe.mResource) {
    
}

ConvInt8TiledExecutor::~ConvInt8TiledExecutor() {
    // Do nothing
}

bool ConvInt8TiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new ConvInt8TiledExecutor(bn, op->main_as_Convolution2D()->common(), *this);
    if (!exe->valid()) {
        return false;
    }
    *dst = exe;
    return true;
}

ErrorCode ConvInt8TiledExecutor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (mDoPostProcess) {
        mResource->updateInputOutputScale(TensorUtils::getQuantInfo(inputs[0]), TensorUtils::getQuantInfo(outputs[0]));
    } else {
        mResource->mInputZeroPoint = 0;
    }
    CPUConvolution::onResize(inputs, outputs);
    auto input  = inputs[0];
    auto output = outputs[0];
    
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    auto convCommon = mCommon;
    const auto kernelCount = convCommon->kernelX() * convCommon->kernelY();
    const auto srcCountUnit = UP_DIV(input->channel(), UNIT);
    const auto totalKernelCountD8Div2 = UP_DIV(srcCountUnit * kernelCount, SRC_UNIT / UNIT);

    mIm2ColParamter.dilateX         = convCommon->dilateX();
    mIm2ColParamter.dilateY         = convCommon->dilateY();
    mIm2ColParamter.strideX         = convCommon->strideX();
    mIm2ColParamter.strideY         = convCommon->strideY();
    mIm2ColParamter.padX            = convCommon->padX();
    mIm2ColParamter.padY            = convCommon->padY();
    mIm2ColParamter.icDiv4          = srcCountUnit;
    mIm2ColParamter.kernelX         = convCommon->kernelX();
    mIm2ColParamter.kernelY         = convCommon->kernelY();
    mIm2ColParamter.kernelCountUnit = totalKernelCountD8Div2;
    mIm2ColParamter.padX = mPadX;
    mIm2ColParamter.padY = mPadY;

    mIm2ColParamter.ih = input->height();
    mIm2ColParamter.iw = input->width();
    mIm2ColParamter.oh = output->height();
    mIm2ColParamter.ow = output->width();
    mIm2ColParamter.srcZStep = input->stride(1) * UNIT * input->batch();
    mIm2ColParamter.srcYStep = input->stride(2) * UNIT;

    mTileCount        = UP_DIV(output->height() * output->width(), DST_XUNIT);
    const int threads = std::max(static_cast<CPUBackend*>(backend())->threadNumber(), 1);
    mThreadNums       = std::min(threads, mTileCount);

    // set im2col tensor info
    mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({mThreadNums, DST_XUNIT, mResource->mWeightInt8->length(1) * SRC_UNIT}));
    bool success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode ConvInt8TiledExecutor::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    
    auto im2ColProcess = core->chooseIm2Col(&mIm2ColParamter, input->channel());

    const int outputPlaneLen = output->height() * output->width();
    const int dstZStep = outputPlaneLen * UNIT * output->batch();
    const int inputPlaneLen = input->width() * input->height();

    const int batch = input->batch();
    const int ocDiv4 = UP_DIV(output->channel(), UNIT);
    const auto kernelCountUnitDouble = mIm2ColParamter.kernelCountUnit;
    //auto remain = outputPlaneLen % GEMM_INT8_DST_XUNIT;
    //FUNC_PRINT(remain);

    const auto inputDataPtr = input->host<int8_t>();
    const auto weightDataPtr = mResource->mWeightInt8->host<int8_t>();
    
    auto im2colPtr           = mTempIm2ColBuffer->host<int8_t>();
    auto outputDataPtr       = output->host<int8_t>();
    QuanPostTreatParameters quanParam;
    quanParam.bias = mResource->mBiasInt32->host<int32_t>();
    if (mDoPostProcess) {
        quanParam.scale = mResource->mScaleFloat->host<float>();
        quanParam.maxValue = mResource->mClampMax;
        if (mResource->mRelu) {
            quanParam.minValue = mResource->mOutputZeroPoint;
        } else {
            quanParam.minValue = mResource->mClampMin;
        }
    } else {
        quanParam.scale = nullptr;
    }
    //MNN_PRINT("max: %d, min: %d\n", quanParam.maxValue, quanParam.minValue);
    
    const int bytes = (mDoPostProcess ? 1 : 4); // int8_t or float

    auto threadFunction = [&](int tId) {
        auto colAddr        = im2colPtr + tId * mTempIm2ColBuffer->stride(0);
        for (int bIndex = 0; bIndex < batch; ++bIndex) {
            const auto srcPtr = inputDataPtr + bIndex * UNIT * bytes * inputPlaneLen;
            auto dstPtr       = outputDataPtr + bIndex * UNIT * bytes * outputPlaneLen;

            for (int tIndex = tId; tIndex < mTileCount; tIndex += mThreadNums) {
                const int xIndexStart  = tIndex * DST_XUNIT;
                const int realDstCount = ALIMIN(outputPlaneLen - xIndexStart, DST_XUNIT);
                // im2col
                im2ColProcess(colAddr, srcPtr, mResource->mInputZeroPoint, &mIm2ColParamter, xIndexStart, realDstCount);
#ifdef MNN_USE_SSE
                const int col_buffer_size = mIm2ColParamter.kernelCountUnit * DST_XUNIT * SRC_UNIT;
                MNNInt8ToUInt8(colAddr, col_buffer_size);
#endif
                auto outputInTilePtr = dstPtr + xIndexStart * UNIT * bytes;
                mGemmKernel(outputInTilePtr, colAddr, weightDataPtr, kernelCountUnitDouble, dstZStep * bytes, ocDiv4, &quanParam, realDstCount);
            }
        }
    };
    MNN_CONCURRENCY_BEGIN(tId, mThreadNums) {
        threadFunction((int)tId);
    }
    MNN_CONCURRENCY_END();

    return NO_ERROR;
}

} // namespace MNN
