//
//  GemmInt8Executor.cpp
//  MNNCPU
//
//  Created by MNN on 2023/3/16.
//
#include "GemmInt8Executor.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"

namespace MNN {

GemmInt8Executor::GemmInt8Executor(Backend* bn, std::shared_ptr<ResourceInt8> resource, const Convolution2D *conv2D, decltype(CoreInt8Functions::Int8GemmKernel) gemmKernel,
                                   std::vector<int32_t> bias):
    CPUConvolution(conv2D->common(), bn), mResource(resource), mMutableResource(resource, bn), mGemmKernel(gemmKernel), mQuantBias(bias){
}

GemmInt8Executor::~GemmInt8Executor() {
    // Do nothing
}

/*
 Deconvolution forward:
 Input  (N⋅IW⋅IH, IC)
 Weight (IC, OC⋅KW⋅KH)
 Output (N⋅IW⋅IH, OC⋅KW⋅KH)
 */
ErrorCode GemmInt8Executor::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto outputQuanInfo = TensorUtils::getQuantInfo(outputs[0]);
    outputQuanInfo[0] = 1.0f;
    mMutableResource.updateInputOutputScale(TensorUtils::getQuantInfo(inputs[0]), outputQuanInfo);
    //CPUConvolution::onResize(inputs, outputs);
    auto input = inputs[0];
    auto output = outputs[0];

    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);

    auto scaleSrc = mMutableResource.mScaleFloat->host<float>();
    auto ocDivUp = UP_DIV(output->channel(), UNIT) * UNIT;
    mKernelY   = mCommon->kernelY();
    mKernelX   = mCommon->kernelX();
    int kernelCount = mKernelX * mKernelY;
    std::vector<float> scaleData(ocDivUp);
    ::memset(scaleData.data(), 1.0, ocDivUp * sizeof(float));
    for (int k = 0; k < ocDivUp / kernelCount; ++k) {
        for (int j = 0; j < kernelCount; ++j) {
            scaleData[k * kernelCount + j] = scaleSrc[k];
        }
    }
    mScaleData = scaleData;
    auto gcore = static_cast<CPUBackend*>(backend())->functions();
    auto pack = gcore->pack;
    const auto IC4 = UP_DIV(input->channel(), pack);

    mIm2ColParamter.strideX         = 1;
    mIm2ColParamter.strideY         = 1;
    mIm2ColParamter.icDiv4          = IC4;
    mIm2ColParamter.kernelX         = 1;
    mIm2ColParamter.kernelY         = 1;
    mIm2ColParamter.padX            = 0;
    mIm2ColParamter.padY            = 0;

    mIm2ColParamter.ih              = input->height();
    mIm2ColParamter.iw              = input->width();
    mIm2ColParamter.oh              = output->height();
    mIm2ColParamter.ow              = output->width();
    mIm2ColParamter.srcZStep        = input->stride(1) * pack * input->batch();
    mIm2ColParamter.srcYStep        = input->stride(2) * pack;
    mIm2ColParamter.packCUnit       = pack;
    const auto srcCountUnit = UP_DIV(input->channel(), UNIT);
    mIm2ColParamter.kernelCountUnit = UP_DIV(srcCountUnit, SRC_UNIT / UNIT); // Here is IC/SRC_UNIT, which is different from (IC·KW·KH)/SRC_UNIT of convolution.

    mTileCnt = UP_DIV(input->height() * input->width(), DST_XUNIT);
    const int threads = std::max(static_cast<CPUBackend*>(backend())->threadNumber(), 1);
    mThreadNums       = std::min(threads, mTileCnt);
    
    mInputCol.reset(Tensor::createDevice<int8_t>({mThreadNums, DST_XUNIT, IC4 * pack}));
    bool success = backend()->onAcquire(mInputCol.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mInputCol.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode GemmInt8Executor::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const auto input = inputs[0];
    auto output = outputs[0];
    auto batch = output->batch();
    const auto kEleCnt = mKernelX * mKernelY;

    const int outplane = output->height() * output->width();
    const int inputplane = input->height() * input->width();

    auto gcore = static_cast<CPUBackend*>(backend())->functions();
    auto arch_pack = gcore->pack;
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    auto im2ColProcess = core->chooseIm2Col(&mIm2ColParamter, input->channel());
    const int dstZStep = outplane * UNIT * output->batch();
    const int ocDiv4 = UP_DIV(output->channel(), UNIT); // Here, output->channel() = oc*kw*kh
    const int oc4 = ocDiv4 / kEleCnt;
    const int icDiv4 = UP_DIV(input->channel(), SRC_UNIT);
    const auto src_depth_quad = mIm2ColParamter.kernelCountUnit;

    const auto inputDataPtr = input->host<int8_t>();
    const auto weightDataPtr = inputs[1]->host<int8_t>();

    auto im2colPtr           = mInputCol->host<int8_t>();
    auto outputDataPtr       = output->host<float>();
    
    auto bias_elesize = ocDiv4 * UNIT;
    QuanPostTreatParameters quanParam;
    quanParam.scale = mScaleData.data();
    quanParam.maxValue = mMutableResource.mClampMax;
    if (mResource->mRelu) {
        quanParam.minValue = mMutableResource.mOutputZeroPoint;
    } else {
        quanParam.minValue = mMutableResource.mClampMin;
    }

    quanParam.useInt8 = 0; // Save result as float data type.
    quanParam.bias = mQuantBias.data();

    auto threadFunction = [&](int tId) {
        auto colAddr        = im2colPtr + tId * mInputCol->stride(0);
        for (int bIndex = 0; bIndex < batch; ++bIndex) {
            const auto srcPtr = inputDataPtr + bIndex * UNIT * inputplane;
            auto dstPtr       = outputDataPtr + bIndex * UNIT * outplane;
            for (int tIndex = tId; tIndex < mTileCnt; tIndex += mThreadNums) {
                const int xIndexStart  = tIndex * DST_XUNIT;
                const int realDstCount = ALIMIN(outplane - xIndexStart, DST_XUNIT);
                // im2col
#ifdef MNN_USE_SSE
                im2ColProcess(colAddr, srcPtr, mMutableResource.mInputZeroPoint + 128, &mIm2ColParamter, xIndexStart, realDstCount);
#else
                im2ColProcess(colAddr, srcPtr, mMutableResource.mInputZeroPoint, &mIm2ColParamter, xIndexStart, realDstCount);
#endif
                auto outputInTilePtr = dstPtr + xIndexStart * UNIT;
                mGemmKernel((int8_t*)outputInTilePtr, colAddr, weightDataPtr, src_depth_quad, dstZStep * sizeof(float), ocDiv4, &quanParam, realDstCount);
            }
        }
    };
    MNN_CONCURRENCY_BEGIN(tId, mThreadNums) {
        threadFunction((int)tId);
    }
    MNN_CONCURRENCY_END();
    
    // MNN_PRINT("deconv int8 execute: cost time: %llu us\n", kernelTimer.durationInUs());
    return NO_ERROR;
}

} // namespace MNN
