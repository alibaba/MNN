//
//  GemmInt8Executor.cpp
//  MNNCPU
//
//  Created by MNN on 2023/3/16.
//
#include "GemmInt8Executor.hpp"
#include "ConvolutionTiledExecutor.hpp"
#include "CommonOptFunction.h"
#include "core/Macro.h"
#include "core/BufferAllocator.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"

namespace MNN {
static void _makeResource(Backend* backend, std::shared_ptr<CPUConvolution::Resource> resource, const MNN::Op *op, std::shared_ptr<CPUConvolution::ResourceInt8> resourceInt8) {
    /* Used to compute weight quant scale and bias and weightKernelSum of type float. */
    auto conv2d = op->main_as_Convolution2D();
    bool quanBuffer = (conv2d->quanParameter() != nullptr && conv2d->quanParameter()->buffer() != nullptr);
    MNN_ASSERT(quanBuffer || resourceInt8);
    resource->backend = backend;
    auto core = static_cast<CPUBackend*>(backend)->functions();
    // common parameters
    int outputCount = conv2d->common()->outputCount();
    int LSize = conv2d->common()->inputCount() * conv2d->common()->kernelX() * conv2d->common()->kernelY();
    int ocUp4 = ROUND_UP(outputCount, core->pack);
    int8_t* weightOrigin;

    // Save weight quant scale and bias: wf=scale*wi+bias
    resource->mDequantize.mScaleBias.reset(Tensor::createDevice<uint8_t>({2 * ocUp4 * core->bytes}));
    auto success = resource->backend->onAcquireBuffer(resource->mDequantize.mScaleBias.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Alloc denquant scaleBias memory error\n");
        return;
    }
    auto alphaPtr = resource->mDequantize.mScaleBias->host<float>();
    auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(alphaPtr) + ocUp4 * core->bytes);
    ::memset(alphaPtr, 0, 2 * ocUp4 * core->bytes);
    auto wZero = resourceInt8->mWeightQuantZero->host<int32_t>(); // has packed to outputUp4
    auto wScale = resourceInt8->mOriginScale->host<float>();
    int h = ocUp4;
    for (int i=0; i< h; ++i) {
        alphaPtr[i] = wScale[i];
        biasPtr[i] = (-1.f) * wZero[i] * wScale[i];
    }
}

GemmInt8Executor::GemmInt8Executor(Backend* bn, std::shared_ptr<ResourceInt8> resource, const Op *op, decltype(CoreInt8Functions::Int8GemmKernel) gemmKernel, std::vector<int32_t> bias) :
    CPUConvolution(op->main_as_Convolution2D()->common(), bn), mResourceInt8(resource), mMutableResource(resource, bn), mGemmKernel(gemmKernel), mQuantBias(bias){
        mResource.reset(new Resource);
        _makeResource(bn, mResource, op, mResourceInt8);
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
    int UNIT__, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT__, &SRC_UNIT, &DST_XUNIT);
    auto gcore = static_cast<CPUBackend*>(backend())->functions();
    auto pack = gcore->pack;

    auto scaleSrc = mMutableResource.mScaleFloat->host<float>();
    int realWeightQuantScaleSize = mResource->mDequantize.mScaleBias->size() / 2;
    auto weightBiasSrc = reinterpret_cast<float*>(mResource->mDequantize.mScaleBias->host<uint8_t>() + realWeightQuantScaleSize);
    auto ocDivUp = UP_DIV(output->channel(), pack) * pack;
    mKernelY   = mCommon->kernelY();
    mKernelX   = mCommon->kernelX();
    int kernelCount = mKernelX * mKernelY;
    std::vector<float> scaleData(ocDivUp);
    mKernelSum.resize(ocDivUp, 0);
    ::memset(scaleData.data(), 0.f, ocDivUp * sizeof(float));
    auto l = mMutableResource.mScaleFloat->length(0);
    auto lU = UP_DIV(l, pack);
    for (int divC = 0; divC < lU; ++divC) {
        auto srcX = scaleSrc + divC * pack;
        auto wbias = weightBiasSrc + divC * pack;
        for (int k = 0; k < kernelCount; ++k) {
            int indexK = divC * kernelCount * pack + k * pack;
            for (int j = 0; j < pack; ++j) {
                scaleData[indexK + j] = srcX[j];
                mKernelSum[indexK + j] = wbias[j];
            }
        }
    }
    float* biasFloat = reinterpret_cast<float*>(mQuantBias.data());
    for (int i = 0; i < mQuantBias.size(); ++i) {
        biasFloat[i] = mQuantBias[i] * scaleData[i];
    }
    mScaleData = scaleData;
    const auto IC4 = UP_DIV(input->channel(), pack);
    ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, input, output, 0, 0, static_cast<CPUBackend*>(backend())->functions(), core);
    auto originKernelCount = mCommon->kernelX() * mCommon->kernelY();
    mIm2ColParamter.strideX         = 1;
    mIm2ColParamter.strideY         = 1;
    mIm2ColParamter.kernelX         = 1;
    mIm2ColParamter.kernelY         = 1;
    mIm2ColParamter.padX            = 0;
    mIm2ColParamter.padY            = 0;
    if (SRC_UNIT > pack) {
        const auto srcCountUnit = UP_DIV(input->channel(), pack);
        mIm2ColParamter.kernelCountUnit = UP_DIV(srcCountUnit, SRC_UNIT / pack);
        mIm2ColParamter.ic = mIm2ColParamter.icDiv4 * pack;
    } else {
        const auto srcCountUnit = UP_DIV(input->channel(), SRC_UNIT);
        mIm2ColParamter.kernelCountUnit = srcCountUnit;
        mIm2ColParamter.ic = srcCountUnit * SRC_UNIT;
    }

    mTileCnt = UP_DIV(input->height() * input->width() * input->batch(), DST_XUNIT);
    const int threads = std::max(static_cast<CPUBackend*>(backend())->threadNumber(), 1);
    mThreadNums       = std::min(threads, mTileCnt);

    mInputCol.reset(Tensor::createDevice<int8_t>({mThreadNums, DST_XUNIT,  mIm2ColParamter.kernelCountUnit * SRC_UNIT}));
    bool success = backend()->onAcquireBuffer(mInputCol.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    auto bufferAlloc = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    auto blitInfoSize = ConvolutionTiledExecutor::computeBlitInfoSize(DST_XUNIT, mIm2ColParamter.ow, mIm2ColParamter.kernelX * mIm2ColParamter.kernelY, mThreadNums);
    mBlitInfo = bufferAlloc->alloc(blitInfoSize.first);
    if (mBlitInfo.invalid()) {
        return OUT_OF_MEMORY;
    }
    bufferAlloc->free(mBlitInfo);
    mBlitInfoStride = blitInfoSize.second;

    backend()->onReleaseBuffer(mInputCol.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode GemmInt8Executor::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const auto input = inputs[0];
    auto output = outputs[0];
    auto batch = output->batch();
    const auto kEleCnt = mKernelX * mKernelY;

    const int outplane = output->height() * output->width() * output->batch();
    const int inputplane = input->height() * input->width();

    auto gcore = static_cast<CPUBackend*>(backend())->functions();
    auto arch_pack = gcore->pack;
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT__, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT__, &SRC_UNIT, &DST_XUNIT);
    int PackUnit = static_cast<CPUBackend*>(backend())->functions()->pack;
    auto blitProc = core->MNNPackC4Int8ForMatMul_A;
    const int dstZStep = outplane * PackUnit;
    const int ocDiv4 = UP_DIV(output->channel(), PackUnit); // Here, output->channel() = oc*kw*kh
    const auto src_depth_quad = mIm2ColParamter.kernelCountUnit;

    const auto inputDataPtr = input->host<int8_t>();
    const auto weightDataPtr = inputs[1]->host<int8_t>();

    auto im2colPtr           = mInputCol->host<int8_t>();
    auto outputDataPtr       = output->host<float>();

    auto bias_elesize = ocDiv4 * PackUnit;
    QuanPostTreatParameters quanParam;
    quanParam.scale = mScaleData.data();
    quanParam.maxValue = mMutableResource.mClampMax;
    if (mResourceInt8->mRelu) {
        quanParam.minValue = mMutableResource.mOutputZeroPoint;
    } else {
        quanParam.minValue = mMutableResource.mClampMin;
    }
    auto postParameters    = getPostParameters();
    std::vector<float> fp32minmax = {postParameters[2], postParameters[3]};
    quanParam.fp32minmax = fp32minmax.data();

    quanParam.useInt8 = 0; // Save result as float data type.
    quanParam.biasFloat = reinterpret_cast<float*>(mQuantBias.data());
    quanParam.weightQuanBias = mKernelSum.data();
    quanParam.extraScale = nullptr;
    float dequantScale = mMutableResource.mResource->mInputScale;

    SumByAxisParams sumParams;
    sumParams.DST_XUNIT = DST_XUNIT;
    sumParams.SRC_UNIT = SRC_UNIT;
    sumParams.blockNum = 1;
    sumParams.kernelCountUnitDouble = mIm2ColParamter.kernelCountUnit;
    sumParams.oneScale = 1;
    sumParams.col_buffer_unit_size = mInputCol->stride(0);

    auto threadFunction = [&](int tId) {
        auto colAddr        = im2colPtr + tId * mInputCol->stride(0);
        auto col_buffer_size = mInputCol->stride(0);
        int32_t info[5];
        info[1] = mIm2ColParamter.iw * mIm2ColParamter.ih * batch;
        info[2] = DST_XUNIT;
        info[3] = mIm2ColParamter.strideX;
        float paramsf[1];
        paramsf[0] = dequantScale;
        auto srcPtr     = (int8_t const **)(mBlitInfo.ptr() + tId * mBlitInfoStride.first);
        auto el         = (int32_t *)(srcPtr + mBlitInfoStride.second);

        for (int tIndex = tId; tIndex < mTileCnt; tIndex += mThreadNums) {
            const int xIndexStart  = tIndex * DST_XUNIT;
            const int realDstCount = ALIMIN(outplane - xIndexStart, DST_XUNIT);
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
            info[4] = realDstCount;
            std::vector<float> xKernelSum(realDstCount);
            if (number > 0) {
                blitProc(colAddr, srcPtr, info, el);
            }
            if (mResourceInt8->mWeightAsymmetricQuant) {
                gcore->MNNSumByAxisLForMatmul_A(xKernelSum.data(), colAddr, &dequantScale, realDstCount, sumParams);
            }
            quanParam.srcKernelSum = xKernelSum.data();
            auto outputInTilePtr = outputDataPtr + xIndexStart * PackUnit;
            mGemmKernel((int8_t*)outputInTilePtr, colAddr, weightDataPtr, src_depth_quad, dstZStep * sizeof(float), ocDiv4, &quanParam, realDstCount);
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
