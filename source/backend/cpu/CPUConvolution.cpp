//
//  CPUConvolution.cpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUConvolution.hpp"
#include <math.h>
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include <limits>
#include "backend/cpu/compute/ConvolutionFloatFactory.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "core/ConvolutionCommon.hpp"

#include "backend/cpu/compute/ConvInt8Winograd.hpp"
#include "backend/cpu/compute/ConvInt8TiledExecutor.hpp"
#include "backend/cpu/compute/SparseConvInt8TiledExecutor.hpp"
#ifdef MNN_USE_ONEDNN
#include "backend/cpu/OneDNNConvInt8.hpp"
#endif

namespace MNN {

bool CPUConvolution::Resource::copyBiasAlign(const float* bias, int outputCount) {
    auto core = static_cast<CPUBackend*>(backend)->functions();
    int bytes = core->bytes;
    int unit = core->pack;
    auto alignOutput = UP_DIV(outputCount, unit) * unit;
    int remain = alignOutput - outputCount;
    mBias.reset(Tensor::createDevice<uint8_t>(std::vector<int>{alignOutput * bytes}));
    bool success = backend->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Error for alloc memory for Alloc Bias\n");
        return false;;
    }
    if (bytes < 4) {
        core->MNNFp32ToLowp(bias, mBias->host<int16_t>(), outputCount);
    } else {
        ::memcpy(mBias->host<float>(), bias, outputCount * bytes);
    }
    if (remain > 0) {
        ::memset(mBias->host<uint8_t>() + outputCount * bytes, 0, remain * bytes);
    }
    return true;
}
CPUConvolution::MutableResourceInt8::MutableResourceInt8(std::shared_ptr<ResourceInt8> res, Backend* backend) : mResource(res) {
    auto outputChannleUp4 = res->mOriginBias->length(0);
    mBiasFloat.reset(Tensor::createDevice<int32_t>({outputChannleUp4}));
    mValid = backend->onAcquireBuffer(mBiasFloat.get(), Backend::STATIC);
    if (!mValid) {
        MNN_ERROR("mBiasFloat buffer allocated error!\n");
        return;
    }
    if (res->mUseConvQuan) {
        mBiasInt32 = res->mOriginBias;
        mScaleFloat = res->mOriginScale;
        mValid = true;
        mInputScale = res->mInputScale;
        mOutputScale = res->mOutputScale;
        mInputZeroPoint = res->mInputZeroPoint;
        mOutputZeroPoint = res->mOutputZeroPoint;
        mClampMax = res->mClampMax;
        mClampMin = res->mClampMin;
        // bias int32 -> bias float
        auto int32BiasPtr = res->mOriginBias->host<int32_t>();
        auto floatBiasPtr = mBiasFloat->host<float>();
        auto weightScale  = res->mOriginScale->host<float>();
        for (int i = 0; i < outputChannleUp4; ++i) {
            if (mInputScale && mOutputScale) { // symmetric quan
                floatBiasPtr[i] = int32BiasPtr[i] * weightScale[i] * mInputScale / mOutputScale;
            } else {
                floatBiasPtr[i] = int32BiasPtr[i] * weightScale[i];
            }
        }
        return;
    }
    mBiasInt32.reset(Tensor::createDevice<int32_t>({outputChannleUp4}));
    mScaleFloat.reset(Tensor::createDevice<int32_t>({outputChannleUp4}));
    mValid = backend->onAcquireBuffer(mBiasInt32.get(), Backend::STATIC);
    if (mValid) {
        mValid = backend->onAcquireBuffer(mScaleFloat.get(), Backend::STATIC);
    }
    
}

void CPUConvolution::MutableResourceInt8::updateInputOutputScale(std::vector<float> inputQuantInfo, std::vector<float> outputQuantInfo) {
    if (mResource->mUseConvQuan) {
        return;
    }
    // new scales and zero points
    float inputScale = inputQuantInfo[0];
    float outputScale = outputQuantInfo[0];
    float inputZeroPoint = inputQuantInfo[1];
    float outputZeroPoint = outputQuantInfo[1];
    mClampMin = int8_t(outputQuantInfo[2]);
    mClampMax = int8_t(outputQuantInfo[3]);

    mInputScale = mResource->mInputScale;
    mOutputScale = mResource->mOutputScale;
    mInputZeroPoint = mResource->mInputZeroPoint;
    mOutputZeroPoint = mResource->mOutputZeroPoint;
//    if (mInputScale == inputScale && mOutputScale == outputScale) {
//        return;
//    }
    if (inputScale != 0 && outputScale != 0) {
        mInputScale = inputScale;
        mOutputScale = outputScale;
        mInputZeroPoint = int8_t(inputZeroPoint);
        mOutputZeroPoint = int8_t(outputZeroPoint);
    }
    if (mInputScale == 0 || mOutputScale == 0) {
        return;
    }
    
    int size = mResource->mOutputCount;
    const int kernelNum = static_cast<int>(mResource->mInt8WeightKernelSum.size());
    auto biasData    = mResource->mOriginBias->host<float>();
    auto alphaData   = mResource->mOriginScale->host<float>();
    auto alphaScale  = mInputScale / mOutputScale;
    auto scale = mScaleFloat->host<float>();
    auto bias = mBiasInt32->host<int32_t>();
    auto biasfloat = mBiasFloat->host<float>();
#ifdef MNN_USE_SSE
    float offset = 128.f;
#else
    float offset = 0.f;
#endif
    for (int i = 0; i < kernelNum; i++) {
        auto alphaValue = alphaData[i];
        if (fabs(alphaValue) < 1e-6) {
            alphaValue = 1e-6;
        }
        scale[i] = alphaValue * alphaScale; // input_scale*weight_scale/output_scale
        // compute outputZeroPointFused in asymmetric quant
        int outputZeroPointFused = static_cast<int32_t>(mOutputZeroPoint / scale[i]);
        bias[i] = static_cast<int32_t>(biasData[i] / (mInputScale * alphaValue)) - mResource->mInt8WeightKernelSum[i] * (mInputZeroPoint + offset) + outputZeroPointFused;
        // biasfloat[i] = biasData[i] / mOutputScale - mResource->mInt8WeightKernelSum[i] * (mInputZeroPoint + offset) * scale[i] + mOutputZeroPoint;
        biasfloat[i] = bias[i] * scale[i];
    }
}
std::shared_ptr<CPUConvolution::ResourceInt8> CPUConvolution::makeResourceInt8(Backend* backend, const MNN::Convolution2D *convParam, int pack) {
    auto core = static_cast<CPUBackend*>(backend)->functions();
    // TODO: use different pack from float
    int UNIT = pack;
    
    std::shared_ptr<CPUConvolution::ResourceInt8> resource(new ResourceInt8);
    // TODO: ConvInt8Winograd need in/out scale, which isn't exist in quantinfo when model construct by V3 API
    const auto convCommon  = convParam->common();
    const auto group = convParam->common()->group();
    const auto outputCount = convCommon->outputCount();
    const auto outputChannleUp4 = UP_DIV(outputCount, UNIT) * UNIT;
    
    int quanCount = outputChannleUp4;
    if (convParam->quanParameter() && convParam->quanParameter()->alpha()) {
        quanCount = convParam->quanParameter()->alpha()->size();
        quanCount = ROUND_UP(quanCount, UNIT);
    }
    resource->mOriginBias.reset(Tensor::createDevice<int32_t>({quanCount}));
    resource->mOriginScale.reset(Tensor::createDevice<uint8_t>({quanCount * core->bytes}));
    resource->mWeightQuantZero.reset(Tensor::createDevice<int32_t>({quanCount}));
    auto allocRes = backend->onAcquireBuffer(resource->mOriginBias.get(), Backend::STATIC);
    allocRes &= backend->onAcquireBuffer(resource->mOriginScale.get(), Backend::STATIC);
    allocRes &= backend->onAcquireBuffer(resource->mWeightQuantZero.get(), Backend::STATIC);
    if (!allocRes) {
        return nullptr;
    }

    auto biasPtr = resource->mOriginBias->host<int32_t>();
    memset(biasPtr, 0, quanCount * sizeof(int32_t));
    auto scalePtr = resource->mOriginScale->host<float>();
    memset(scalePtr, 0, quanCount * sizeof(uint8_t) * core->bytes);
    auto betaPtr = resource->mWeightQuantZero->host<int32_t>();
    memset(betaPtr, 0, quanCount * sizeof(int32_t));

    resource->mActBits = 8;
    if (convParam->symmetricQuan()) {
        resource->mActBits = convParam->symmetricQuan()->nbits();
    }
    const int8_t* weightSrc = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    resource->mOutputCount = outputCount;
    if (!ConvolutionCommon::getConvInt8Parameters(convParam, quanCommon, backend, weightSrc, weightSize, scalePtr, biasPtr, betaPtr)) {
        return nullptr;
    }
    if (convParam->bias() && convParam->quanParameter()->alpha()) {
        resource->mUseConvQuan = false;
    }
    if (quanCommon.get()) {
        resource->mWeightAsymmetricQuant = quanCommon->asymmetric;
    }

    resource->mWeightInt8.reset(Tensor::createDevice<int8_t>({weightSize}));
    allocRes = backend->onAcquireBuffer(resource->mWeightInt8.get(), Backend::STATIC);
    if (!allocRes) {
        return nullptr;
    }
    const int kernelNum = outputCount;
    const int kernelSize = weightSize / kernelNum;
    resource->mInt8WeightKernelSum.resize(outputChannleUp4);
    bool checkWeightQuantZero = false;
    for (int i = 0; i < kernelNum; i++) {
        int temp = 0;
        int offset = i * kernelSize;
        if (static_cast<int32_t>(betaPtr[i]) != 0) {
            checkWeightQuantZero = true;
        }
        for (int j = 0; j < kernelSize; j++) {
            temp += (static_cast<int>(weightSrc[offset + j]) - betaPtr[i]);
        }
        resource->mInt8WeightKernelSum[i] = temp;
#ifdef MNN_USE_SSE
        if (resource->mUseConvQuan) {
            resource->mOriginBias->host<int32_t>()[i] -= 128 * temp;
        }
#endif
    }
    if (false == checkWeightQuantZero) { // All weight quant bias is 0, do not need to compute related term in gemm kernel.
        resource->mWeightAsymmetricQuant = false;
    }
    resource->mInputZeroPoint = 0;
    resource->mOutputZeroPoint = 0;
    resource->mClampMin = -128;
    resource->mClampMax = 127;
    if (convParam->symmetricQuan()) {
        resource->mInputZeroPoint = convParam->symmetricQuan()->zeroPoint();
        resource->mOutputZeroPoint = convParam->symmetricQuan()->outputZeroPoint();
        resource->mClampMin = convParam->symmetricQuan()->clampMin();
        resource->mClampMax = convParam->symmetricQuan()->clampMax();
    }
    if (convParam->quanParameter() != nullptr) {
        resource->mInputScale = convParam->quanParameter()->scaleIn();
        resource->mOutputScale = convParam->quanParameter()->scaleOut();
    }
    auto weightDst = resource->mWeightInt8->host<int8_t>();
    memcpy(weightDst, weightSrc, resource->mWeightInt8->size());
    resource->mRelu = convCommon->relu() || convCommon->relu6();
    if (convParam->symmetricQuan() && convParam->symmetricQuan()->outputDataType() == MNN::DataType_DT_FLOAT) {
        resource->mOutputZeroPoint = 0;
        resource->mOutputScale = 1.0f;
    }
    return resource;
}

void CPUConvolution::makeResource(Backend* backend, std::shared_ptr<Resource> resource, const Convolution2D* conv2d, std::shared_ptr<ResourceInt8> resourceInt8) {
    /* Used to compute weight quant scale and bias and weightKernelSum of type float. */
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
    
    std::shared_ptr<ConvolutionCommon::Int8Common> quantCommon;
    // Load quant scale and bias
    if (quanBuffer) {
        quantCommon = ConvolutionCommon::load(conv2d, backend, false, true);
        weightOrigin = quantCommon->weight.get(); // weight before reorder

        int h = quantCommon->alpha.size();
        if (core->bytes == 2) {
            if (quantCommon->asymmetric) {
                std::unique_ptr<int16_t[]> tmp(new int16_t[h]);
                core->MNNFp32ToLowp(quantCommon->alpha.get(), tmp.get(), h);
                for (int i=0; i< h/2; ++i) {
                    reinterpret_cast<int16_t*>(alphaPtr)[i] = tmp[2 * i + 1];
                    reinterpret_cast<int16_t*>(biasPtr)[i] = tmp[2 * i];
                }
            } else {
                core->MNNFp32ToLowp(quantCommon->alpha.get(), reinterpret_cast<int16_t*>(alphaPtr), h);
            }
        } else {
            if (quantCommon->asymmetric) {
                h = h / 2;
                for (int i=0; i<h; ++i) {
                    alphaPtr[i] = quantCommon->alpha.get()[2 * i + 1];
                    biasPtr[i] = quantCommon->alpha.get()[2 * i];
                }
            } else {
                for (int i=0; i<h; ++i) {
                    alphaPtr[i] = quantCommon->alpha.get()[i];
                    biasPtr[i] = 0.f;
                }
            }
        }
    } else {
        weightOrigin = resourceInt8->mWeightInt8->host<int8_t>();
        auto wZero = resourceInt8->mWeightQuantZero->host<int32_t>(); // has packed to outputUp4
        auto wScale = resourceInt8->mOriginScale->host<float>();
        int h = ocUp4;
        if (core->bytes == 2) {
            std::unique_ptr<int16_t[]> tmp(new int16_t[h]);
            core->MNNFp32ToLowp(wScale, tmp.get(), h);
            for (int i=0; i< h; ++i) {
                reinterpret_cast<int16_t*>(alphaPtr)[i] = tmp[i];
                reinterpret_cast<int16_t*>(biasPtr)[i] = (-1.f) * wZero[i] * tmp[i];
            }
        } else {
            for (int i=0; i< h; ++i) {
                alphaPtr[i] = wScale[i];
                biasPtr[i] = (-1.f) * wZero[i] * wScale[i];
            }
        }
    }
    
    // Compute float weightKernelSum
    resource->mWeightKernelSum.reset(Tensor::createDevice<uint8_t>({ocUp4 * 4}));
    success = resource->backend->onAcquireBuffer(resource->mWeightKernelSum.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Alloc denquant mWeightKernelSum memory error\n");
        return;
    }
    auto weightKernelSum = resource->mWeightKernelSum->host<float>();
    for (int i = 0; i < outputCount; ++i) {
        int sum = 0;
        for (int j = 0; j < LSize; ++j) {
            sum = sum + static_cast<int>(weightOrigin[j + i * LSize]);
        }
        if(core->bytes == 2) {
            auto scale = reinterpret_cast<int16_t*>(alphaPtr)[i];
            auto bias = reinterpret_cast<int16_t*>(biasPtr)[i];
            weightKernelSum[i] = static_cast<float>(sum) * scale + LSize * bias;
        } else {
            auto scale = alphaPtr[i];
            auto bias = biasPtr[i];
            weightKernelSum[i] = static_cast<float>(sum) * scale + LSize * bias;
        }
    }
}

CPUConvolution::CPUConvolution(const Convolution2DCommon *convOp, Backend *b) : MNN::Execution(b), mCommon(convOp) {
    // Do nothing
}
std::vector<float> CPUConvolution::getPostParameters() const {
    std::vector<float> postParameters = {
        1.0f,
        1.0f,
        -std::numeric_limits<float>().max(),
        std::numeric_limits<float>().max(),
    };
    if (mCommon->relu()) {
        postParameters[2] = 0.0f;
    }
    if (mCommon->relu6()) {
        postParameters[2] = 0.0f;
        postParameters[3] = 6.0f;
    }
    return postParameters;
}

int CPUConvolution::reorderWeightSize(int depth, int outputCount, int kernelSize, int unitDepth, int unitOC) {
    return UP_DIV(outputCount, unitOC) * UP_DIV(depth, unitDepth) * kernelSize * unitDepth * unitOC;
}



ErrorCode CPUConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto pad = ConvolutionCommon::convolutionPad(input, output, mCommon);
    mPadY = pad.second;
    mPadX = pad.first;
    return NO_ERROR;
}

class ConvolutionFactory : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return ConvolutionFloatFactory::create(inputs, outputs, op, backend);
    }
};

class CPUConvInt8Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto convOp = op->main_as_Convolution2D();
#ifdef MNN_USE_ONEDNN
        return OneDNNConvInt8::create(backend, convOp, inputs, outputs);
#endif
        auto core = static_cast<CPUBackend*>(backend)->functions();
        auto res = CPUConvolution::makeResourceInt8(backend, convOp, core->pack);
#ifdef MNN_USE_SPARSE_COMPUTE
        if (static_cast<CPUBackend*>(backend)->functions()->pack == 4 && convOp->sparseParameter() && SparseConvInt8TiledExecutor::shouldUseSparse(convOp)) {
            return new SparseConvInt8TiledExecutor(backend, convOp, res);
        }
#endif
        if (ConvInt8Winograd::mustUse(convOp)) {
            return new ConvInt8Winograd(backend, convOp, res);
        }
        return new DenseConvInt8TiledExecutor(backend, convOp, res, false);
    }
};

REGISTER_CPU_OP_CREATOR(ConvolutionFactory, OpType_Convolution);
REGISTER_CPU_OP_CREATOR(CPUConvInt8Creator, OpType_ConvInt8);
} // namespace MNN
