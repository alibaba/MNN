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

static void unpackScaleFromBuffer(float* scaleBuffer, const int8_t* srcbuffer, const int32_t* info, int infoBytes) {
    int blockNum    = info[0];
    int ocDiv       = info[1]; // ocDiv = UP_DIV(oc, UNIT)
    int stride1     = info[2]; // int stride1 = blockL * UNIT * SRC_UNIT; // Int8 weight size per block
    int UNIT        = info[3];


    size_t packedUnitSize = stride1 + 2 * UNIT * infoBytes;

    int8_t* scaleWritePtr = reinterpret_cast<int8_t*>(scaleBuffer);

    for (int hU = 0; hU < ocDiv; ++hU) {
        const int8_t* huPtr = srcbuffer + hU * blockNum * packedUnitSize;
        for (int bl = 0; bl < blockNum; ++bl) {
            const int8_t* blockPtr = huPtr + bl * packedUnitSize;
            const int8_t* scaleReadPtr = blockPtr + stride1;
            memcpy(scaleWritePtr, scaleReadPtr, UNIT * infoBytes);
            scaleWritePtr += UNIT * infoBytes;
        }
    }
}

void CPUConvolution::Resource::copyBias(float* dst, const float* bias, int outputCount, Backend* backend) {
    auto core = static_cast<CPUBackend*>(backend)->functions();
    int bytes = core->bytes;
    int unit = core->pack;
    auto alignOutput = UP_DIV(outputCount, unit) * unit;
    int remain = alignOutput - outputCount;
    if (bytes < 4) {
        core->MNNFp32ToLowp(bias, (int16_t*)dst, outputCount);
    } else {
        ::memcpy(dst, bias, outputCount * bytes);
    }
    if (remain > 0) {
        ::memset((uint8_t*)dst + outputCount * bytes, 0, remain * bytes);
    }
}

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
    copyBias(mBias->host<float>(), bias, outputCount, backend);
    return true;
}
CPUConvolution::MutableResourceInt8::MutableResourceInt8(std::shared_ptr<ResourceInt8> res, Backend* backend, float* scalePtr) : mResource(res) {
    auto outputChannelUp4 = res->mOriginBias->length(0); // outputChannelUp4 = ROUND_UP(oc, pack)
    const int ocUpHp = (int)(res->mWeightKernelSum->length(0) / res->mBlockNum / sizeof(float));
    mBiasFloat.reset(Tensor::createDevice<int32_t>({outputChannelUp4}));
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
        auto weightScale = scalePtr;

        auto blockNum = res->mBlockNum;
        AutoStorage<int8_t> tmpBuffer(ocUpHp * blockNum * 4);
        if (!tmpBuffer.get()) {
             MNN_ERROR("Memory not enough for allocating a temp buffer for weight scale\n");
             return;
         }
        if (!scalePtr && !res->mOriginScale) { // if convolution, res->mOriginScale == nullptr
            // get weight scale from packed res->mWeightInt8
             int UNIT, SRC_UNIT, DST_XUNIT;
             auto int8Core = static_cast<CPUBackend*>(backend)->int8Functions();
             int8Core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
             int32_t perBlockWeightSize = (res->mWeightInt8->size() - 2 * ocUpHp * sizeof(float)) / (blockNum * UP_DIV(ocUpHp, UNIT));
             int32_t info[4] = {blockNum, UP_DIV(ocUpHp, UNIT), perBlockWeightSize, UNIT};
             unpackScaleFromBuffer((float*)tmpBuffer.get(), res->mWeightInt8->host<int8_t>(), info, 4);

            weightScale  = (float*)tmpBuffer.get();
        } else if (!scalePtr) { // if depthwiseInt8, res->mOriginScale != nullptr
            weightScale = res->mOriginScale->host<float>();
        }
        for (int i = 0; i < outputChannelUp4; ++i) {
            if (mInputScale && mOutputScale) { // symmetric quan
                floatBiasPtr[i] = int32BiasPtr[i] * weightScale[i] * mInputScale / mOutputScale;
            } else {
                floatBiasPtr[i] = int32BiasPtr[i] * weightScale[i];
            }
        }
        return;
    }
    mBiasInt32.reset(Tensor::createDevice<int32_t>({outputChannelUp4}));
    mScaleFloat.reset(Tensor::createDevice<int32_t>({outputChannelUp4}));
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
    if (inputScale != 0 && outputScale != 0) {
        mInputScale = inputScale;
        mOutputScale = outputScale;
        mInputZeroPoint = int8_t(inputZeroPoint);
        mOutputZeroPoint = int8_t(outputZeroPoint);
    }
    if (mInputScale == 0 || mOutputScale == 0) {
        return;
    }

    const int ocUp4 = mResource->mOriginBias->length(0);
    auto biasData    = mResource->mOriginBias->host<float>();
    auto scaleDiv  = mInputScale / mOutputScale;
    auto scale = mScaleFloat->host<float>();
    auto bias = mBiasInt32->host<int32_t>();
    auto biasfloat = mBiasFloat->host<float>();
#ifdef MNN_USE_SSE
    float offset = 128.f;
#else
    float offset = 0.f;
#endif
    if (mResource->mOriginScale) { // Only depthwiseInt8 has mOriginScale
        auto weightScalePtr   = mResource->mOriginScale->host<float>();
        for (int i = 0; i < ocUp4; i++) {
            auto weightScale = weightScalePtr[i];
            if (fabs(weightScale) < 1e-6) {
                weightScale = 1e-6;
            }
            scale[i] = weightScale * scaleDiv; // input_scale*weight_scale/output_scale
            // compute outputZeroPointFused in asymmetric quant
            int outputZeroPointFused = static_cast<int32_t>(mOutputZeroPoint / scale[i]);
            bias[i] = static_cast<int32_t>(biasData[i] / (mInputScale * weightScale)) - mResource->mInt8WeightKernelSum[i] * (mInputZeroPoint + offset) + outputZeroPointFused;
        }
    } else {
        auto outputScale = mResource->mWeightBits == 4 ? 1.f : mOutputScale;
        int32_t outputZero = mResource->mWeightBits == 4 ? 0 : mOutputZeroPoint;
        for (int i = 0; i < ocUp4; ++i) {
            biasfloat[i] = (biasData[i] - mResource->mWeightKernelSum->host<float>()[i] * (mInputZeroPoint + offset) * mInputScale) / outputScale + outputZero;

        }
    }
}
std::shared_ptr<CPUConvolution::ResourceInt8> CPUConvolution::makeResourceInt8(Backend* backend, const MNN::Op* op, int pack) {
    auto convParam = op->main_as_Convolution2D();
    auto core = static_cast<CPUBackend*>(backend)->functions();
    // TODO: use different pack from float
    int UNIT = pack;

    std::shared_ptr<CPUConvolution::ResourceInt8> resource(new ResourceInt8);
    // TODO: ConvInt8Winograd need in/out scale, which isn't exist in quantinfo when model construct by V3 API
    const auto convCommon  = convParam->common();
    const auto group = convParam->common()->group();
    const auto outputCount = convCommon->outputCount();
    const auto ocUpUnit = UP_DIV(outputCount, UNIT) * UNIT;

    resource->mOriginBias.reset(Tensor::createDevice<int32_t>({ocUpUnit}));
    resource->mOriginScale.reset(Tensor::createDevice<uint8_t>({2 * ocUpUnit * core->bytes})); // TO DO: blockNum>1
    resource->mWeightKernelSum.reset(Tensor::createDevice<uint8_t>({ocUpUnit * 4}));
    auto allocRes = backend->onAcquireBuffer(resource->mOriginBias.get(), Backend::STATIC);
    allocRes &= backend->onAcquireBuffer(resource->mOriginScale.get(), Backend::STATIC);
    allocRes &= backend->onAcquireBuffer(resource->mWeightKernelSum.get(), Backend::STATIC);
    if (!allocRes) {
        return nullptr;
    }

    auto biasPtr = resource->mOriginBias->host<int32_t>();
    memset(biasPtr, 0, ocUpUnit * sizeof(int32_t));
    auto scalePtr = resource->mOriginScale->host<float>();
    memset(scalePtr, 0, ocUpUnit * 2 * sizeof(float));

    resource->mWeightBits = 8;
    if (convParam->symmetricQuan()) {
        resource->mWeightBits = convParam->symmetricQuan()->nbits();
    }
    const int8_t* weightSrc = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    if (!ConvolutionCommon::getConvInt8Parameters(op, quanCommon, backend, weightSrc, weightSize, scalePtr, biasPtr, ocUpUnit)) {
        return nullptr;
    }
    if (convParam->bias() && (convParam->quanParameter()->alpha() || quanCommon->alpha.get())) {
        resource->mUseConvQuan = false;
    }
    if (quanCommon.get()) {
        resource->mWeightAsymmetricQuant = quanCommon->asymmetric;
    }

    // TODO: first alloc.
    resource->mWeightInt8.reset(Tensor::createDevice<int8_t>({weightSize}));
    allocRes = backend->onAcquireBuffer(resource->mWeightInt8.get(), Backend::STATIC);
    if (!allocRes) {
        return nullptr;
    }
    const int kernelNum = outputCount;
    const int kernelSize = weightSize / kernelNum;
    resource->mInt8WeightKernelSum.resize(ocUpUnit); // for cpu, only used by depthwiseInt8
    auto weightBiasPtr = scalePtr + ocUpUnit;
    for (int i = 0; i < kernelNum; i++) {
        int temp = 0;
        int offset = i * kernelSize;
        for (int j = 0; j < kernelSize; j++) {
            temp += static_cast<int>(weightSrc[offset + j]);
        }
        resource->mInt8WeightKernelSum[i] = (temp + kernelSize * (weightBiasPtr[i] / scalePtr[i]));
#ifdef MNN_USE_SSE
        if (resource->mUseConvQuan) {
            resource->mOriginBias->host<int32_t>()[i] -= 128 * temp;
        }
#endif
    }
    ConvInt8TiledExecutor::initializeConvInt8QuantInfo(resource, convParam, quanCommon);
    auto weightDst = resource->mWeightInt8->host<int8_t>();
    memcpy(weightDst, weightSrc, resource->mWeightInt8->size());
    return resource;
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
        return OneDNNConvInt8::create(backend, op, inputs, outputs);
#endif
        auto core = static_cast<CPUBackend*>(backend)->functions();

#ifdef MNN_USE_SPARSE_COMPUTE
        if (static_cast<CPUBackend*>(backend)->functions()->pack == 4 && convOp->sparseParameter() && SparseConvInt8TiledExecutor::shouldUseSparse(convOp)) {
            auto res = CPUConvolution::makeResourceInt8(backend, op, core->pack);
            return new SparseConvInt8TiledExecutor(backend, op, res);
        }
#endif
#ifndef MNN_REDUCE_SIZE
        if (ConvInt8Winograd::mustUse(convOp)) {
            auto res = CPUConvolution::makeResourceInt8(backend, op, core->pack);
            return new ConvInt8Winograd(backend, convOp, res);
        }
#endif
        std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
        if (convOp->quanParameter() && (convOp->quanParameter()->buffer() || convOp->external())) { // int8 weight
            quanCommon = ConvolutionCommon::load(op, backend, false, true);

        }
        // auto res = CPUConvolution::makeResourceInt8(backend, op, core->pack);
        // return new DenseConvInt8TiledExecutor(backend, op, res);

       return new DenseConvInt8TiledExecutor(backend, op, quanCommon, false);
    }
};

REGISTER_CPU_OP_CREATOR(ConvolutionFactory, OpType_Convolution);
REGISTER_CPU_OP_CREATOR(CPUConvInt8Creator, OpType_ConvInt8);
} // namespace MNN
