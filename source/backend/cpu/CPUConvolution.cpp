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

void CPUConvolution::ResourceInt8::updateInputOutputScale(std::vector<float> inputQuantInfo, std::vector<float> outputQuantInfo) {
    std::call_once(flag, [&](){
        // new scales and zero points
        float inputScale = inputQuantInfo[0];
        float outputScale = outputQuantInfo[0];
        float inputZeroPoint = inputQuantInfo[1];
        float outputZeroPoint = outputQuantInfo[1];

        if (inputScale == 0.f || outputScale == 0.f) {
            return;
        }
        if (mInputScale == inputScale && mOutputScale == outputScale) {
            return;
        }
        auto scalePtr = mScaleFloat->host<float>();
        auto biasPtr = mBiasInt32->host<int>();
        int size = mOutputCount;
        float is = mInputScale / inputScale;
        float os = mOutputScale / outputScale;

        const int kernelNum = mInt8WeightKernelSum.size();

        // compute remains used in asymmetric quant
        std::vector<int> remainsCorrection;
        for (int i = 0; i < kernelNum; i++) {
            int temp = (int(inputZeroPoint) - mInputZeroPoint) * mInt8WeightKernelSum[i];
            remainsCorrection.emplace_back(temp);
        }

        for (int i = kernelNum; i < size; i++) {
            remainsCorrection.emplace_back(0);
        }

        for (int i = 0; i < size; i++) {
            // compute outputZeroPointFused in asymmetric quant
            int correction1 = static_cast<int32_t>(mOutputZeroPoint / scalePtr[i]);
            scalePtr[i] = scalePtr[i] * os / is;
            int correction2 = static_cast<int32_t>(outputZeroPoint / scalePtr[i]);
            int outputZeroPointFusedCorrection = correction2 - correction1;
#ifdef MNN_USE_SSE
            if (offsets.empty()) {
                biasPtr[i] = biasPtr[i] - remainsCorrection[i] + outputZeroPointFusedCorrection;
                biasPtr[i] = static_cast<int32_t>(biasPtr[i] * is);
            } else {
                biasPtr[i] = biasPtr[i] - offsets[i];
                biasPtr[i] = biasPtr[i] - remainsCorrection[i] + outputZeroPointFusedCorrection;
                biasPtr[i] = static_cast<int32_t>(biasPtr[i] * is + offsets[i]);
            }
#else
            biasPtr[i] = biasPtr[i] - remainsCorrection[i] + outputZeroPointFusedCorrection;
            biasPtr[i] = static_cast<int32_t>(biasPtr[i] * is);
#endif
        }
        mInputScale = inputScale;
        mOutputScale = outputScale;
        mInputZeroPoint = int8_t(inputZeroPoint);
        mOutputZeroPoint = int8_t(outputZeroPoint);
        mClampMin = int8_t(outputQuantInfo[2]);
        mClampMax = int8_t(outputQuantInfo[3]);
    });
}
CPUConvolution::ResourceInt8::~ResourceInt8() {
    // Do nothing
}
std::shared_ptr<CPUConvolution::ResourceInt8> CPUConvolution::makeResourceInt8(Backend* backend, const MNN::Convolution2D *convParam,
                                                                               std::vector<float> inputQuantInfo, std::vector<float> outputQuantInfo) {
    if (inputQuantInfo.empty() && outputQuantInfo.empty()) {
        inputQuantInfo = {0.f, 0.f, -127, 127};
        outputQuantInfo = {0.f, 0.f, -127, 127};
    }
    auto core = static_cast<CPUBackend*>(backend)->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);

    std::shared_ptr<CPUConvolution::ResourceInt8> resource(new ResourceInt8);
    resource->backend = backend;
    auto inputScale = inputQuantInfo[0], outputScale = outputQuantInfo[0];
    // TODO: ConvInt8Winograd need in/out scale, which isn't exist in quantinfo when model construct by V3 API
    if (convParam->quanParameter() != nullptr) {
        if (inputScale == 0) {
            inputScale = convParam->quanParameter()->scaleIn();
        }
        if (outputScale == 0) {
            outputScale = convParam->quanParameter()->scaleOut();
        }
    }
    resource->mInputScale = inputScale;
    resource->mOutputScale = outputScale;
    const auto convCommon  = convParam->common();
    const auto group = convParam->common()->group();
    const auto outputCount = convCommon->outputCount();
    const auto outputChannleUp4 = UP_DIV(outputCount, UNIT) * UNIT;

    resource->mBiasInt32.reset(Tensor::createDevice<int32_t>({outputChannleUp4}));
    resource->mScaleFloat.reset(Tensor::createDevice<float>({outputChannleUp4}));
    auto allocRes = backend->onAcquireBuffer(resource->mBiasInt32.get(), Backend::STATIC);
    allocRes &= backend->onAcquireBuffer(resource->mScaleFloat.get(), Backend::STATIC);
    if (!allocRes) {
        return nullptr;
    }

    auto biasPtr = resource->mBiasInt32->host<int32_t>();
    memset(biasPtr, 0, outputChannleUp4 * sizeof(int32_t));
    auto scalePtr = resource->mScaleFloat->host<float>();
    memset(scalePtr, 0, outputChannleUp4 * sizeof(float));

    resource->mActBits = convParam->symmetricQuan()->nbits();
    const int8_t* weightSrc = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    resource->mOutputCount = outputCount;
    if (!ConvolutionCommon::getConvInt8Parameters(convParam, quanCommon, weightSrc, weightSize, scalePtr, biasPtr,
                                                  inputScale, outputScale,
                                                  convParam->symmetricQuan()->zeroPoint(),
                                                  convParam->symmetricQuan()->outputZeroPoint())) {
        return nullptr;
    }

    resource->mWeightInt8.reset(Tensor::createDevice<int8_t>({weightSize}));
    allocRes = backend->onAcquireBuffer(resource->mWeightInt8.get(), Backend::STATIC);
    if (!allocRes) {
        return nullptr;
    }
    const int kernelNum = outputCount;
    const int kernelSize = weightSize / kernelNum;
    for (int i = 0; i < kernelNum; i++) {
        int temp = 0;
        int offset = i * kernelSize;
        for (int j = 0; j < kernelSize; j++) {
            temp += int(weightSrc[offset + j]);
        }
        resource->mInt8WeightKernelSum.emplace_back(temp);
    }

#ifdef MNN_USE_SSE
    resource->offsets.resize(outputCount);
    // For SSE use uint8_t, int8_t -> uint8_t, x + 128 -> x', x * w + b = (x' - 128) * w + b = x' * w + (-128 * w) + b
    for (int x = 0; x < outputCount; ++x) {
        int offset = resource->mInt8WeightKernelSum[x] * (-128);
        resource->offsets[x] = offset;
        if (convParam->symmetricQuan()->winogradAttr() == nullptr) {
            biasPtr[x] = biasPtr[x] + offset;
        }
    }
#endif
    auto weightDst = resource->mWeightInt8->host<int8_t>();
    memcpy(weightDst, weightSrc, resource->mWeightInt8->size());
    resource->mInputZeroPoint = convParam->symmetricQuan()->zeroPoint();
    resource->mOutputZeroPoint = convParam->symmetricQuan()->outputZeroPoint();
    resource->mClampMin = convParam->symmetricQuan()->clampMin();
    resource->mClampMax = convParam->symmetricQuan()->clampMax();
    resource->mRelu = convCommon->relu() || convCommon->relu6();
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

template<typename T>
void CPUConvolution::reorderWeightSlow(T* dest, const T* source, size_t depth, size_t outputCount, size_t kernelSize,
                                       size_t unitDepth, size_t unitOC, bool transpose) {
    memset(dest, 0, reorderWeightSize(depth, outputCount, kernelSize, unitDepth, unitOC) * sizeof(T));
    for (int dz = 0; dz < outputCount; ++dz) {
        auto dz_unit = dz / unitOC;
        auto mx      = dz % unitOC;
        auto dst_dz = dest + dz_unit * UP_DIV(depth, unitDepth) * kernelSize * unitDepth * unitOC;
        for (int sz = 0; sz < depth; ++sz) {
            auto sz_unit = sz / unitDepth;
            auto my      = sz % unitDepth;
            auto dst_sz = dst_dz + sz_unit * kernelSize * unitDepth * unitOC;
            auto src    = source + kernelSize * (sz + dz * depth);
            for (int ki = 0; ki < kernelSize; ++ki) {
                auto dst_i         = dst_sz + ki * unitDepth * unitOC;
                if (transpose) {
                    dst_i[unitDepth * mx + my] = src[ki];
                } else {
                    dst_i[unitOC * my + mx] = src[ki];
                }
            }
        }
    }
}

template void CPUConvolution::reorderWeightSlow<int8_t>(int8_t*, const int8_t*, size_t, size_t, size_t, size_t, size_t, bool);


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
        std::vector<float> inputQuantInfo;
        std::vector<float> outputQuantInfo;
        if (inputs.size() > 0) {
            inputQuantInfo = TensorUtils::getQuantInfo(inputs[0]);
            outputQuantInfo = TensorUtils::getQuantInfo(outputs[0]);
        }
        auto convOp = op->main_as_Convolution2D();
#ifdef MNN_USE_ONEDNN
        return OneDNNConvInt8::create(backend, convOp, inputs, outputs);
#endif
        auto res = CPUConvolution::makeResourceInt8(backend, convOp, inputQuantInfo, outputQuantInfo);
#ifdef MNN_USE_SPARSE_COMPUTE
        auto core = static_cast<CPUBackend*>(backend)->int8Functions();
        if (static_cast<CPUBackend*>(backend)->functions()->pack == 4 && convOp->sparseParameter() && SparseConvInt8TiledExecutor::shouldUseSparse(convOp)) {
            return new SparseConvInt8TiledExecutor(backend, convOp, res);
        }
#endif
        if (ConvInt8Winograd::mustUse(convOp)) {
            return new ConvInt8Winograd(backend, convOp, res);
        }
        return new DenseConvInt8TiledExecutor(backend, convOp, res);
    }
};

REGISTER_CPU_OP_CREATOR(ConvolutionFactory, OpType_Convolution);
REGISTER_CPU_OP_CREATOR(CPUConvInt8Creator, OpType_ConvInt8);
} // namespace MNN
