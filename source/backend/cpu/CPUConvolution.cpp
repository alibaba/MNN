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
#include <limits>
#include "backend/cpu/compute/ConvolutionFloatFactory.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "core/ConvolutionCommon.hpp"

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
template void CPUConvolution::reorderWeightSlow<int16_t>(int16_t*, const int16_t*, size_t, size_t, size_t, size_t, size_t, bool); // FLOAT16(__fp16) is not available here, so use int16_t (2 byte also)

template<typename T, typename U> // T -> U
bool CPUConvolution::acquireMemoryAndCopy(std::shared_ptr<Tensor> dest, const T* source, size_t count, Backend* backend) {
    bool allocRes = ((CPUBackend*)backend)->onAcquireBuffer(dest.get(), Backend::STATIC);
    if (!allocRes) {
        return false;
    }
    auto dataPtr = dest->host<U>();
    memset(dataPtr, 0, dest->size());
    for (int i = 0; i < count; ++i) {
        dataPtr[i] = source[i]; // type cast T -> U elementwise
    }
    return true;
}

template bool CPUConvolution::acquireMemoryAndCopy<int32_t, float>(std::shared_ptr<Tensor>, const int32_t*, size_t, Backend*);
template bool CPUConvolution::acquireMemoryAndCopy<float, float>(std::shared_ptr<Tensor>, const float*, size_t, Backend*);


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

REGISTER_CPU_OP_CREATOR(ConvolutionFactory, OpType_Convolution);
} // namespace MNN
