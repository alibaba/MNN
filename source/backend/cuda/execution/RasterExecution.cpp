//
//  RasterExecution.cpp
//  MNN
//
//  Created by MNN on 2020/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "RasterExecution.hpp"
#include "Raster.cuh"
#include "core/Concurrency.h"
#include "core/OpCommonUtils.hpp"
namespace MNN {
namespace CUDA {

ErrorCode RasterExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input     = inputs[0];
    auto output    = outputs[0];
    auto des       = TensorUtils::getDescribe(input);
    auto outputDes = TensorUtils::getDescribe(output);
    mNeedZero      = !TensorUtils::regionIsFull(input);
    mTempInputCopy.clear();
    for (int i = 0; i < des->regions.size(); ++i) {
        auto& slice = des->regions[i];
        if (nullptr == slice.origin) {
            continue;
        }
        mTempInputCopy.emplace_back(std::make_pair((void*)slice.origin->deviceId(), &slice));
    }
    return NO_ERROR;
}

ErrorCode RasterExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input   = inputs[0];
    auto output  = outputs[0];
    auto bytes   = input->getType().bytes();
    if (mNeedZero) {
        runtime->memset((void*)output->deviceId(), 0, output->size());
    }
    for (int u = 0; u < mTempInputCopy.size(); ++u) {
        auto& iter  = mTempInputCopy[u];
        auto& slice = *(iter.second);
        auto srcPtr = (uint8_t*)iter.first + slice.src.offset * bytes;
        auto dstPtr = (uint8_t*)output->deviceId() + slice.dst.offset * bytes;
        RasterBlit(dstPtr, srcPtr, slice.size, slice.src.stride, slice.dst.stride, bytes, runtime);
    }
    return NO_ERROR;
}

RasterExecution::RasterExecution(Backend* backend) : Execution(backend) {
    // Do nothing
}
RasterExecution::~RasterExecution() {
    // Do nothing
}
class RasterCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new RasterExecution(backend);
    }
};

static CUDACreatorRegister<RasterCreator> __init(OpType_Raster);
} // namespace CUDA
} // namespace MNN