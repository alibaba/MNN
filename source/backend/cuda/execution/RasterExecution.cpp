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

    mFuseRaster.first = false;
    if(des->regions.size() > 1) {
        mFuseRaster.first = true;
        mFuseRaster.second = des->regions.size();
        auto& slice0 = des->regions[0];
        for (int i = 1; i < des->regions.size(); ++i) {
            auto& slice = des->regions[i];
            if (slice0.origin->deviceId() != slice.origin->deviceId()) {
                mFuseRaster.first = false;
                break;
            }
            if (slice0.src.stride[0] != slice.src.stride[0] || slice0.dst.stride[0] != slice.dst.stride[0]) {
                mFuseRaster.first = false;
                break;
            }
            if (slice0.src.stride[1] != slice.src.stride[1] || slice0.dst.stride[1] != slice.dst.stride[1]) {
                mFuseRaster.first = false;
                break;
            }      
            if (slice0.src.stride[2] != slice.src.stride[2] || slice0.dst.stride[2] != slice.dst.stride[2]) {
                mFuseRaster.first = false;
                break;
            }
            if (slice0.size[0] != slice.size[0] || slice0.size[1] != slice.size[1] || slice0.size[2] != slice.size[2]) {
                mFuseRaster.first = false;
                break;
            }
        }
    }
    //mFuseRaster.first = false;
    if(!mFuseRaster.first) {
        for (int i = 0; i < des->regions.size(); ++i) {
            auto& slice = des->regions[i];
            if (nullptr == slice.origin) {
                continue;
            }
            mTempInputCopy.emplace_back(std::make_pair((void*)slice.origin->deviceId(), &slice));
        }
    } else {
        auto& slice0 = des->regions[0];
        if (nullptr != slice0.origin) {
            mTempInputCopy.emplace_back(std::make_pair((void*)slice0.origin->deviceId(), &slice0));
        }

        int regionSize = des->regions.size();
        std::vector<int32_t> temp(2*regionSize, 0);
        for (int i = 0; i < regionSize; ++i) {
            auto& slice = des->regions[i];
            temp[i] = slice.src.offset;
            temp[regionSize+i] = slice.dst.offset;
            //printf("%d-", tmpSrc[i]);
        }
        //save srcOffset/dstOffset to Device
        offsetTensor.reset(Tensor::createDevice<int32_t>({2*regionSize}));
        backend()->onAcquireBuffer(offsetTensor.get(), Backend::STATIC);
        mOffset = (void *)offsetTensor.get()->buffer().device;
        cuda_check(cudaMemcpy(mOffset, temp.data(), 2*regionSize*sizeof(int32_t), cudaMemcpyHostToDevice));
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
    if(mFuseRaster.first) {
        MNN_ASSERT(mTempInputCopy.size() == 1);
        auto& iter  = mTempInputCopy[0];
        auto& slice = *(iter.second);
        auto srcPtr = (uint8_t*)iter.first;
        auto dstPtr = (uint8_t*)output->deviceId();
        //printf("fuseRaster:%p-%p\n", mSrcOffset, mDstOffset);

        FuseRasterBlit(dstPtr, srcPtr, slice.size, slice.src.stride, slice.dst.stride, mFuseRaster.second, mOffset, bytes, runtime);
        return NO_ERROR;
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