//
//  MetalPooling.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalPooling.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalPooling::MetalPooling(Backend *backend, const Pool *pooling)
    : MetalExecution(backend),
      mGlobal(pooling->isGlobal()),
      mPoolType(pooling->type()),
      mKernelX(pooling->kernelX()),
      mKernelY(pooling->kernelY()),
      mStrideX(pooling->strideX()),
      mStrideY(pooling->strideY()),
      mPadX(pooling->padX()),
      mPadY(pooling->padY()) {
    // nothing to do
}

ErrorCode MetalPooling::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];

    int strideWidth  = mStrideX;
    int strideHeight = mStrideY;
    int padWidth     = mPadX;
    int padHeight    = mPadY;
    int kernelWidth  = (int)MIN(mKernelX, input->width());
    int kernelHeight = (int)MIN(mKernelY, input->height());
    if (mGlobal) {
        kernelWidth  = (int)input->width();
        kernelHeight = (int)input->height();
        strideWidth  = (int)input->width();
        strideHeight = (int)input->height();
        padWidth     = 0;
        padHeight    = 0;
    }

    mConstBuffer                       = [context newDeviceBuffer:11 * sizeof(int) access:CPUWriteOnly];
    ((int *)mConstBuffer.contents)[0]  = input->width();
    ((int *)mConstBuffer.contents)[1]  = input->height();
    ((int *)mConstBuffer.contents)[2]  = output->width();
    ((int *)mConstBuffer.contents)[3]  = output->height();
    ((int *)mConstBuffer.contents)[4]  = UP_DIV(output->channel(), 4) * output->batch();
    ((int *)mConstBuffer.contents)[5]  = kernelWidth;
    ((int *)mConstBuffer.contents)[6]  = kernelHeight;
    ((int *)mConstBuffer.contents)[7]  = strideWidth;
    ((int *)mConstBuffer.contents)[8]  = strideHeight;
    ((int *)mConstBuffer.contents)[9]  = padWidth;
    ((int *)mConstBuffer.contents)[10] = padHeight;
    auto ow = output->width(), oh = output->height(), slice = UP_DIV(output->channel(), 4) * output->batch();
    mPipeline = [context pipelineWithName:(mPoolType == PoolType_MAXPOOL) ? @"pooling_max" : @"pooling_avg" fp16:backend->useFp16InsteadFp32()];
    auto size = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(ow, oh, slice)];
    mLocal = size.second;
    mGroup = size.first;
    return NO_ERROR;
}

void MetalPooling::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto input = inputs[0], output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:1];
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    [encoder dispatchThreadgroups:mGroup threadsPerThreadgroup:mLocal];
}

class MetalPoolingCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        return new MetalPooling(backend, op->main_as_Pool());
    }
};
REGISTER_METAL_OP_CREATOR(MetalPoolingCreator, OpType_Pooling);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
