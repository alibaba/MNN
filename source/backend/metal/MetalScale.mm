//
//  MetalScale.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalScale.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalScale::MetalScale(Backend *backend, const Scale *scale) : MetalExecution(backend) {
    auto context  = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    auto mtbn = static_cast<MetalBackend *>(backend);
    auto channel4 = UP_DIV(scale->channels(), 4) * 4;
    mScale = [context newDeviceBuffer:channel4 * sizeof(float) bytes:scale->scaleData()->data() access:CPUWriteOnly];
    mBias  = scale->biasData()
                ? [context newDeviceBuffer:channel4 * sizeof(float) bytes:scale->biasData()->data() access:CPUWriteOnly]
                : [context newDeviceBuffer:channel4 * sizeof(float) access:CPUTransparent];
    mConst = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    mPipeline = [context pipelineWithName:@"scale_ca" fp16:mtbn->useFp16InsteadFp32()];
}

ErrorCode MetalScale::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto output = outputs[0];

    // shape
    int w   = output->width();
    int h   = output->height();
    int c   = output->channel();
    int z   = UP_DIV(c, 4);
    ((int *)mConst.contents)[0] = w*h;
    ((int *)mConst.contents)[1] = z;
    ((int *)mConst.contents)[2] = output->batch();
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(w*h, z * outputs[0]->batch(), 1)];
    return NO_ERROR;
}

void MetalScale::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto input = inputs[0], output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:1];
    [encoder setBuffer:mConst offset:0 atIndex:2];
    [encoder setBuffer:mScale offset:0 atIndex:3];
    [encoder setBuffer:mBias offset:0 atIndex:4];
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];

}

class MetalScaleCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        return new MetalScale(backend, op->main_as_Scale());
    }
};
REGISTER_METAL_OP_CREATOR(MetalScaleCreator, OpType_Scale);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
