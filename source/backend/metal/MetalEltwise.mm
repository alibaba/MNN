//
//  MetalEltwise.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalEltwise.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalEltwise::MetalEltwise(Backend *backend, EltwiseType type) : MetalExecution(backend) {
    auto metal   = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)metal->context();
    mConst             = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    NSString *kernel = nil;
    switch (type) {
        case EltwiseType_PROD:
            kernel = @"eltwise_prod";
            break;
        case EltwiseType_SUM:
            kernel = @"eltwise_add";
            break;
        case EltwiseType_MAXIMUM:
            kernel = @"eltwise_max";
            break;
        default:
            break;
    }
    mPipeline = [context pipelineWithName:kernel fp16:metal->useFp16InsteadFp32()];
}
ErrorCode MetalEltwise::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    ((int*)(mConst.contents))[0] = outputs[0]->elementSize();
    auto metal   = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)metal->context();

    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(outputs[0]->elementSize(), 1, 1)];
    return NO_ERROR;
}

void MetalEltwise::encode(const Tensor *input0, const Tensor *input1, const Tensor *output, id<MTLComputeCommandEncoder> encoder) {
    [encoder setComputePipelineState:mPipeline];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input0->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input0)->extra.offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input1->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input1)->extra.offset atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:2];
    [encoder setBuffer:mConst offset:0 atIndex:3];
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
}

void MetalEltwise::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto output = outputs[0];
    encode(inputs[0], inputs[1], output, encoder);
    for (int i = 2; i < inputs.size(); i++) {
        encode(inputs[i], output, output, encoder);
    }
}

class MetalEltwiseCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        auto eltwise = op->main_as_Eltwise();
        return new MetalEltwise(backend, eltwise->type());
    }
};
REGISTER_METAL_OP_CREATOR(MetalEltwiseCreator, OpType_Eltwise);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
