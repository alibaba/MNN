//
//  MetalGather.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalGather.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalGather::MetalGather(Backend *backend) : Execution(backend) {
    // nothing to do
}

ErrorCode MetalGather::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], indices = inputs[1], output = outputs[0];
    auto constants                 = [context newDeviceBuffer:3 * sizeof(int) access:CPUWriteOnly];
    ((int *)constants.contents)[0] = input->length(0);
    ((int *)constants.contents)[1] = input->stride(0);
    ((int *)constants.contents)[2] = indices->elementSize();

    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"gather" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)indices->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:2];
    [encoder setBuffer:constants offset:0 atIndex:3];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) indices->elementSize(), 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalGatherCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalGather(backend);
    }
};
REGISTER_METAL_OP_CREATOR(MetalGatherCreator, OpType_Gather);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
