//
//  MetalGatherV2.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalGatherV2.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalGatherV2::MetalGatherV2(Backend *backend, DataType type) : Execution(backend), mType(type) {
    // nothing to do
}

ErrorCode MetalGatherV2::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], indices = inputs[1], output = outputs[0];

    // create dims
    auto dims                 = [context newDeviceBuffer:sizeof(int) * (input->dimensions() + 1) access:CPUWriteOnly];
    ((int *)dims.contents)[0] = input->dimensions();
    for (int i = 0; i < input->dimensions(); i++) {
        ((int *)dims.contents)[i + 1] = input->length(i);
    }

    // get axis ready
    id<MTLBuffer> axis = nil;
    if (inputs.size() > 2) {
        axis = (__bridge id<MTLBuffer>)(void *)inputs[2]->deviceId();
    } else {
        int defaults = 0;
        axis         = [context newDeviceBuffer:sizeof(defaults) bytes:&defaults access:CPUWriteOnly];
    }

    // choose kernel
    NSString *kernel = nil;
    if (mType == DataType_DT_INT32) {
        kernel = @"gatherv2_int32";
    } else if (mType == DataType_DT_FLOAT) {
        kernel = @"gatherv2_float";
    } else {
        return NOT_SUPPORT;
    }

    // encode
    auto encoder   = [context encoder];
    auto bandwidth = [context load:kernel encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:dims offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)indices->deviceId() offset:0 atIndex:2];
    [encoder setBuffer:axis offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:4];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) indices->elementSize(), 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalGatherV2Creator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalGatherV2(backend, op->main_as_GatherV2()->Tparams());
    }
};
REGISTER_METAL_OP_CREATOR(MetalGatherV2Creator, OpType_GatherV2);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
