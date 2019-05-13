//
//  MetalRange.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalRange.hpp"
#import "MNNMetalContext.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalRange::MetalRange(Backend *backend, DataType type) : Execution(backend), mType(type) {
    // nothing to do
}

ErrorCode MetalRange::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto start = inputs[0], delta = inputs[2], flat = outputs[0];

    auto encoder   = [context encoder];
    auto bandwidth = (MetalBandwidth){};
    switch (mType) {
        case DataType_DT_INT32:
            bandwidth = [context load:@"range_int32" encoder:encoder];
            break;
        case DataType_DT_FLOAT:
            bandwidth = [context load:@"range_float" encoder:encoder];
            break;
        case DataType_DT_INT64:
        case DataType_DT_DOUBLE:
        default:
            return NOT_SUPPORT;
    }
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)start->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)delta->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)flat->deviceId() offset:0 atIndex:2];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) flat->elementSize(), 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalRangeCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalRange(backend, op->main_as_Range()->Tidx());
    }
};
REGISTER_METAL_OP_CREATOR(MetalRangeCreator, OpType_Range);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
