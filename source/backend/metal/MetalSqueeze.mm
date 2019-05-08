//
//  MetalSqueeze.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalSqueeze.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalSqueeze::MetalSqueeze(Backend *backend) : Execution(backend) {
    // nothing to do
}

ErrorCode MetalSqueeze::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];

    auto flt       = output->getType().code == halide_type_float;
    auto size      = flt ? output->elementSize() : output->size();
    auto encoder   = [context encoder];
    auto bandwidth = [context load:flt ? @"copy_float" : @"copy_byte" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) size, 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalSqueezeCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalSqueeze(backend);
    }
};
REGISTER_METAL_OP_CREATOR(MetalSqueezeCreator, OpType_Squeeze);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
