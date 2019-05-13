//
//  MetalEltwise.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalEltwise.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalEltwise::MetalEltwise(Backend *backend, EltwiseType type) : Execution(backend), mType(type) {
    // nothing to do
}

void MetalEltwise::encode(NSString *kernel, const Tensor *input0, const Tensor *input1, const Tensor *output) {
    auto metal   = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)metal->context();
    auto encoder   = [context encoder];
    auto bandwidth = [context load:kernel encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input0->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input1->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:2];
    [context dispatchEncoder:encoder threads:{ (NSUInteger)output->elementSize() / 4, 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
}

ErrorCode MetalEltwise::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    NSString *kernel = nil;
    switch (mType) {
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

    auto output = outputs[0];
    encode(kernel, inputs[0], inputs[1], output);
    for (int i = 2; i < inputs.size(); i++) {
        encode(kernel, inputs[i], output, output);
    }
    return NO_ERROR;
}

class MetalEltwiseCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto eltwise = op->main_as_Eltwise();
        return new MetalEltwise(backend, eltwise->type());
    }
};
REGISTER_METAL_OP_CREATOR(MetalEltwiseCreator, OpType_Eltwise);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
