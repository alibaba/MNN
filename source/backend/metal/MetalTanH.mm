//
//  MetalTanH.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalTanH.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalTanH::MetalTanH(Backend *backend) : Execution(backend) {
    // nothing to do
}

ErrorCode MetalTanH::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    NSUInteger size = output->elementSize();
    auto simd       = size % 4 == 0;
    if (simd) {
        size /= 4;
    }

    auto encoder   = [context encoder];
    auto bandwidth = [context load:simd ? @"tanh_x4" : @"tanh_x1" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [context dispatchEncoder:encoder threads:{ size, 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalTanHCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalTanH(backend);
    }
};
REGISTER_METAL_OP_CREATOR(MetalTanHCreator, OpType_TanH);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
