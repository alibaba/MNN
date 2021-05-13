//
//  MetalReLU.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalReLU.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalReLU::MetalReLU(Backend *backend, float slope) : Execution(backend) {
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    mSlope       = [context newDeviceBuffer:sizeof(float) bytes:&slope access:CPUWriteOnly];
}

ErrorCode MetalReLU::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    
    if(backend->isCommandEncoderSet()) {
        return NO_ERROR;
    }

    auto func = [=](){
        auto input = inputs[0], output = outputs[0];
        NSUInteger size = output->elementSize();
        auto simd       = size % 4 == 0;
        if (simd) {
            size /= 4;
        }

        MNN_ASSERT(mSlope.length == sizeof(float));
        auto encoder   = backend->encoder();
        auto bandwidth = [context load:simd ? @"relu_x4" : @"relu_x1" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:mSlope offset:0 atIndex:2];
        [context dispatchEncoder:encoder threads:{ size, 1, 1 } bandwidth:bandwidth];
        MNN_PRINT_ENCODER(context, encoder);

        auto context = (__bridge MNNMetalContext *)backend->context();
        if(context.isCommitEachShader) {
            backend->flushEncoder();
            [context commit_net];
        }
    };
    func();
    backend->addOpEncoder(func);
    return NO_ERROR;
}

class MetalReLUCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalReLU(backend, op->main_as_Relu()->slope());
    }
};
REGISTER_METAL_OP_CREATOR(MetalReLUCreator, OpType_ReLU);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
