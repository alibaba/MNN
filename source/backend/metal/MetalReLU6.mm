//
//  MetalReLU6.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalReLU6.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalReLU6::MetalReLU6(Backend *backend, float minV, float maxV) : Execution(backend) {
    auto metal   = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)metal->context();
    mConst             = [context newDeviceBuffer:4 * sizeof(float) access:CPUWriteOnly];
    ((float*)mConst.contents)[0] = minV;
    ((float*)mConst.contents)[1] = maxV;
}
ErrorCode MetalReLU6::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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

        auto encoder   = backend->encoder();
        auto bandwidth = [context load:simd ? @"relu6_x4" : @"relu6_x1" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:mConst offset:0 atIndex:2];
        [context dispatchEncoder:encoder threads:{ size, 1, 1 } bandwidth:bandwidth];

        if(context.isCommitEachShader) {
            backend->flushEncoder();
            [context commit_net];
        }
    };
    func();
    backend->addOpEncoder(func);
    return NO_ERROR;
}

class MetalReLU6Creator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        float minV = 0.0f;
        float maxV = 6.0f;
        if (nullptr != op->main()) {
            auto p = op->main_as_Relu6();
            minV = p->minValue();
            maxV = p->maxValue();
        }
        return new MetalReLU6(backend, minV, maxV);
    }
};
REGISTER_METAL_OP_CREATOR(MetalReLU6Creator, OpType_ReLU6);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
