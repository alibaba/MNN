//
//  MetalPReLU.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalPReLU.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalPReLU::MetalPReLU(Backend *backend, const float *slope, int count) : Execution(backend) {
    auto context  = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    mSlope        = [context newDeviceBuffer:UP_DIV(count, 4) * 4 * sizeof(float) bytes:slope access:CPUWriteOnly];
    mShareChannel = 1 == count;
    if (!mShareChannel) {
        mShape = [context newDeviceBuffer:3 * sizeof(int) access:CPUWriteOnly];
    }
    mPipeline = [context pipelineWithName:mShareChannel ? @"prelu" : @"prelu_slopes"];
}

ErrorCode MetalPReLU::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto output = outputs[0];
    int w = output->width(), h = output->height(), z = UP_DIV(output->channel(), 4), b = output->batch();
    if (mShareChannel) {
        mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(w * h * z * b, 1, 1)];
    } else {
        ((int *)mShape.contents)[0] = w * h;
        ((int *)mShape.contents)[1] = z;
        ((int *)mShape.contents)[2] = b;
        mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(w * h, z, b)];
    }
    return NO_ERROR;
}

ErrorCode MetalPReLU::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    
    if(backend->isCommandEncoderSet()) {
        return NO_ERROR;
    }
    
    auto func = [=](){
        auto input = inputs[0], output = outputs[0];

        auto encoder   = backend->encoder();
        [encoder setComputePipelineState:mPipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:mSlope offset:0 atIndex:2];
        if (!mShareChannel) {
            [encoder setBuffer:mShape offset:0 atIndex:3];
        }
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
        
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

class MetalPReLUCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto prelu = op->main_as_PRelu();
        return new MetalPReLU(backend, prelu->slope()->data(), prelu->slopeCount());
    }
};
REGISTER_METAL_OP_CREATOR(MetalPReLUCreator, OpType_PReLU);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
