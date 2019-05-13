//
//  MetalPReLU.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalPReLU.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalPReLU::MetalPReLU(Backend *backend, const float *slope, int count) : Execution(backend) {
    auto context  = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    mSlope        = [context newDeviceBuffer:UP_DIV(count, 4) * 4 * sizeof(float) bytes:slope access:CPUWriteOnly];
    mShareChannel = 1 == count;
}

ErrorCode MetalPReLU::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    int w = output->width(), h = output->height(), z = UP_DIV(output->channel(), 4), b = output->batch();

    auto encoder   = [context encoder];
    auto bandwidth = [context load:mShareChannel ? @"prelu" : @"prelu_slopes" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mSlope offset:0 atIndex:2];
    if (mShareChannel) {
        [context dispatchEncoder:encoder
                         threads:{ (NSUInteger) w * h * z * b, (NSUInteger)1, (NSUInteger)1 }
                       bandwidth:bandwidth];
    } else {
        auto shape                 = [context newDeviceBuffer:3 * sizeof(int) access:CPUWriteOnly];
        ((int *)shape.contents)[0] = w * h;
        ((int *)shape.contents)[1] = z;
        ((int *)shape.contents)[2] = b;
        [encoder setBuffer:shape offset:0 atIndex:3];
        [context dispatchEncoder:encoder
                         threads:{ (NSUInteger) w * h, (NSUInteger)z, (NSUInteger)b }
                       bandwidth:bandwidth];
    }
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
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
