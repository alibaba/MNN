//
//  MetalNormalize.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalNormalize.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalNormalize::MetalNormalize(Backend *backend, const Normalize *nor)
    : Execution(backend), mAcrossSpatial(nor->acrossSpatial()), mChannelShared(nor->channelShared()), mEps(nor->eps()) {
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    mScale =
        [context newDeviceBuffer:nor->scale()->size() * sizeof(float) bytes:nor->scale()->data() access:CPUWriteOnly];
}

ErrorCode MetalNormalize::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    int w = output->width(), h = output->height(), c = output->channel(), z = UP_DIV(c, 4), b = output->batch();

    auto constBuffer                   = [context newDeviceBuffer:4 * sizeof(int) + sizeof(float) access:CPUWriteOnly];
    ((int *)constBuffer.contents)[0]   = w * h;
    ((int *)constBuffer.contents)[1]   = c;
    ((int *)constBuffer.contents)[2]   = z * b;
    ((int *)constBuffer.contents)[3]   = mChannelShared;
    ((float *)constBuffer.contents)[4] = mEps;

    auto encoder = [context encoder];
    auto bandwidth =
        [context load:mAcrossSpatial ? @"normalize_across_spatial" : @"normalize_across_channel" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mScale offset:0 atIndex:2];
    [encoder setBuffer:constBuffer offset:0 atIndex:3];
    if (mAcrossSpatial) {
        [context dispatchEncoder:encoder threads:{ 1, 1, 1 } bandwidth:bandwidth];
    } else {
        [context dispatchEncoder:encoder threads:{ (NSUInteger)w * h, 1, 1 } bandwidth:bandwidth];
    }
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);

    return NO_ERROR;
}

class MetalNormalizeCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalNormalize(backend, op->main_as_Normalize());
    }
};
REGISTER_METAL_OP_CREATOR(MetalNormalizeCreator, OpType_Normalize);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
