//
//  MetalROIPooling.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalROIPooling.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalROIPooling::MetalROIPooling(Backend *backend, float spatialScale)
    : Execution(backend), mSpatialScale(spatialScale) {
    // nothing to do
}

ErrorCode MetalROIPooling::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], roi = inputs[1], output = outputs[0];
    int iw = input->width(), ih = input->height();
    int ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4), ob = output->batch();

    auto shape                   = [context newDeviceBuffer:7 * sizeof(int) + sizeof(float) access:CPUWriteOnly];
    ((int *)shape.contents)[0]   = iw;
    ((int *)shape.contents)[1]   = ih;
    ((int *)shape.contents)[2]   = iw * ih;
    ((int *)shape.contents)[3]   = ow;
    ((int *)shape.contents)[4]   = oh;
    ((int *)shape.contents)[5]   = ow * oh;
    ((int *)shape.contents)[6]   = oz;
    ((float *)shape.contents)[7] = mSpatialScale;

    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"ROI_pooling" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)roi->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:2];
    [encoder setBuffer:shape offset:0 atIndex:3];
    [context dispatchEncoder:encoder
                     threads:{ (NSUInteger) ow, (NSUInteger)oh, (NSUInteger)oz *ob }
                   bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalROIPoolingCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalROIPooling(backend, op->main_as_RoiPooling()->spatialScale());
    }
};
REGISTER_METAL_OP_CREATOR(MetalROIPoolingCreator, OpType_ROIPooling);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
