//
//  MetalResize.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalResize.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalResize::MetalResize(Backend *backend, float xScale, float yScale)
    : Execution(backend), mXScale(xScale), mYScale(yScale) {
    // nothing to do
}

ErrorCode MetalResize::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    int iw = input->width(), ih = input->height();
    int ow = output->width(), oh = output->height(), slice = UP_DIV(output->channel(), 4) * output->batch();

    auto shape                 = [context newDeviceBuffer:6 * sizeof(int) access:CPUWriteOnly];
    ((int *)shape.contents)[0] = iw;
    ((int *)shape.contents)[1] = ih;
    ((int *)shape.contents)[2] = iw * ih;
    ((int *)shape.contents)[3] = ow;
    ((int *)shape.contents)[4] = oh;
    ((int *)shape.contents)[5] = ow * oh;

    auto scale                   = [context newDeviceBuffer:2 * sizeof(float) access:CPUWriteOnly];
    ((float *)scale.contents)[0] = 1.f / mXScale;
    ((float *)scale.contents)[1] = 1.f / mYScale;

    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"resize_bilinear" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:shape offset:0 atIndex:2];
    [encoder setBuffer:scale offset:0 atIndex:3];
    [context dispatchEncoder:encoder
                     threads:{ (NSUInteger) ow, (NSUInteger)oh, (NSUInteger)slice }
                   bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalResizeCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto resize = op->main_as_Resize();
        return new MetalResize(backend, resize->xScale(), resize->yScale());
    }
};
REGISTER_METAL_OP_CREATOR(MetalResizeCreator, OpType_Resize);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
