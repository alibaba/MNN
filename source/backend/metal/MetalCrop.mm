//
//  MetalCrop.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalCrop.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalCrop::MetalCrop(Backend *backend, const Crop *crop) : Execution(backend), mAxis(crop->axis()) {
    auto length = crop->offset()->size();
    mOffsetY    = mAxis <= 2 ? crop->offset()->data()[MIN(2 - mAxis, length - 1)] : 0;
    mOffsetX    = mAxis <= 3 ? crop->offset()->data()[0] : 0;
    MNN_ASSERT(mAxis >= 2); // crop width or height only
}

ErrorCode MetalCrop::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // axis <  crop, output.extent = input.extent. ==> offset = 0.
    // axis >= crop, output.extent = crop.extent.  ==> input.extent >= offset + crop.extent
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    auto iw = input->width(), ih = input->height();
    auto ow = output->width(), oh = output->height();
    auto z = UP_DIV(output->channel(), 4), b = output->batch();

    auto shape                 = [context newDeviceBuffer:7 * sizeof(int) access:CPUWriteOnly];
    ((int *)shape.contents)[0] = ow;
    ((int *)shape.contents)[1] = oh;
    ((int *)shape.contents)[2] = ow * oh;
    ((int *)shape.contents)[3] = iw;
    ((int *)shape.contents)[4] = iw * ih;
    ((int *)shape.contents)[5] = mOffsetX;
    ((int *)shape.contents)[6] = mOffsetY;

    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"crop" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:shape offset:0 atIndex:2];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) ow, (NSUInteger)oh, (NSUInteger)z *b } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalCropCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalCrop(backend, op->main_as_Crop());
    }
};
REGISTER_METAL_OP_CREATOR(MetalCropCreator, OpType_Crop);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
