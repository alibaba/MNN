//
//  MetalCropAndResize.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalCropAndResize.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalCropAndResize::MetalCropAndResize(Backend *backend, float extrapolation, CropAndResizeMethod method)
    : Execution(backend), mExtrapolation(extrapolation), mMethod(method) {
    // nothing to do
}

ErrorCode MetalCropAndResize::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto image = inputs[0], crops = outputs[0];
    auto ib = image->length(0), ih = image->length(1), iw = image->length(2), ic = image->length(3);
    auto cb = crops->length(0), ch = crops->length(1), cw = crops->length(2), cc = crops->length(3);

    mShape        = [context newDeviceBuffer:8 * sizeof(int) + sizeof(float) access:CPUWriteOnly];
    auto contents = (int *)mShape.contents;
    contents[0]   = ib;
    contents[1]   = ih;
    contents[2]   = iw;
    contents[3]   = ic;
    contents[4]   = cb;
    contents[5]   = ch;
    contents[6]   = cw;
    contents[7]   = cc;

    ((float *)mShape.contents)[8] = mExtrapolation;
    return NO_ERROR;
}

ErrorCode MetalCropAndResize::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto image = inputs[0], boxes = inputs[1], indexes = inputs[2], crops = outputs[0];
    auto cb = crops->length(0), ch = crops->length(1), cw = crops->length(2), cc = crops->length(3);

    auto encoder             = [context encoder];
    MetalBandwidth bandwidth = {};
    if (mMethod == CropAndResizeMethod_BILINEAR) {
        bandwidth = [context load:@"crop_and_resize_bilinear" encoder:encoder];
    } else if (mMethod == CropAndResizeMethod_NEAREST) {
        bandwidth = [context load:@"crop_and_resize_nearest" encoder:encoder];
    } else {
        return NOT_SUPPORT;
    }
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)image->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)boxes->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)indexes->deviceId() offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)crops->deviceId() offset:0 atIndex:3];
    [encoder setBuffer:mShape offset:0 atIndex:4];
    [context dispatchEncoder:encoder
                     threads:{ (NSUInteger) cc, (NSUInteger)cw, (NSUInteger)ch *cb }
                   bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalCropAndResizeCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto car = op->main_as_CropAndResize();
        return new MetalCropAndResize(backend, car->extrapolationValue(), car->method());
    }
};
REGISTER_METAL_OP_CREATOR(MetalCropAndResizeCreator, OpType_CropAndResize);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
