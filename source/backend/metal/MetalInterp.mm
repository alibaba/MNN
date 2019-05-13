//
//  MetalInterp.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalInterp.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalInterp::MetalInterp(Backend *backend, float widthScale, float heightScale, int32_t outputWidth,
                         int32_t outputHeight, int32_t reiszeType, bool alignCorner)
    : Execution(backend),
      mWidthScale(widthScale),
      mHeightScale(heightScale),
      mOutputWidth(outputWidth),
      mOutputHeight(outputHeight),
      mReiszeType(reiszeType),
      mAlignCorner(alignCorner) {
    // nothing to do
}

ErrorCode MetalInterp::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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

    // encode
    auto encoder   = [context encoder];
    auto bandwidth = (MetalBandwidth){};
    if (mReiszeType == 2) {
        bandwidth  = [context load:@"resize_bilinear" encoder:encoder];
        auto scale = [context newDeviceBuffer:2 * sizeof(float) access:CPUWriteOnly];
        if (mAlignCorner) {
            ((float *)scale.contents)[0] = (float)(iw - 1) / (float)(ow - 1);
            ((float *)scale.contents)[1] = (float)(ih - 1) / (float)(oh - 1);
        } else {
            ((float *)scale.contents)[0] = (float)iw / (float)ow;
            ((float *)scale.contents)[1] = (float)ih / (float)oh;
        }
        [encoder setBuffer:scale offset:0 atIndex:3];
    } else if (mReiszeType == 3) {
        bandwidth = [context load:@"resize_cubic" encoder:encoder];
    } else {
        MNN_ASSERT(false);
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:shape offset:0 atIndex:2];
    [context dispatchEncoder:encoder
                     threads:{ (NSUInteger) ow, (NSUInteger)oh, (NSUInteger)slice }
                   bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalInterpCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto interp = op->main_as_Interp();
        return new MetalInterp(backend, interp->widthScale(), interp->heightScale(), interp->outputWidth(),
                               interp->outputHeight(), interp->resizeType(), interp->alignCorners());
    }
};
REGISTER_METAL_OP_CREATOR(MetalInterpCreator, OpType_Interp);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
