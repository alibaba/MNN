//
//  MetalInterp.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalInterp.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalInterp::MetalInterp(Backend *backend, const Op* op)
    : Execution(backend) {
    auto interpParam = op->main_as_Interp();
    auto mBk = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)mBk->context();
    mCordTransform = [context newDeviceBuffer:4 * sizeof(float) access:CPUWriteOnly];
    ((float *)mCordTransform.contents)[0] = interpParam->widthScale();
    ((float *)mCordTransform.contents)[1] = interpParam->widthOffset();
    ((float *)mCordTransform.contents)[2] = interpParam->heightScale();
    ((float *)mCordTransform.contents)[3] = interpParam->heightOffset();
    mReiszeType = interpParam->resizeType();
    mShape = [context newDeviceBuffer:7 * sizeof(int) access:CPUWriteOnly];
}

ErrorCode MetalInterp::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    int iw = input->width(), ih = input->height();
    int ow = output->width(), oh = output->height(), slice = UP_DIV(output->channel(), 4) * output->batch();

    ((int *)mShape.contents)[0] = iw;
    ((int *)mShape.contents)[1] = ih;
    ((int *)mShape.contents)[2] = iw * ih;
    ((int *)mShape.contents)[3] = ow;
    ((int *)mShape.contents)[4] = oh;
    ((int *)mShape.contents)[5] = ow * oh;
    ((int *)mShape.contents)[6] = slice;


    // encode
    auto encoder   = [context encoder];
    auto bandwidth = (MetalBandwidth){};
    if (mReiszeType == 2 || mReiszeType == 1) {
        if (2 == mReiszeType) {
            bandwidth  = [context load:@"resize_bilinear" encoder:encoder];
        } else {
            bandwidth  = [context load:@"resize_nearest" encoder:encoder];
        }
    } else if (mReiszeType == 3) {
        bandwidth = [context load:@"resize_cubic" encoder:encoder];
    } else {
        MNN_ASSERT(false);
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mShape offset:0 atIndex:2];
    [encoder setBuffer:mCordTransform offset:0 atIndex:3];
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
        return new MetalInterp(backend, op);
    }
};
REGISTER_METAL_OP_CREATOR(MetalInterpCreator, OpType_Interp);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
