//
//  MetalLRN.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalLRN.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalLRN::MetalLRN(Backend *backend, int regionType, int localSize, float alpha, float beta)
    : Execution(backend), mRegionType(regionType), mLocalSize(localSize), mAlpha(alpha), mBeta(beta) {
    // nothing to do
}

ErrorCode MetalLRN::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto encoder = [context encoder];
    auto input = inputs[0], output = outputs[0];
    int iw = input->width(), ih = input->height();
    int ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4) * output->batch();

    // acorss channel
    auto constantsBuffer = [context newDeviceBuffer:2 * sizeof(metal_float) + 8 * sizeof(int) access:CPUWriteOnly];
    auto floats          = (metal_float *)constantsBuffer.contents;
    auto ints            = (int *)((uint8_t *)constantsBuffer.contents + sizeof(metal_float) * 2);
    auto bandwidth       = (MetalBandwidth){};
    if (mRegionType == 0) {
        bandwidth = [context load:@"lrn_across_channel" encoder:encoder];
        floats[0] = mAlpha / mLocalSize;  // alpha
        floats[1] = mBeta;                // beta
        ints[0]   = mLocalSize;           // local size
        ints[1]   = inputs[0]->channel(); // channel
    }
    // within channel
    else if (mRegionType == 1) {
        bandwidth = [context load:@"lrn_within_channel" encoder:encoder];
        floats[0] = mAlpha / mLocalSize / mLocalSize; // alpha
        floats[1] = mBeta;                            // beta
        ints[0]   = mLocalSize;                       // local size
        ints[1]   = inputs[0]->channel();             // channel (unused)
    }
    // unsupport
    else {
        MNN_ASSERT(false);
    }
    ints[2] = iw;
    ints[3] = ih;
    ints[4] = iw * ih;
    ints[5] = ow;
    ints[6] = oh;
    ints[7] = ow * oh;
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:constantsBuffer offset:0 atIndex:2];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) ow, (NSUInteger)oh, (NSUInteger)oz } bandwidth:bandwidth];

    // commit
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalLRNCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto lrn = op->main_as_LRN();
        return new MetalLRN(backend, lrn->regionType(), lrn->localSize(), lrn->alpha(), lrn->beta());
    }
};
REGISTER_METAL_OP_CREATOR(MetalLRNCreator, OpType_LRN);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
