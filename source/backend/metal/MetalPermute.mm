//
//  MetalPermute.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalPermute.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalPermute::MetalPermute(Backend *backend, const Permute *permute) : Execution(backend) {
    MNN_ASSERT(permute->dims()->size() == 4);
    for (int i = 0; i < permute->dims()->size(); i++) {
        mDims.push_back(permute->dims()->data()[i]);
    }
}

ErrorCode MetalPermute::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];

    // to chw
    static int DimC = 1, DimH = 2, DimW = 3;
    if (mDims[1] == DimC && mDims[2] == DimH && mDims[3] == DimW) {
        auto encoder   = [context encoder];
        auto bandwidth = [context load:@"copy_float" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
        [context dispatchEncoder:encoder threads:{ (NSUInteger) output->elementSize(), 1, 1 } bandwidth:bandwidth];
        [encoder endEncoding];
        MNN_PRINT_ENCODER(context, encoder);
        return NO_ERROR;
    }

    int iw = input->width(), ih = input->height(), iz = UP_DIV(input->channel(), 4);
    int ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4), ob = output->batch();
    auto shape                 = [context newDeviceBuffer:8 * sizeof(int) access:CPUWriteOnly];
    ((int *)shape.contents)[0] = iw;
    ((int *)shape.contents)[1] = iw * ih;
    ((int *)shape.contents)[2] = iz;
    ((int *)shape.contents)[3] = ow;
    ((int *)shape.contents)[4] = ow * oh;
    ((int *)shape.contents)[5] = oz;
    ((int *)shape.contents)[6] = oh;
    ((int *)shape.contents)[7] = ob;

    auto encoder   = [context encoder];
    auto bandwidth = (MetalBandwidth){};
    if (mDims[1] == DimC && mDims[2] == DimW && mDims[3] == DimH) {
        bandwidth = [context load:@"permute_to_cwh" encoder:encoder];
    } else if (mDims[1] == DimH && mDims[2] == DimC && mDims[3] == DimW) {
        bandwidth = [context load:@"permute_to_hcw" encoder:encoder];
    } else if (mDims[1] == DimH && mDims[2] == DimW && mDims[3] == DimC) {
        bandwidth = [context load:@"permute_to_hwc" encoder:encoder];
    } else if (mDims[1] == DimW && mDims[2] == DimC && mDims[3] == DimH) {
        bandwidth = [context load:@"permute_to_wch" encoder:encoder];
    } else if (mDims[1] == DimW && mDims[2] == DimH && mDims[3] == DimC) {
        bandwidth = [context load:@"permute_to_whc" encoder:encoder];
    } else {
        MNN_ASSERT(false); // unsupported
    }
    bandwidth.zAxisProtected = YES;
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:shape offset:0 atIndex:2];
    [context dispatchEncoder:encoder
                     threads:{ (NSUInteger) ow, (NSUInteger)oh, (NSUInteger)oz *ob }
                   bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalPermuteCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalPermute(backend, op->main_as_Permute());
    }
};
REGISTER_METAL_OP_CREATOR(MetalPermuteCreator, OpType_Permute);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
