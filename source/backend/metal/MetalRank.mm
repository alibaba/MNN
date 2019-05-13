//
//  MetalRank.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalRank.hpp"
#import "MNNMetalContext.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalRank::MetalRank(Backend *backend) : Execution(backend) {
    // nothing to do
}

ErrorCode MetalRank::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto output  = outputs[0];
    int dims     = inputs[0]->dimensions();

    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"rank" encoder:encoder];
    [encoder setBuffer:[context newDeviceBuffer:sizeof(int) bytes:&dims access:CPUWriteOnly] offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [context dispatchEncoder:encoder threads:{ 1, 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalRankCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalRank(backend);
    }
};
REGISTER_METAL_OP_CREATOR(MetalRankCreator, OpType_Rank);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
