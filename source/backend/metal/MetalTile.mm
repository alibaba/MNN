//
//  MetalTile.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalTile.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalTile::MetalTile(Backend *backend) : Execution(backend) {
    // nothing to do
}

ErrorCode MetalTile::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    auto ib = input->batch(), ih = input->height(), iw = input->width(), ic = input->channel();
    auto ob = output->batch(), oh = output->height(), ow = output->width(), oc = output->channel();

    auto shape    = [context newDeviceBuffer:7 * sizeof(int) access:CPUWriteOnly];
    auto contents = (int *)shape.contents;
    contents[0]   = oh;
    contents[1]   = ow;
    contents[2]   = oc;
    contents[3]   = ib;
    contents[4]   = ih;
    contents[5]   = iw;
    contents[6]   = ic;

    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"tile" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:shape offset:0 atIndex:2];
    [context dispatchEncoder:encoder
                     threads:{ (NSUInteger) oc, (NSUInteger)ow, (NSUInteger)oh *ob }
                   bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalTileCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalTile(backend);
    }
};
REGISTER_METAL_OP_CREATOR(MetalTileCreator, OpType_Tile);
} // namespace MNN

#endif /* MNN_METAL_ENABLED */
