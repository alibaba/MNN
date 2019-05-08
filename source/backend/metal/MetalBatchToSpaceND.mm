//
//  MetalBatchToSpaceND.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalBatchToSpaceND.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalBatchToSpaceND::MetalBatchToSpaceND(Backend *backend, int blockHeight, int blockWidth, int paddingTop,
                                         int paddingLeft)
    : Execution(backend),
      mBlockHeight(blockHeight),
      mBlockWidth(blockWidth),
      mPaddingTop(paddingTop),
      mPaddingLeft(paddingLeft) {
    // nothing to do
}

ErrorCode MetalBatchToSpaceND::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];

    mConst    = [context newDeviceBuffer:12 * sizeof(int) access:CPUWriteOnly];
    auto data = (int *)mConst.contents;
    data[0]   = mBlockWidth;
    data[1]   = mBlockHeight;
    data[2]   = mPaddingLeft;
    data[3]   = mPaddingTop;
    data[4]   = input->width();
    data[5]   = input->height();
    data[6]   = UP_DIV(input->channel(), 4);
    data[7]   = input->batch();
    data[8]   = output->width();
    data[9]   = output->height();
    data[10]  = UP_DIV(output->channel(), 4);
    data[11]  = output->batch();

    return NO_ERROR;
}

ErrorCode MetalBatchToSpaceND::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    auto w = output->width(), h = output->height(), c = output->channel(), z = UP_DIV(c, 4), b = output->batch();

    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"batch_to_space_nd" encoder:encoder];
    bandwidth.zAxisProtected = YES;
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mConst offset:0 atIndex:2];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) w, (NSUInteger)h, (NSUInteger)z *b } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalBatchToSpaceNDCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto sb      = op->main_as_SpaceBatch();
        auto block   = sb->blockShape()->int32s()->data();
        auto padding = sb->padding()->int32s()->data();
        return new MetalBatchToSpaceND(backend, block[0], block[1], padding[0], padding[2]);
    }
};
REGISTER_METAL_OP_CREATOR(MetalBatchToSpaceNDCreator, OpType_BatchToSpaceND);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
