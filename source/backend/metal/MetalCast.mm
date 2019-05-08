//
//  MetalCast.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalCast.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalCast::MetalCast(Backend *backend, DataType srcType, DataType dstType)
    : Execution(backend), mSrcType(srcType), mDstType(dstType) {
    // nothing to do
}

ErrorCode MetalCast::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];

    NSString *kernel = nil;
    if (mSrcType == DataType_DT_FLOAT && mDstType == DataType_DT_INT32) {
        kernel = @"cast_float_to_int32";
    } else if (mSrcType == DataType_DT_INT32 && mDstType == DataType_DT_FLOAT) {
        kernel = @"cast_int32_to_float";
    } else if (mSrcType == DataType_DT_UINT8 && mDstType == DataType_DT_FLOAT) {
        kernel = @"cast_uint8_to_float";
    } else {
        return NOT_SUPPORT;
    }

    auto encoder   = [context encoder];
    auto bandwidth = [context load:kernel encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [context dispatchEncoder:encoder
                     threads:{ (NSUInteger) output->elementSize(), (NSUInteger)1, (NSUInteger)1 }
                   bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalCastCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto cast = op->main_as_CastParam();
        auto src = cast->srcT(), dst = cast->dstT();

        if (src == DataType_DT_FLOAT && dst == DataType_DT_INT32) {
            return new MetalCast(backend, src, dst);
        }
        if (src == DataType_DT_INT32 && dst == DataType_DT_FLOAT) {
            return new MetalCast(backend, src, dst);
        }
        if (src == DataType_DT_FLOAT && dst == DataType_DT_UINT8) {
            return new MetalCast(backend, src, dst);
        }
        if (src == DataType_DT_UINT8 && dst == DataType_DT_FLOAT) {
            return new MetalCast(backend, src, dst);
        }
        return NULL;
    }
};
REGISTER_METAL_OP_CREATOR(MetalCastCreator, OpType_Cast);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
