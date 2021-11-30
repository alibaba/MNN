//
//  MetalCast.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalCast.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

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
    } else if (mSrcType == DataType_DT_UINT8 && mDstType == DataType_DT_INT32) {
        kernel = @"cast_uint8_to_int";
    } else {
        return NOT_SUPPORT;
    }

    auto encoder   = backend->encoder();
    auto bandwidth = [context load:kernel encoder:encoder];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:1];
    [context dispatchEncoder:encoder
                     threads:{ (NSUInteger) output->elementSize(), (NSUInteger)1, (NSUInteger)1 }
                   bandwidth:bandwidth];
    return NO_ERROR;
}
static DataType _mapDataType(DataType src) {
    if (DataType_DT_BOOL == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_INT64 == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_DOUBLE == src) {
        return DataType_DT_FLOAT;
    }
    return src;
}
class MetalCastCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto cast = op->main_as_CastParam();
        auto srcType = inputs[0]->getType();
        auto dst = _mapDataType(cast->dstT());

        if (srcType.code == halide_type_float && dst == DataType_DT_INT32) {
            return new MetalCast(backend, DataType_DT_FLOAT, dst);
        }
        if (srcType.code == halide_type_int && srcType.bits == 32 && dst == DataType_DT_FLOAT) {
            return new MetalCast(backend, DataType_DT_INT32, dst);
        }
        if (srcType.code == halide_type_float && dst == DataType_DT_UINT8) {
            return new MetalCast(backend, DataType_DT_FLOAT, dst);
        }
        if (srcType.code == halide_type_uint && srcType.bits == 8 && dst == DataType_DT_FLOAT) {
            return new MetalCast(backend, DataType_DT_UINT8, dst);
        }
        if (srcType.code == halide_type_uint && srcType.bits == 8 && dst == DataType_DT_INT32) {
            return new MetalCast(backend, DataType_DT_UINT8, dst);
        }
        MNN_PRINT("%d, %d - %d\n", srcType.code, srcType.bits, dst);
        return NULL;
    }
};
REGISTER_METAL_OP_CREATOR(MetalCastCreator, OpType_Cast);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
