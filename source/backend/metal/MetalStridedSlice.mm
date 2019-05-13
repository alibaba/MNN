//
//  MetalStridedSlice.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalStridedSlice.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalStridedSlice::MetalStridedSlice(Backend *backend, const StridedSliceParam *s) : Execution(backend), mParam(s) {
    // nothing to do
}

ErrorCode MetalStridedSlice::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], begin = inputs[1];

    // dim
    mDims                      = [context newDeviceBuffer:sizeof(int) * (input->dimensions() + 1) access:CPUWriteOnly];
    ((int *)mDims.contents)[0] = input->dimensions();
    for (int i = 0; i < input->dimensions(); i++) {
        ((int *)mDims.contents)[i + 1] = input->length(i);
    }

    // mask
    mMask                      = [context newDeviceBuffer:sizeof(int) * (begin->length(0) * 3 + 1) access:CPUWriteOnly];
    ((int *)mMask.contents)[0] = begin->length(0);
    for (int i = 0; i < begin->length(0); i++) {
        ((int *)mMask.contents)[3 * i + 1] = mParam->beginMask() & (1 << i);
        ((int *)mMask.contents)[3 * i + 2] = mParam->endMask() & (1 << i);
        ((int *)mMask.contents)[3 * i + 3] = mParam->shrinkAxisMask() & (1 << i);
    }
    return NO_ERROR;
}

ErrorCode MetalStridedSlice::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], begin = inputs[1], end = inputs[2], strided = inputs[3], output = outputs[0];
    auto slices  = [context newHeapBuffer:input->dimensions() * 2 * sizeof(int) access:CPUReadWrite];
    auto strides = [context newHeapBuffer:input->dimensions() * 2 * sizeof(int) access:CPUReadWrite];

    // prepare
    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"strided_slice_prepare" encoder:encoder];
    [encoder setBuffer:mDims offset:0 atIndex:0];
    [encoder setBuffer:mMask offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)begin->deviceId() offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)end->deviceId() offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)strided->deviceId() offset:0 atIndex:4];
    [encoder setBuffer:slices offset:0 atIndex:5];
    [encoder setBuffer:strides offset:0 atIndex:6];
    [context dispatchEncoder:encoder threads:{ 1, 1, 1 } bandwidth:bandwidth];

    // stride slice
    auto type = mParam->T();
    if (type == DataType_DT_INT32) {
        bandwidth = [context load:@"strided_slice_int32" encoder:encoder];
    } else if (type == DataType_DT_FLOAT) {
        bandwidth = [context load:@"strided_slice_float" encoder:encoder];
    } else {
        return NOT_SUPPORT;
    }
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mDims offset:0 atIndex:2];
    [encoder setBuffer:strides offset:0 atIndex:3];
    [encoder setBuffer:slices offset:0 atIndex:4];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) output->elementSize(), 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalStridedSliceCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalStridedSlice(backend, op->main_as_StridedSliceParam());
    }
};
REGISTER_METAL_OP_CREATOR(MetalStridedSliceCreator, OpType_StridedSlice);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
