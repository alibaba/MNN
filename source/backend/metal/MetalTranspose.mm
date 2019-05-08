//
//  MetalTranspose.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalTranspose.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalTranspose::MetalTranspose(Backend *backend, DataType type) : Execution(backend), mType(type) {
    // nothing to do
}

ErrorCode MetalTranspose::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];

    // dims
    mInDims = [context newDeviceBuffer:sizeof(int) * (input->dimensions() + 1) access:CPUWriteOnly];
    ((int *)mInDims.contents)[0] = input->dimensions();
    for (int i = 0; i < input->dimensions(); i++) {
        ((int *)mInDims.contents)[i + 1] = input->length(i);
    }

    // stride
    mOutStrides = [context newDeviceBuffer:sizeof(int) * (output->dimensions() + 1) access:CPUWriteOnly];
    ((int *)mOutStrides.contents)[0] = output->dimensions();
    for (int i = 0; i < output->dimensions(); i++) {
        ((int *)mOutStrides.contents)[i + 1] = output->stride(i);
    }
    return NO_ERROR;
}

ErrorCode MetalTranspose::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mType != DataType_DT_INT32)
        return NOT_SUPPORT;

    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], perm = inputs[1], output = outputs[0];
    auto permStrides = [context newHeapBuffer:sizeof(int) * perm->elementSize() access:CPUTransparent];

    // prepare
    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"transpose_prepare" encoder:encoder];
    [encoder setBuffer:mInDims offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)perm->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:permStrides offset:0 atIndex:2];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) perm->elementSize(), 1, 1 } bandwidth:bandwidth];

    // transpose
    bandwidth = [context load:@"transpose" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:permStrides offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:2];
    [encoder setBuffer:mOutStrides offset:0 atIndex:3];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) output->elementSize(), 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalTransposeCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalTranspose(backend, op->main_as_Transpose()->Tperm());
    }
};
REGISTER_METAL_OP_CREATOR(MetalTransposeCreator, OpType_Transpose);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
