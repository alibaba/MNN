//
//  MetalSliceTF.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalSliceTF.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalSliceTF::MetalSliceTF(Backend *backend, DataType type) : Execution(backend), mType(type) {
    // nothing to do
}

ErrorCode MetalSliceTF::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0], begin = inputs[1], output = outputs[0];
    int dims = output->dimensions();
    if (dims == 0)
        return NO_ERROR;

    auto backend   = static_cast<MetalBackend *>(this->backend());
    auto context   = (__bridge MNNMetalContext *)backend->context();
    auto encoder   = [context encoder];
    auto bandwidth = (MetalBandwidth){};
    if (mType == DataType_DT_INT32) {
        bandwidth = [context load:@"slice_tf_int32" encoder:encoder];
    } else if (mType == DataType_DT_FLOAT) {
        bandwidth = [context load:@"slice_tf_float" encoder:encoder];
    } else {
        return NOT_SUPPORT;
    }

    // create buffers
    auto dimsBuffer = [context newDeviceBuffer:sizeof(int) bytes:&dims access:CPUWriteOnly];
    auto inStrides  = [context newDeviceBuffer:dims * sizeof(int) access:CPUWriteOnly];
    auto outStrides = [context newDeviceBuffer:dims * sizeof(int) access:CPUWriteOnly];
    for (int i = 0; i < dims; i++) {
        ((int *)inStrides.contents)[i]  = input->stride(i);
        ((int *)outStrides.contents)[i] = output->stride(i);
    }

    // encode
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:inStrides offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:2];
    [encoder setBuffer:outStrides offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)begin->deviceId() offset:0 atIndex:4];
    [encoder setBuffer:dimsBuffer offset:0 atIndex:5];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) output->elementSize(), 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalSliceTFCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalSliceTF(backend, op->main_as_SliceTf()->T());
    }
};
REGISTER_METAL_OP_CREATOR(MetalSliceTFCreator, OpType_SliceTf);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
