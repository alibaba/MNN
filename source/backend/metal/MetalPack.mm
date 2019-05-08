//
//  MetalPack.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalPack.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"
#import "TensorUtils.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalPack::MetalPack(Backend *bn, DataType type, int axis) : Execution(bn), mType(type), mAxis(axis) {
    // nothing to do
}

ErrorCode MetalPack::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    // set up blits
    int N = (int)inputs.size();
    if (inputs[0]->dimensions() > 0) {
        mBlits     = [context newDeviceBuffer:5 * inputs.size() * sizeof(int) access:CPUReadWrite];
        auto blits = (int *)mBlits.contents;
        for (int i = 0; i < inputs.size(); i++, blits += 5) {
            auto input = inputs[i];
            auto axis  = mAxis;
            blits[0]   = 1;
            for (int d = axis + 1; d < input->buffer().dimensions; d++)
                blits[0] *= input->length(d);
            blits[1] = input->length(axis);
            blits[2] = 1;
            for (int d = 0; d < axis; d++)
                blits[2] *= input->length(d);
            blits[3] = blits[0] * blits[1];
            blits[4] = blits[3] * N;
        }
    }
    return NO_ERROR;
}

ErrorCode MetalPack::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto output  = outputs[0];
    auto scalar  = inputs[0]->dimensions() == 0;

    NSUInteger unit  = 0;
    NSString *kernel = nil;
    switch (mType) {
        case DataType_DT_INT32:
            kernel = scalar ? @"copy_int" : @"pack_int32";
            unit   = sizeof(int);
            break;
        case DataType_DT_FLOAT:
            kernel = scalar ? @"copy_float" : @"pack_float";
            unit   = sizeof(metal_float);
            break;
        default:
            return NOT_SUPPORT;
    }

    auto encoder             = [context encoder];
    auto bandwidth           = [context load:kernel encoder:encoder];
    bandwidth.zAxisProtected = YES;
    auto start               = 0;
    for (int i = 0; i < inputs.size(); i++) {
        auto input = inputs[i];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)(input->buffer().device) offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)(output->buffer().device) offset:start atIndex:1];

        if (scalar) {
            [context dispatchEncoder:encoder threads:{ (NSUInteger) input->elementSize(), 1, 1 } bandwidth:bandwidth];
            start += input->elementSize() * unit;
        } else {
            auto blits = (int *)mBlits.contents + i * 5;
            [encoder setBuffer:mBlits offset:i * 5 * sizeof(int) atIndex:2];
            [context dispatchEncoder:encoder
                             threads:{ (NSUInteger) blits[0], (NSUInteger)blits[1], (NSUInteger)blits[2] }
                           bandwidth:bandwidth];
            start += blits[3] * unit;
        }
    }
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalPackCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto pack = op->main_as_PackParam();
        return new MetalPack(backend, pack->dataType(), pack->axis());
    }
};
REGISTER_METAL_OP_CREATOR(MetalPackCreator, OpType_Pack);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
