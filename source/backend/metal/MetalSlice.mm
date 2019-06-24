//
//  MetalSlice.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalSlice.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalSlice::MetalSlice(Backend *backend, int axis) : Execution(backend), mAxis(axis) {
    // nothing to do
}

ErrorCode MetalSlice::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input   = (__bridge id<MTLBuffer>)(void *)inputs[0]->deviceId();
    auto tf      = inputs[0]->getDimensionType() == Tensor::TENSORFLOW;
    auto encoder = [context encoder];
    auto start   = 0;

    // tensorflow
    if (tf) {
        auto iw = inputs[0]->width(), ih = inputs[0]->height(), ic = inputs[0]->channel();
        for (int i = 0; i < outputs.size(); i++) {
            auto output = (__bridge id<MTLBuffer>)(void *)outputs[i]->deviceId();

            if (mAxis == 1) { // h
                NSUInteger size = outputs[i]->elementSize();
                auto bandwidth  = [context load:@"copy_float" encoder:encoder];
                [encoder setBuffer:input offset:start * sizeof(metal_float) atIndex:0];
                [encoder setBuffer:output offset:0 atIndex:1];
                [context dispatchEncoder:encoder threads:{ size, 1, 1 } bandwidth:bandwidth];
                start += size;
                continue;
            }

            auto ow = outputs[i]->width(), oc = outputs[i]->channel();
            auto shape                 = [context newDeviceBuffer:6 * sizeof(int) access:CPUWriteOnly];
            ((int *)shape.contents)[0] = ow;
            ((int *)shape.contents)[1] = ih;
            ((int *)shape.contents)[2] = ow * oc; // output steps
            ((int *)shape.contents)[3] = oc;
            ((int *)shape.contents)[4] = iw * ic; //  input steps
            ((int *)shape.contents)[5] = ic;

            auto bandwidth = [context load:@"slice_tf" encoder:encoder];
            [encoder setBuffer:input offset:start * sizeof(metal_float) atIndex:0];
            [encoder setBuffer:output offset:0 atIndex:1];
            [encoder setBuffer:shape offset:0 atIndex:2];
            [context dispatchEncoder:encoder
                             threads:{ (NSUInteger) ow, (NSUInteger)ih, (NSUInteger)oc }
                           bandwidth:bandwidth];

            if (mAxis == 2) { // w
                start += ow * oc;
            } else if (mAxis == 3) { // c
                start += oc;
            } else {
                MNN_ASSERT(false);
            }
        }
    }
    // caffe
    else {
        auto iw = inputs[0]->width(), ih = inputs[0]->height(), iz = UP_DIV(inputs[0]->channel(), 4);
        switch (mAxis) {
            case 1: { // c
                auto shape                 = [context newDeviceBuffer:6 * sizeof(int) access:CPUWriteOnly];
                ((int *)shape.contents)[0] = iw;
                ((int *)shape.contents)[1] = ih;
                ((int *)shape.contents)[2] = iw * ih; // output steps
                ((int *)shape.contents)[3] = iw;
                ((int *)shape.contents)[4] = iw * ih; //  input steps
                ((int *)shape.contents)[5] = iw;

                auto bandwidth = [context load:@"slice_channel" encoder:encoder];
                for (int i = 0; i < outputs.size(); i++) {
                    auto output = (__bridge id<MTLBuffer>)(void *)outputs[i]->deviceId();
                    auto oc = outputs[i]->channel(), oz = UP_DIV(oc, 4);
                    int range[2] = {start, start + oc - 1};

                    [encoder setBuffer:input offset:0 atIndex:0];
                    [encoder setBuffer:output offset:0 atIndex:1];
                    [encoder setBuffer:shape offset:0 atIndex:2];
                    [encoder setBuffer:[context newDeviceBuffer:sizeof(range) bytes:range access:CPUWriteOnly]
                                offset:0
                               atIndex:3];
                    [context dispatchEncoder:encoder
                                     threads:{ (NSUInteger) iw, (NSUInteger)ih, (NSUInteger)oz }
                                   bandwidth:bandwidth];
                    start += oc;
                }
            } break;
            case 2: { // h
                auto bandwidth = [context load:@"copy_float" encoder:encoder];
                auto is        = iw * ih * 4 * sizeof(metal_float);
                for (int i = 0; i < outputs.size(); i++) {
                    auto output = (__bridge id<MTLBuffer>)(void *)outputs[i]->deviceId();
                    auto oh = outputs[i]->height(), num = iw * oh * 4, os = num * (int)sizeof(metal_float);

                    for (int j = 0; j < iz; j++) {
                        [encoder setBuffer:input offset:start + j * is atIndex:0];
                        [encoder setBuffer:output offset:j * os atIndex:1];
                        [context dispatchEncoder:encoder threads:{ (NSUInteger) num, 1, 1 } bandwidth:bandwidth];
                    }
                    start += os;
                }
            } break;
            case 3: { // w
                auto bandwidth = [context load:@"slice_width" encoder:encoder];
                for (int i = 0; i < outputs.size(); i++) {
                    auto output = (__bridge id<MTLBuffer>)(void *)outputs[i]->deviceId();
                    auto ow     = outputs[i]->width();

                    auto shape                 = [context newDeviceBuffer:6 * sizeof(int) access:CPUWriteOnly];
                    ((int *)shape.contents)[0] = ow;
                    ((int *)shape.contents)[1] = ih;
                    ((int *)shape.contents)[2] = ow * ih; // output steps
                    ((int *)shape.contents)[3] = ow;
                    ((int *)shape.contents)[4] = iw * ih; //  input steps
                    ((int *)shape.contents)[5] = iw;

                    [encoder setBuffer:input offset:start * 4 * sizeof(metal_float) atIndex:0];
                    [encoder setBuffer:output offset:0 atIndex:1];
                    [encoder setBuffer:shape offset:0 atIndex:2];
                    [context dispatchEncoder:encoder
                                     threads:{ (NSUInteger) ow, (NSUInteger)ih, (NSUInteger)iz }
                                   bandwidth:bandwidth];
                    start += ow;
                }
            } break;
            default:
                MNN_ASSERT(false);
                break;
        }
    }

    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);

    return NO_ERROR;
}

class MetalSliceCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto axis = op->main_as_Slice()->axis();
        if (0 > axis) {
            axis = inputs[0]->dimensions() + axis;
        }
        return new MetalSlice(backend, axis);
    }
};
REGISTER_METAL_OP_CREATOR(MetalSliceCreator, OpType_Slice);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
