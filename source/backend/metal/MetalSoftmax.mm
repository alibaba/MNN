//
//  MetalSoftmax.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalSoftmax.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalSoftmax::MetalSoftmax(Backend *backend, int32_t axis) : Execution(backend), mAxis(axis) {
    // nothing to do
}

ErrorCode MetalSoftmax::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    const int dimensions = input->buffer().dimensions;
    auto reAxis          = mAxis < 0 ? dimensions - 1 : mAxis;
    // shape
    auto inside = 1, flat = input->length(reAxis), axis = flat, outside = 1;
    for (int i = 0; i < reAxis; i++) {
        outside *= input->buffer().dim[i].flags ? UP_DIV(input->length(i), 4) : input->length(i);
    }
    for (int i = reAxis + 1; i < input->dimensions(); i++) {
        inside *= input->buffer().dim[i].flags ? UP_DIV(input->length(i), 4) : input->length(i);
    }
    auto reorder = input->buffer().dim[reAxis].flags;
    if (reorder) {
        axis = UP_DIV(axis, 4);
    }

    auto shape                 = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    ((int *)shape.contents)[0] = inside;
    ((int *)shape.contents)[1] = axis;
    ((int *)shape.contents)[2] = outside;
    ((int *)shape.contents)[3] = flat;

    auto multiplex = axis >= 128;

    // encode
    auto tf     = input->getDimensionType() == Tensor::TENSORFLOW;
    auto kernel = multiplex ? (tf ? @"softmax_m_tf" : reorder ? @"softmax_m_on_reorder" : @"softmax_m_off_reorder")
                            : (tf ? @"softmax_tf" : reorder ? @"softmax_on_reorder" : @"softmax_off_reorder");
    auto encoder   = [context encoder];
    auto bandwidth = [context load:kernel encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:shape offset:0 atIndex:2];

    if (multiplex) {
        auto unit    = (!tf && reorder) ? sizeof(float) : 4 * sizeof(float);
        auto threads = MIN(pow(log2(UP_DIV(axis, 64)), 2), bandwidth.threadExecutionWidth);
        if (unit * bandwidth.maxThreadsPerThreadgroup > context.maxThreadgroupMemoryLength) {
            bandwidth.maxThreadsPerThreadgroup /= context.maxThreadgroupMemoryLength / unit;
        }
        bandwidth.zAxisProtected = YES;
        [encoder setThreadgroupMemoryLength:unit * bandwidth.maxThreadsPerThreadgroup atIndex:0];
        [context dispatchEncoder:encoder
                         threads:{(NSUInteger)threads, (NSUInteger)inside, (NSUInteger)outside}
                       bandwidth:bandwidth];
    } else {
        [context dispatchEncoder:encoder threads:{(NSUInteger)inside, (NSUInteger)outside, 1} bandwidth:bandwidth];
    }
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalSoftmaxCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto softmax = op->main_as_Axis();
        return new MetalSoftmax(backend, softmax->axis());
    }
};
REGISTER_METAL_OP_CREATOR(MetalSoftmaxCreator, OpType_Softmax);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
