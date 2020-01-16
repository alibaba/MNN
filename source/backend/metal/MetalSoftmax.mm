//
//  MetalSoftmax.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MNNMetalContext.h"
#if MNN_METAL_ENABLED
#import "backend/metal/MetalSoftmax.hpp"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#import "core/TensorUtils.hpp"
namespace MNN {

MetalSoftmax::MetalSoftmax(Backend *backend, int32_t axis) : Execution(backend), mAxis(axis) {
    // nothing to do
}

ErrorCode MetalSoftmax::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    auto dimensions    = input->buffer().dimensions;
    auto realAxis      = mAxis < 0 ? dimensions + mAxis : mAxis;
    auto channelPacked = TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4; // even dims != 4
    auto reorder       = realAxis == 1 && channelPacked;
    // shape
    auto inside = 1, flat = input->length(realAxis), axis = flat, outside = 1;
    for (int i = 0; i < realAxis; i++) {
        auto length = input->length(i);
        if (1 == i && channelPacked) {
            length = UP_DIV(length, 4);
        }
        outside *= length;
    }
    for (int i = realAxis + 1; i < input->dimensions(); i++) {
        auto length = input->length(i);
        if (1 == i && channelPacked) {
            length = UP_DIV(length, 4);
        }
        inside *= length;
    }
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
        return nullptr;
//        auto softmax = op->main_as_Axis();
//        return new MetalSoftmax(backend, softmax->axis());
    }
};
REGISTER_METAL_OP_CREATOR(MetalSoftmaxCreator, OpType_Softmax);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
