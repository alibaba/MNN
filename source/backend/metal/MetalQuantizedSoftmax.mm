//
//  MetalQuantizedSoftmax.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalQuantizedSoftmax.hpp"
#import "CPUQuantizationUtils.hpp"
#import "MNNMetalContext.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

const int kScaledDiffIntegerBits = 5;

MetalQuantizedSoftmax::MetalQuantizedSoftmax(Backend *backend, float beta, float scale) : Execution(backend) {
    PreprocessSoftmaxScaling(beta, scale, kScaledDiffIntegerBits, &mInputMultiplier, &mInputLeftShift);
    mDiffMin = -1.0 * CalculateInputRadius(kScaledDiffIntegerBits, mInputLeftShift);
}

ErrorCode MetalQuantizedSoftmax::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input   = inputs[0];

    int outer = 0, inner = 0;
    if (input->dimensions() == 4) {
        outer = input->length(0) * input->length(1) * input->length(2);
        inner = input->length(3);
    } else {
        outer = input->length(0);
        inner = input->length(1);
    }

    mConst                      = [context newDeviceBuffer:5 * sizeof(int) access:CPUWriteOnly];
    ((int *)mConst.contents)[0] = outer;
    ((int *)mConst.contents)[1] = inner;
    ((int *)mConst.contents)[2] = mDiffMin;
    ((int *)mConst.contents)[3] = mInputMultiplier;
    ((int *)mConst.contents)[4] = mInputLeftShift;
    return NO_ERROR;
}

ErrorCode MetalQuantizedSoftmax::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];

    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"quantized_softmax" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mConst offset:0 atIndex:2];
    [context dispatchEncoder:encoder threads:{ (NSUInteger) output->size(), 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalQuantizedSoftmaxCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto qs = op->main_as_QuantizedSoftmax();
        return new MetalQuantizedSoftmax(backend, qs->beta(), qs->inputScale());
    }
};
REGISTER_METAL_OP_CREATOR(MetalQuantizedSoftmaxCreator, OpType_QuantizedSoftmax);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
