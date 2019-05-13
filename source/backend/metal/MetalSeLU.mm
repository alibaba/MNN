//
//  MetalSeLU.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalSeLU.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalSeLU::MetalSeLU(Backend *backend, float scale, float alpha) : Execution(backend) {
    auto context                  = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    mConst                        = [context newDeviceBuffer:2 * sizeof(float) access:CPUWriteOnly];
    ((float *)mConst.contents)[0] = scale;
    ((float *)mConst.contents)[1] = alpha;
}

ErrorCode MetalSeLU::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    NSUInteger size = output->elementSize();
    auto simd       = size % 4 == 0;
    if (simd) {
        size /= 4;
    }

    auto encoder   = [context encoder];
    auto bandwidth = [context load:simd ? @"selu_x4" : @"selu_x1" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mConst offset:0 atIndex:2];
    [context dispatchEncoder:encoder threads:{ size, 1, 1 } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalSeLUCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        auto selu = op->main_as_Selu();
        return new MetalSeLU(backend, selu->scale(), selu->alpha());
    }
};
REGISTER_METAL_OP_CREATOR(MetalSeLUCreator, OpType_Selu);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
