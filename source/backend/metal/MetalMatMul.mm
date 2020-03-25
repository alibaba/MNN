//
//  MetalMatMul.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalMatMul.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalMatMul::MetalMatMul(Backend *backend, const MatMul *matmul) : Execution(backend) {
    // nothing to do
}
ErrorCode MetalMatMul::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();

    auto input0 = inputs[0], input1 = inputs[1], output = outputs[0];
    int M = input0->length(0);
    int N = input0->length(1);
    int K = input1->length(1);
    
    auto shape                 = [context newDeviceBuffer:3 * sizeof(int) access:CPUWriteOnly];
    ((int *)shape.contents)[0] = M;
    ((int *)shape.contents)[1] = N;
    ((int *)shape.contents)[2] = K;
    
    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"matmul" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input0->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input1->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:2];
    [encoder setBuffer:shape offset:0 atIndex:3];
    [context dispatchEncoder:encoder
                     threads:{ (NSUInteger)K, (NSUInteger)M, (NSUInteger)1 }
                   bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

class MetalMatMulCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op,
                                Backend *backend) const override {
        return new MetalMatMul(backend, op->main_as_MatMul());
    }
};
REGISTER_METAL_OP_CREATOR(MetalMatMulCreator, OpType_MatMul);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
