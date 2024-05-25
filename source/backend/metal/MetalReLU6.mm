//
//  MetalReLU6.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalReLU6.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalReLU6::MetalReLU6(Backend *backend, float minV, float maxV, bool isRelu) : MetalExecution(backend) {
    // For Relu use minV and slope
    auto metal   = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)metal->context();
    mConst = [context newDeviceBuffer:4 * sizeof(float) access:CPUWriteOnly];
    ((float*)mConst.contents)[0] = minV;
    ((float*)mConst.contents)[1] = maxV;
    if (isRelu) {
        mPipeline = [context pipelineWithName:@"relu" fp16:metal->useFp16InsteadFp32()];
    } else {
        mPipeline = [context pipelineWithName:@"relu6" fp16:metal->useFp16InsteadFp32()];
    }
}
void MetalReLU6::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto input = inputs[0], output = outputs[0];
    int size = output->elementSize();
    size = UP_DIV(size, 4);
    ((int*)mConst.contents)[2] = size;
    ((int*)mConst.contents)[3] = size;

    [encoder setComputePipelineState:mPipeline];
    MetalBackend::setTensor(input, encoder, 0);
    MetalBackend::setTensor(output, encoder, 1);
    [encoder setBuffer:mConst offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(size, 256), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

class MetalReLU6Creator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        if (op->type() == OpType_ReLU) {
            return new MetalReLU6(backend, op->main_as_Relu()->slope(), 0.0f, true);
        }
        float minV = 0.0f;
        float maxV = 6.0f;
        if (nullptr != op->main()) {
            auto p = op->main_as_Relu6();
            minV = p->minValue();
            maxV = p->maxValue();
        }
        return new MetalReLU6(backend, minV, maxV, false);
    }
};
REGISTER_METAL_OP_CREATOR(MetalReLU6Creator, OpType_ReLU6);
REGISTER_METAL_OP_CREATOR(MetalReLU6Creator, OpType_ReLU);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
