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
struct matP {
    int size[4];
    int stride[4];
};
MetalMatMul::MetalMatMul(Backend *backend, const MatMul *matmul, bool withBias) : MetalExecution(backend) {
    mTransposeA = matmul->transposeA();
    mTransposeB = matmul->transposeB();
    auto mkbn = static_cast<MetalBackend *>(backend);
    mConstBuffer = mkbn->getConstBuffer(sizeof(matP));
    auto context = (__bridge MNNMetalContext *)mkbn->context();
    if (withBias) {
        mPipeline = [context pipelineWithName:@"matmul_bias" fp16:mkbn->useFp16InsteadFp32()];
    } else {
        mPipeline = [context pipelineWithName:@"matmul" fp16:mkbn->useFp16InsteadFp32()];
    }
}
MetalMatMul::~MetalMatMul() {
    auto mkbn = static_cast<MetalBackend *>(backend());
    mkbn->returnConstBuffer(mConstBuffer);
}

ErrorCode MetalMatMul::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor* C       = outputs[0];
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    auto e = C->length(0);
    auto h = C->length(1);
    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }
    matP buffer;
    buffer.size[0] = h;
    buffer.size[1] = e;
    buffer.size[2] = l;
    if (mTransposeA) {
        buffer.stride[0] = 1;
        buffer.stride[1] = e;
    } else {
        buffer.stride[0] = l;
        buffer.stride[1] = 1;
    }
    if (mTransposeB) {
        buffer.stride[2] = l;
        buffer.stride[3] = 1;
    } else {
        buffer.stride[2] = 1;
        buffer.stride[3] = h;
    }
    
    ::memcpy(mConstBuffer.contents, &buffer, sizeof(matP));
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    mThreads = [context computeBestGroupAndLocal:mPipeline threads: MTLSizeMake(h, e, 1)];
    return NO_ERROR;
}

void MetalMatMul::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    auto input0 = inputs[0], input1 = inputs[1], output = outputs[0];
    Tensor* C       = outputs[0];
    auto e = C->length(0);
    auto h = C->length(1);
    
    if (inputs.size() > 2) {
        [encoder setComputePipelineState:mPipeline];
        MetalBackend::setTensor(input0, encoder, 0);
        MetalBackend::setTensor(input1, encoder, 1);
        MetalBackend::setTensor(inputs[2], encoder, 2);
        MetalBackend::setTensor(output, encoder, 3);
        [encoder setBuffer:mConstBuffer offset:0 atIndex:4];
    } else {
        [encoder setComputePipelineState:mPipeline];
        MetalBackend::setTensor(input0, encoder, 0);
        MetalBackend::setTensor(input1, encoder, 1);
        MetalBackend::setTensor(output, encoder, 2);
        [encoder setBuffer:mConstBuffer offset:0 atIndex:3];
    }
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
}

class MetalMatMulCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op,
                                Backend *backend, const std::vector<Tensor *>& outputs) const override {
        if(inputs.size() < 2) {
            MNN_PRINT("metal not support matmul inpt size less than 2\n");
            return nullptr;
        }
        return new MetalMatMul(backend, op->main_as_MatMul(), inputs.size() > 2);
    }
};
REGISTER_METAL_OP_CREATOR(MetalMatMulCreator, OpType_MatMul);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
