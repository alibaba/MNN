//
//  MetalUnary.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalUnary.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

static NSString *kernelForType(UnaryOpOperation type) {
#define op_case(type, imp)        \
    case UnaryOpOperation_##type: \
        return @"unary_" #imp "_x4"
    switch (type) {
        op_case(ABS, abs);
        op_case(NEG, neg);
        op_case(FLOOR, floor);
        op_case(CEIL, ceil);
        op_case(ROUND, round);
        op_case(SQUARE, square);
        op_case(SQRT, sqrt);
        op_case(RSQRT, rsqrt);
        op_case(EXP, exp);
        op_case(EXPM1, expm1);
        op_case(LOG, log);
        op_case(SIN, sin);
        op_case(COS, cos);
        op_case(TAN, tan);
        op_case(TANH, tanh);
        op_case(SIGMOID, sigmoid);
        op_case(ASIN, asin);
        op_case(ACOS, acos);
        op_case(ATAN, atan);
        op_case(SIGN, sign);
        op_case(RECIPROCAL, reciprocal);
        op_case(LOG1P, log1p);
        op_case(ACOSH, acosh);
        op_case(COSH, cosh);
        op_case(SINH, sinh);
        op_case(ASINH, asinh);
        op_case(ATANH, atanh);
        op_case(HARDSWISH, hardswish);
        op_case(GELU, gelu);
        default:
            FUNC_PRINT_ALL(EnumNameUnaryOpOperation(type), s);
            return nil;
    }
}

MetalUnary::MetalUnary(Backend *backend, UnaryOpOperation optype) : Execution(backend), mOpType(optype) {
    auto mtbn = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mConstBuffer                 = [context newDeviceBuffer:3 * sizeof(int) access:CPUWriteOnly];
    mPipeline = [context pipelineWithName:kernelForType(mOpType)];
}
ErrorCode MetalUnary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    auto input = inputs[0];
    auto element = input->elementSize();
    auto sizeDiv4 = UP_DIV(element, 4);
    ((int *)mConstBuffer.contents)[0] = sizeDiv4;
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(sizeDiv4, 1, 1)];
    return NO_ERROR;
}

ErrorCode MetalUnary::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    
    if(backend->isCommandEncoderSet()) {
        return NO_ERROR;
    }
    
    auto func = [=](){
        auto input = inputs[0], output = outputs[0];
        auto encoder   = backend->encoder();
        [encoder setComputePipelineState:mPipeline];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:1];
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
        
        auto context = (__bridge MNNMetalContext *)backend->context();
        if(backend->isCmdBufferCommit()) {
            backend->flushEncoder();
            [context commit_net];
        }
    };
    func();
    backend->addOpEncoder(func);
    return NO_ERROR;
}

class MetalUnaryCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        if (op->type() == OpType_TanH) {
            return new MetalUnary(backend, UnaryOpOperation_TANH);
        }
        if (op->type() == OpType_Sigmoid) {
            return new MetalUnary(backend, UnaryOpOperation_SIGMOID);
        }
        auto optype = op->main_as_UnaryOp()->opType();
        if (UnaryOpOperation_ERF == optype || UnaryOpOperation_ERFC == optype || UnaryOpOperation_ERFINV == optype) {
            return nullptr;
        }
        return new MetalUnary(backend, optype);
    }
};
REGISTER_METAL_OP_CREATOR(MetalUnaryCreator, OpType_UnaryOp);
REGISTER_METAL_OP_CREATOR(MetalUnaryCreator, OpType_TanH);
REGISTER_METAL_OP_CREATOR(MetalUnaryCreator, OpType_Sigmoid);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
