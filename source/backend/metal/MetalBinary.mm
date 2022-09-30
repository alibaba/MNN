//
//  MetalBinary.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalBinary.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalBinary::MetalBinary(Backend *backend, std::string type, const MNN::Op *op) : Execution(backend) {
    auto mKernelName = "binary_" + type + "_x1";
    auto mtbn = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mConstBuffer             = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    auto kn = [NSString stringWithCString:mKernelName.c_str() encoding:[NSString defaultCStringEncoding]];
    mPipeline = [context pipelineWithName:kn];
    mActivationType = op->main_as_BinaryOp()->activationType();
}
ErrorCode MetalBinary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input0 = inputs[0], input1 = inputs[1], output = outputs[0];
    const int input0_data_count = (int)input0->elementSize();
    const int input1_data_count = (int)input1->elementSize();

    int outdatacount = output->elementSize();
    ((int *)mConstBuffer.contents)[0] = input0_data_count == 1 ? 0 : 1;
    ((int *)mConstBuffer.contents)[1] = input1_data_count == 1 ? 0 : 1;
    ((int *)mConstBuffer.contents)[2] = outdatacount;
    ((int *)mConstBuffer.contents)[3] = mActivationType;
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(outdatacount, 1, 1)];
    return NO_ERROR;
}

ErrorCode MetalBinary::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());

    if(backend->isCommandEncoderSet()) {
        return NO_ERROR;
    }
    
    auto func = [=](){
        auto input0 = inputs[0], input1 = inputs[1], output = outputs[0];
        auto encoder           = backend->encoder();
        [encoder setComputePipelineState:mPipeline];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input0->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input0)->extra.offset atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input1->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input1)->extra.offset atIndex:1];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:2];
        [encoder setBuffer:mConstBuffer offset:0 atIndex:3];
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

#define CHECK(t, i) if (originOp == t) return i;
static std::string _convert(int originOp) {
    CHECK(BinaryOpOperation_ADD, "add");
    CHECK(BinaryOpOperation_SUB, "sub");
    CHECK(BinaryOpOperation_MUL, "mul");
    CHECK(BinaryOpOperation_MOD, "mod");
    CHECK(BinaryOpOperation_FLOORMOD, "floormod");
    CHECK(BinaryOpOperation_MINIMUM, "min");
    CHECK(BinaryOpOperation_MAXIMUM, "max");
    CHECK(BinaryOpOperation_DIV, "div");
    CHECK(BinaryOpOperation_REALDIV, "div");
    CHECK(BinaryOpOperation_POW, "pow");
    CHECK(BinaryOpOperation_SquaredDifference, "squared_diff");
    return "";
}

class MetalBinaryCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        auto binaryop = op->main_as_BinaryOp();
        auto type = _convert(binaryop->opType());
        if (inputs[0]->getType().code != halide_type_float) {
            return nullptr;
        }
        if (type.empty()) {
            FUNC_PRINT(binaryop->opType());
            return nullptr;
        }
        return new MetalBinary(backend, type, op);
    }
};
REGISTER_METAL_OP_CREATOR(MetalBinaryCreator, OpType_BinaryOp);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
