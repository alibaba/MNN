//
//  MetalFuse.mm
//  MNN
//
//  Created by MNN on 2022/11/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalFuse.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#import "AllShader.hpp"
#include <sstream>

#if MNN_METAL_ENABLED
namespace MNN {
// #define MNN_FUSE_DEBUG
MetalFuse::MetalFuse(Backend *backend, const Op* op) : Execution(backend), mOp(op) {
    auto mtbn = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mConstBuffer                 = [context newDeviceBuffer:3 * sizeof(int) access:CPUWriteOnly];
    auto extra = op->main_as_Extra();
    const char* srcCode = reinterpret_cast<const char*>(extra->info()->data());
    std::ostringstream ss;
    ss << shader_MetalDefine_metal << "\n" << srcCode;
#ifdef MNN_FUSE_DEBUG
    MNN_PRINT("MetalFuse srcCode:\n%s\n", srcCode);
#endif
    auto source = [[NSString alloc] initWithUTF8String:ss.str().c_str()];
    auto name = [[NSString alloc] initWithUTF8String:extra->type()->c_str()];
    mPipeline = [context pipelineWithSource:source name:name];
}

ErrorCode MetalFuse::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    auto input = inputs[0];
    auto element = input->elementSize();
    auto sizeDiv4 = UP_DIV(element, 4);
    ((int *)mConstBuffer.contents)[0] = sizeDiv4;
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(sizeDiv4, 1, 1)];
    return NO_ERROR;
}

ErrorCode MetalFuse::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    
    if(backend->isCommandEncoderSet()) {
        return NO_ERROR;
    }
    
    auto func = [=](){
        auto input = inputs[0], output = outputs[0];
        auto encoder   = backend->encoder();
        [encoder setComputePipelineState:mPipeline];
        int i = 0;
        for (; i < inputs.size(); i++) {
            [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)inputs[i]->deviceId())->getBuffer() offset:TensorUtils::getDescribe(inputs[i])->extra.offset atIndex:i];
        }
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:i++];
        [encoder setBuffer:mConstBuffer offset:0 atIndex:i++];
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
#ifdef MNN_FUSE_DEBUG
        auto dump = [&backend](const Tensor* t) {
            auto outDimType = t->getDimensionType();
            auto expectTensor = new MNN::Tensor(t, outDimType);
            backend->onCopyBuffer(t, expectTensor);
            MNN_PRINT("[ ");
            for (int i = 0; i < 10; i++) {
                MNN_PRINT("%f, ", expectTensor->host<float>()[i]);
            }
            MNN_PRINT(" ]\n");
            delete expectTensor;
        };
        {
            MNN_PRINT("=============================\n");
            for (int i = 0; i < inputs.size(); i++) {
                inputs[i]->wait(Tensor::MAP_TENSOR_READ, true);
                dump(inputs[i]);
            }
            output->wait(Tensor::MAP_TENSOR_READ, true);
            dump(output);
            MNN_PRINT("=============================\n");
        }
#endif
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

class MetalFuseCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        return new MetalFuse(backend, op);
    }
};
REGISTER_METAL_OP_CREATOR(MetalFuseCreator, OpType_Extra);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
