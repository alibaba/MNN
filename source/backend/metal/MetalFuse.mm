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
MetalFuse::MetalFuse(Backend *backend, const Op* op) : MetalExecution(backend), mOp(op) {
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
    mPipeline = mtbn->makeComputePipelineWithSourceOption(ss.str().c_str(), extra->type()->c_str(), nil);
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

void MetalFuse::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto input = inputs[0], output = outputs[0];
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
}

static bool _isStandardFuse(const Op* op) {
    if (op->type() != OpType_Extra) {
        return false;
    }
    if (nullptr == op->main_as_Extra()) {
        return false;
    }
    auto extra = op->main_as_Extra();
    if (nullptr == extra->attr()) {
        return false;
    }
    for (int i=0; i<extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr->key()->str() == "version") {
            if (nullptr != attr->s()) {
                std::string cont = attr->s()->str();
                return cont == "common";
            }
            return false;
        }
    }
    return false;
}
class MetalFuseV2 : public MetalExecution {
private:
    MTLSize mGroupSize;
    MTLSize mThreadSize;
    std::vector<int> mInputBinding;
    std::vector<int> mOutputBinding;
    std::vector<std::pair<int, std::pair<id<MTLBuffer>, size_t>>> mConstIndides;
    id<MTLComputePipelineState> mPipeline;
    MTLSize mGlobalSize;
    bool mSupportTuning = false;

public:
    MetalFuseV2(Backend *backend, const Op* op, int outputSize, int inputSize) : MetalExecution(backend) {
        mOutputBinding.resize(outputSize);
        mInputBinding.resize(inputSize);
        auto mtbn = static_cast<MetalBackend*>(backend);
        auto context = (__bridge MNNMetalContext *)mtbn->context();
        auto extra = op->main_as_Extra();
        // Find shader
        const char* source = nil;
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "metal") {
                source = attr->s()->c_str();
                break;
            }
        }
        mPipeline = mtbn->makeComputePipelineWithSourceOption(source, "main0", nil);

        // Init size
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "group_size") {
                auto ptr = attr->tensor()->int32s()->data();
                mGroupSize.width = ptr[0];
                mGroupSize.height = ptr[1];
                mGroupSize.depth = ptr[2];
                break;
            }
        }
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "global_size") {
                auto ptr = attr->tensor()->int32s()->data();
                mGlobalSize.width = ptr[0];
                mGlobalSize.height = ptr[1];
                mGlobalSize.depth = ptr[2];
                mSupportTuning = true;
                break;
            }
        }
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "local_size") {
                auto ptr = attr->tensor()->int32s()->data();
                mThreadSize.width = ptr[0];
                mThreadSize.height = ptr[1];
                mThreadSize.depth = ptr[2];
                break;
            }
        }
        int maxIndex = -1;
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "input") {
                maxIndex = ALIMAX(maxIndex, attr->i());
            } else if (attr->key()->str() == "const") {
                maxIndex = ALIMAX(maxIndex, attr->i());
            }
        }
        for (int i=0; i<extra->attr()->size(); ++i) {
            auto attr = extra->attr()->GetAs<Attribute>(i);
            if (attr->key()->str() == "input") {
                auto list = attr->list()->i()->data();
                if (list[1] >= 0) {
                    if (0 == list[0]) {
                        mInputBinding[list[1]] = attr->i();
                    } else {
                        mOutputBinding[list[1]] = attr->i();
                    }
                }
                continue;
            }
            if (attr->key()->str() == "const") {
                auto b = attr->tensor();
                void* result = nullptr;
                size_t bufferSize = 0;
                switch (b->dataType()) {
                    case DataType_DT_FLOAT:
                        result = (void*)b->float32s()->Data();
                        bufferSize = b->float32s()->size() * sizeof(float);
                        break;
                    case DataType_DT_INT32:
                        result = (void*)b->int32s()->Data();
                        bufferSize = b->int32s()->size() * sizeof(float);
                        break;
                    default:
                        MNN_ASSERT(false);
                        break;
                }
                // TODO: Fuse All Const Buffer to One buffer
                id<MTLBuffer> constBuffer = [context newDeviceBuffer:bufferSize access:CPUWriteOnly];
                ::memcpy([constBuffer contents], result, bufferSize);
                
                mConstIndides.emplace_back(std::make_pair(attr->i(), std::make_pair(constBuffer, 0)));
                continue;
            }
        }
    }
    virtual ~MetalFuseV2() = default;
    void encode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder, MTLSize localSize, MTLSize groupSize) {
        [encoder setComputePipelineState:mPipeline];
        for (int i=0; i<inputs.size(); ++i) {
            [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)inputs[i]->deviceId())->getBuffer() offset:TensorUtils::getDescribe(inputs[i])->extra.offset atIndex:mInputBinding[i]];
        }
        for (int i=0; i<outputs.size(); ++i) {
            [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)outputs[i]->deviceId())->getBuffer() offset:TensorUtils::getDescribe(outputs[i])->extra.offset atIndex:mOutputBinding[i]];
        }
        for (int i=0; i<mConstIndides.size(); ++i) {
            [encoder setBuffer:mConstIndides[i].second.first offset:0 atIndex:mConstIndides[i].first];
        }
        [encoder dispatchThreadgroups:groupSize threadsPerThreadgroup:localSize];
    }
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override {
        encode(inputs, outputs, encoder, mThreadSize, mGroupSize);
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        if (!mSupportTuning) {
            return NO_ERROR;
        }
        auto backend = static_cast<MetalBackend *>(this->backend());
        auto rt = (MetalRuntime*)(backend->runtime());
        auto context = (__bridge MNNMetalContext *)backend->context();

        //get original trick time
        std::pair<MTLSize, MTLSize> thread = std::make_pair(mGroupSize, mThreadSize);
        int count = 10;
        auto cmdqueue = backend->queue();
        NSUInteger min_time = UINT_MAX;
        {
            id<MTLCommandBuffer> commamd_buffer = [context newCmdBuffer:thread.second queue:cmdqueue];
            id<MTLComputeCommandEncoder> encoder = [commamd_buffer computeCommandEncoder];
            
            int loop = count;
            while(loop--) {
                encode(inputs, outputs, encoder, thread.second, thread.first);
            }
            [encoder endEncoding];
            min_time = [context timeUsed: commamd_buffer];
            //MNN_PRINT("orig prit: %d   us, %d %d %d\n", min_time, thread.second.width, thread.second.height, thread.second.depth);
        }
        
        bool isMuchTime = (min_time > 8000) ? true : false;
        NSUInteger magic_l = 1;
        NSUInteger magic_z = 4;
        NSUInteger magic_y = 4;
        NSUInteger magic_x = 4;
        
        NSUInteger gid_x = mGlobalSize.width;
        NSUInteger gid_y = mGlobalSize.height;
        NSUInteger gid_z = mGlobalSize.depth;
        auto maxSize = [[context device] maxThreadsPerThreadgroup];
        int limitSize = [mPipeline maxTotalThreadsPerThreadgroup];

        for(NSUInteger z = 1; z <= gid_z * magic_l && z <= magic_z && z < maxSize.depth; z *= 4) {
            for(NSUInteger y = 1; y <= gid_y * magic_l && y <= magic_y && y <= maxSize.height; y *= 4) {
                for(NSUInteger x = 1; x < gid_x * magic_l && x <= magic_x && x<= maxSize.width; x *= 4) {
                    if(x * y * z <= limitSize) {
                        if(x==1 && y==1 && z==1) {
                            continue;
                        }
                        MTLSize local = {x, y, z};
                        MTLSize global = {UP_DIV(gid_x, x), UP_DIV(gid_y, y), UP_DIV(gid_z, z)};
                        id<MTLCommandBuffer> commamd_buffer = [cmdqueue commandBuffer];
                        id<MTLComputeCommandEncoder> encoder = [commamd_buffer computeCommandEncoder];

                        int loop = count;
                        while(loop--) {
                            encode(inputs, outputs, encoder, local, global);
                        }
                        [encoder endEncoding];
                        auto time = [context timeUsed :commamd_buffer];
                        if(time < min_time) {
                            min_time = time;
                            thread.first = global;
                            thread.second = local;
                        }
                    }
                }
            }
        }
        mThreadSize = thread.second;
        mGroupSize = thread.first;

        return NO_ERROR;
    }
};

class MetalFuseCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        if (_isStandardFuse(op)) {
            return new MetalFuseV2(backend, op, (int)outputs.size(), (int)inputs.size());
        }
        return new MetalFuse(backend, op);
    }
};
REGISTER_METAL_OP_CREATOR(MetalFuseCreator, OpType_Extra);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
