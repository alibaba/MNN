//
//  FuseExecution.cpp
//  MNN
//
//  Created by MNN on 2023/06/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_CODEGEN_CUDA
#include "FuseExecution.hpp"
#include "FuseExecutionV2.hpp"
#include "core/OpCommonUtils.hpp"
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"

namespace MNN {
namespace CUDA {

FuseExecution::FuseExecution(const Op* op, Backend *backend) : Execution(backend) {
    // AUTOTIME;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime(); 
    auto extra = op->main_as_Extra();
    std::string source(reinterpret_cast<const char*>(extra->info()->data()));
    mSource = source;
    mName = extra->type()->c_str();
    mVectorize = extra->vector();
    // MNN_PRINT("\n\n%s\n\n%s \n\n", mSource.c_str(), mName);

    auto kernelInfoMap = static_cast<CUDABackend*>(backend)->kernelCuModuleMap();
    auto module = kernelInfoMap[std::pair<std::string, std::string>(mName, mSource)];
    MNN_CUDA_SAFE_CALL(cuModuleGetFunction(&mKernel, module, mName));
}

ErrorCode FuseExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime(); 
    auto output =outputs[0];
    auto format = TensorUtils::getDescribe(output)->dimensionFormat;
    auto dims = output->dimensions();
    batch = output->length(0);
    if (format == MNN_DATA_FORMAT_NHWC) {
        channel = output->length(dims-1);
        channel_pack = UP_DIV(channel, PACK_NUMBER) * PACK_NUMBER;
        area = 1;
        for(int i = 1; i < dims-1; i++) {
            area *= output->length(i);
        }

        if (mVectorize) { // Fast vectorize
            if(static_cast<CUDABackend*>(backend())->useFp16()) { // half2
                channel = channel / 2;
                channel_pack = channel_pack / 2; 
            } else { // float4
                channel = channel / 4;
                channel_pack = channel_pack / 4;
            }
        }
    } else if(format == MNN_DATA_FORMAT_NCHW || format == MNN_DATA_FORMAT_NC4HW4) {
        channel = output->length(1);
        channel_pack = UP_DIV(channel, PACK_NUMBER) * PACK_NUMBER;
        area = 1;
        for(int i = 2; i < dims; i++) {
            area *= output->length(i);
        }
    } else {
        MNN_ERROR("FuseExecution not support format:%d\n", format);
        MNN_ASSERT(false);
    }

    #if 0 // TODO : Optimize raster
    DivModFast d_area(area);
    DivModFast d_channel(channel);

    mDivChannelStorage = static_cast<CUDABackend*>(backend())->getStaticBufferPool()->alloc(sizeof(DivModFast));
    mDivAreaStorage = static_cast<CUDABackend*>(backend())->getStaticBufferPool()->alloc(sizeof(DivModFast));
    runtime->memcpy((uint8_t*)mDivAreaStorage.first + mDivAreaStorage.second, &d_area, sizeof(DivModFast), MNNMemcpyHostToDevice, true);
    runtime->memcpy((uint8_t*)mDivChannelStorage.first + mDivChannelStorage.second, &d_channel, sizeof(DivModFast), MNNMemcpyHostToDevice, true);
    #endif

    return NO_ERROR;
}

ErrorCode FuseExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto count = CUDABackend::realSize(outputs[0]);

    if(mVectorize) {
        if(static_cast<CUDABackend*>(backend())->useFp16()) { // half2
            count = count / 2;
        } else {
            count = count / 4;
        }
    }
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto& prop = runtime->prop();
    int threads_num = runtime->threads_num();//prop.maxThreadsPerBlock;
    int block_num = runtime->blocks_num(count);// prop.multiProcessorCount;

    std::vector<void *> args;

    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        for (int i=0; i < inputs.size(); i++) {
            auto inputPtr = (const half*)inputs[i]->deviceId();
            args.emplace_back((void *)inputPtr);
        }
        for (int i=0; i < outputs.size(); i++) {
            auto outputPtr = (const half*)outputs[i]->deviceId();
            args.emplace_back((void *)outputPtr);
        }
    } else {
        for (int i=0; i < inputs.size(); i++) {
            auto inputPtr = (const float*)inputs[i]->deviceId();
            args.emplace_back((void *)inputPtr);
        }
        for (int i=0; i < outputs.size(); i++) {
            auto outputPtr = (const float*)outputs[i]->deviceId();
            args.emplace_back((void *)outputPtr);
        }
    }

    args.emplace_back((void *)count);

    //TODO : when can do not pass these params
    args.emplace_back((void *)batch);
    args.emplace_back((void *)area);
    args.emplace_back((void *)channel);
    args.emplace_back((void *)channel_pack);
    // args.emplace_back((void *)(DivModFast *)((uint8_t*)mDivAreaStorage.first + mDivAreaStorage.second));
    // args.emplace_back((void *)(DivModFast *)((uint8_t*)mDivChannelStorage.first + mDivChannelStorage.second));
    
    std::vector<void*> argsPtr;
    for(int i=0; i<args.size(); i++) {
        argsPtr.emplace_back(args.data() + i);
    }

    //printf("size  %p-%p-%p-%d   %p-%p-%p-%d\n\n", (const float*)inputs[0]->deviceId(), (const float*)inputs[1]->deargsviceId(), (const float*)outputs[0]->deviceId(), count, argsPtr[0], argsPtr[1], argsPtr[2], *((argsPtr[3])));

    MNN_CUDA_SAFE_CALL(
        cuLaunchKernel(mKernel,
        block_num, 1, 1, // grid dim
        threads_num, 1, 1, // block dim
        0, NULL, // shared mem 
        &(argsPtr[0]), 0)); // arguments

    return NO_ERROR;
}
class FuseCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (FuseExecutionV2::check(op)) {
            return FuseExecutionV2::create(op, backend, inputs.size(), outputs.size());
        }
        return new FuseExecution(op, backend);
    }
};

static CUDACreatorRegister<FuseCreator> __init(OpType_Extra);

};
};
#endif