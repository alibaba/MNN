//
//  FmhaV2Execution.cpp
//  MNN
//
//  Created by MNN on 2023/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include "FmhaV2Execution.hpp"
#include "../FmhaCommon/FmhaCommonExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace CUDA {

bool FmhaV2Execution::isValid(const MNN::Op* op, Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto fmha_v2_param = op->main_as_FmhaV2Param();
    int head_num = fmha_v2_param->heads();
    int head_size = outputs[0]->length(2)/head_num;
    if(head_size != 16 && head_size != 32 && head_size != 40 && head_size != 64 && head_size != 80 && head_size != 128) {
        return false;
    }
    // If need acc with fp32, do not use
    return true;
}
FmhaV2Execution::FmhaV2Execution(const MNN::Op* op, Backend* backend) : Execution(backend) {
    auto fmha_v2_param = op->main_as_FmhaV2Param();
    mNumHeads = fmha_v2_param->heads();    
}
ErrorCode FmhaV2Execution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto pool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];
    MNN_ASSERT(input->dimensions() == 3);
    MNN_ASSERT(output->dimensions() == 3);

    mBatchSize = output->length(0);
    mSeqLen = output->length(1);
    auto buffer_data = pool->alloc((mBatchSize+1) * sizeof(int32_t));
    mSeqLenDevPtr = (void*)((uint8_t*)buffer_data.first + buffer_data.second);
    std::vector<int32_t> cuSeqLens(mBatchSize + 1, 0);
    // Compute the prefix sum of the1
    for (int32_t it = 0; it < mBatchSize; it++) {
        cuSeqLens[it + 1] = cuSeqLens[it] + mSeqLen;
    }

    runtime->memcpy(mSeqLenDevPtr, cuSeqLens.data(), sizeof(int32_t) * cuSeqLens.size(), MNNMemcpyHostToDevice);
    checkKernelErrors;
    mSM = runtime->compute_capability();

    if(static_cast<CUDABackend*>(backend())->useFp16()) {
        mKernels = getFMHAFlashCubinKernels(DATA_TYPE_FP16, mSM);
    } else {
        mKernels = getFMHAFlashCubinKernels(DATA_TYPE_FP32, mSM);
    }
    return NO_ERROR;
}

int32_t FmhaV2Execution::runFMHFAKernel(void const* devQKV, void* cuSeqlens, void* devOutput, size_t total, int32_t sm,
    FusedMultiHeadFlashAttentionKernel const* kernels, int32_t b, int32_t h, int32_t d, int32_t s, cudaStream_t stream)
{
    Fused_multihead_flash_attention_params_v2 params
        = getMHFAParams(/* data_type */ DATA_TYPE_FP16, /* acc_type */ DATA_TYPE_FP16, b, s, h, d, total, devQKV,
            cuSeqlens, devOutput, /* p_d */ nullptr, /* s_d */ nullptr,
            /* scale_bmm1 */ 1.F / sqrtf(d), /* scale_softmax */ 1.F, /* scale_bmm2 */ 1.F,
            /* interleaved */ false,
            /* ignore_b1opt */ false,
            /* force_unroll */ true,
            /* use_int8_scale_max  */ false);
    //MNN_PRINT("%p %p %p, %d %d, %p, %d %d %d %d %p\n", devQKV, cuSeqlens, devOutput, total, sm, kernels, b, h, d, s, stream);

    kernels->run(params, stream);
    checkKernelErrors;
    return 0;
}

ErrorCode FmhaV2Execution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start FmhaV2Execution onExecute...");
#endif

    //MNN_PRINT("fmha format:%d %d\n", MNN::TensorUtils::getDescribe(inputs[0])->dimensionFormat, MNN::TensorUtils::getDescribe(outputs[0])->dimensionFormat);

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    // launch kernel.
    int32_t const head_num = mNumHeads;
    int32_t const size_per_head = outputs[0]->length(2)/head_num;

    size_t const total = mBatchSize * mSeqLen;
    // printf("fmha shape  b:%d s:%d h_num:%d h_size:%d, %d\n", mBatchSize, mSeqLen, head_num, size_per_head, inputs[0]->dimensions());
    runFMHFAKernel((const void *)inputs[0]->deviceId(), mSeqLenDevPtr, (void *)outputs[0]->deviceId(), total, mSM, mKernels,
        mBatchSize, head_num, size_per_head, mSeqLen);
    checkKernelErrors;
#ifdef LOG_VERBOSE
    MNN_PRINT("end FmhaV2Execution onExecute...");
#endif
    return NO_ERROR;
}


class FmhaV2Creator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if(!static_cast<CUDABackend*>(backend)->useFp16()) {
            MNN_PRINT("CUDA FmhaV2 only support fp16 now!\n");
            return nullptr;
        }
        if(FmhaV2Execution::isValid(op, backend, inputs, outputs)) {
            return new FmhaV2Execution(op, backend);
        }
        return new FmhaCommonExecution(op, backend);
    }
};

CUDACreatorRegister<FmhaV2Creator> __FmhaV2Execution(OpType_FmhaV2);
} // namespace CUDA
} // namespace MNN
#endif