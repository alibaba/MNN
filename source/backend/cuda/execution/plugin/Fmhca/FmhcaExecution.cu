//
//  FmhcaExecution.cpp
//  MNN
//
//  Created by MNN on 2023/09/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "FmhcaExecution.hpp"
#include "../FmhaCommon/FmhaCommonExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace CUDA {
bool FmhcaExecution::isValid(const MNN::Op* op, Backend *backend, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto fmhca_param = op->main_as_FmhcaParam();
    int head_num = fmhca_param->heads();
    int head_size = outputs[0]->length(2)/head_num;
    int seq_kv = inputs[1]->length(2)/head_num;
    if(head_size != 64 && head_size != 128 && head_size != 256) {
        return false;
    }
    if(seq_kv > 128) {
        return false;
    }
    // If need acc with fp32, do not use
    return true;
}

FmhcaExecution::FmhcaExecution(const MNN::Op* op, Backend* backend) : Execution(backend) {
    auto fmhca_param = op->main_as_FmhcaParam();
    mNumHeads = fmhca_param->heads(); 
}
ErrorCode FmhcaExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto pool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    MNN_ASSERT(inputs.size() == 2);
    MNN_ASSERT(outputs.size() == 1);
    auto input0 = inputs[0];    
    auto input1 = inputs[1];
    auto output = outputs[0];

    mBatchSize = output->length(0);
    mSeqLenQ = output->length(1);
    mSeqLenKV = input1->length(1);
    if(mSeqLenKV > 128) {
        MNN_ERROR("MNN CUDA Fmhca only support sequence len <= 128 now!\n");
    }
    auto buffer_q = pool->alloc((mBatchSize+1) * sizeof(int32_t));
    mSeqLenQDevPtr = (void*)((uint8_t*)buffer_q.first + buffer_q.second);
    std::vector<int32_t> cuSeqLensQ(mBatchSize + 1, 0);
    // Compute the prefix sum of the1
    for (int32_t it = 0; it < mBatchSize; it++) {
        cuSeqLensQ[it + 1] = cuSeqLensQ[it] + mSeqLenQ;
    }

    runtime->memcpy(mSeqLenQDevPtr, cuSeqLensQ.data(), sizeof(int32_t) * cuSeqLensQ.size(), MNNMemcpyHostToDevice);
    checkKernelErrors;

    auto buffer_kv = pool->alloc((mBatchSize+1) * sizeof(int32_t));
    mSeqLenKVDevPtr = (void*)((uint8_t*)buffer_kv.first + buffer_kv.second);
    std::vector<int32_t> cuSeqLensKV(mBatchSize + 1, 0);
    // Compute the prefix sum of the1
    for (int32_t it = 0; it < mBatchSize; it++) {
        cuSeqLensKV[it + 1] = cuSeqLensKV[it] + mSeqLenKV;
    }

    runtime->memcpy(mSeqLenKVDevPtr, cuSeqLensKV.data(), sizeof(int32_t) * cuSeqLensKV.size(), MNNMemcpyHostToDevice);
    checkKernelErrors;

    mSM = runtime->compute_capability();

    if(static_cast<CUDABackend*>(backend())->useFp16()) {
        mKernels = getFMHCACubinKernels(DATA_TYPE_FP16, mSM);
    } else {
        mKernels = getFMHCACubinKernels(DATA_TYPE_FP32, mSM);
    }
    return NO_ERROR;
}

int32_t FmhcaExecution::runFMHCAKernel(void const* devQ, void const* devKV, void* cuSeqlensQ, void* cuSeqlensKV, void* devOutput,
    int32_t sm, FusedMultiHeadCrossAttentionKernel const* kernels, int32_t b, int32_t h, int32_t d, int32_t seqQ,
    int32_t seqKV, cudaStream_t stream)
{
    MNN_ASSERT(sm != 75 || d < 160);

    // Run kernel.
    Fused_multihead_attention_params_mhca params = getMHCAParams(/* dType */ DATA_TYPE_FP16,
        /* accType */ DATA_TYPE_FP16, b, seqQ, seqKV, h, d, /* total */ 0, devQ, devKV, cuSeqlensQ, cuSeqlensKV,
        devOutput, /* devP */ nullptr, /* devS */ nullptr, /* scaleBmm1 */ 1.F / sqrtf(d), /* scaleSoftmax */ 1.F,
        /* scaleBmm2 */ 1.F, /* interleaved */ false, /* ignoreB1Opt */ false,
        /* forceUnroll */ true, /* useInt8ScaleMax */ false, /* useTMA */ false);

    kernels->run(params, stream);
    checkKernelErrors;
    return 0;
}

ErrorCode FmhcaExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start FmhcaExecution onExecute...");
#endif

    //MNN_PRINT("fmha format:%d %d\n", MNN::TensorUtils::getDescribe(inputs[0])->dimensionFormat, MNN::TensorUtils::getDescribe(outputs[0])->dimensionFormat);

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    // launch kernel.
    constexpr int32_t seqLenKvPadded = 128;
    int32_t const headNum = mNumHeads;
    int32_t const sizePerHead = outputs[0]->length(2) / headNum;

    //printf("fmha shape  b:%d s:%d h_num:%d h_size:%d, %d\n", mBatchSize, mSeqLen, head_num, size_per_head, inputs[0]->length(3));
    runFMHCAKernel((const void *)inputs[0]->deviceId(), (const void *)inputs[1]->deviceId(), 
        mSeqLenQDevPtr, mSeqLenKVDevPtr, (void *)outputs[0]->deviceId(), mSM, mKernels,
        mBatchSize, headNum, sizePerHead, mSeqLenQ, seqLenKvPadded);
    checkKernelErrors;
#ifdef LOG_VERBOSE
    MNN_PRINT("end FmhcaExecution onExecute...");
#endif
    return NO_ERROR;
}


class FmhcaCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if(!static_cast<CUDABackend*>(backend)->useFp16()) {
            MNN_PRINT("CUDA Fmhca only support fp16 now!\n");
            return nullptr;
        }
        if(FmhcaExecution::isValid(op, backend, inputs, outputs)) {
            return new FmhcaExecution(op, backend);
        }
        return new FmhaCommonExecution(op, backend);
    }
};

CUDACreatorRegister<FmhcaCreator> __FmhcaExecution(OpType_Fmhca);
} // namespace CUDA
} // namespace MNN
#endif