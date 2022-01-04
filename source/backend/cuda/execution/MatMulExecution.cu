#include "MatMulExecution.hpp"
namespace MNN {
namespace CUDA {
#define PACK_MATMUL 16

MatMulExecution::MatMulExecution(bool transposeA, bool transposeB, Backend *backend) : Execution(backend) {
    mTransposeA = transposeA;
    mTransposeB = transposeB;
    auto staticPool = static_cast<CUDABackend*>(backend)->getStaticBufferPool();
    mParameters = staticPool->alloc(sizeof(MatMulParam));
}
MatMulExecution::~ MatMulExecution() {
    auto staticPool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
    staticPool->free(mParameters);
}

ErrorCode MatMulExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    auto C = outputs[0];

    auto e = C->length(0);
    auto h = C->length(1);
    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }
    MatMulParam& param = mParam;
    param.elh[0] = e;
    param.elh[1] = l;
    param.elh[2] = h;
    auto eU = UP_DIV(e, PACK_MATMUL);
    auto lU = UP_DIV(l, PACK_MATMUL);
    auto hU = UP_DIV(h, PACK_MATMUL);

    param.elhPack[0] = eU;
    param.elhPack[1] = lU;
    param.elhPack[2] = hU;

    // compute src stride
    param.aStride[2] = 0;
    if (mTransposeA) {
        param.aStride[0] = 1;
        param.aStride[1] = e;
    } else {
        param.aStride[0] = l;
        param.aStride[1] = 1;
    }

    param.bStride[0] = 0;
    if (mTransposeB) {
        param.bStride[1] = 1;
        param.bStride[2] = l;
    } else {
        param.bStride[1] = h;
        param.bStride[2] = 1;
    }
    param.cStride[0] = h;
    param.cStride[1] = 0;
    param.cStride[2] = 1;
    param.split[0] = 1;
    param.split[1] = 1;
    param.split[2] = 1;
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    runtime->memcpy((uint8_t*)mParameters.first + mParameters.second, &param, sizeof(MatMulParam), MNNMemcpyHostToDevice);

    // Alloc for temp buffer
    auto aPackSize = eU * lU * PACK_MATMUL * PACK_MATMUL;
    auto bPackSize = lU * hU * PACK_MATMUL * PACK_MATMUL;

    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    mTempA = pool->alloc(aPackSize * sizeof(__half), false, 256);
    mTempB = pool->alloc(bPackSize * sizeof(__half), false, 256);
    pool->free(mTempA);
    pool->free(mTempB);
    return NO_ERROR;
}

ErrorCode MatMulExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    auto C = outputs[0];

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto APtr = (const float*)A->deviceId();
    auto BPtr = (const float*)B->deviceId();
    auto CDestPtr = (float*)C->deviceId();

    auto aP = (__half*)((uint8_t*)mTempA.first + mTempA.second);
    auto bP = (__half*)((uint8_t*)mTempB.first + mTempB.second);
    const float* biasPtr = nullptr;
    if (inputs.size() > 2) {
        biasPtr = (const float*)inputs[2]->deviceId();
    }
    auto param = (MatMulParam*)((uint8_t*)mParameters.first + mParameters.second);
    GemmPrepareRerange(runtime, &mParam, param, APtr, aP, BPtr, bP);
    GemmPackedMain(runtime, &mParam, param, CDestPtr, aP, bP, biasPtr);
    return NO_ERROR;

    auto blasHandle = runtime->cublas_handle();
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);

    auto e = C->length(0);
    auto h = C->length(1);
    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    auto tranB = CUBLAS_OP_N;
    auto ldB = h;
    if (mTransposeB) {
        ldB = l;
        tranB = CUBLAS_OP_T;
    }
    auto tranA = CUBLAS_OP_N;
    auto ldA = l;
    if (mTransposeA) {
        ldA = e;
        tranA = CUBLAS_OP_T;
    }
    int block_num = runtime->blocks_num(e*h);
    int threads_num = runtime->threads_num();
    
    //[e, l] x [l, h] -> [e, h]
        auto status = cublasSgemm(blasHandle, tranB, tranA, h, e, l, &alpha, BPtr, ldB, APtr, ldA, &beta, CDestPtr, h);
        cublas_check(status);
        //cudaThreadSynchronize();
    // } else {
    //     auto CPtr = (float*)mTempOutput->deviceId();
    //     auto status = cublasSgemm(blasHandle, tranB, tranA, h, e, l, &alpha, BPtr, ldB, APtr, ldA, &beta, CPtr, h);
    //     cublas_check(status);
    //     //cudaThreadSynchronize();

    //     //bias: [e, h] + [h] -> [e, h]
    //     add_bias<<<block_num, threads_num>>>((float*)CPtr, (float*)CDestPtr, (const float*)inputs[2]->deviceId(), e, h);
    // }

    return NO_ERROR;
}

class MatMulCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_MatMul();
        return new MatMulExecution(param->transposeA(), param->transposeB(), backend);
    }
};

static CUDACreatorRegister<MatMulCreator> __init(OpType_MatMul);

}
}