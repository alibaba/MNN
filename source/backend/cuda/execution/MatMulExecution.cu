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
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto C = outputs[0];
    auto dimensions = C->dimensions();
    int batch = 1;
    for (int i = 0; i < dimensions - 2; ++i) {
        batch *= C->length(i);
    }
    auto e = C->length(dimensions-2);
    auto h = C->length(dimensions-1);
    auto w0 = inputs[0]->length(dimensions-1);
    auto h0 = inputs[0]->length(dimensions-2);

    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }
    MatMulParam& param = mParam;
    param.elh[0] = e;
    param.elh[1] = l;
    param.elh[2] = h;
    param.batch = batch;
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
    param.aPStride[0] = 256 * lU;
    param.aPStride[1] = 16;
    param.aPStride[2] = 16 * lU;
    param.bPStride[0] = 256 * lU;
    param.bPStride[1] = 16;
    param.bPStride[2] = 16 * lU;
    runtime->memcpy((uint8_t*)mParameters.first + mParameters.second, &param, sizeof(MatMulParam), MNNMemcpyHostToDevice);

    // Alloc for temp buffer
    auto aPackSize = eU * lU * PACK_MATMUL * PACK_MATMUL * batch;
    auto bPackSize = lU * hU * PACK_MATMUL * PACK_MATMUL * batch;

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
    int e = mParam.elh[0];
    int l = mParam.elh[1];
    int h = mParam.elh[2];
    int batch = mParam.batch;
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);

    auto aP = (__half*)((uint8_t*)mTempA.first + mTempA.second);
    auto bP = (__half*)((uint8_t*)mTempB.first + mTempB.second);
    const float* biasPtr = nullptr;
    if (inputs.size() > 2) {
        biasPtr = (const float*)inputs[2]->deviceId();
    }
    auto param = (MatMulParam*)((uint8_t*)mParameters.first + mParameters.second);
    GemmPrepareRerange(runtime, &mParam, param, APtr, aP, BPtr, bP, bytes);
    GemmPackedMain(runtime, &mParam, param, CDestPtr, aP, bP, biasPtr, bytes, false, false);

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