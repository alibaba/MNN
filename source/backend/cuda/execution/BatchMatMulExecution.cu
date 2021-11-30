#include "BatchMatMulExecution.hpp"
namespace MNN {
namespace CUDA {

template <typename T>
__global__ void add_bias(T *input, T *output, const T* bias, int batch, int e, int h) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch * e * h; index += blockDim.x * gridDim.x) {
        int i = index % (e*h);
        int b = index / (e*h);
        int y = i % h;
        output[index] = input[index] + bias[b * h + y];
    }
    return;
}
BatchMatMulExecution::BatchMatMulExecution(bool transposeA, bool transposeB, Backend *backend) : Execution(backend) {
    mTransposeA = transposeA;
    mTransposeB = transposeB;
}
BatchMatMulExecution::~ BatchMatMulExecution() {
    // do nothing
}

ErrorCode BatchMatMulExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto C = outputs[0];

    auto dimensions = C->dimensions();
    int batch = 1;
    for (int i = 0; i < dimensions - 2; ++i) {
        batch *= C->length(i);
    }
    auto e = C->length(dimensions-2);
    auto h = C->length(dimensions-1);
    if(inputs.size() > 2) {
        mTempOutput.reset(Tensor::createDevice<float>({batch*h*e}));
        auto res = backend()->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        backend()->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

ErrorCode BatchMatMulExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto blasHandle = runtime->cublas_handle();
    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];

    auto dimensions = A->dimensions();
    int batch = 1;
    for (int i = 0; i < dimensions - 2; ++i) {
        batch *= A->length(i);
    }

    auto w0         = inputs[0]->length(dimensions-1);
    auto h0         = inputs[0]->length(dimensions-2);
    auto C = outputs[0];

    auto e = C->length(dimensions-2);
    auto h = C->length(dimensions-1);
    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }
    auto APtr = (const float*)A->deviceId();
    auto BPtr = (const float*)B->deviceId();
    auto CDestPtr = (float*)C->deviceId();

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

    // [b, e, l] x [b, l, h] -> [b, e, h]
    if(inputs.size() == 2) {    
        auto status = cublasSgemmStridedBatched(blasHandle, tranB, tranA, h, e, l, &alpha, BPtr, ldB, l*h, APtr, ldA, e*l, &beta, CDestPtr, h, e*h, batch);
        cublas_check(status);
        //cudaThreadSynchronize();

    } else {
        auto CPtr = (float*)mTempOutput->deviceId();
        auto status = cublasSgemmStridedBatched(blasHandle, tranB, tranA, h, e, l, &alpha, BPtr, ldB, l*h, APtr, ldA, e*l, &beta, CPtr, h, e*h, batch);
        cublas_check(status);
        //cudaThreadSynchronize();

        //add bias: [b, e, h] + [b, h] -> [b, e, h]
        int block_num = runtime->blocks_num(batch*e*h);
        int threads_num = runtime->threads_num();
        add_bias<<<block_num, threads_num>>>((float*)CPtr, (float*)CDestPtr, (const float*)inputs[2]->deviceId(), batch, e, h);
    }

    return NO_ERROR;
}

class BatchMatMulCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_BatchMatMulParam();
        return new BatchMatMulExecution(param->adjX(), param->adjY(), backend);
    }
};

static CUDACreatorRegister<BatchMatMulCreator> __init(OpType_BatchMatMul);

}
}
