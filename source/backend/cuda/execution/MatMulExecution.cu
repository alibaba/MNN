#include "MatMulExecution.hpp"
namespace MNN {
namespace CUDA {

template <typename T>
__global__ void transpose(T *input, T *output, size_t e, size_t h) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < e * h; i += blockDim.x * gridDim.x) {
        int y = i % e;
        int x = i / e;
        output[y * h + x] = input[i];
    }
    return;
}
template <typename T>
__global__ void transpose_bias(T *input, T *output, const T* bias, size_t e, size_t h) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < e * h; i += blockDim.x * gridDim.x) {
        int y = i % e;
        int x = i / e;
        output[y * h + x] = input[i] + bias[x];
    }
    return;
}
MatMulExecution::MatMulExecution(bool transposeA, bool transposeB, Backend *backend) : Execution(backend) {
    mTransposeA = transposeA;
    mTransposeB = transposeB;
}
MatMulExecution::~ MatMulExecution() {
    // do nothing
}

ErrorCode MatMulExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto C = outputs[0];
    auto e = C->length(0);
    auto h = C->length(1);
    mTempOutput.reset(Tensor::createDevice<float>({e, h}));
    auto res = backend()->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode MatMulExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto blasHandle = runtime->cublas_handle();
    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    auto C = outputs[0];

    auto e = C->length(0);
    auto h = C->length(1);
    auto l = w0;
    if (mTransposeA) {
        l = h0;
    }
    auto APtr = (const float*)A->deviceId();
    auto BPtr = (const float*)B->deviceId();
    auto CPtr = (float*)mTempOutput->deviceId();

    float alpha = 1.0f;
    float beta = 0.0f;
    auto tranA = CUBLAS_OP_T;
    auto ldA = l;
    if (mTransposeA) {
        ldA = e;
        tranA = CUBLAS_OP_N;
    }
    auto tranB = CUBLAS_OP_T;
    auto ldB = h;
    if (mTransposeB) {
        ldB = l;
        tranB = CUBLAS_OP_N;
    }
    auto status = cublasSgemm(blasHandle, tranA, tranB, e, h, l, &alpha, APtr, ldA, BPtr, ldB, &beta, CPtr, e);
    //cudaThreadSynchronize();
    // Transpose h, e -> e, h
    int block_num = runtime->blocks_num(e*h);
    int threads_num = runtime->threads_num();
    auto CDestPtr = (float*)C->deviceId();
    if (inputs.size() > 2) {
        transpose_bias<<<block_num, threads_num>>>((float*)CPtr, (float*)CDestPtr, (const float*)inputs[2]->deviceId(), e, h);
    } else {
        transpose<<<block_num, threads_num>>>((float*)CPtr, (float*)CDestPtr, e, h);
    }
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