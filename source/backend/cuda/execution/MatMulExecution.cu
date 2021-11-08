#include "MatMulExecution.hpp"
namespace MNN {
namespace CUDA {

template <typename T>
__global__ void add_bias(T *input, T *output, const T* bias, int e, int h) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < e * h; i += blockDim.x * gridDim.x) {
        int y = i % h;
        output[i] = input[i] + bias[y];
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
    if(inputs.size() > 2) {
        mTempOutput.reset(Tensor::createDevice<float>({e, h}));
        auto res = backend()->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        backend()->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    }
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
    int block_num = runtime->blocks_num(e*h);
    int threads_num = runtime->threads_num();
    
    //[e, l] x [l, h] -> [e, h]
    if(inputs.size() == 2) {
        auto status = cublasSgemm(blasHandle, tranB, tranA, h, e, l, &alpha, BPtr, ldB, APtr, ldA, &beta, CDestPtr, h);
        cublas_check(status);
        //cudaThreadSynchronize();
    } else {
        auto CPtr = (float*)mTempOutput->deviceId();
        auto status = cublasSgemm(blasHandle, tranB, tranA, h, e, l, &alpha, BPtr, ldB, APtr, ldA, &beta, CPtr, h);
        cublas_check(status);
        //cudaThreadSynchronize();

        //bias: [e, h] + [h] -> [e, h]
        add_bias<<<block_num, threads_num>>>((float*)CPtr, (float*)CDestPtr, (const float*)inputs[2]->deviceId(), e, h);
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