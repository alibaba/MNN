#include "FuseExecution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void FuseReluKernel(const T* input, T* output, int totalSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < totalSize) {
        T val = input[index];
        output[index] = val > 0 ? val : 0;
    }
}

template<typename T>
__global__ void FuseRelu6Kernel(const T* input, T* output, int totalSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < totalSize) {
        T val = input[index];
        T clipped = val > 6.0 ? 6.0 : val;
        output[index] = clipped > 0 ? clipped : 0;
    }
}

template<typename T>
__global__ void FuseSigmoidKernel(const T* input, T* output, int totalSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < totalSize) {
        T val = input[index];
        output[index] = 1.0 / (1.0 + exp(-val));
    }
}

template<typename T>
__global__ void FuseTanhKernel(const T* input, T* output, int totalSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < totalSize) {
        T val = input[index];
        T expVal = exp(2.0 * val);
        output[index] = (expVal - 1.0) / (expVal + 1.0);
    }
}

FuseExecution::FuseExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
    mOp = op->main_as_Fuse();
}

ErrorCode FuseExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    mTotalSize = 1;
    for (int i = 0; i < input->dimensions(); i++) {
        mTotalSize *= input->length(i);
    }
    
    int threads = 256;
    int blocks = (mTotalSize + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode FuseExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    auto inputPtr = input->host<float>();
    auto outputPtr = output->host<float>();
    
    auto opType = mOp->fuseType();
    
    if (opType == 0) { // ReLU
        FuseReluKernel<<<mDim3Grid, mDim3Block>>>(inputPtr, outputPtr, mTotalSize);
    } else if (opType == 1) { // ReLU6
        FuseRelu6Kernel<<<mDim3Grid, mDim3Block>>>(inputPtr, outputPtr, mTotalSize);
    } else if (opType == 2) { // Sigmoid
        FuseSigmoidKernel<<<mDim3Grid, mDim3Block>>>(inputPtr, outputPtr, mTotalSize);
    } else if (opType == 3) { // Tanh
        FuseTanhKernel<<<mDim3Grid, mDim3Block>>>(inputPtr, outputPtr, mTotalSize);
    } else {
        return COMPUTE_NO_SUPPORT;
    }
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class FuseCreator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new FuseExecution(inputs, op, backend);
    }
};

MNNCreatorRegister<FuseCreator> gFuseRegistration(OpType_Fuse);

} // namespace MUSA
} // namespace MNN
