#include "TransposeExecution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void TransposeKernel(const T* input, T* output, const int* perm,
                                 int dims, const int* inputStrides, const int* outputStrides,
                                 int totalSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < totalSize) {
        int tmp = index;
        int inputIdx = 0;
        
        // Decode output index to multi-dimensional index
        for (int i = dims - 1; i >= 0; i--) {
            int coord = tmp % outputStrides[i];
            tmp /= outputStrides[i];
            inputIdx += coord * inputStrides[perm[i]];
        }
        
        output[index] = input[inputIdx];
    }
}

TransposeExecution::TransposeExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
    mOp = op->main_as_Transpose();
}

ErrorCode TransposeExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    mDims = input->dimensions();
    mTotalSize = 1;
    for (int i = 0; i < mDims; i++) {
        mTotalSize *= input->length(i);
    }
    
    // Compute strides
    mInputStrides.resize(mDims);
    mOutputStrides.resize(mDims);
    
    int inputStride = 1;
    int outputStride = 1;
    for (int i = mDims - 1; i >= 0; i--) {
        mInputStrides[i] = inputStride;
        mOutputStrides[i] = outputStride;
        inputStride *= input->length(i);
        outputStride *= output->length(i);
    }
    
    // Get perm
    auto permData = mOp->perm();
    mPerm.resize(mDims);
    for (int i = 0; i < mDims; i++) {
        mPerm[i] = (i < permData->size()) ? permData->Get(i) : (mDims - 1 - i);
    }
    
    int threads = 256;
    int blocks = (mTotalSize + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode TransposeExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    auto inputPtr = input->host<float>();
    auto outputPtr = output->host<float>();
    
    TransposeKernel<<<mDim3Grid, mDim3Block>>>(
        inputPtr, outputPtr, mPerm.data(),
        mDims, mInputStrides.data(), mOutputStrides.data(),
        mTotalSize
    );
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class TransposeCreator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new TransposeExecution(inputs, op, backend);
    }
};

MNNCreatorRegister<TransposeCreator> gTransposeRegistration(OpType_Transpose);

} // namespace MUSA
} // namespace MNN
