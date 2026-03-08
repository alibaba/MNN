#include "SelectExecution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void SelectKernel(const bool* condition, const T* x, const T* y, T* output, int totalSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < totalSize) {
        output[index] = condition[index] ? x[index] : y[index];
    }
}

SelectExecution::SelectExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
}

ErrorCode SelectExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto output = outputs[0];
    
    mTotalSize = 1;
    for (int i = 0; i < output->dimensions(); i++) {
        mTotalSize *= output->length(i);
    }
    
    int threads = 256;
    int blocks = (mTotalSize + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode SelectExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto condition = inputs[0];
    auto x = inputs[1];
    auto y = inputs[2];
    auto output = outputs[0];
    
    auto conditionPtr = condition->host<bool>();
    auto xPtr = x->host<float>();
    auto yPtr = y->host<float>();
    auto outputPtr = output->host<float>();
    
    SelectKernel<<<mDim3Grid, mDim3Block>>>(
        conditionPtr, xPtr, yPtr, outputPtr, mTotalSize
    );
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class SelectCreator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new SelectExecution(inputs, op, backend);
    }
};

MNNCreatorRegister<SelectCreator> gSelectRegistration(OpType_Select);

} // namespace MUSA
} // namespace MNN
