#include "ArgMaxExecution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void ArgMaxKernel(const T* input, int* output,
                              int outerSize, int axisSize, int innerSize) {
    int outerIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outerIdx < outerSize) {
        T maxVal = input[outerIdx * axisSize * innerSize];
        int maxIdx = 0;
        
        for (int i = 0; i < axisSize; i++) {
            for (int j = 0; j < innerSize; j++) {
                int idx = (outerIdx * axisSize + i) * innerSize + j;
                if (input[idx] > maxVal) {
                    maxVal = input[idx];
                    maxIdx = i;
                }
            }
        }
        
        output[outerIdx] = maxIdx;
    }
}

ArgMaxExecution::ArgMaxExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
    mOp = op->main_as_ArgMax();
}

ErrorCode ArgMaxExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    mAxis = mOp->axis();
    if (mAxis < 0) {
        mAxis += input->dimensions();
    }
    
    mOuterSize = 1;
    for (int i = 0; i < mAxis; i++) {
        mOuterSize *= input->length(i);
    }
    
    mAxisSize = input->length(mAxis);
    
    mInnerSize = 1;
    for (int i = mAxis + 1; i < input->dimensions(); i++) {
        mInnerSize *= input->length(i);
    }
    
    int threads = 256;
    int blocks = (mOuterSize + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode ArgMaxExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    auto inputPtr = input->host<float>();
    auto outputPtr = output->host<int>();
    
    ArgMaxKernel<<<mDim3Grid, mDim3Block>>>(
        inputPtr, outputPtr,
        mOuterSize, mAxisSize, mInnerSize
    );
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class ArgMaxCreator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new ArgMaxExecution(inputs, op, backend);
    }
};

MNNCreatorRegister<ArgMaxCreator> gArgMaxRegistration(OpType_ArgMax);

} // namespace MUSA
} // namespace MNN
