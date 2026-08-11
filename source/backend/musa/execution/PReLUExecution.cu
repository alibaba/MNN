#include "PReLUExecution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void PReLUKernel(const T* input, const T* slope, T* output,
                            int totalSize, int channels, int innerDims,
                            int slopeSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < totalSize) {
        int tmp = index;
        int inner = tmp % innerDims;
        tmp /= innerDims;
        int c = tmp % channels;
        
        T slopeVal = (slopeSize == 1) ? slope[0] : slope[c];
        T inVal = input[index];
        output[index] = (inVal > 0) ? inVal : (inVal * slopeVal);
    }
}

PReLUExecution::PReLUExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
    mOp = op->main_as_PReLU();
}

ErrorCode PReLUExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    mTotalSize = 1;
    mChannels = input->channel();
    mInnerDims = 1;
    
    for (int i = 0; i < input->dimensions(); i++) {
        if (i == 1) {
            continue;
        }
        if (i > 1) {
            mInnerDims *= input->length(i);
        }
        mTotalSize *= input->length(i);
    }
    
    mSlopeSize = 1;
    if (mOp->slope() != nullptr) {
        mSlopeSize = mOp->slope()->size();
    }
    
    int threads = 256;
    int blocks = (mTotalSize + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode PReLUExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    auto inputPtr = input->host<float>();
    auto outputPtr = output->host<float>();
    
    const float* slopePtr = nullptr;
    if (mOp->slope() != nullptr && mOp->slope()->size() > 0) {
        slopePtr = mOp->slope()->data();
    }
    
    PReLUKernel<<<mDim3Grid, mDim3Block>>>(
        inputPtr, slopePtr, outputPtr,
        mTotalSize, mChannels, mInnerDims,
        mSlopeSize
    );
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class PReLUCreator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new PReLUExecution(inputs, op, backend);
    }
};

MNNCreatorRegister<PReLUCreator> gPReLURegistration(OpType_PReLU);

} // namespace MUSA
} // namespace MNN
