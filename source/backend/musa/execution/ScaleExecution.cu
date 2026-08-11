#include "ScaleExecution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void ScaleKernel(const T* input, const T* scale, const T* bias, T* output,
                            int outerDims, int channels, int innerDims,
                            int scaleOuter, int scaleInner) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = outerDims * channels * innerDims;
    
    if (index < totalSize) {
        int tmp = index;
        int inner = tmp % innerDims;
        tmp /= innerDims;
        int c = tmp % channels;
        int outer = tmp / channels;
        
        T scaleVal = (scale != nullptr) ? scale[c] : 1.0f;
        T biasVal = (bias != nullptr) ? bias[c] : 0.0f;
        
        int inputIndex = (outer * channels + c) * innerDims + inner;
        output[index] = input[inputIndex] * scaleVal + biasVal;
    }
}

ScaleExecution::ScaleExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
    mOp = op->main_as_Scale();
}

ErrorCode ScaleExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    mOuterDims = 1;
    mChannels = input->channel();
    mInnerDims = 1;
    
    for (int i = 0; i < input->dimensions(); i++) {
        if (i == 1) {
            continue;
        }
        if (i < 1) {
            mOuterDims *= input->length(i);
        } else {
            mInnerDims *= input->length(i);
        }
    }
    
    int threads = 256;
    int totalSize = mOuterDims * mChannels * mInnerDims;
    int blocks = (totalSize + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode ScaleExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    auto inputPtr = input->host<float>();
    auto outputPtr = output->host<float>();
    
    const float* scalePtr = nullptr;
    const float* biasPtr = nullptr;
    
    if (mOp->scaleData() != nullptr && mOp->scaleData()->size() > 0) {
        scalePtr = mOp->scaleData()->data();
    }
    if (mOp->biasData() != nullptr && mOp->biasData()->size() > 0) {
        biasPtr = mOp->biasData()->data();
    }
    
    ScaleKernel<<<mDim3Grid, mDim3Block>>>(
        inputPtr, scalePtr, biasPtr, outputPtr,
        mOuterDims, mChannels, mInnerDims,
        1, 1
    );
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class ScaleCreator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new ScaleExecution(inputs, op, backend);
    }
};

MNNCreatorRegister<ScaleCreator> gScaleRegistration(OpType_Scale);

} // namespace MUSA
} // namespace MNN
