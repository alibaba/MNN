#include "TopKV2Execution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void TopKKernel(const T* input, T* outValues, int* outIndices,
                           int outerSize, int k, int innerSize) {
    int outerIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outerIdx < outerSize) {
        const T* inputPtr = input + outerIdx * k * innerSize;
        T* outValPtr = outValues + outerIdx * k * innerSize;
        int* outIdxPtr = outIndices + outerIdx * k * innerSize;
        
        // Simple selection sort for top k
        for (int i = 0; i < k; i++) {
            T maxVal = inputPtr[i * innerSize];
            int maxIdx = i;
            
            for (int j = i + 1; j < innerSize; j++) {
                if (inputPtr[j] > maxVal) {
                    maxVal = inputPtr[j];
                    maxIdx = j;
                }
            }
            
            // Swap
            if (maxIdx != i) {
                T tempVal = inputPtr[i * innerSize];
                inputPtr[i * innerSize] = maxVal;
                inputPtr[maxIdx * innerSize] = tempVal;
            }
            
            outValPtr[i * innerSize] = maxVal;
            outIdxPtr[i * innerSize] = maxIdx;
        }
    }
}

TopKV2Execution::TopKV2Execution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
    mOp = op->main_as_TopKV2();
}

ErrorCode TopKV2Execution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto kTensor = inputs[1];
    
    mAxis = mOp->axis();
    if (mAxis < 0) {
        mAxis += input->dimensions();
    }
    
    mK = kTensor->host<int>()[0];
    
    mOuterSize = 1;
    for (int i = 0; i < mAxis; i++) {
        mOuterSize *= input->length(i);
    }
    
    mInnerSize = input->length(mAxis);
    
    int threads = 256;
    int blocks = (mOuterSize + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode TopKV2Execution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto outputValues = outputs[0];
    auto outputIndices = outputs[1];
    
    auto inputPtr = input->host<float>();
    auto outputValuesPtr = outputValues->host<float>();
    auto outputIndicesPtr = outputIndices->host<int>();
    
    TopKKernel<<<mDim3Grid, mDim3Block>>>(
        inputPtr, outputValuesPtr, outputIndicesPtr,
        mOuterSize, mK, mInnerSize
    );
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class TopKV2Creator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new TopKV2Execution(inputs, op, backend);
    }
};

MNNCreatorRegister<TopKV2Creator> gTopKV2Registration(OpType_TopKV2);

} // namespace MUSA
} // namespace MNN
