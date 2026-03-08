#include "GatherV2Execution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void GatherV2Kernel(const T* input, const int* indices, T* output,
                                int outerDims, int indicesCount, int innerDims,
                                int axis) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = outerDims * indicesCount * innerDims;
    
    if (index < totalSize) {
        int tmp = index;
        int inner = tmp % innerDims;
        tmp /= innerDims;
        int idx = tmp % indicesCount;
        int outer = tmp / indicesCount;
        
        int srcIndex = indices[idx];
        srcIndex = (srcIndex < 0) ? (outerDims + srcIndex) : srcIndex;
        
        int inputIndex = (outer * outerDims + srcIndex) * innerDims + inner;
        output[index] = input[inputIndex];
    }
}

GatherV2Execution::GatherV2Execution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
    mOp = op->main_as_GatherV2();
}

ErrorCode GatherV2Execution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto indices = inputs[1];
    auto output = outputs[0];
    
    mAxis = mOp->axis();
    if (mAxis < 0) {
        mAxis += input->dimensions();
    }
    
    mOuterDims = 1;
    for (int i = 0; i < mAxis; i++) {
        mOuterDims *= input->length(i);
    }
    
    mIndicesCount = 1;
    for (int i = 0; i < indices->dimensions(); i++) {
        mIndicesCount *= indices->length(i);
    }
    
    mInnerDims = 1;
    for (int i = mAxis + 1; i < input->dimensions(); i++) {
        mInnerDims *= input->length(i);
    }
    
    int threads = 256;
    int totalSize = mOuterDims * mIndicesCount * mInnerDims;
    int blocks = (totalSize + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode GatherV2Execution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto indices = inputs[1];
    auto output = outputs[0];
    
    auto inputPtr = input->host<float>();
    auto indicesPtr = indices->host<int>();
    auto outputPtr = output->host<float>();
    
    GatherV2Kernel<<<mDim3Grid, mDim3Block>>>(
        inputPtr, indicesPtr, outputPtr,
        mOuterDims, mIndicesCount, mInnerDims,
        mAxis
    );
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class GatherV2Creator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new GatherV2Execution(inputs, op, backend);
    }
};

MNNCreatorRegister<GatherV2Creator> gGatherV2Registration(OpType_GatherV2);

} // namespace MUSA
} // namespace MNN
