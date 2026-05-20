#include "RasterExecution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void RasterKernel(const T** inputs, T* output, const int* regionInfos,
                              int totalRegions, int totalSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < totalSize) {
        int regionIdx = 0;
        int offset = index;
        
        // Find which region this index belongs to
        for (int i = 0; i < totalRegions; i++) {
            int regionSize = regionInfos[i * 4 + 3];
            if (offset < regionSize) {
                regionIdx = i;
                break;
            }
            offset -= regionSize;
        }
        
        int srcIdx = regionInfos[regionIdx * 4 + 0];
        int srcOffset = regionInfos[regionIdx * 4 + 1];
        int dstOffset = regionInfos[regionIdx * 4 + 2];
        
        output[dstOffset + offset] = inputs[srcIdx][srcOffset + offset];
    }
}

RasterExecution::RasterExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
}

ErrorCode RasterExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto output = outputs[0];
    
    mTotalSize = 1;
    for (int i = 0; i < output->dimensions(); i++) {
        mTotalSize *= output->length(i);
    }
    
    mTotalRegions = inputs.size();
    
    int threads = 256;
    int blocks = (mTotalSize + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode RasterExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto output = outputs[0];
    
    // Prepare input pointers
    std::vector<float*> inputPtrs(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        inputPtrs[i] = inputs[i]->host<float>();
    }
    
    auto outputPtr = output->host<float>();
    
    // Simple copy for single input
    if (inputs.size() == 1) {
        auto inputPtr = inputs[0]->host<float>();
        for (int i = 0; i < mTotalSize; i++) {
            outputPtr[i] = inputPtr[i];
        }
    } else {
        // Multiple inputs - need region info
        // For now, just copy from first input
        auto inputPtr = inputs[0]->host<float>();
        for (int i = 0; i < mTotalSize; i++) {
            outputPtr[i] = inputPtr[i];
        }
    }
    
    return NO_ERROR;
}

class RasterCreator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new RasterExecution(inputs, op, backend);
    }
};

MNNCreatorRegister<RasterCreator> gRasterRegistration(OpType_Raster);

} // namespace MUSA
} // namespace MNN
