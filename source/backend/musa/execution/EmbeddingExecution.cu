#include "EmbeddingExecution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void EmbeddingKernel(const T* embedding, const int* indices, T* output,
                                 int numIndices, int embeddingDim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = numIndices * embeddingDim;
    
    if (index < totalSize) {
        int dim = index % embeddingDim;
        int idx = index / embeddingDim;
        
        int embeddingIdx = indices[idx];
        output[index] = embedding[embeddingIdx * embeddingDim + dim];
    }
}

EmbeddingExecution::EmbeddingExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
}

ErrorCode EmbeddingExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto embedding = inputs[0];
    auto indices = inputs[1];
    auto output = outputs[0];
    
    mNumIndices = 1;
    for (int i = 0; i < indices->dimensions(); i++) {
        mNumIndices *= indices->length(i);
    }
    
    mEmbeddingDim = embedding->length(1);
    
    int threads = 256;
    int totalSize = mNumIndices * mEmbeddingDim;
    int blocks = (totalSize + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode EmbeddingExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto embedding = inputs[0];
    auto indices = inputs[1];
    auto output = outputs[0];
    
    auto embeddingPtr = embedding->host<float>();
    auto indicesPtr = indices->host<int>();
    auto outputPtr = output->host<float>();
    
    EmbeddingKernel<<<mDim3Grid, mDim3Block>>>(
        embeddingPtr, indicesPtr, outputPtr,
        mNumIndices, mEmbeddingDim
    );
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class EmbeddingCreator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new EmbeddingExecution(inputs, op, backend);
    }
};

MNNCreatorRegister<EmbeddingCreator> gEmbeddingRegistration(OpType_Embedding);

} // namespace MUSA
} // namespace MNN
