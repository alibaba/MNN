#include "LayerNormExecution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void LayerNormKernel(const T* input, const T* gamma, const T* beta, T* output,
                                 int outerSize, int innerSize,
                                 T epsilon, int gammaSize, int betaSize) {
    int outerIdx = blockIdx.x;
    
    if (outerIdx < outerSize) {
        // Compute mean
        T sum = 0;
        for (int i = 0; i < innerSize; i++) {
            int idx = outerIdx * innerSize + i;
            sum += input[idx];
        }
        T mean = sum / innerSize;
        
        // Compute variance
        T var = 0;
        for (int i = 0; i < innerSize; i++) {
            int idx = outerIdx * innerSize + i;
            T diff = input[idx] - mean;
            var += diff * diff;
        }
        var = var / innerSize;
        
        // Normalize
        T invStd = 1.0 / sqrt(var + epsilon);
        
        for (int i = 0; i < innerSize; i++) {
            int idx = outerIdx * innerSize + i;
            T normalized = (input[idx] - mean) * invStd;
            
            T g = (gamma != nullptr && gammaSize > 0) ? gamma[i % gammaSize] : 1.0;
            T b = (beta != nullptr && betaSize > 0) ? beta[i % betaSize] : 0.0;
            
            output[idx] = normalized * g + b;
        }
    }
}

LayerNormExecution::LayerNormExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
    mOp = op->main_as_LayerNorm();
}

ErrorCode LayerNormExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    mEpsilon = mOp->eps();
    
    mOuterSize = 1;
    for (int i = 0; i < input->dimensions() - 1; i++) {
        mOuterSize *= input->length(i);
    }
    mInnerSize = input->length(input->dimensions() - 1);
    
    mGammaSize = 0;
    mBetaSize = 0;
    if (mOp->gamma() != nullptr) {
        mGammaSize = mOp->gamma()->size();
    }
    if (mOp->beta() != nullptr) {
        mBetaSize = mOp->beta()->size();
    }
    
    int threads = 256;
    dim3 grid(mOuterSize, 1, 1);
    dim3 block(threads, 1, 1);
    
    mDim3Grid = grid;
    mDim3Block = block;
    
    return NO_ERROR;
}

ErrorCode LayerNormExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    auto inputPtr = input->host<float>();
    auto outputPtr = output->host<float>();
    
    const float* gammaPtr = nullptr;
    const float* betaPtr = nullptr;
    
    if (mGammaSize > 0 && mOp->gamma() != nullptr) {
        gammaPtr = mOp->gamma()->data();
    }
    if (mBetaSize > 0 && mOp->beta() != nullptr) {
        betaPtr = mOp->beta()->data();
    }
    
    LayerNormKernel<<<mDim3Grid, mDim3Block>>>(
        inputPtr, gammaPtr, betaPtr, outputPtr,
        mOuterSize, mInnerSize,
        static_cast<float>(mEpsilon), mGammaSize, mBetaSize
    );
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class LayerNormCreator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new LayerNormExecution(inputs, op, backend);
    }
};

MNNCreatorRegister<LayerNormCreator> gLayerNormRegistration(OpType_LayerNorm);

} // namespace MUSA
} // namespace MNN
