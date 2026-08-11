//
//  BatchNormExecution.cu
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "core/MusaBackend.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include <musa_runtime.h>

namespace MNN {
namespace MUSA {

// MUSA kernel for batch normalization
__global__ void BatchNormKernel(const float* input, float* output, 
                                const float* scale, const float* bias,
                                const float* mean, const float* variance,
                                float epsilon, int batchSize, int channels, int spatialSize) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (c >= channels || s >= spatialSize) return;
    
    float invStd = 1.0f / sqrtf(variance[c] + epsilon);
    float m = mean[c];
    float b = bias[c];
    float s_val = scale[c];
    
    for (int b = 0; b < batchSize; ++b) {
        int idx = (b * channels + c) * spatialSize + s;
        float x = input[idx];
        float y = (x - m) * invStd * s_val + b;
        output[idx] = y;
    }
}

class BatchNormExecution : public Execution {
public:
    BatchNormExecution(Backend* backend) : Execution(backend) {}
    
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto input = inputs[0];
        auto inputShape = input->shape();
        
        mBatchSize = inputShape[0];
        mChannels = inputShape[1];
        mSpatialSize = 1;
        for (size_t i = 2; i < inputShape.size(); ++i) {
            mSpatialSize *= inputShape[i];
        }
        
        return NO_ERROR;
    }
    
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
#ifdef LOG_VERBOSE
        MNN_PRINT("start BatchNormExecution onExecute...\n");
#endif
        
        auto input = inputs[0];
        auto output = outputs[0];
        auto op = this->op();
        
        auto batchNorm = op->main_as_BatchNorm();
        
        void* inputPtr = (void*)input->deviceId();
        void* outputPtr = (void*)output->deviceId();
        
        // Get scale, bias, mean, variance from the op
        auto scaleData = batchNorm->scaleData();
        auto biasData = batchNorm->biasData();
        auto meanData = batchNorm->meanData();
        auto varianceData = batchNorm->varianceData();
        float epsilon = batchNorm->eps();
        
        // Copy parameters to device
        float *dScale, *dBias, *dMean, *dVariance;
        size_t dataSize = sizeof(float) * mChannels;
        
        musaMalloc(&dScale, dataSize);
        musaMalloc(&dBias, dataSize);
        musaMalloc(&dMean, dataSize);
        musaMalloc(&dVariance, dataSize);
        
        musaMemcpy(dScale, scaleData->data(), dataSize, MNNMemcpyHostToDevice);
        musaMemcpy(dBias, biasData->data(), dataSize, MNNMemcpyHostToDevice);
        musaMemcpy(dMean, meanData->data(), dataSize, MNNMemcpyHostToDevice);
        musaMemcpy(dVariance, varianceData->data(), dataSize, MNNMemcpyHostToDevice);
        
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((mChannels + 15) / 16, (mSpatialSize + 15) / 16);
        
        BatchNormKernel<<<blocksPerGrid, threadsPerBlock>>>(
            (const float*)inputPtr, (float*)outputPtr,
            dScale, dBias, dMean, dVariance,
            epsilon, mBatchSize, mChannels, mSpatialSize);
        
        // Check for kernel launch errors
        musaError_t err = musaGetLastError();
        if (err != musaSuccess) {
            MNN_ERROR("MUSA BatchNorm kernel launch failed: %s\n", musaGetErrorString(err));
        }
        
        // Synchronize to ensure completion
        auto musaBackend = static_cast<MusaBackend*>(backend());
        musaBackend->getMusaRuntime()->device_sync();
        
        // Free temporary device memory
        musaFree(dScale);
        musaFree(dBias);
        musaFree(dMean);
        musaFree(dVariance);
        
#ifdef LOG_VERBOSE
        MNN_PRINT("end BatchNormExecution onExecute...\n");
#endif
        
        return NO_ERROR;
    }
    
private:
    int mBatchSize;
    int mChannels;
    int mSpatialSize;
};

// Creator for BatchNorm operations
class BatchNormCreator : public MusaBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new BatchNormExecution(backend);
    }
};

MusaCreatorRegister<BatchNormCreator> __BatchNormExecution(OpType_BatchNorm);

} // namespace MUSA
} // namespace MNN
