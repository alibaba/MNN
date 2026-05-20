//
//  MatMulExecution.cu
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

// MUSA kernel for matrix multiplication
__global__ void MatMulKernel(const float* A, const float* B, float* C, 
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    
    C[row * N + col] = sum;
}

// MUSA kernel for batched matrix multiplication
__global__ void BatchMatMulKernel(const float* A, const float* B, float* C,
                                  int batch, int M, int N, int K) {
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += A[b * M * K + row * K + i] * B[b * K * N + i * N + col];
    }
    
    C[b * M * N + row * N + col] = sum;
}

class MatMulExecution : public Execution {
public:
    MatMulExecution(Backend* backend) : Execution(backend) {}
    
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        mShapeChanged = true;
        return NO_ERROR;
    }
    
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
#ifdef LOG_VERBOSE
        MNN_PRINT("start MatMulExecution onExecute...\n");
#endif
        
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        auto output = outputs[0];
        
        auto input0Shape = input0->shape();
        auto input1Shape = input1->shape();
        auto outputShape = output->shape();
        
        void* input0Ptr = (void*)input0->deviceId();
        void* input1Ptr = (void*)input1->deviceId();
        void* outputPtr = (void*)output->deviceId();
        
        if (input0Shape.size() == 2 && input1Shape.size() == 2) {
            // 2D matrix multiplication
            int M = input0Shape[0];
            int K = input0Shape[1];
            int N = input1Shape[1];
            
            dim3 threadsPerBlock(16, 16);
            dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);
            
            MatMulKernel<<<blocksPerGrid, threadsPerBlock>>>(
                (const float*)input0Ptr, (const float*)input1Ptr, (float*)outputPtr, M, N, K);
        } else {
            // Batched matrix multiplication
            int batch = 1;
            int M = input0Shape[input0Shape.size() - 2];
            int K = input0Shape[input0Shape.size() - 1];
            int N = input1Shape[input1Shape.size() - 1];
            
            for (size_t i = 0; i < input0Shape.size() - 2; ++i) {
                batch *= input0Shape[i];
            }
            
            dim3 threadsPerBlock(16, 16);
            dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16, batch);
            
            BatchMatMulKernel<<<blocksPerGrid, threadsPerBlock>>>(
                (const float*)input0Ptr, (const float*)input1Ptr, (float*)outputPtr, batch, M, N, K);
        }
        
        // Check for kernel launch errors
        musaError_t err = musaGetLastError();
        if (err != musaSuccess) {
            MNN_ERROR("MUSA MatMul kernel launch failed: %s\n", musaGetErrorString(err));
        }
        
        // Synchronize to ensure completion
        auto musaBackend = static_cast<MusaBackend*>(backend());
        musaBackend->getMusaRuntime()->device_sync();
        
#ifdef LOG_VERBOSE
        MNN_PRINT("end MatMulExecution onExecute...\n");
#endif
        
        return NO_ERROR;
    }
    
private:
    bool mShapeChanged{false};
};

// Creator for MatMul operations
class MatMulCreator : public MusaBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new MatMulExecution(backend);
    }
};

MusaCreatorRegister<MatMulCreator> __MatMulExecution(OpType_MatMul);
MusaCreatorRegister<MatMulCreator> __MatMulInt8Execution(OpType_MatMulInt8);

} // namespace MUSA
} // namespace MNN
