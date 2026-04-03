//
//  ConvExecution.cu
//  MNN
//
//  Created by MNN on 2026/02/25.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "ConvExecution.hpp"
#include "core/TensorUtils.hpp"
#include "backend/musa/core/MusaBackend.hpp"
#include <musa_runtime.h>

namespace MNN {
namespace MUSA {

// MUSA kernel for 1x1 convolution (GEMM-based)
__global__ void Conv1x1Kernel(const float* input, float* output, const float* weight, const float* bias,
                              int batch, int channels, int height, int width, int outputChannels,
                              int stride, int pad) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width * outputChannels || y >= height * batch) return;
    
    int outX = x % width;
    int outCh = x / width;
    int outY = y % height;
    int outB = y / height;
    
    float sum = bias ? bias[outCh] : 0.0f;
    
    int inX = outX * stride;
    int inY = outY * stride;
    
    if (inX >= width || inY >= height) {
        output[outB * outputChannels * height * width + outCh * height * width + outY * width + outX] = sum;
        return;
    }
    
    for (int ic = 0; ic < channels; ++ic) {
        float inVal = input[outB * channels * height * width + ic * height * width + inY * width + inX];
        float wVal = weight[outCh * channels + ic];
        sum += inVal * wVal;
    }
    
    output[outB * outputChannels * height * width + outCh * height * width + outY * width + outX] = sum;
}

// MUSA kernel for general convolution (im2col + GEMM)
__global__ void Conv2dKernel(const float* input, float* output, const float* weight, const float* bias,
                             int batch, int channels, int height, int width, int outputChannels,
                             int kernelSize, int stride, int pad, int dilation) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (outX >= width || outY >= height) return;
    
    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < outputChannels; ++oc) {
            float sum = bias ? bias[oc] : 0.0f;
            
            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    int inX = outX * stride + kx * dilation - pad;
                    int inY = outY * stride + ky * dilation - pad;
                    
                    if (inX >= 0 && inX < width && inY >= 0 && inY < height) {
                        for (int ic = 0; ic < channels; ++ic) {
                            float inVal = input[b * channels * height * width + ic * height * width + inY * width + inX];
                            int wIdx = oc * channels * kernelSize * kernelSize + ic * kernelSize * kernelSize + ky * kernelSize + kx;
                            float wVal = weight[wIdx];
                            sum += inVal * wVal;
                        }
                    }
                }
            }
            
            int outIdx = b * outputChannels * height * width + oc * height * width + outY * width + outX;
            output[outIdx] = sum;
        }
    }
}

ConvExecution::ConvExecution(const MNN::Op* op, Backend* backend) : Execution(backend) {
    auto conv2d = op->main_as_Convolution2D();
    mResource = ConvolutionCommon::getConvolutionResource(op);
    
    auto common = conv2d->common();
    mIsDepthWise = common->depthwise();
    mIsConv1x1 = (common->kernelX() == 1 && common->kernelY() == 1 && 
                  common->strideX() == 1 && common->strideY() == 1 &&
                  common->dilateX() == 1 && common->dilateY() == 1);
    
    mIm2ColParams.kernelX = common->kernelX();
    mIm2ColParams.kernelY = common->kernelY();
    mIm2ColParams.strideX = common->strideX();
    mIm2ColParams.strideY = common->strideY();
    mIm2ColParams.padX = common->padX();
    mIm2ColParams.padY = common->padY();
    mIm2ColParams.dilateX = common->dilateX();
    mIm2ColParams.dilateY = common->dilateY();
}

ErrorCode ConvExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return NO_ERROR;
}

ErrorCode ConvExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ConvExecution onExecute...\n");
#endif
    
    auto input = inputs[0];
    auto output = outputs[0];
    auto conv2d = op()->main_as_Convolution2D();
    
    auto inputShape = input->shape();
    auto outputShape = output->shape();
    
    int batch = inputShape[0];
    int channels = inputShape[1];
    int height = inputShape[2];
    int width = inputShape[3];
    
    int outputChannels = outputShape[1];
    int outHeight = outputShape[2];
    int outWidth = outputShape[3];
    
    auto common = conv2d->common();
    int kernelSize = common->kernelX();
    int stride = common->strideX();
    int pad = common->padX();
    int dilation = common->dilateX();
    
    auto weight = mResource->weight.get();
    auto bias = mResource->bias.get();
    
    void* inputPtr = (void*)input->deviceId();
    void* outputPtr = (void*)output->deviceId();
    
    if (mIsConv1x1 && stride == 1 && pad == 0 && dilation == 1) {
        // Use optimized 1x1 convolution kernel
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((outWidth * outputChannels + 15) / 16, (outHeight * batch + 15) / 16);
        
        Conv1x1Kernel<<<blocksPerGrid, threadsPerBlock>>>(
            (const float*)inputPtr, (float*)outputPtr, (const float*)weight, 
            bias ? (const float*)bias : nullptr,
            batch, channels, height, width, outputChannels, stride, pad);
    } else {
        // Use general convolution kernel
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((outWidth + 15) / 16, (outHeight + 15) / 16);
        
        Conv2dKernel<<<blocksPerGrid, threadsPerBlock>>>(
            (const float*)inputPtr, (float*)outputPtr, (const float*)weight,
            bias ? (const float*)bias : nullptr,
            batch, channels, height, width, outputChannels,
            kernelSize, stride, pad, dilation);
    }
    
    // Check for kernel launch errors
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        MNN_ERROR("MUSA Conv kernel launch failed: %s\n", musaGetErrorString(err));
    }
    
    // Synchronize to ensure completion
    auto musaBackend = static_cast<MusaBackend*>(backend());
    musaBackend->getMusaRuntime()->device_sync();
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onExecute...\n");
#endif
    
    return NO_ERROR;
}

// Creator for Conv operations
class ConvCreator : public MusaBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new ConvExecution(op, backend);
    }
};

MusaCreatorRegister<ConvCreator> __ConvExecution(OpType_Convolution);

} // namespace MUSA
} // namespace MNN
