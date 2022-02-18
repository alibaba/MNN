#include <cuda_fp16.h>
#include "PoolExecution.hpp"
#include <float.h>
#include "MNNCUDADefine.hpp"
namespace MNN {
namespace CUDA {
#define HALF_MIN  half(-65504)
#define HALF2_MIN half2(-65504, -65504)
#define MNN_CUDA_HALF2_MAX(a, b)                     \
    do {                                             \
        (a).x = __hgt((a).x, (b).x) ? (a).x : (b).x; \
        (a).y = __hgt((a).y, (b).y) ? (a).y : (b).y; \
    } while (0)

__global__ void maxpool_halfC16(const half* uInput, half* uOutput,
    int bc,
    int ih, int iw,
    int oh, int ow,
    int padX, int padY,
    int kernelX, int kernelY,
    int strideX, int strideY
    ) {
    int total = bc * oh * ow * 8;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int x = i % ow;
        int tmp = i / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int zC = z / 8;
        int zR = z % 8;
        int ix = x * strideX - padX;
        int iy = y * strideY - padY;
        int sx = max(0, -ix);
        int sy = max(0, -iy);
        int ex = min(kernelX, iw - ix);
        int ey = min(kernelY, ih - iy);
        float div = (float)(ey-sy)* (float)(ex-sx);
        half2 sumValue = HALF2_MIN;
        for (int fy=sy; fy<ey; ++fy) {
            for (int fx=sx; fx<ex; ++fx) {
                int currentX = ix + fx;
                int currentY = iy + fy;
                const half2* input = (const half2*)(uInput
                    + zR * 2
                    + currentX * 16
                    + currentY * iw * 16
                    + zC * iw * ih * 16
                );
                half2 inputV = *input;
                MNN_CUDA_HALF2_MAX(sumValue, inputV);
            }
        }
        half2* dst = (half2*)(uOutput
            + zC * ow * oh * 16
            + y * ow * 16
            + x * 16
            + zR * 2
        );
        *dst = sumValue;
    }
}

__global__ void avgpool_halfC16(const half* uInput, half* uOutput,
    int bc,
    int ih, int iw,
    int oh, int ow,
    int padX, int padY,
    int kernelX, int kernelY,
    int strideX, int strideY
    ) {
    int total = bc * oh * ow * 8;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int x = i % ow;
        int tmp = i / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int zC = z / 8;
        int zR = z % 8;
        int ix = x * strideX - padX;
        int iy = y * strideY - padY;
        int sx = max(0, -ix);
        int sy = max(0, -iy);
        int ex = min(kernelX, iw - ix);
        int ey = min(kernelY, ih - iy);
        float div = (float)(ey-sy)* (float)(ex-sx);
        half2 sumValue = half2(0.0f, 0.0f);
        half2 mulValue = half2(1.0f / div, 1.0f/div);
        for (int fy=sy; fy<ey; ++fy) {
            for (int fx=sx; fx<ex; ++fx) {
                int currentX = ix + fx;
                int currentY = iy + fy;
                const half2* input = (const half2*)(uInput
                    + zR * 2
                    + currentX * 16
                    + currentY * iw * 16
                    + zC * iw * ih * 16
                );
                sumValue = __hadd2(sumValue, (*input) * mulValue);
            }
        }
        half2* dst = (half2*)(uOutput
            + zC * ow * oh * 16
            + y * ow * 16
            + x * 16
            + zR * 2
        );
        *dst = sumValue;
    }
}



__global__ void maxpool_floatC16(const float* uInput, float* uOutput,
    int bc,
    int ih, int iw,
    int oh, int ow,
    int padX, int padY,
    int kernelX, int kernelY,
    int strideX, int strideY
    ) {
    int total = bc * oh * ow * PACK_NUMBER;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int x = i % ow;
        int tmp = i / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int zC = z / PACK_NUMBER;
        int zR = z % PACK_NUMBER;
        int ix = x * strideX - padX;
        int iy = y * strideY - padY;
        int sx = max(0, -ix);
        int sy = max(0, -iy);
        int ex = min(kernelX, iw - ix);
        int ey = min(kernelY, ih - iy);
        float maxValue = -FLT_MAX;
        for (int fy=sy; fy<ey; ++fy) {
            for (int fx=sx; fx<ex; ++fx) {
                int currentX = ix + fx;
                int currentY = iy + fy;
                const float* input = (const float*)(uInput
                    + zR
                    + currentX * PACK_NUMBER
                    + currentY * iw * PACK_NUMBER
                    + zC * iw * ih * PACK_NUMBER
                );
                maxValue = max(maxValue, *input);
            }
        }
        float* dst = (float*)(uOutput
            + zC * ow * oh * PACK_NUMBER
            + y * ow * PACK_NUMBER
            + x * PACK_NUMBER
            + zR
        );
        *dst = maxValue;
    }
}

__global__ void avgpool_floatC16(const float* uInput, float* uOutput,
    int bc,
    int ih, int iw,
    int oh, int ow,
    int padX, int padY,
    int kernelX, int kernelY,
    int strideX, int strideY
    ) {
    int total = bc * oh * ow * PACK_NUMBER;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int x = i % ow;
        int tmp = i / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int zC = z / PACK_NUMBER;
        int zR = z % PACK_NUMBER;
        int ix = x * strideX - padX;
        int iy = y * strideY - padY;
        int sx = max(0, -ix);
        int sy = max(0, -iy);
        int ex = min(kernelX, iw - ix);
        int ey = min(kernelY, ih - iy);
        float div = (float)(ey-sy)* (float)(ex-sx);
        float sumValue = 0.0f;
        float mulValue = 1.0f/div;
        for (int fy=sy; fy<ey; ++fy) {
            for (int fx=sx; fx<ex; ++fx) {
                int currentX = ix + fx;
                int currentY = iy + fy;
                const float* input = (const float*)(uInput
                    + zR
                    + currentX * PACK_NUMBER
                    + currentY * iw * PACK_NUMBER
                    + zC * iw * ih * PACK_NUMBER
                );
                sumValue = sumValue + (*input) * mulValue;
            }
        }
        float* dst = (float*)(uOutput
            + zC * ow * oh * 16
            + y * ow * 16
            + x * 16
            + zR
        );
        *dst = sumValue;
    }
}

ErrorCode PoolExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto layer       = mParameter;
    int strideWidth  = layer->strideX();
    int strideHeight = layer->strideY();
    int padWidth     = layer->padX();
    int padHeight    = layer->padY();

    // edit const if global
    auto input       = inputs[0];
    auto output      = outputs[0];
    int kernelWidth  = std::min(layer->kernelX(), input->width());
    int kernelHeight = std::min(layer->kernelY(), input->height());
    if (layer->isGlobal()) {
        kernelWidth  = input->width();
        kernelHeight = input->height();
        strideWidth  = input->width();
        strideHeight = input->height();
        padWidth     = 0;
        padHeight    = 0;
    }
    if (layer->padType() == PoolPadType_SAME) {
        int padNeededWidth  = (output->width() - 1) * strideWidth + kernelWidth - input->width();
        int padNeededHeight = (output->height() - 1) * strideHeight + kernelHeight - input->height();
        padWidth            = padNeededWidth > 0 ? padNeededWidth / 2 : 0;
        padHeight           = padNeededHeight > 0 ? padNeededHeight / 2 : 0;
    } else if (layer->padType() == PoolPadType_VALID) {
        padWidth = padHeight = 0;
    }
    mPoolType      = layer->type();
    auto padType           = layer->padType();
    if (layer->pads() != nullptr && padType == PoolPadType_CAFFE) {
        padType = PoolPadType_VALID;
    }
    mPadType = padType;
    mPaddings = {padWidth, padHeight};
    mStrides = {strideWidth, strideHeight};
    mKernels = {kernelWidth, kernelHeight};
    return NO_ERROR;
}

ErrorCode PoolExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto iw = inputs[0]->width();
    auto ih = inputs[0]->height();
    auto bc = inputs[0]->batch() * UP_DIV(inputs[0]->channel(), PACK_NUMBER);
    auto ow = outputs[0]->width();
    auto oh = outputs[0]->height();
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto& prop = runtime->prop();
    int threads_num = prop.maxThreadsPerBlock;
    int block_num = prop.multiProcessorCount;
    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        auto inputPtr = (const half*)inputs[0]->deviceId();
        auto outputPtr = (half*)outputs[0]->deviceId();
        switch (mPoolType) {
            case PoolType_AVEPOOL:
                avgpool_halfC16<<<block_num, threads_num>>>(inputPtr, outputPtr, 
                    bc, 
                    ih, iw,
                    oh, ow,
                    mPaddings[0], mPaddings[1],
                    mKernels[0], mKernels[1],
                    mStrides[0], mStrides[1]
                );
                return NO_ERROR;
            case PoolType_MAXPOOL:
                maxpool_halfC16<<<block_num, threads_num>>>(inputPtr, outputPtr, 
                    bc, 
                    ih, iw,
                    oh, ow,
                    mPaddings[0], mPaddings[1],
                    mKernels[0], mKernels[1],
                    mStrides[0], mStrides[1]
                );
                return NO_ERROR;
        }        
        return NO_ERROR;
    }
    auto inputPtr = (const float*)inputs[0]->deviceId();
    auto outputPtr = (float*)outputs[0]->deviceId();
    switch (mPoolType) {
        case PoolType_AVEPOOL:
            avgpool_floatC16<<<block_num, threads_num>>>(inputPtr, outputPtr, 
                bc, 
                ih, iw,
                oh, ow,
                mPaddings[0], mPaddings[1],
                mKernels[0], mKernels[1],
                mStrides[0], mStrides[1]
            );
            return NO_ERROR;
        case PoolType_MAXPOOL:
            maxpool_floatC16<<<block_num, threads_num>>>(inputPtr, outputPtr, 
                bc, 
                ih, iw,
                oh, ow,
                mPaddings[0], mPaddings[1],
                mKernels[0], mKernels[1],
                mStrides[0], mStrides[1]
            );
            return NO_ERROR;
    }
    return NOT_SUPPORT;
}
class PoolCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new PoolExecution(op->main_as_Pool(), backend);
    }
};

static CUDACreatorRegister<PoolCreator> __init(OpType_Pooling);


};
};