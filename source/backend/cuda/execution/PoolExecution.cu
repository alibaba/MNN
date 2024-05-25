#include <cuda_fp16.h>
#include "PoolExecution.hpp"
#include <float.h>
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"

namespace MNN {
namespace CUDA {
#define HALF_MIN  half(-65504)
#define HALF2_MIN half2(-65504, -65504)
#define MNN_CUDA_HALF2_MAX(a, b)                     \
    do {                                             \
        (a).x = __hgt((a).x, (b).x) ? (a).x : (b).x; \
        (a).y = __hgt((a).y, (b).y) ? (a).y : (b).y; \
    } while (0)

template<typename T>
__global__ void maxpool_C8(const T* uInput, T* uOutput,
    const int ib, const int ic_p,
    const int ih, const int iw,
    const int oh, const int ow,
    const int padX, const int padY,
    const int kernelX, const int kernelY,
    const int strideX, const int strideY
    ) {
    int total = ib * oh * ow * ic_p;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int ic_idx = i % ic_p;
        int tmp0 = i / ic_p;
        int ow_idx = tmp0 % ow;
        int tmp1 = tmp0 / ow;
        int ib_idx = tmp1 / oh;
        int oh_idx = tmp1 % oh;

        int iw_idx = ow_idx * strideX - padX;
        int ih_idx = oh_idx * strideY - padY;
        int sx = max(0, -iw_idx);
        int sy = max(0, -ih_idx);
        int ex = min(kernelX, iw - iw_idx);
        int ey = min(kernelY, ih - ih_idx);
        T maxValue = HALF_MIN;
        for (int fy=sy; fy<ey; ++fy) {
            for (int fx=sx; fx<ex; ++fx) {
                int currentX = iw_idx + fx;
                int currentY = ih_idx + fy;
                const T* input = (const T*)(uInput
                    + ib_idx * ih * iw * ic_p 
                    + currentY * iw * ic_p
                    + currentX * ic_p
                    + ic_idx
                );
                T val = *input;
                maxValue = maxValue > val ? maxValue : val;
            }
        }
        T* dst = (T*)(uOutput
            + ib_idx * oh * ow * ic_p 
            + oh_idx * ow * ic_p
            + ow_idx * ic_p
            + ic_idx
        );
        *dst = maxValue;
    }
}

template<typename T>
__global__ void avgpool_C8(const T* uInput, T* uOutput,
    const int ib, const int ic_p,
    const int ih, const int iw,
    const int oh, const int ow,
    const int padX, const int padY,
    const int kernelX, const int kernelY,
    const int strideX, const int strideY
    ) {
    int total = ib * oh * ow * ic_p;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int ic_idx = i % ic_p;
        int tmp0 = i / ic_p;
        int ow_idx = tmp0 % ow;
        int tmp1 = tmp0 / ow;
        int ib_idx = tmp1 / oh;
        int oh_idx = tmp1 % oh;

        int iw_idx = ow_idx * strideX - padX;
        int ih_idx = oh_idx * strideY - padY;
        int sx = max(0, -iw_idx);
        int sy = max(0, -ih_idx);
        int ex = min(kernelX, iw - iw_idx);
        int ey = min(kernelY, ih - ih_idx);
        T div = (float)(ey-sy)* (float)(ex-sx);
        T sumValue = (T)0.0f;
        for (int fy=sy; fy<ey; ++fy) {
            for (int fx=sx; fx<ex; ++fx) {
                int currentX = iw_idx + fx;
                int currentY = ih_idx + fy;
                const T* input = (const T*)(uInput
                    + ib_idx * ih * iw * ic_p 
                    + currentY * iw * ic_p
                    + currentX * ic_p
                    + ic_idx
                );
                T val = *input;
                sumValue += val;
            }
        }
        sumValue /= div; 
        T* dst = (T*)(uOutput
            + ib_idx * oh * ow * ic_p 
            + oh_idx * ow * ic_p
            + ow_idx * ic_p
            + ic_idx
        );
        *dst = sumValue;
    }
}

template <typename T>
__global__ void global_avgpool_C8(const T *input, T *output,
    const int outside,
    const int axis,
    const int inside,
    const int per_block_size,
    const int calc_multi_num
) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x -  idx_outside * inside;

    const T* src = input + idx_outside * axis * inside + idx_inside;
    int tid = threadIdx.x;

    float local_src = 0.0;
    __shared__ float sumValue;
    for(int i=0; i<calc_multi_num; i++) {
        if(tid + i * per_block_size < axis) {
            local_src += (float)(src[(tid + i * per_block_size) * inside]);
        }
    }
    float maxRes = blockReduceSum<float>(local_src);
    if(tid == 0)
        sumValue = maxRes;
    __syncthreads();

    output[idx_outside * inside + idx_inside] = (T)(sumValue / (float)axis);
    return;
}

template <typename T>
__global__ void global_maxpool_C8(const T *input, T *output,
    const int outside,
    const int axis,
    const int inside,
    const int per_block_size,
    const int calc_multi_num
) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x -  idx_outside * inside;

    const T* src = input + idx_outside * axis * inside + idx_inside;
    int tid = threadIdx.x;

    float local_src = -FLT_MAX;;
    __shared__ float maxValue;
    for(int i=0; i<calc_multi_num; i++) {
        if(tid + i * per_block_size < axis) {
            local_src = max(local_src, (float)src[(tid + i * per_block_size) * inside]);
        }
    }
    float maxRes = blockReduceMax<float>(local_src);
    if(tid == 0)
        maxValue = maxRes;
    __syncthreads();

    output[idx_outside * inside + idx_inside] = (T)(maxValue);
    return;
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
    auto ic = inputs[0]->channel();
    auto ic_p = UP_DIV(inputs[0]->channel(), PACK_NUMBER) * PACK_NUMBER;
    auto ib = inputs[0]->batch();
    auto bc = inputs[0]->batch() * UP_DIV(inputs[0]->channel(), PACK_NUMBER);
    auto ow = outputs[0]->width();
    auto oh = outputs[0]->height();
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto& prop = runtime->prop();
    int threads_num = prop.maxThreadsPerBlock;
    int block_num = prop.multiProcessorCount;
    // MNN_PRINT("%d %d, %d %d %d %d %d %d, %d %d\n", ih, iw, mKernels[0], mKernels[1], mPaddings[0], mPaddings[1], mStrides[0], mStrides[1], oh, ow);

    #ifdef ENABLE_CUDA_BF16
    if (static_cast<CUDABackend*>(backend())->getPrecision() == 3) {
        auto inputPtr = (const __nv_bfloat16*)inputs[0]->deviceId();
        auto outputPtr = (__nv_bfloat16*)outputs[0]->deviceId();
        switch (mPoolType) {
            case PoolType_AVEPOOL:
                avgpool_C8_BF16<<<block_num, threads_num>>>(inputPtr, outputPtr, 
                    ib, ic_p, 
                    ih, iw,
                    oh, ow,
                    mPaddings[0], mPaddings[1],
                    mKernels[0], mKernels[1],
                    mStrides[0], mStrides[1]
                );
                return NO_ERROR;
            case PoolType_MAXPOOL:
                maxpool_C8_BF16<<<block_num, threads_num>>>(inputPtr, outputPtr, 
                    ib, ic_p, 
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
    #endif

    // Global Pooling
    if(ih == mKernels[0] && iw == mKernels[1] && mPaddings[0] == 0 && mPaddings[1] == 0 && oh == 1 && ow == 1) {
        auto outside = ib;
        auto axis = ih * iw;
        auto inside = ic_p;
        int count = outside * inside;
        int calc_multi_num = (axis + 127) / 128;
        if (static_cast<CUDABackend*>(backend())->useFp16()) {
            auto inputPtr = (const half*)inputs[0]->deviceId();
            auto outputPtr = (half*)outputs[0]->deviceId();
            if (mPoolType == PoolType_AVEPOOL) {
                global_avgpool_C8<<<count, 128>>>(inputPtr, outputPtr, outside, axis, inside, 128, calc_multi_num);
                checkKernelErrors;
            } else if (mPoolType == PoolType_MAXPOOL) {
                global_maxpool_C8<<<count, 128>>>(inputPtr, outputPtr, outside, axis, inside, 128, calc_multi_num);
                checkKernelErrors;
            } else {
                MNN_ERROR("MNN not support pool type:%d\n", mPoolType);
            }
        } else {
            auto inputPtr = (const float*)inputs[0]->deviceId();
            auto outputPtr = (float*)outputs[0]->deviceId();
            if (mPoolType == PoolType_AVEPOOL) {
                global_avgpool_C8<<<count, 128>>>(inputPtr, outputPtr, outside, axis, inside, 128, calc_multi_num);
                checkKernelErrors;
            } else if (mPoolType == PoolType_MAXPOOL) {
                global_maxpool_C8<<<count, 128>>>(inputPtr, outputPtr, outside, axis, inside, 128, calc_multi_num);
                checkKernelErrors;
            } else {
                MNN_ERROR("MNN not support pool type:%d\n", mPoolType);
            }
        }
        return NO_ERROR;
    }

    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        auto inputPtr = (const half*)inputs[0]->deviceId();
        auto outputPtr = (half*)outputs[0]->deviceId();
        switch (mPoolType) {
            case PoolType_AVEPOOL:
                avgpool_C8<<<block_num, threads_num>>>(inputPtr, outputPtr, 
                    ib, ic_p, 
                    ih, iw,
                    oh, ow,
                    mPaddings[0], mPaddings[1],
                    mKernels[0], mKernels[1],
                    mStrides[0], mStrides[1]
                );
                return NO_ERROR;
            case PoolType_MAXPOOL:
                maxpool_C8<<<block_num, threads_num>>>(inputPtr, outputPtr, 
                    ib, ic_p, 
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

    //MNN_PRINT("Pool pad:%d-%d, kernel:%d-%d, stride:%d-%d\n", mPaddings[1], mPaddings[0], mKernels[1], mKernels[0], mStrides[1], mStrides[0]);
    //MNN_PRINT("Feature input size:%d-%d-%d-%d, output size:%d-%d\n", ib, ic_p, ih, iw, oh, ow);
    auto inputPtr = (const float*)inputs[0]->deviceId();
    auto outputPtr = (float*)outputs[0]->deviceId();
    switch (mPoolType) {
        case PoolType_AVEPOOL:
            avgpool_C8<<<block_num, threads_num>>>(inputPtr, outputPtr, 
                ib, ic_p, 
                ih, iw,
                oh, ow,
                mPaddings[0], mPaddings[1],
                mKernels[0], mKernels[1],
                mStrides[0], mStrides[1]
            );
            return NO_ERROR;
        case PoolType_MAXPOOL:
            maxpool_C8<<<block_num, threads_num>>>(inputPtr, outputPtr, 
                ib, ic_p, 
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
