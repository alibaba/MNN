//
//  PoolBf16.cuh
//  MNN
//
//  Created by MNN on 2023/05/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_BF16

#ifndef CONV_DEPTHWISE_BF16_CUH_
#define CONV_DEPTHWISE_BF16_CUH_

#include "../MNNCUDADefine.hpp"
#include "../MNNCUDAFunction.cuh"

namespace MNN {
namespace CUDA {

template<typename T>
__global__ void maxpool_C8_BF16(const T* uInput, T* uOutput,
    const int ib, const int ic_p,
    const int ih, const int iw,
    const int oh, const int ow,
    const int padX, const int padY,
    const int kernelX, const int kernelY,
    const int strideX, const int strideY
) {
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))    
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
        T maxValue = uInput[0];
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
    #endif
}

template<typename T>
__global__ void avgpool_C8_BF16(const T* uInput, T* uOutput,
    const int ib, const int ic_p,
    const int ih, const int iw,
    const int oh, const int ow,
    const int padX, const int padY,
    const int kernelX, const int kernelY,
    const int strideX, const int strideY
) {
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
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
    #endif
}
} //namespace CUDA
} //namespace MNN
#endif
#endif