//
//  ConvBaseKernel.cu
//  MNN
//
//  Created by MNN on 2023/03/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvBaseKernel.cuh"
#include "MNNCUDAFunction.cuh"
#include "MNNCUDADefine.hpp"

namespace MNN {
namespace CUDA {

template<typename T0, typename T1>
__global__ void Im2Col_packC(
    const int sw,
    const int sh,
    const int dw,
    const int dh,
    const int pw,
    const int ph,
    const int icDiv4,
    const int iw,
    const int ih,
    const size_t maxCount,
    const int pack,
    const int e,
    const int l,
    const T0* A,
    T1* AP,
    DivModFast d_lp,
    DivModFast d_ow,
    DivModFast d_oh,
    DivModFast d_fxy,
    DivModFast d_fx
) {
    
    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int eIndex, lpIndex;
        d_lp.divmod(indexO, eIndex, lpIndex);

        if(eIndex >= e || lpIndex >= l) {
            *(AP + indexO) = (T1)0.0f;
            continue;
        }
        // Compute for source
        int ox, oby, ob, oy, ic, kI, ksx, ksy;
        d_ow.divmod(eIndex, oby, ox);
        d_oh.divmod(oby, ob, oy);
        d_fxy.divmod(lpIndex, ic, kI);
        d_fx.divmod(kI, ksy, ksx);

        size_t sx = ox * sw + ksx * dw - pw;
        size_t sy = oy * sh + ksy * dh- ph;

        const int ic_p = icDiv4 * pack;
        if (sx >= 0 && sx < iw) {
            if (sy >=0 && sy < ih) {
                size_t offset = ((ob * ih + sy) * iw + sx) * ic_p + ic;
                *(AP + indexO) = (T1)(*(A + offset));
                continue;
            }
        }
        *(AP + indexO) = (T1)0.0f;
    }
}
    
template<typename T0, typename T>
__global__ void WeightPackFill(const T0* param,
    T* output,
    const size_t maxCount,
    const int l,
    const int h,
    DivModFast d_lp
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int lpIndex, hpIndex;
        d_lp.divmod(index, hpIndex, lpIndex);

        if(lpIndex >= l || hpIndex >= h) {
            output[index] = (T)0.0f;
            continue;
        }
        output[index] = param[hpIndex * l + lpIndex];
    }
}
    
__global__ void Float22Half2(const float* param,
    half* output,
    const size_t maxCount
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        float2* srcPtr = (float2 *)(param + (index << 2));
        half2* dstPtr = (half2*)(output + (index << 2));
        dstPtr[0] = __float22half2_rn(srcPtr[0]);
        dstPtr[1] = __float22half2_rn(srcPtr[1]);
    }
}


void callFloat2Half(const void* input, void* output, const int count, CUDARuntime* runtime) {
    int thread_count = count / 4;
    int block_num = runtime->blocks_num(thread_count);
    int block_size = runtime->threads_num();
    Float22Half2<<<block_num, block_size>>>((const float*)input, (half *)output, thread_count);
    checkKernelErrors;
}

void callWeightFill(const void* input, void* output, const int l, const int h, const int lp, const int hp, const int precision, CUDARuntime* runtime) {
    DivModFast lpD(lp);
    int block_num = runtime->blocks_num(lp*hp);
    int block_size = runtime->threads_num();

    if(precision == 1) {
        WeightPackFill<<<block_num, block_size>>>((const float*)input, (float*)output, lp*hp, l, h, lpD);
        checkKernelErrors;
    } else if(precision == 0) {
        WeightPackFill<<<block_num, block_size>>>((const float*)input, (half*)output, lp*hp, l, h, lpD);
        checkKernelErrors;
    } else {
        WeightPackFill<<<block_num, block_size>>>((const half*)input, (half*)output, lp*hp, l, h, lpD);
        checkKernelErrors;    
    }
}

void callIm2ColPack(const void* input, void* output, const ConvolutionCommon::Im2ColParameter* info, const int e, const int l, const int ep, const int lp, const int precision, CUDARuntime* runtime) {
    DivModFast lpD(lp);
    DivModFast fxyD((info->kernelX * info->kernelY));
    DivModFast fxD(info->kernelX);
    DivModFast owD(info->ow);
    DivModFast ohD(info->oh);

    const int sw = info->strideX;
    const int sh = info->strideY;
    const int dw = info->dilateX;
    const int dh = info->dilateY;
    const int pw = info->padX;
    const int ph = info->padY;
    const int icDiv4 = info->icDiv4;
    const int iw = info->iw;
    const int ih = info->ih;

    size_t maxCount = e * lp;
    size_t block_num = runtime->blocks_num(maxCount);
    size_t block_size = runtime->threads_num();

    if(precision == 1) {
        Im2Col_packC<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih,
            maxCount, PACK_NUMBER, e, l, (const float*)input, (float *)output, \
            lpD, owD, ohD, fxyD, fxD);
        checkKernelErrors;
    } else if(precision == 0) {
        Im2Col_packC<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, 
            maxCount, PACK_NUMBER, e, l, (const float*)input, (half *)output, \
            lpD, owD, ohD, fxyD, fxD);
        checkKernelErrors;
    } else {
        Im2Col_packC<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, 
            maxCount, PACK_NUMBER, e, l, (const half*)input, (half *)output, \
            lpD, owD, ohD, fxyD, fxD);
        checkKernelErrors;
    }
}



} //namespace CUDA
} //namespace MNN
