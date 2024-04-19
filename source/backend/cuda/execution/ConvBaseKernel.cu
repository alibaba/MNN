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
__global__ void Im2Col_FilterC(
    const int sw,
    const int sh,
    const int dw,
    const int dh,
    const int pw,
    const int ph,
    const int icDiv4,
    const int iw,
    const int ih,
    const int ic,
    const size_t maxCount,
    const int pack,
    const int e,
    const int l,
    const int l_p,
    const T0* A,
    T1* AP,
    DivModFast d_lp,
    DivModFast d_ow,
    DivModFast d_oh,
    DivModFast d_fx,
    DivModFast d_ic
) {
    
    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int eIndex, lpIndex;
        d_lp.divmod(indexO, eIndex, lpIndex);

        if(eIndex >= e || lpIndex >= l) {
            *(AP + indexO) = (T1)0.0f;
            continue;
        }
        // Compute for source
        int ox, oby, ob, oy, iz, kI, ksx, ksy;
        d_ow.divmod(eIndex, oby, ox);
        d_oh.divmod(oby, ob, oy);
        d_ic.divmod(lpIndex, kI, iz);
        d_fx.divmod(kI, ksy, ksx);

        size_t sx = ox * sw + ksx * dw - pw;
        size_t sy = oy * sh + ksy * dh- ph;

        const int ic_p = icDiv4 * pack;
        size_t dst_offset = eIndex * l_p + kI * ic + iz;

        if (sx >= 0 && sx < iw) {
            if (sy >=0 && sy < ih) {
                size_t offset = ((ob * ih + sy) * iw + sx) * ic_p + iz;
                *(AP + dst_offset) = (T1)(*(A + offset));
                continue;
            }
        }
        *(AP + dst_offset) = (T1)0.0f;
    }
}

#define DATA_CONVERT_COPY(precision) \
    if(precision == 1) { *((float4*)((float*)AP + dst_offset)) = *((float4*)((float*)A + src_offset)); }\
    else if(precision == 2) { *((int64_t*)((half*)AP + dst_offset)) = *((int64_t*)((half*)A + src_offset)); }\
    else if(precision == 3) { *((int64_t*)((half*)AP + dst_offset)) = *((int64_t*)((half*)A + src_offset)); }\
    else if(precision == 0) { *((half2*)((half*)AP + dst_offset)) = __float22half2_rn(*((float2*)((float*)A + src_offset))); \
        *((half2*)((half*)AP + dst_offset + 2)) = __float22half2_rn(*((float2*)((float*)A + src_offset + 2)));}

#define DATA_MEMSET_ZERO(precision) \
    if(precision == 1) { float4 zeros; zeros.x = 0.0f; zeros.y = 0.0f; zeros.z = 0.0f; zeros.w = 0.0f; *((float4*)((float*)AP + dst_offset)) = zeros; }\
    else if(precision == 2 || precision == 0) { half2 zeros; zeros.x = (half)0.0f; zeros.y = (half)0.0f; *((half2*)((half*)AP + dst_offset)) = zeros;  *((half2*)((half*)AP + dst_offset + 2)) = zeros;}


template<typename T0, typename T>
__global__ void Im2Col_FilterC_Vec4(
    const int sw,
    const int sh,
    const int dw,
    const int dh,
    const int pw,
    const int ph,
    const int icDiv4,
    const int iw,
    const int ih,
    const int ic,
    const size_t maxCount,
    const int pack,
    const int e,
    const int l,
    const int l_p,
    const T0* A,
    T* AP,
    const int precision,
    DivModFast d_lp,
    DivModFast d_ow,
    DivModFast d_oh,
    DivModFast d_fx,
    DivModFast d_ic4
) {
    
    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int eIndex, lpIndex;
        d_lp.divmod(indexO, eIndex, lpIndex);

        // Compute for source
        int ox, oby, ob, oy, iz_4, kI, ksx, ksy;
        d_ow.divmod(eIndex, oby, ox);
        d_oh.divmod(oby, ob, oy);
        d_ic4.divmod(lpIndex, kI, iz_4);
        d_fx.divmod(kI, ksy, ksx);

        size_t sx = ox * sw + ksx * dw - pw;
        size_t sy = oy * sh + ksy * dh- ph;

        const int ic_p = icDiv4 * pack;

        const int iz = iz_4 << 2;
        size_t dst_offset = eIndex * l_p + kI * ic + iz;

        if (sx >= 0 && sx < iw) {
            if (sy >=0 && sy < ih) {
                size_t src_offset = ((ob * ih + sy) * iw + sx) * ic_p + iz;
                DATA_CONVERT_COPY(precision);
                continue;
            }
        }
        DATA_MEMSET_ZERO(precision);
        if(precision == 3) {
            #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
            __nv_bfloat162 zeros; zeros.x = (__nv_bfloat16)0.0f; zeros.y = (__nv_bfloat16)0.0f; *((__nv_bfloat162*)((__nv_bfloat16*)AP + dst_offset)) = zeros;  *((__nv_bfloat162*)((__nv_bfloat16*)AP + dst_offset + 2)) = zeros;
            #endif
        }
    }
}

template<typename T0, typename T>
__global__ void WeightPackFill(const T0* param,
    T* output,
    const int khw,
    const size_t maxCount,
    const int l,
    const int h,
    DivModFast d_lp,
    DivModFast d_ic
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int lpIndex, hpIndex, icIndex, khwIndex;
        d_lp.divmod(index, hpIndex, lpIndex);
        if(lpIndex >= l || hpIndex >= h) {
            output[index] = (T)0.0f;
            continue;
        }
        d_ic.divmod(lpIndex, khwIndex, icIndex);

        // [Co, Ci, KhKw] -> [Co, KhKw, Ci], Ci available for vectorize
        output[index] = param[hpIndex * l + icIndex * khw + khwIndex];
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

#ifdef ENABLE_CUDA_BF16
__global__ void Float22BFloat16(const float* param,
    __nv_bfloat16* output,
    const size_t maxCount
) {
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        float2* srcPtr = (float2 *)(param + (index << 2));
        __nv_bfloat162* dstPtr = (__nv_bfloat162*)(output + (index << 2));
        dstPtr[0] = __float22bfloat162_rn(srcPtr[0]);
        dstPtr[1] = __float22bfloat162_rn(srcPtr[1]);
    }
    #endif
}
#endif

void callFloat2Half(const void* input, void* output, const int count, CUDARuntime* runtime) {
    int thread_count = count / 4;
    int block_num = runtime->blocks_num(thread_count);
    int block_size = runtime->threads_num();
    Float22Half2<<<block_num, block_size>>>((const float*)input, (half *)output, thread_count);
    checkKernelErrors;
}

#ifdef ENABLE_CUDA_BF16
void callFloat2BFloat16(const void* input, void* output, const int count, CUDARuntime* runtime) {
    int thread_count = count / 4;
    int block_num = runtime->blocks_num(thread_count);
    int block_size = runtime->threads_num();
    Float22BFloat16<<<block_num, block_size>>>((const float*)input, (__nv_bfloat16 *)output, thread_count);
    checkKernelErrors;
}
#endif

void callWeightFill(const void* input, void* output, const int ic, const int l, const int h, const int lp, const int hp, const int precision, CUDARuntime* runtime, int quant_int_bit) {
    DivModFast lpD(lp);
    DivModFast icD(ic);

    int block_num = runtime->blocks_num(lp*hp);
    int block_size = runtime->threads_num();

    if (quant_int_bit == 8) {
        WeightPackFill<<<block_num, block_size>>>((const char*)input, (char*)output, l/ic, lp*hp, l, h, lpD, icD);
        checkKernelErrors;
        return;
    }

    if(precision == 1) {
        WeightPackFill<<<block_num, block_size>>>((const float*)input, (float*)output, l/ic, lp*hp, l, h, lpD, icD);
        checkKernelErrors;
    } else if(precision == 0) {
        WeightPackFill<<<block_num, block_size>>>((const float*)input, (half*)output, l/ic, lp*hp, l, h, lpD, icD);
        checkKernelErrors;
    } else if(precision == 2){
        WeightPackFill<<<block_num, block_size>>>((const half*)input, (half*)output, l/ic, lp*hp, l, h, lpD, icD);
        checkKernelErrors;    
    } else {
        MNN_ASSERT(precision == 3);
        #ifdef ENABLE_CUDA_BF16
        WeightPackFill<<<block_num, block_size>>>((const float*)input, (__nv_bfloat16*)output, l/ic, lp*hp, l, h, lpD, icD);
        checkKernelErrors;
        #endif
    }
}

void callIm2ColPack(const void* input, void* output, const ConvolutionCommon::Im2ColParameter* info, const int e, const int l, const int ep, const int lp, const int precision, CUDARuntime* runtime) {
    DivModFast lpD(lp);
    DivModFast fxD(info->kernelX);
    DivModFast owD(info->ow);
    DivModFast ohD(info->oh);
    DivModFast icD(info->ic);

    const int sw = info->strideX;
    const int sh = info->strideY;
    const int dw = info->dilateX;
    const int dh = info->dilateY;
    const int pw = info->padX;
    const int ph = info->padY;
    const int icDiv4 = info->icDiv4;
    const int iw = info->iw;
    const int ih = info->ih;
    const int ic = info->ic;

    size_t maxCount = e * lp;
    size_t block_num = runtime->blocks_num(maxCount);
    size_t block_size = runtime->threads_num();

    if(ic % 4 == 0) {
        maxCount /= 4;
        block_num = runtime->blocks_num(maxCount);
        block_size = runtime->threads_num();
        DivModFast lpD_4(lp/4);
        DivModFast icD_4(ic/4);
        if(precision == 1) {
            Im2Col_FilterC_Vec4<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, ic,
                maxCount, PACK_NUMBER, e, l, lp, (const float*)input, (float *)output, precision,
                lpD_4, owD, ohD, fxD, icD_4);
            checkKernelErrors;
            return;
        } else if(precision == 0) {
            Im2Col_FilterC_Vec4<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, ic,
                maxCount, PACK_NUMBER, e, l, lp, (const float*)input, (half *)output, precision,
                lpD_4, owD, ohD, fxD, icD_4);
            checkKernelErrors;
            return;
        } else if(precision == 2) {
            Im2Col_FilterC_Vec4<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, ic,
                maxCount, PACK_NUMBER, e, l, lp, (const half*)input, (half *)output, precision,
                lpD_4, owD, ohD, fxD, icD_4);
            checkKernelErrors;
            return;
        } else {
            MNN_ASSERT(precision == 3);
            #ifdef ENABLE_CUDA_BF16
            Im2Col_FilterC_Vec4<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, ic,
                maxCount, PACK_NUMBER, e, l, lp, (const __nv_bfloat16*)input, (__nv_bfloat16 *)output, precision,
                lpD_4, owD, ohD, fxD, icD_4);
            checkKernelErrors;
            return;
            #endif
        }
    }

    if(precision == 1) {
        Im2Col_FilterC<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, ic,
            maxCount, PACK_NUMBER, e, l, lp, (const float*)input, (float *)output, \
            lpD, owD, ohD, fxD, icD);
        checkKernelErrors;
    } else if(precision == 0) {
        Im2Col_FilterC<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, ic,
            maxCount, PACK_NUMBER, e, l, lp, (const float*)input, (half *)output, \
            lpD, owD, ohD, fxD, icD);
        checkKernelErrors;
    } else if(precision == 2) {
        Im2Col_FilterC<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, ic,
            maxCount, PACK_NUMBER, e, l, lp, (const half*)input, (half *)output, \
            lpD, owD, ohD, fxD, icD);
        checkKernelErrors;
    } else {
        MNN_ASSERT(precision == 3);
        #ifdef ENABLE_CUDA_BF16
        Im2Col_FilterC<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, ic,
            maxCount, PACK_NUMBER, e, l, lp, (const __nv_bfloat16*)input, (__nv_bfloat16 *)output, \
            lpD, owD, ohD, fxD, icD);
        checkKernelErrors;
        #endif
    }
}



} //namespace CUDA
} //namespace MNN
