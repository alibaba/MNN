//
//  DeconvBaseKernel.cu
//  MNN
//
//  Created by MNN on 2023/04/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//
 
#include "DeconvBaseKernel.cuh"
#include "MNNCUDAFunction.cuh"
#include "MNNCUDADefine.hpp"
 
namespace MNN {
namespace CUDA {
 
template<typename T0, typename T1>
__global__ void DeconvKernelReorder(const T0* B, T1* BP, int kw, int kh, int ic, int oc, int icPack) {
    int kernelCount = kw * kh;
    int e = oc * kernelCount;
    int l = ic;
    int lAlign = icPack;

    int maxCount = e * lAlign;
    // l * e  --> e * lp
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int lp_idx = index % lAlign;
        int e_idx = index / lAlign;

        if(lp_idx >= l) {
            BP[index] = (T1)0.0f;
            continue;
        }
        BP[index] = (T1)(B[lp_idx * e + e_idx]);
    }
}

template <typename Stype, typename Dtype>
__global__ void Col2Im(const int n, const Stype* data_col,
    const int batch, const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    const Dtype* bias, Dtype* data_im
) {
    const int channel_pack = ((channels+7) / 8) * 8;

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x) {
        Dtype val = 0;        
        const int c_p = index % channel_pack;
        const int idx_tmp = index / channel_pack;
        const int b_im = idx_tmp / (width * height);
        const int hw = idx_tmp % (width * height);
        const int c_im = c_p;
        const int w_im = hw % width + pad_w;
        const int h_im = hw / width + pad_h;

        if(c_im >= channels) {
            data_im[index] = val;
            break;
        }
        if(nullptr != bias) {
            val += (Dtype)bias[c_im];
        }
        int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
        int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
        // compute the start and end of the output
        const int w_col_start =
            (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);
        const int h_col_start =
            (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);
        // TODO: use LCM of stride and dilation to avoid unnecessary loops
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
                if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                    h_k /= dilation_h;
                    w_k /= dilation_w;

                    const int data_col_index = ((((c_im * kernel_h + h_k) * kernel_w + w_k) * batch + b_im) *
                                            height_col + h_col) * width_col + w_col;
                    val += (Dtype)data_col[data_col_index];
                }
            }
        }
        data_im[index] = val;
    }
}

void callWeightReorder(const void* input, void* output, const KernelInfo kernel_info, const int icPack, const int precision, CUDARuntime* runtime) {
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    int threadNumbers = prop.maxThreadsPerBlock;
 
    if(precision == 1) {
        DeconvKernelReorder<<<cores, threadNumbers>>>((const float*)input, (float*)output,
            kernel_info.kernelX, kernel_info.kernelY, kernel_info.kernelC, kernel_info.kernelN, icPack);
        checkKernelErrors;
    } else if(precision == 0) {
        DeconvKernelReorder<<<cores, threadNumbers>>>((const float*)input, (half*)output,
            kernel_info.kernelX, kernel_info.kernelY, kernel_info.kernelC, kernel_info.kernelN, icPack);
        checkKernelErrors;
    } else {
        DeconvKernelReorder<<<cores, threadNumbers>>>((const half*)input, (half*)output,
            kernel_info.kernelX, kernel_info.kernelY, kernel_info.kernelC, kernel_info.kernelN, icPack);
        checkKernelErrors;
    }
}
 
void callCol2ImFunc(const void* input, const void* bias, void* output, const Col2ImParameter* col2im_param, const int precision, CUDARuntime* runtime) {
    // Do Col2Im trans
    int height_col = col2im_param->ih;
    int width_col = col2im_param->iw;
    int num_kernels = col2im_param->ob * UP_DIV(col2im_param->oc, 8) * col2im_param->oh * col2im_param->ow * 8;

    int col2im_block_num = runtime->blocks_num(num_kernels);
    int col2im_thread_num = runtime->threads_num();

    // printf("col2im:%d, %d-%d-%d-%d-%d-%d\n %d-%d-%d-%d-%d-%d\n %d-%d\n", col2im_param->ob, col2im_param->oh, col2im_param->ow, col2im_param->oc, \
    //     col2im_param->ih, col2im_param->iw, col2im_param->ic, \
    //     col2im_param->padX, col2im_param->padY, col2im_param->kernelX, col2im_param->kernelY, col2im_param->strideX, col2im_param->strideY, \
    //     col2im_block_num, col2im_thread_num);
    
    if(precision == 1) {
        Col2Im<<<col2im_block_num, col2im_thread_num>>>(
            num_kernels, (const float*)input, col2im_param->ob, col2im_param->oh, col2im_param->ow, col2im_param->oc, 
            col2im_param->kernelY, col2im_param->kernelX, col2im_param->padY, col2im_param->padX, 
            col2im_param->strideY, col2im_param->strideX, col2im_param->dilateY, col2im_param->dilateX,
            height_col, width_col, (const float*)bias, (float *)output);
        checkKernelErrors;
    } else if(precision == 0) {
        Col2Im<<<col2im_block_num, col2im_thread_num>>>(
            num_kernels, (const half*)input, col2im_param->ob, col2im_param->oh, col2im_param->ow, col2im_param->oc, 
            col2im_param->kernelY, col2im_param->kernelX, col2im_param->padY, col2im_param->padX, 
            col2im_param->strideY, col2im_param->strideX, col2im_param->dilateY, col2im_param->dilateX,
            height_col, width_col, (const float*)bias, (float *)output);
        checkKernelErrors;
    } else {
        Col2Im<<<col2im_block_num, col2im_thread_num>>>(
            num_kernels, (const half*)input, col2im_param->ob, col2im_param->oh, col2im_param->ow, col2im_param->oc, 
            col2im_param->kernelY, col2im_param->kernelX, col2im_param->padY, col2im_param->padX, 
            col2im_param->strideY, col2im_param->strideX, col2im_param->dilateY, col2im_param->dilateX,
            height_col, width_col, (const half*)bias, (half *)output);
        checkKernelErrors;
    }

}
 
 
 
} //namespace CUDA
} //namespace MNN