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
__global__ void DeconvKernelReorder(const T0* B, T1* BP, int kw, int kh, int ic, int oc, int icPack, int ocPack) {
    int kernelCount = kw * kh;
    int maxCount = kernelCount * icPack * oc;
    // l = Cip, h = Cop * KhKw
    // [Ci, Co, KhKw] -> [KhKw, Co, Cip]
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int l_idx = index % icPack;
        int h_idx = index / icPack;
        int oc_idx = h_idx % oc;
        int khw_idx = h_idx / oc;

        if(l_idx >= ic) {
            BP[index] = (T1)0.0f;
            continue;
        }
        BP[index] = (T1)(B[(l_idx * oc + oc_idx) * kernelCount + khw_idx]);
    }
}

#define DATA_CONVERT_COPY(precision) \
    if(precision == 1 || precision == 0) { *((float4*)((float*)data_im + dst_offset)) = val; }\
    else if(precision == 2) { float2 tmp; tmp.x = val.x; tmp.y = val.y; *(half2 *)((half *)(data_im + dst_offset))= __float22half2_rn(tmp); \
        tmp.x = val.z; tmp.y = val.w; *(half2 *)((half *)(data_im + dst_offset + 2))= __float22half2_rn(tmp);}

template <typename Stype, typename Dtype>
__global__ void Col2Im_Vec4(const int n, const Stype* data_col,
    const int batch, const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    const Dtype* bias, Dtype* data_im,
    const int precision,
    DivModFast d_ocp,
    DivModFast d_ow,
    DivModFast d_oh,
    DivModFast d_ob
) {
    const int channel_pack = ((channels+7) / 8) * 8;

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x) {
        float4 val;
        val.x = 0.0;
        val.y = 0.0;
        val.z = 0.0;
        val.w = 0.0;
        int idx_ocp, idx_bhw, idx_bh, b_im, idx_w, idx_h;
        d_ocp.divmod(index, idx_bhw, idx_ocp);
        const int c_im = idx_ocp << 2;

        if(c_im >= channels) {
            continue;
        }

        d_ow.divmod(idx_bhw, idx_bh, idx_w);
        d_oh.divmod(idx_bh, b_im, idx_h);

        const int w_im = idx_w + pad_w;
        const int h_im = idx_h + pad_h;

        if(nullptr != bias) {
            if(precision == 2) {
                val.x += (float)bias[c_im];
                val.y += (float)bias[c_im+1];
                val.z += (float)bias[c_im+2];
                val.w += (float)bias[c_im+3];
            } else {
                val = *((float4*)((float*)bias + c_im));
            }
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
        const int src_offset_add = kernel_h * kernel_w * batch * height_col * width_col;
        // TODO: use LCM of stride and dilation to avoid unnecessary loops
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
                if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                    h_k /= dilation_h;
                    w_k /= dilation_w;

                    const int data_col_index = ((((b_im * height_col + h_col) * width_col + w_col) * kernel_h + h_k) * kernel_w + w_k) * channel_pack + c_im;
                    val.x += (float)data_col[data_col_index];
                    val.y += (float)data_col[data_col_index + 1];
                    val.z += (float)data_col[data_col_index + 2];
                    val.w += (float)data_col[data_col_index + 3];
                }
            }
        }
        int dst_offset = index << 2;
        DATA_CONVERT_COPY(precision);
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
    const Dtype* bias, Dtype* data_im,
    const int precision,
    DivModFast d_ocp,
    DivModFast d_ow,
    DivModFast d_oh,
    DivModFast d_ob
) {
    const int channel_pack = ((channels+7) / 8) * 8;

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x) {
        Dtype val = 0;        
        int idx_ocp, idx_tmp, idx_bh, b_im, idx_w, idx_h;
        d_ocp.divmod(index, idx_tmp, idx_ocp);
        const int c_im = idx_ocp;
        if(c_im >= channels) {
            data_im[index] = val;
            break;
        }

        d_ow.divmod(idx_tmp, idx_bh, idx_w);
        d_oh.divmod(idx_bh, b_im, idx_h);

        const int w_im = idx_w + pad_w;
        const int h_im = idx_h + pad_h;

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

                    const int data_col_index = ((((b_im * height_col + h_col) * width_col + w_col) * kernel_h + h_k) * kernel_w + w_k) * channels + c_im;

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
 
    int ocPack = UP_DIV(kernel_info.kernelN, PACK_NUMBER) * PACK_NUMBER;
    if(precision == 1) {
        DeconvKernelReorder<<<cores, threadNumbers>>>((const float*)input, (float*)output,
            kernel_info.kernelX, kernel_info.kernelY, kernel_info.kernelC, kernel_info.kernelN, icPack, ocPack);
        checkKernelErrors;
    } else if(precision == 0) {
        DeconvKernelReorder<<<cores, threadNumbers>>>((const float*)input, (half*)output,
            kernel_info.kernelX, kernel_info.kernelY, kernel_info.kernelC, kernel_info.kernelN, icPack, ocPack);
        checkKernelErrors;
    } else {
        DeconvKernelReorder<<<cores, threadNumbers>>>((const half*)input, (half*)output,
            kernel_info.kernelX, kernel_info.kernelY, kernel_info.kernelC, kernel_info.kernelN, icPack, ocPack);
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

    DivModFast ocpD((UP_DIV(col2im_param->oc, 8) * 8));
    DivModFast ocpD_4((UP_DIV(col2im_param->oc, 8) * 2));
    DivModFast owD(col2im_param->ow);
    DivModFast ohD(col2im_param->oh);
    DivModFast obD( col2im_param->ob);

    // printf("col2im:%d, %d-%d-%d-%d-%d-%d\n %d-%d-%d-%d-%d-%d\n %d-%d\n", col2im_param->ob, col2im_param->oh, col2im_param->ow, col2im_param->oc, \
    //     col2im_param->ih, col2im_param->iw, col2im_param->ic, \
    //     col2im_param->padX, col2im_param->padY, col2im_param->kernelX, col2im_param->kernelY, col2im_param->strideX, col2im_param->strideY, \
    //     col2im_block_num, col2im_thread_num);
    
    if(col2im_param->oc % 4 == 0) {
        num_kernels /= 4;
        col2im_block_num = runtime->blocks_num(num_kernels);
        col2im_thread_num = runtime->threads_num();
        if(precision == 1) {
            Col2Im_Vec4<<<col2im_block_num, col2im_thread_num>>>(
                num_kernels, (const float*)input, col2im_param->ob, col2im_param->oh, col2im_param->ow, col2im_param->oc, 
                col2im_param->kernelY, col2im_param->kernelX, col2im_param->padY, col2im_param->padX, 
                col2im_param->strideY, col2im_param->strideX, col2im_param->dilateY, col2im_param->dilateX,
                height_col, width_col, (const float*)bias, (float *)output, precision, ocpD_4, owD, ohD, obD);
            checkKernelErrors;
            return;
        } else if(precision == 0) {
            Col2Im_Vec4<<<col2im_block_num, col2im_thread_num>>>(
                num_kernels, (const half*)input, col2im_param->ob, col2im_param->oh, col2im_param->ow, col2im_param->oc, 
                col2im_param->kernelY, col2im_param->kernelX, col2im_param->padY, col2im_param->padX, 
                col2im_param->strideY, col2im_param->strideX, col2im_param->dilateY, col2im_param->dilateX,
                height_col, width_col, (const float*)bias, (float *)output, precision, ocpD_4, owD, ohD, obD);
            checkKernelErrors;
            return;
        } else {
            Col2Im_Vec4<<<col2im_block_num, col2im_thread_num>>>(
                num_kernels, (const half*)input, col2im_param->ob, col2im_param->oh, col2im_param->ow, col2im_param->oc, 
                col2im_param->kernelY, col2im_param->kernelX, col2im_param->padY, col2im_param->padX, 
                col2im_param->strideY, col2im_param->strideX, col2im_param->dilateY, col2im_param->dilateX,
                height_col, width_col, (const half*)bias, (half *)output, precision, ocpD_4, owD, ohD, obD);
            checkKernelErrors;
            return;
        }
    }

    if(precision == 1) {
        Col2Im<<<col2im_block_num, col2im_thread_num>>>(
            num_kernels, (const float*)input, col2im_param->ob, col2im_param->oh, col2im_param->ow, col2im_param->oc, 
            col2im_param->kernelY, col2im_param->kernelX, col2im_param->padY, col2im_param->padX, 
            col2im_param->strideY, col2im_param->strideX, col2im_param->dilateY, col2im_param->dilateX,
            height_col, width_col, (const float*)bias, (float *)output, precision, ocpD, owD, ohD, obD);
        checkKernelErrors;
        return;
    } else if(precision == 0) {
        Col2Im<<<col2im_block_num, col2im_thread_num>>>(
            num_kernels, (const half*)input, col2im_param->ob, col2im_param->oh, col2im_param->ow, col2im_param->oc, 
            col2im_param->kernelY, col2im_param->kernelX, col2im_param->padY, col2im_param->padX, 
            col2im_param->strideY, col2im_param->strideX, col2im_param->dilateY, col2im_param->dilateX,
            height_col, width_col, (const float*)bias, (float *)output, precision, ocpD, owD, ohD, obD);
        checkKernelErrors;
        return;
    } else {
        Col2Im<<<col2im_block_num, col2im_thread_num>>>(
            num_kernels, (const half*)input, col2im_param->ob, col2im_param->oh, col2im_param->ow, col2im_param->oc, 
            col2im_param->kernelY, col2im_param->kernelX, col2im_param->padY, col2im_param->padX, 
            col2im_param->strideY, col2im_param->strideX, col2im_param->dilateY, col2im_param->dilateX,
            height_col, width_col, (const half*)bias, (half *)output, precision, ocpD, owD, ohD, obD);
        checkKernelErrors;
        return;
    }
}
 
 
 
} //namespace CUDA
} //namespace MNN