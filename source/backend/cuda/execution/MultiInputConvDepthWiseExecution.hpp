//
//  MultiInputConvDepthWiseExecution.hpp
//  MNN
//
//  Created by MNN on 2023/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MultiInputConvDepthWiseExecution_hpp
#define MultiInputConvDepthWiseExecution_hpp

#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "ConvBaseKernel.cuh"
#include "MNNCUDAFunction.cuh"
#include "MNNCUDADefine.hpp"
#include <float.h>

namespace MNN {
namespace CUDA {


template<typename T>
__global__ void CONV_DW(const T* input, const half* kernel, const half* bias, T *output, const float maxV, const float minV, const int iw, const int ih, const int c, const int c_p, const int ow, const int oh, const int kw, const int kh, const int dw, const int dh, const int sw, const int sh, const int pw, const int ph, const int total, DivModFast d_oc, DivModFast d_ow, DivModFast d_oh);

__global__ void CONV_DW_HALF2_OPT(const half2* input, const half2* kernel, const half2* bias, half2 *output, const float maxV, const float minV, const int iw, const int ih, const int c, const int c_p, const int ow, const int oh, const int kw, const int kh, const int dw, const int dh, const int sw, const int sh, const int pw, const int ph, const int total, DivModFast d_oc, DivModFast d_ow, DivModFast d_oh);

__global__ void CONV_DW3x3_HALF2_OPT(const half2* input, const half2* kernel, const half2* bias, half2 *output,  const float maxV, const float minV, const int iw, const int ih, const int c, const int c_p, const int ow, const int oh, const int kw, const int kh, const int dw, const int dh, const int sw, const int sh, const int pw, const int ph, const int total, DivModFast d_oc, DivModFast d_ow, DivModFast d_oh);

__global__ void CONV_DW_OPT(const float* input, const half* kernel, const half* bias, float *output, const float maxV, const float minV, const int iw, const int ih, const int c, const int c_p, const int ow, const int oh, const int kw, const int kh, const int dw, const int dh, const int sw, const int sh, const int pw, const int ph, const int total, DivModFast d_oc, DivModFast d_ow, DivModFast d_oh);

template<typename T>
__global__ void CONV_DW_MULTI_WIDTH4(const T* input, const half* kernel, const half* bias, T *output, const float maxV, const float minV, const int iw, const int ih, const int c, const int c_p, const int ow, const int oh, const int kw, const int kh, const int total, DivModFast d_oc, DivModFast d_ow_4, DivModFast d_oh);

__global__ void CONV_DW_MULTI_WIDTH_CHANNEL(const float* input, const half* kernel, const half* bias, float *output, const float maxV, const float minV, const int iw, const int ih, const int c, const int c_p, const int ow, const int oh, const int kw, const int kh, const int total, DivModFast d_oc_2, DivModFast d_ow_2, DivModFast d_oh );

#ifdef ENABLE_CUDA_BF16
__global__ void CONV_DW_BF16(const __nv_bfloat16* input, const __nv_bfloat16* kernel, const __nv_bfloat16* bias, __nv_bfloat16 *output, const float maxV, const float minV, const int iw, const int ih, const int c, const int c_p, const int ow, const int oh, const int kw, const int kh, const int dw, const int dh, const int sw, const int sh, const int pw, const int ph, const int total, DivModFast d_oc, DivModFast d_ow, DivModFast d_oh);

__global__ void CONV_DW_BF162_OPT(const __nv_bfloat162* input, const __nv_bfloat162* kernel, const __nv_bfloat162* bias, __nv_bfloat162 *output, const float maxV, const float minV, const int iw, const int ih, const int c, const int c_p, const int ow, const int oh, const int kw, const int kh, const int dw, const int dh, const int sw, const int sh, const int pw, const int ph, const int total, DivModFast d_oc, DivModFast d_ow, DivModFast d_oh);


__global__ void CONV_DW3x3_BF162_OPT(const __nv_bfloat162* input, const __nv_bfloat162* kernel, const __nv_bfloat162* bias, __nv_bfloat162 *output, const float maxV, const float minV, const int iw, const int ih, const int c, const int c_p, const int ow, const int oh, const int kw, const int kh, const int dw, const int dh, const int sw, const int sh, const int pw, const int ph, const int total, DivModFast d_oc, DivModFast d_ow, DivModFast d_oh);

template<typename T>
__global__ void CONV_DW_BF16_MULTI_WIDTH4(const T* input, const __nv_bfloat16* kernel, const __nv_bfloat16* bias, T *output, const float maxV, const float minV, const int iw, const int ih, const int c, const int c_p, const int ow, const int oh, const int kw, const int kh, const int total, DivModFast d_oc, DivModFast d_ow_4, DivModFast d_oh);
#endif

ErrorCode ConvDepthWiseCompute(Backend* bn, const int blockNum, const int threadNum, const void * inputAddr, const void * filterAddr, const void * biasAddr, void * outputAddr, const float maxV, const float minV, const int iw, const int ih, const int c, const int c_p, const int ow, const int oh, const int kw, const int kh, const int dw, const int dh, const int sw, const int sh, const int pw, const int ph, const int total, DivModFast d_oc, DivModFast d_ow, DivModFast d_oh);

template<typename type1, typename type2>
__global__ void WeightPrepare(const type1 * inputWeightDevice, type2 * outputWeightDevice, const int numTotal, const int numChannel, const int kernelHeight, const int kernelWeight, DivModFast divNumChannelPack, DivModFast divKernelWeight);

template<typename type1, typename type2>
__global__ void BiasPrepare(const type1 * inputBiasDevice, type2 * outputBiasDevice, const int numTotal, const int numChannel);

class MultiInputConvDepthWiseExecution : public Execution {
public:
    MultiInputConvDepthWiseExecution(const Op *op, Backend *bn);
    virtual ~MultiInputConvDepthWiseExecution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    struct MultiInputConvDepthWiseParams {
        void* mFilter;
        void* mBias;
        int inputSize[2];
        int outputSize[2];
        int kernelSize[2];
        int stride[2];
        int pad[2];
        int dilate[2];
        int channel_raw;
        int channel_div;
        int channel_pack;
        int batch;
        int numWeightPackTotal;
        int numBiasPackTotal;
        int numOutputTotal;
        float minValue = -65504.0f;
        float maxValue = 65504.0f;
    };
    const Op *mOp;
    MultiInputConvDepthWiseParams mParams;
};


class MultiInputDeconvDepthWiseExecution : public MultiInputConvDepthWiseExecution {
public:
    MultiInputDeconvDepthWiseExecution(const Op *op, Backend *bn) : MultiInputConvDepthWiseExecution(op, bn) {
        // Do nothing
    }
    virtual ~MultiInputDeconvDepthWiseExecution() {
        // Do nothing
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};


} // namespace CUDA
} // namespace MNN
#endif