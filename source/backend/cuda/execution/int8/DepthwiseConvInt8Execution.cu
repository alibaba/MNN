//
//  DepthwiseConvInt8Execution.cpp
//  MNN
//
//  Created by MNN on 2023/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "DepthwiseConvInt8Execution.hpp"
#include "../Raster.cuh"
#include "../MNNCUDADefine.hpp"
#include "../MNNCUDAFunction.cuh"
#include <sm_61_intrinsics.h>

namespace MNN {
namespace CUDA {

__inline__ __device__
int32_t vecDot(char4 inp0, char4 inp1, int32_t val)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610))
    return __dp4a(inp0, inp1, val);
#else
    int32_t res = val;
    res += inp0.x * inp1.x;
    res += inp0.y * inp1.y;
    res += inp0.z * inp1.z;
    res += inp0.w * inp1.w;
    return res;
#endif
}

__global__ void CONV_DW_INT8_(const int8_t* input, 
    const int8_t* kernel, 
    const int32_t* bias, 
    const float*  scale,
    int8_t *output, 
    const int8_t maxV,
    const int8_t minV,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
    const int k_p,
    const int dw,
    const int dh,
    const int sw,
    const int sh,
    const int pw,
    const int ph,
    const int total,
    DivModFast d_oc,
    DivModFast d_ow,
    DivModFast d_oh
) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total/4; index += blockDim.x * gridDim.x) {
        int oz_4, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_4);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);
        
        int oz = oz_4 << 2;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;

        int4 bias4 = ((int4 *)(bias + oz))[0];
        float color0 = (float)bias4.x;
        float color1 = (float)bias4.y;
        float color2 = (float)bias4.z;
        float color3 = (float)bias4.w;

        int fxSta = max(0, (UP_DIV(-ix, dw)));
        int fySta = max(0, (UP_DIV(-iy, dh)));
        int fxEnd = min(kw, UP_DIV(iw - ix, dw));
        int fyEnd = min(kh, UP_DIV(ih - iy, dh));
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy*dh + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx*dw + ix;
                int src_offset = ((ob * ih + sy) * iw + sx) * c_p + oz;

                char4 inp4 = ((char4 *)(input + src_offset))[0];
                char4 ker4 = ((char4 *)(kernel + (fy * kw + fx) * c_p + oz))[0];;

                color0 = color0 + (float)inp4.x * (float)ker4.x;
                color1 = color1 + (float)inp4.y * (float)ker4.y;
                color2 = color2 + (float)inp4.z * (float)ker4.z;
                color3 = color3 + (float)inp4.w * (float)ker4.w;

            }
        }

        float4 scale4 = ((float4 *)(scale + oz))[0];
        color0 = color0 * scale4.x;
        color1 = color1 * scale4.y;
        color2 = color2 * scale4.z;
        color3 = color3 * scale4.w;

        color0 = max(color0, (float)minV);
        color0 = min(color0, (float)maxV);

        color1 = max(color1, (float)minV);
        color1 = min(color1, (float)maxV);

        color2 = max(color2, (float)minV);
        color2 = min(color2, (float)maxV);

        color3 = max(color3, (float)minV);
        color3 = min(color3, (float)maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        char4 res = make_char4(__float2int_rn(color0), __float2int_rn(color1), __float2int_rn(color2), __float2int_rn(color3));
        ((char4*)(output + dst_offset))[0] = res;
    }
}


__global__ void CONV_DW3x3S1_INT8_OPT(const int8_t* input, 
    const int8_t* kernel, 
    const int32_t* bias, 
    const float*  scale,
    int8_t *output, 
    const int8_t maxV,
    const int8_t minV,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
    const int k_p,
    const int dw,
    const int dh,
    const int sw,
    const int sh,
    const int pw,
    const int ph,
    const int total,
    DivModFast d_oc,
    DivModFast d_ow,
    DivModFast d_oh
) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total/8; index += blockDim.x * gridDim.x) {
        int oz_4, tmp2, oy, ox_2, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_4);
        d_ow.divmod(tmp1, tmp2, ox_2);
        d_oh.divmod(tmp2, ob, oy);
        
        int ox = ox_2 << 1;
        int oz = oz_4 << 2;
        int ix = ox - 1;
        int iy = oy - 1;

        int4 bias4 = ((int4 *)(bias + oz))[0];
        float color0_0 = (float)bias4.x;
        float color0_1 = color0_0;
        float color1_0 = (float)bias4.y;
        float color1_1 = color1_0;
        float color2_0 = (float)bias4.z;
        float color2_1 = color2_0;
        float color3_0 = (float)bias4.w;
        float color3_1 = color3_0;

        char4 zero4 = make_char4(0, 0, 0, 0);
        char4 inp4[12], ker4[3][3];
        for(int j=0; j<3; j++) {
            if(iy < 0 && j==0) {
                for(int i=0; i<4; i++) {
                    inp4[i] = zero4;
                }
                continue;
            }
            if(iy+2 > ih-1 && j==2) {
                for(int i=0; i<4; i++) {
                    inp4[8+i] = zero4;
                }
                continue;
            }

            for(int i=0; i<4; i++) {
                if(ix < 0 && i==0) {
                    for(int j=0; j<3; j++) {
                        inp4[4*j+0] = zero4;
                    }
                    continue;
                }
                if(ix+3 > iw-1 && i==3) {
                    for(int j=0; j<3; j++) {
                        inp4[4*j+3] = zero4;
                    }
                    continue;
                }
                int src_offset = ((ob * ih + iy+j) * iw + ix+i) * c_p + oz;
                inp4[4*j+i] = ((char4 *)(input + src_offset))[0];
            }
        }

        for(int j=0; j<3; j++) {
            for(int i=0; i<3; i++) {
                ker4[j][i] = ((char4 *)(kernel + (j * 3 + i) * c_p + oz))[0];// kernel[(j * 3 + i) * c_p + oz];
            }
        }

        // 1st channel
        char4 tmp0_inp4 = make_char4(inp4[0].x, inp4[1].x, inp4[2].x, inp4[4].x);
        char4 tmp1_inp4 = make_char4(inp4[1].x, inp4[2].x, inp4[3].x, inp4[5].x);
        char4 tmp_ker4 = make_char4(ker4[0][0].x, ker4[0][1].x, ker4[0][2].x, ker4[1][0].x);
        int32_t tmp0_res = vecDot(tmp0_inp4, tmp_ker4, 0);
        int32_t tmp1_res = vecDot(tmp1_inp4, tmp_ker4, 0);

        tmp0_inp4 = make_char4(inp4[5].x, inp4[6].x, inp4[8].x, inp4[9].x);
        tmp1_inp4 = make_char4(inp4[6].x, inp4[7].x, inp4[9].x, inp4[10].x);
        tmp_ker4 = make_char4(ker4[1][1].x, ker4[1][2].x, ker4[2][0].x, ker4[2][1].x);
        tmp0_res += vecDot(tmp0_inp4, tmp_ker4, 0);
        tmp1_res += vecDot(tmp1_inp4, tmp_ker4, 0);

        tmp0_res += inp4[10].x * ker4[2][2].x;
        tmp1_res += inp4[11].x * ker4[2][2].x;

        color0_0 += (float)tmp0_res;
        color0_1 += (float)tmp1_res;

        // 2nd channel
        tmp0_inp4 = make_char4(inp4[0].y, inp4[1].y, inp4[2].y, inp4[4].y);
        tmp1_inp4 = make_char4(inp4[1].y, inp4[2].y, inp4[3].y, inp4[5].y);
        tmp_ker4 = make_char4(ker4[0][0].y, ker4[0][1].y, ker4[0][2].y, ker4[1][0].y);
        tmp0_res = vecDot(tmp0_inp4, tmp_ker4, 0);
        tmp1_res = vecDot(tmp1_inp4, tmp_ker4, 0);

        tmp0_inp4 = make_char4(inp4[5].y, inp4[6].y, inp4[8].y, inp4[9].y);
        tmp1_inp4 = make_char4(inp4[6].y, inp4[7].y, inp4[9].y, inp4[10].y);
        tmp_ker4 = make_char4(ker4[1][1].y, ker4[1][2].y, ker4[2][0].y, ker4[2][1].y);
        tmp0_res += vecDot(tmp0_inp4, tmp_ker4, 0);
        tmp1_res += vecDot(tmp1_inp4, tmp_ker4, 0);

        tmp0_res += inp4[10].y * ker4[2][2].y;
        tmp1_res += inp4[11].y * ker4[2][2].y;

        color1_0 += (float)tmp0_res;
        color1_1 += (float)tmp1_res;


        // 3rd channel
        tmp0_inp4 = make_char4(inp4[0].z, inp4[1].z, inp4[2].z, inp4[4].z);
        tmp1_inp4 = make_char4(inp4[1].z, inp4[2].z, inp4[3].z, inp4[5].z);
        tmp_ker4 = make_char4(ker4[0][0].z, ker4[0][1].z, ker4[0][2].z, ker4[1][0].z);
        tmp0_res = vecDot(tmp0_inp4, tmp_ker4, 0);
        tmp1_res = vecDot(tmp1_inp4, tmp_ker4, 0);

        tmp0_inp4 = make_char4(inp4[5].z, inp4[6].z, inp4[8].z, inp4[9].z);
        tmp1_inp4 = make_char4(inp4[6].z, inp4[7].z, inp4[9].z, inp4[10].z);
        tmp_ker4 = make_char4(ker4[1][1].z, ker4[1][2].z, ker4[2][0].z, ker4[2][1].z);
        tmp0_res += vecDot(tmp0_inp4, tmp_ker4, 0);
        tmp1_res += vecDot(tmp1_inp4, tmp_ker4, 0);

        tmp0_res += inp4[10].z * ker4[2][2].z;
        tmp1_res += inp4[11].z * ker4[2][2].z;

        color2_0 += (float)tmp0_res;
        color2_1 += (float)tmp1_res;


        // 4th channel
        tmp0_inp4 = make_char4(inp4[0].w, inp4[1].w, inp4[2].w, inp4[4].w);
        tmp1_inp4 = make_char4(inp4[1].w, inp4[2].w, inp4[3].w, inp4[5].w);
        tmp_ker4 = make_char4(ker4[0][0].w, ker4[0][1].w, ker4[0][2].w, ker4[1][0].w);
        tmp0_res = vecDot(tmp0_inp4, tmp_ker4, 0);
        tmp1_res = vecDot(tmp1_inp4, tmp_ker4, 0);

        tmp0_inp4 = make_char4(inp4[5].w, inp4[6].w, inp4[8].w, inp4[9].w);
        tmp1_inp4 = make_char4(inp4[6].w, inp4[7].w, inp4[9].w, inp4[10].w);
        tmp_ker4 = make_char4(ker4[1][1].w, ker4[1][2].w, ker4[2][0].w, ker4[2][1].w);
        tmp0_res += vecDot(tmp0_inp4, tmp_ker4, 0);
        tmp1_res += vecDot(tmp1_inp4, tmp_ker4, 0);

        tmp0_res += inp4[10].w * ker4[2][2].w;
        tmp1_res += inp4[11].w * ker4[2][2].w;

        color3_0 += (float)tmp0_res;
        color3_1 += (float)tmp1_res;



        float4 scale4 = ((float4 *)(scale + oz))[0];
        color0_0 = color0_0 * scale4.x;
        color0_1 = color0_1 * scale4.x;

        color1_0 = color1_0 * scale4.y;
        color1_1 = color1_1 * scale4.y;

        color2_0 = color2_0 * scale4.z;
        color2_1 = color2_1 * scale4.z;

        color3_0 = color3_0 * scale4.w;
        color3_1 = color3_1 * scale4.w;

        color0_0 = max(color0_0, (float)minV);
        color0_0 = min(color0_0, (float)maxV);
        color0_1 = max(color0_1, (float)minV);
        color0_1 = min(color0_1, (float)maxV);

        color1_0 = max(color1_0, (float)minV);
        color1_0 = min(color1_0, (float)maxV);
        color1_1 = max(color1_1, (float)minV);
        color1_1 = min(color1_1, (float)maxV);

        color2_0 = max(color2_0, (float)minV);
        color2_0 = min(color2_0, (float)maxV);
        color2_1 = max(color2_1, (float)minV);
        color2_1 = min(color2_1, (float)maxV);

        color3_0 = max(color3_0, (float)minV);
        color3_0 = min(color3_0, (float)maxV);
        color3_1 = max(color3_1, (float)minV);
        color3_1 = min(color3_1, (float)maxV);
        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        char4 res0 = make_char4(__float2int_rn(color0_0), __float2int_rn(color1_0), __float2int_rn(color2_0), __float2int_rn(color3_0));
        char4 res1 = make_char4(__float2int_rn(color0_1), __float2int_rn(color1_1), __float2int_rn(color2_1), __float2int_rn(color3_1));

        ((char4*)(output + dst_offset))[0] = res0;
        ((char4*)(output + dst_offset + c_p))[0] = res1;

    }
}

DepthwiseConvInt8Execution::DepthwiseConvInt8Execution(Backend* backend, const Op* op, std::shared_ptr<ConvInt8CutlassExecution::Resource> res) : ConvInt8CutlassExecution(backend, op, res) {
    mOp = op;
    mResource = res;//
}
DepthwiseConvInt8Execution::~DepthwiseConvInt8Execution() {
    // Do nothing
}

bool DepthwiseConvInt8Execution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new DepthwiseConvInt8Execution(bn, op, mResource);
    *dst = exe;
    return true;
}

ErrorCode DepthwiseConvInt8Execution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    std::vector<float> inputQuantInfo = TensorUtils::getQuantInfo(input);
    std::vector<float> outputQuantInfo = TensorUtils::getQuantInfo(output);
    mResource->updateInputOutputScale(inputQuantInfo, outputQuantInfo);
    runtime->memcpy(mResource->mBiasInt32Ptr, mResource->mBiasInt32Vec, mResource->mOutputChannelPack*sizeof(int32_t), MNNMemcpyHostToDevice);
    runtime->memcpy(mResource->mScaleFloatPtr, mResource->mScaleFloatVec, mResource->mOutputChannelPack*sizeof(float), MNNMemcpyHostToDevice);


    mPads = ConvolutionCommon::convolutionPad(input, output, mOp->main_as_Convolution2D()->common());

    auto mCommon = mOp->main_as_Convolution2D()->common();

    const int src_width      = input->width();
    const int src_height     = input->height();
    const int dst_width      = output->width();
    const int dst_height     = output->height();
    const int strideY        = mCommon->strideY();
    const int strideX        = mCommon->strideX();
    const int dilateY        = mCommon->dilateY();
    const int dilateX        = mCommon->dilateX();
    const int kernel_height  = mCommon->kernelY();
    const int kernel_width   = mCommon->kernelX();

    mStrides = std::make_pair(strideX, strideY);
    mDilates = std::make_pair(dilateX, dilateY);
    mKernels = std::make_pair(kernel_width, kernel_height);

    auto clamp_max = mResource->mClampMax;
    auto clamp_min = mResource->mClampMin;

    if (mCommon->relu()) {
        clamp_min = 0;
    }
    if (mCommon->relu6()) {
        clamp_min = 0;
        clamp_max = 6;
    }
    mClamps = std::make_pair(clamp_max, clamp_min);
    // MNN_PRINT("%d-%d-%d-%d, %d-%d-%d-%d\n", mKernels.first, mKernels.second, mStrides.first, mStrides.second, mDilates.first, mDilates.second, mPads.first, mPads.second);

    return NO_ERROR;
}

ErrorCode DepthwiseConvInt8Execution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto& prop = runtime->prop();

    auto input        = inputs[0];
    auto output       = outputs[0];
    const int batch   = input->batch();
    const int c       = input->channel();
    const int c_p     = UP_DIV(c, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;
    const int iw      = input->width();
    const int ih      = input->height();
    const int ow      = output->width();
    const int oh      = output->height();
    const int total   = batch * c_p * oh * ow;

    const int k_p     = UP_DIV(mKernels.first * mKernels.second, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;
    const auto weightPtr   = mResource->mWeightInt8Ptr;
    const auto biasPtr     = mResource->mBiasInt32Ptr;
    const auto scalePtr    = mResource->mScaleFloatPtr;

    int limitThreads = UP_DIV(total, prop.multiProcessorCount);
    int threads_num = ALIMIN(prop.maxThreadsPerBlock/2, limitThreads);
    int block_num = prop.multiProcessorCount;

    DivModFast d_oc(c_p / 4);
    DivModFast d_ow(ow);
    DivModFast d_oh(oh);


    if(mKernels.first==3 && mKernels.second==3 && mStrides.first==1 && mStrides.second==1 && mPads.first==1 && mPads.second==1 && ow % 2 ==0) {
        DivModFast d_ow2(ow/2);

        CONV_DW3x3S1_INT8_OPT<<<block_num, threads_num>>>((const int8_t*)inputs[0]->deviceId(), (const int8_t*)weightPtr,
            (const int32_t*)biasPtr, (const float*)scalePtr, (int8_t*)outputs[0]->deviceId(),
            mClamps.first, mClamps.second, iw, ih, c, c_p, ow, oh, mKernels.first, mKernels.second, k_p,
            mDilates.first, mDilates.second, mStrides.first, mStrides.second, mPads.first, mPads.second,
            total, d_oc, d_ow2, d_oh);
        checkKernelErrors;
        return NO_ERROR;
    }

    CONV_DW_INT8_<<<block_num, threads_num>>>((const int8_t*)inputs[0]->deviceId(), (const int8_t*)weightPtr,
        (const int32_t*)biasPtr, (const float*)scalePtr, (int8_t*)outputs[0]->deviceId(),
        mClamps.first, mClamps.second, iw, ih, c, c_p, ow, oh, mKernels.first, mKernels.second, k_p,
        mDilates.first, mDilates.second, mStrides.first, mStrides.second, mPads.first, mPads.second,
        total, d_oc, d_ow, d_oh);
    checkKernelErrors;

    return NO_ERROR;
}

class DepthWiseConvInt8ExecutionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (inputs.size() > 1) {
            MNN_PRINT("OpType_DepthwiseConvInt8 CUDA not support multi input!, fall back...\n");
            return nullptr;
        }
        std::shared_ptr<ConvInt8CutlassExecution::Resource> resource(new ConvInt8CutlassExecution::Resource(backend, op));
        return new DepthwiseConvInt8Execution(backend, op, resource);
    }
};

static CUDACreatorRegister<DepthWiseConvInt8ExecutionCreator> __init(OpType_DepthwiseConvInt8);

} // namespace CUDA
} // namespace MNN
