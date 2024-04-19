//
//  ConvFpAIntBExecution.cpp
//  MNN
//
//  Created by MNN on 2024/03/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvFpAIntBExecution.hpp"
#include "../Raster.cuh"
#include "../ConvBaseKernel.cuh"
#include <float.h>

//#define DEBUG

namespace MNN {
namespace CUDA {

template<typename T>
__global__ void CONV_FpAInt8B(const T* input,
    const int8_t* kernel,
    const T* scale,
    const T* offset,
    const T* bias,
    T *output,
    const float maxV,
    const float minV,
    const int ic,
    const int ic_p,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
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

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);
        
        int oz = oz_2;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color0 = bias[oz];
        float x_scale = scale[oz];
        float x_offset = offset[oz];

        int fxSta = max(0, (UP_DIV(-ix, dw)));
        int fySta = max(0, (UP_DIV(-iy, dh)));
        int fxEnd = min(kw, UP_DIV(iw - ix, dw));
        int fyEnd = min(kh, UP_DIV(ih - iy, dh));
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy*dh + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx*dw + ix;
                for(int sz = 0; sz < ic_p; sz++) {
                    int src_offset = ((ob * ih + sy) * iw + sx) * ic_p + sz;
                    float inp0 = input[src_offset];
                    //[Cop, KhKw, Cip]
                    int8_t ker0 = kernel[((oz * kh + fy) * kw + fx) * ic_p + sz];

                    color0 = color0 + inp0 * ((float)ker0 * x_scale + x_offset);
                }
            }
        }
        color0 = max(color0, minV);
        color0 = min(color0, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        output[dst_offset] = color0;
    }
}

template<typename T>
__global__ void CONV_FpAInt4B(const T* input,
    const uint8_t* kernel,
    const T* scale,
    const T* offset,
    const T* bias,
    T *output,
    const float maxV,
    const float minV,
    const int ic,
    const int ic_p,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
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

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);
        
        int oz = oz_2;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color0 = bias[oz];
        float x_scale = scale[oz];
        float x_offset = offset[oz];

        int fxSta = max(0, (UP_DIV(-ix, dw)));
        int fySta = max(0, (UP_DIV(-iy, dh)));
        int fxEnd = min(kw, UP_DIV(iw - ix, dw));
        int fyEnd = min(kh, UP_DIV(ih - iy, dh));
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy*dh + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx*dw + ix;
                for(int sz = 0; sz < ic_p/2; sz++) {
                    int src_offset = ((ob * ih + sy) * iw + sx) * ic_p + 2*sz;
                    float inp0 = input[src_offset];
                    float inp1 = input[src_offset+1];

                    //[Cop, KhKw, Cip]
                    uint8_t ker = kernel[((oz * kh + fy) * kw + fx) * ic_p / 2 + sz];
                    int8_t ker0 = (ker >> 4) - 8;
                    int8_t ker1 = (ker & 15) - 8;
                    color0 = color0 + inp0 * ((float)ker0 * x_scale + x_offset);
                    color0 = color0 + inp1 * ((float)ker1 * x_scale + x_offset);
                }
            }
        }
        color0 = max(color0, minV);
        color0 = min(color0, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        output[dst_offset] = color0;
    }
}

__global__ void Rearrange_Weight_Int4(const int8_t* param,
    uint8_t* output,
    const int khw,
    const size_t maxCount,
    const int oc,
    const int ic,
    DivModFast d_khw,
    DivModFast d_icp2
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int icp2Index, temp, ocpIndex, khwIndex;
        d_icp2.divmod(index, temp, icp2Index);
        d_khw.divmod(temp, ocpIndex, khwIndex);
        if(2*icp2Index >= ic || ocpIndex >= oc) {
            output[index] = 0;
            continue;
        }
        // [Co, Ci, KhKw] -> [Cop, KhKw, Cip/2], Ci available for vectorize
        output[index] = ((uint8_t)(param[(ocpIndex * ic + 2*icp2Index+0) * khw + khwIndex] + 8 )) * 16;
        if(2*icp2Index+1 < ic) {
            output[index] += ((uint8_t)(param[(ocpIndex * ic + 2*icp2Index+1) * khw + khwIndex] + 8 ));
        }
    }
}

__global__ void Rearrange_Weight_Int8(const int8_t* param,
    int8_t* output,
    const int khw,
    const size_t maxCount,
    const int oc,
    const int ic,
    DivModFast d_khw,
    DivModFast d_icp
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int icpIndex, temp, ocpIndex, khwIndex;
        d_icp.divmod(index, temp, icpIndex);
        d_khw.divmod(temp, ocpIndex, khwIndex);
        if(icpIndex >= ic || ocpIndex >= oc) {
            output[index] = 0;
            continue;
        }
        // [Co, Ci, KhKw] -> [Cop, KhKw, Cip], Ci available for vectorize
        output[index] = param[(ocpIndex * ic + icpIndex) * khw + khwIndex];
    }
}

bool ConvFpAIntBExecution::isValid(const Convolution2D* conv, Backend* backend) {
    return true;
}

    
ConvFpAIntBExecution::Resource::Resource(Backend* bn, const MNN::Op* op) {
    mBackend = bn;
    auto runtime = static_cast<CUDABackend*>(bn)->getCUDARuntime();

    auto conv       = op->main_as_Convolution2D();
    auto common     = conv->common();

    //weight host->device
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon = ConvolutionCommon::load(conv, mBackend, false, true);

    auto oc = common->outputCount();
    auto weightSize = quanCommon->weight.size();
    int l = weightSize / oc;
    int h = oc;
    int ic = common->inputCount();
    if(ic == 0) {
        ic = l / common->kernelX() / common->kernelY();
    }

    int lp = UP_DIV(l, 8) * 8;
    int hp = UP_DIV(h, 8) * 8;

    // set dequant scale/offset
    {
        float * dequantAlpha = quanCommon->alpha.get();
        std::vector<float> dequantScale(hp, 0.0);
        std::vector<float> dequantOffset(hp, 0.0);

        for (int o = 0; o < oc; o++) {
            float min = 0.0f;
            float alpha = 0.0f;
            if (quanCommon->asymmetric) {
                min = dequantAlpha[2*o];
                alpha = dequantAlpha[2*o+1];
            } else {
                alpha = dequantAlpha[o];
            }
            dequantScale[o] = alpha;
            dequantOffset[o] = min;
        }

        if(static_cast<CUDABackend*>(bn)->useFp16()) {

            auto tempScaleStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(hp*sizeof(float));
            auto scaleTemp = (float*)((uint8_t*)tempScaleStorage.first + tempScaleStorage.second);
            cuda_check(cudaMemcpy(scaleTemp, dequantScale.data(), hp*sizeof(float), cudaMemcpyHostToDevice));

            scaleTensor.reset(Tensor::createDevice<int16_t>({hp}));
            bn->onAcquireBuffer(scaleTensor.get(), Backend::STATIC);
            mScale = (void *)scaleTensor.get()->buffer().device;
            callFloat2Half((const void*)scaleTemp, (void*)mScale, hp, runtime);

            // Reuse scaleTemp buffer
            cuda_check(cudaMemcpy(scaleTemp, dequantOffset.data(), hp*sizeof(float), cudaMemcpyHostToDevice));

            offsetTensor.reset(Tensor::createDevice<int16_t>({hp}));
            bn->onAcquireBuffer(offsetTensor.get(), Backend::STATIC);
            mOffset = (void *)offsetTensor.get()->buffer().device;
            callFloat2Half((const void*)scaleTemp, (void*)mOffset, hp, runtime);

            static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempScaleStorage);
        } else {

            scaleTensor.reset(Tensor::createDevice<int32_t>({hp}));
            bn->onAcquireBuffer(scaleTensor.get(), Backend::STATIC);
            mScale = (void *)scaleTensor.get()->buffer().device;
            cuda_check(cudaMemcpy(mScale, dequantScale.data(), hp*sizeof(float), cudaMemcpyHostToDevice));

            offsetTensor.reset(Tensor::createDevice<int32_t>({hp}));
            bn->onAcquireBuffer(offsetTensor.get(), Backend::STATIC);
            mOffset = (void *)offsetTensor.get()->buffer().device;
            cuda_check(cudaMemcpy(mOffset, dequantOffset.data(), hp*sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    // Reorder weight
    {
        int khw = common->kernelX() * common->kernelY();
        int icp = UP_DIV(ic, 8) * 8;
        DivModFast khwD(khw);
        DivModFast icp2D(icp/2);
        DivModFast icpD(icp);

        if(quanCommon->canUseInt4) {
            mIsWeightInt4 = true;

            auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(int8_t));
            float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
            runtime->memcpy(cacheWeight, quanCommon->weight.get(), weightSize * sizeof(int8_t), MNNMemcpyHostToDevice);

            weightTensor.reset(Tensor::createDevice<uint8_t>({khw * icp/2 * hp}));
            bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
            mFilter = (void *)weightTensor.get()->buffer().device;

            //[Co, Ci, KhKw] -> [Cop, KhKw, Cip/2]
            int block_num = runtime->blocks_num(khw*icp/2*hp);
            int block_size = runtime->threads_num();
            Rearrange_Weight_Int4<<<block_num, block_size>>>((const int8_t*)cacheWeight, (uint8_t*)mFilter, khw, khw*icp/2*hp, oc, ic, khwD, icp2D);
            checkKernelErrors;

            static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempCacheBuffer);
        } else {
            auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(int8_t));
            float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
            runtime->memcpy(cacheWeight, quanCommon->weight.get(), weightSize * sizeof(int8_t), MNNMemcpyHostToDevice);
            weightTensor.reset(Tensor::createDevice<int8_t>({lp * hp}));

            bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
            mFilter = (void *)weightTensor.get()->buffer().device;

            //[Co, Ci, KhKw] -> [Cop, KhKw, Cip]
            int block_num = runtime->blocks_num(khw*icp*hp);
            int block_size = runtime->threads_num();
            Rearrange_Weight_Int8<<<block_num, block_size>>>((const int8_t*)cacheWeight, (int8_t*)mFilter, khw, khw*icp*hp, oc, ic, khwD, icpD);
            checkKernelErrors;
            static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempCacheBuffer);
        }
    }

    // Copy Bias
    {
        if(static_cast<CUDABackend*>(bn)->useFp16()) {
            int biasSize = conv->bias()->size();
            int hp = UP_DIV(biasSize, 8) * 8;

            auto tempBiasStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(hp*sizeof(float));
            auto biasTemp = (float*)((uint8_t*)tempBiasStorage.first + tempBiasStorage.second);
            runtime->memset(biasTemp, 0, hp * sizeof(int32_t));
            cuda_check(cudaMemcpy(biasTemp, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));

            biasTensor.reset(Tensor::createDevice<int16_t>({hp}));
            bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
            mBias = (void *)biasTensor.get()->buffer().device;
            callFloat2Half((const void*)biasTemp, (void*)mBias, hp, runtime);

            static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempBiasStorage);
        } else {
            int biasSize = conv->bias()->size();
            int hp = UP_DIV(biasSize, 8) * 8;
            biasTensor.reset(Tensor::createDevice<int32_t>({hp}));
            bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
            mBias = (void *)biasTensor.get()->buffer().device;
            runtime->memset(mBias, 0, hp * sizeof(int32_t));
            cuda_check(cudaMemcpy(mBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));
        }
    }
}

ConvFpAIntBExecution::Resource::~Resource() {
    // Do nothing
}
ConvFpAIntBExecution::ConvFpAIntBExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res) : CutlassConvCommonExecution(backend) {
    mOp = op;
    mResource = res;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    mPrecisonLevel = static_cast<CUDABackend*>(backend)->getPrecision();
    mFp16Infer = (mPrecisonLevel == 2);
    mFp32Infer = (mPrecisonLevel == 1);
    mFp16Fp32MixInfer = (mPrecisonLevel == 0);
    mBf16Infer = (mPrecisonLevel == 3);
}

ConvFpAIntBExecution::~ConvFpAIntBExecution() {

}
bool ConvFpAIntBExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvFpAIntBExecution(bn, op, mResource);
    *dst = dstExe;
    return true;
}


ErrorCode ConvFpAIntBExecution::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input = inputs[0], output = outputs[0];
    const int UNIT = PACK_NUMBER;
    auto convCommon = mOp->main_as_Convolution2D()->common();
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mOp->main_as_Convolution2D()->common());
    int ic = input->channel();
    auto icDiv = UP_DIV(ic, UNIT);

    mIm2ColParamter.dilateX         = convCommon->dilateX();
    mIm2ColParamter.dilateY         = convCommon->dilateY();
    mIm2ColParamter.strideX         = convCommon->strideX();
    mIm2ColParamter.strideY         = convCommon->strideY();
    mIm2ColParamter.icDiv4          = icDiv;
    mIm2ColParamter.ic              = ic;
    mIm2ColParamter.kernelX         = convCommon->kernelX();
    mIm2ColParamter.kernelY         = convCommon->kernelY();
    mIm2ColParamter.padX = std::get<0>(pads);
    mIm2ColParamter.padY = std::get<1>(pads);

    mIm2ColParamter.ih = input->height();
    mIm2ColParamter.iw = input->width();
    mIm2ColParamter.oh = output->height();
    mIm2ColParamter.ow = output->width();
    mIm2ColParamter.srcZStep = input->height() * input->width() * UNIT * input->batch();
    mIm2ColParamter.srcYStep = input->width() * UNIT;
    mIm2ColParamter.packCUnit = UNIT;

    mActivationType = convCommon->relu() ? 1 : convCommon->relu6() ? 2 : 0;

    //MNN_PRINT("conv size:%d-%d, %d-%d-%d, %d-%d-%d\n", mIm2ColParamter.kernelX, mIm2ColParamter.strideX, input->height(), input->width(), input->channel(), output->height(), output->width(), output->channel());
    int e = output->height() * output->width() * output->batch();
    int l = ic * mIm2ColParamter.kernelX * mIm2ColParamter.kernelY;
    int h = output->channel();
    mGemmInfo.elh[0] = e;
    mGemmInfo.elh[1] = l;
    mGemmInfo.elh[2] = h;
    mGemmInfo.elhPad[0] = UP_DIV(e, 8) * 8;
    mGemmInfo.elhPad[1] = UP_DIV(l, 8) * 8;
    mGemmInfo.elhPad[2] = UP_DIV(h, 8) * 8;

    return NO_ERROR;
}

ErrorCode ConvFpAIntBExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];

    //printf("convcutlass:%p %p\n", input->deviceId(), output->deviceId());
    //MNN_PRINT("cutlass hw:%d-%d\n", input->height(), input->width());
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    const void *input_addr = (const void*)inputs[0]->deviceId();
    const void *filter_addr = mResource->mFilter;
    const void *bias_addr = mResource->mBias;
    auto bn = backend();
    void *output_addr = (void*)outputs[0]->deviceId();

    const int sw = mIm2ColParamter.strideX;
    const int sh = mIm2ColParamter.strideY;
    const int kw = mIm2ColParamter.kernelX;
    const int kh = mIm2ColParamter.kernelY;
    const int dw = mIm2ColParamter.dilateX;
    const int dh = mIm2ColParamter.dilateY;
    const int pw = mIm2ColParamter.padX;
    const int ph = mIm2ColParamter.padY;
    const int ic = mIm2ColParamter.ic;
    const int icp = UP_DIV(ic, 8) * 8;
    const int iw = mIm2ColParamter.iw;
    const int ih = mIm2ColParamter.ih;

    const int oc = mGemmInfo.elh[2];
    const int ocp = mGemmInfo.elhPad[2];
    const int ow = mIm2ColParamter.ow;
    const int oh = mIm2ColParamter.oh;

    float maxV = FLT_MAX;
    float minV = -FLT_MAX;
    if (mActivationType == 1) {
        minV = 0.0f;
    }
    if (mActivationType == 2) {
        minV = 0.0f;
        maxV = 6.0f;
    }

    auto total = outputs[0]->batch() * oh * ow * ocp; 
    auto& prop = runtime->prop();
    int limitThreads = UP_DIV(total, prop.multiProcessorCount);
    int threadNum = ALIMIN(prop.maxThreadsPerBlock/2, limitThreads);
    int blockNum = prop.multiProcessorCount;

    DivModFast d_oc(ocp);
    DivModFast d_ow(ow);
    DivModFast d_oh(oh);

    if(mResource->mIsWeightInt4) {
        if(mFp16Infer) {
            CONV_FpAInt4B<<<blockNum, threadNum>>>((const half*)input_addr, (const uint8_t*)mResource->mFilter,
                (const half*)mResource->mScale,  (const half*)mResource->mOffset, (const half*)bias_addr, (half*)output_addr,
                maxV, minV, ic, icp, iw, ih, oc, ocp, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
                d_oc, d_ow, d_oh);
            checkKernelErrors;
        } else {
            CONV_FpAInt4B<<<blockNum, threadNum>>>((const float*)input_addr, (const uint8_t*)mResource->mFilter,
                (const float*)mResource->mScale,  (const float*)mResource->mOffset, (const float*)bias_addr, (float*)output_addr,
                maxV, minV, ic, icp, iw, ih, oc, ocp, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
                d_oc, d_ow, d_oh);
            checkKernelErrors;  
        }
        
        return NO_ERROR;
    }

    if(mFp16Infer) {
        CONV_FpAInt8B<<<blockNum, threadNum>>>((const half*)input_addr, (const int8_t*)mResource->mFilter,
            (const half*)mResource->mScale,  (const half*)mResource->mOffset, (const half*)bias_addr, (half*)output_addr,
            maxV, minV, ic, icp, iw, ih, oc, ocp, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
            d_oc, d_ow, d_oh);
        checkKernelErrors;
    } else {
        CONV_FpAInt8B<<<blockNum, threadNum>>>((const float*)input_addr, (const int8_t*)mResource->mFilter,
            (const float*)mResource->mScale,  (const float*)mResource->mOffset, (const float*)bias_addr, (float*)output_addr,
            maxV, minV, ic, icp, iw, ih, oc, ocp, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
            d_oc, d_ow, d_oh);
        checkKernelErrors;  
    }
    
    return NO_ERROR;
}


}// namespace CUDA
}// namespace MNN
