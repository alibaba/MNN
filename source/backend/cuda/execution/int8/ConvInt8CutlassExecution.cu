//
//  ConvInt8CutlassExecution.cpp
//  MNN
//
//  Created by MNN on 2023/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_QUANT

#include "ConvInt8CutlassExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace CUDA {

__global__ void Im2Col_packC_16(
    const int sw,
    const int sh,
    const int dw,
    const int dh,
    const int pw,
    const int ph,
    const int ic,
    const int iw,
    const int ih,
    const size_t maxCount,
    const int iBlock,
    const int icDiv4,
    const int e,
    const int l,
    const int32_t* A,
    int32_t* AP,
    DivModFast d_lp,
    DivModFast d_ow,
    DivModFast d_oh,
    DivModFast d_icp,
    DivModFast d_fx
) {

    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int eIndex, lpIndex;
        d_lp.divmod(indexO, eIndex, lpIndex);

        if(eIndex >= e) {
            *(((int4 *)AP) + indexO) = make_int4(0, 0, 0, 0);
            continue;
        }
        // Compute for source
        int ox, oby, ob, oy, sz, kI, ksx, ksy;
        d_ow.divmod(eIndex, oby, ox);
        d_oh.divmod(oby, ob, oy);
        d_icp.divmod(lpIndex, kI, sz);
        d_fx.divmod(kI, ksy, ksx);

        // No need to check ci boundry, for weight is already set zero.
        size_t sx = ox * sw + ksx * dw - pw;
        size_t sy = oy * sh + ksy * dh- ph;

        if (sx >= 0 && sx < iw) {
            if (sy >=0 && sy < ih) {
                size_t offset = ((ob * ih + sy) * iw + sx) * icDiv4 + sz;
                *(((int4 *)AP) + indexO) = (*(((int4 *)A) + offset));
                continue;
            }
        }
        *(((int4 *)AP) + indexO) =  make_int4(0, 0, 0, 0);
    }
}

template<typename T>
__global__ void WeightInt8PackFill(const int8_t* param,
    T* output,
    const int maxCount,
    const int l,
    const int h,
    const int hp,
    const int ic,
    DivModFast d_lp,
    DivModFast d_hp,
    DivModFast d_icp,
    const bool ocMajor
) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        if(ocMajor) { // Depthwise Weight
            int lIndex, hpIndex;
            d_hp.divmod(index, lIndex, hpIndex);
            if(hpIndex >= h) {
                output[index] = (T)0;
                continue;
            }
            output[index] = param[hpIndex * l + lIndex];
        } else { // Convolution Weight
            int lpIndex, fxyIndex, icpIndex, hpIndex;
            d_lp.divmod(index, hpIndex, lpIndex);
            d_icp.divmod(lpIndex, fxyIndex, icpIndex);
    
            if(icpIndex >= ic || hpIndex >= h) {
                output[index] = (T)0;
                continue;
            }
    
            output[index] = param[hpIndex * l + icpIndex * (l / ic) + fxyIndex];
        }
    }
}

void ConvInt8CutlassExecution::Resource::updateInputOutputScale(std::vector<float> inputQuantInfo, std::vector<float> outputQuantInfo) {
    if(mUseConvQuan) {
        return;
    }
    // new scales and zero points
    float inputScale = inputQuantInfo[0];
    float outputScale = outputQuantInfo[0];
    float inputZeroPoint = inputQuantInfo[1];
    float outputZeroPoint = outputQuantInfo[1];
    mClampMin = int8_t(outputQuantInfo[2]);
    mClampMax = int8_t(outputQuantInfo[3]);

    if (inputScale == 0.f || outputScale == 0.f) {
        return;
    }

    mInputScale = inputScale;
    mOutputScale = outputScale;
    mInputZeroPoint = int8_t(inputZeroPoint);
    mOutputZeroPoint = int8_t(outputZeroPoint);
    const int kernelNum = static_cast<int>(mInt8WeightKernelSum.size());

    auto alphaScale  = inputScale / outputScale;
    auto alphaData = mScaleFloatVec;
    auto biasData = (float *)mBiasInt32Vec;

    for (int i = 0; i < kernelNum; i++) {
        auto alphaValue = alphaData[i];
        if (fabs(alphaValue) < 1e-6) {
            alphaValue = 1e-6;
        }
        mScaleFloatVec[i] = alphaValue * alphaScale;
        // compute outputZeroPointFused in asymmetric quant
        int outputZeroPointFused = static_cast<int32_t>(outputZeroPoint / mScaleFloatVec[i]);
        mBiasInt32Vec[i] = static_cast<int32_t>(biasData[i] / (alphaScale * alphaValue)) - mInt8WeightKernelSum[i] * inputZeroPoint + outputZeroPointFused;
    }

}

ConvInt8CutlassExecution::Resource::Resource(Backend* bn, const MNN::Op* op) {
    mBackend = bn;
    auto runtime = static_cast<CUDABackend*>(bn)->getCUDARuntime();

    auto conv       = op->main_as_Convolution2D();
    auto common     = conv->common();

    float inputScale = 0.0f;
    float outputScale = 0.0f;
    if (conv->quanParameter() != nullptr) {
        inputScale = conv->quanParameter()->scaleIn();
        outputScale = conv->quanParameter()->scaleOut();
    }

    const auto outputCount = common->outputCount();
    mOutputChannelPack = UP_DIV(outputCount, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;

    mInputScale = inputScale;
    mOutputScale = outputScale;
    mActBits = conv->symmetricQuan()->nbits();

    mBiasInt32Vec = new int32_t[mOutputChannelPack];
    mScaleFloatVec = new float[mOutputChannelPack];

    memset(mBiasInt32Vec, 0, mOutputChannelPack * sizeof(int32_t));
    memset(mScaleFloatVec, 0, mOutputChannelPack * sizeof(float));

    mScaleFloatTensor.reset(Tensor::createDevice<float>({(int)(mOutputChannelPack * sizeof(float))}));
    static_cast<CUDABackend*>(bn)->onAcquireBuffer(mScaleFloatTensor.get(), Backend::STATIC);
    mScaleFloatPtr = (void *)mScaleFloatTensor.get()->buffer().device;

    mBiasInt32Tensor.reset(Tensor::createDevice<int32_t>({(int)(mOutputChannelPack * sizeof(int32_t))}));
    static_cast<CUDABackend*>(bn)->onAcquireBuffer(mBiasInt32Tensor.get(), Backend::STATIC);
    mBiasInt32Ptr = (void *)mBiasInt32Tensor.get()->buffer().device;

    // MNN_PRINT("resource init %p-%p\n", mScaleFloatPtr, mBiasInt32Ptr);

    //weight host->device
    const int8_t* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;

    bool res = ConvolutionCommon::getConvInt8Parameters(conv, quanCommon, bn, filterDataPtr, weightSize, 
                                    mScaleFloatVec, 
                                    mBiasInt32Vec);
                                    // inputScale, 
                                    // outputScale, 
                                    // conv->symmetricQuan()->zeroPoint(),
                                    // conv->symmetricQuan()->outputZeroPoint());
    if(!res) {
        MNN_PRINT("CUDA Error getConvInt8Parameters!\n");
        return;
    }

    const int kernelNum = outputCount;
    const int kernelSize = weightSize / kernelNum;
    for (int i = 0; i < kernelNum; i++) {
        int temp = 0;
        int offset = i * kernelSize;
        for (int j = 0; j < kernelSize; j++) {
            temp += int(filterDataPtr[offset + j]);
        }
        mInt8WeightKernelSum.emplace_back(temp);
    }

    if (conv->bias() && conv->quanParameter() && conv->quanParameter()->alpha()) {
        mUseConvQuan = false;
    }


    mInputZeroPoint = conv->symmetricQuan()->zeroPoint();
    mOutputZeroPoint = conv->symmetricQuan()->outputZeroPoint();
    mClampMin = conv->symmetricQuan()->clampMin();
    mClampMax = conv->symmetricQuan()->clampMax();

    auto oc = common->outputCount();
    auto ic = common->inputCount();

    int l = weightSize / oc;
    int h = oc;
    int ic_p = UP_DIV(ic, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;
    int lp = (l / ic) * ic_p;
    int hp = UP_DIV(h, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;

    if(op->type() == OpType_DepthwiseConvInt8  || op->type() == OpType_ConvolutionDepthwise) {
        lp = l;
    }
    // Reorder weight
    {
        auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(int8_t));
        int8_t* cacheWeight = (int8_t*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
        runtime->memcpy(cacheWeight, filterDataPtr, weightSize * sizeof(int8_t), MNNMemcpyHostToDevice);

        mWeightInt8Tensor.reset(Tensor::createDevice<int8_t>({lp * hp}));
        bn->onAcquireBuffer(mWeightInt8Tensor.get(), Backend::STATIC);
        mWeightInt8Ptr = (void *)mWeightInt8Tensor.get()->buffer().device;

        DivModFast lpD(lp);
        DivModFast hpD(hp);
        DivModFast icpD(ic_p);
        int block_num = runtime->blocks_num(lp*hp);
        int block_size = runtime->threads_num();

        // DepthwiseConv --> [KhKw, (Oc)p]
        // Conv          --> [(Oc)p, KhKw(Ic)p]
        bool ocMajor = false;
        if(op->type() == OpType_DepthwiseConvInt8 || op->type() == OpType_ConvolutionDepthwise) {
            ocMajor = true;
        }
        
        WeightInt8PackFill<<<block_num, block_size>>>((int8_t*)cacheWeight, (int8_t*)mWeightInt8Ptr, lp*hp, l, h, hp, ic, lpD, hpD, icpD, ocMajor);
        checkKernelErrors;

        static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempCacheBuffer);
    }

}

ConvInt8CutlassExecution::Resource::~Resource() {
    if(nullptr != mBiasInt32Vec) {
        delete[] mBiasInt32Vec;
        mBiasInt32Vec = nullptr;
    }
    if(nullptr != mScaleFloatVec) {
        delete[] mScaleFloatVec;
        mScaleFloatVec = nullptr;
    }
}
ConvInt8CutlassExecution::ConvInt8CutlassExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res) : Execution(backend), mOp(op) {
    mResource = res;
}

ConvInt8CutlassExecution::~ConvInt8CutlassExecution() {

}
bool ConvInt8CutlassExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvInt8CutlassExecution(bn, op, mResource);
    *dst = dstExe;
    return true;
}


ErrorCode ConvInt8CutlassExecution::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input = inputs[0], output = outputs[0];

    std::vector<float> inputQuantInfo = TensorUtils::getQuantInfo(input);
    std::vector<float> outputQuantInfo = TensorUtils::getQuantInfo(output);
    mResource->updateInputOutputScale(inputQuantInfo, outputQuantInfo);

    runtime->memcpy(mResource->mScaleFloatPtr, mResource->mScaleFloatVec, mResource->mOutputChannelPack*sizeof(float), MNNMemcpyHostToDevice);
    runtime->memcpy(mResource->mBiasInt32Ptr, mResource->mBiasInt32Vec, mResource->mOutputChannelPack*sizeof(int32_t), MNNMemcpyHostToDevice);

    const int UNIT = INT8_PACK_NUMBER;
    auto convCommon = mOp->main_as_Convolution2D()->common();
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mOp->main_as_Convolution2D()->common());
    int ic = input->channel();
    auto icDiv = UP_DIV(ic, UNIT);

    mIm2ColParamter.dilateX         = convCommon->dilateX();
    mIm2ColParamter.dilateY         = convCommon->dilateY();
    mIm2ColParamter.strideX         = convCommon->strideX();
    mIm2ColParamter.strideY         = convCommon->strideY();
    mIm2ColParamter.icDiv4          = icDiv;
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
    mGemmInfo.elhPad[0] = UP_DIV(e, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;
    mGemmInfo.elhPad[1] = mIm2ColParamter.kernelX * mIm2ColParamter.kernelY * UP_DIV(ic, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;
    mGemmInfo.elhPad[2] = UP_DIV(h, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;

    mIsConv1x1S1D1P0 = (mIm2ColParamter.kernelX == 1 && mIm2ColParamter.kernelY == 1 && \
                        mIm2ColParamter.strideX == 1 && mIm2ColParamter.strideY == 1 && \
                        mIm2ColParamter.dilateX == 1 && mIm2ColParamter.dilateY == 1 && \
                        mIm2ColParamter.padX == 0 && mIm2ColParamter.padY == 0);
    mNeedIm2Col = !mIsConv1x1S1D1P0;

    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    if(mNeedIm2Col) {
        auto buffer = pool->alloc(sizeof(int8_t) * (size_t)mGemmInfo.elh[0] * (size_t)mGemmInfo.elhPad[1]);
        mIm2ColBuffer = (void*)((uint8_t*)buffer.first + buffer.second);
        pool->free(buffer);
    }

    if(mGpuComputeCap < 80) {
        if(mGemmInfo.elh[0] < 128 || mGemmInfo.elhPad[2] < 64 || mGemmInfo.elhPad[1] < 64) { // M-N-K
            mGemmShapeSizeLevel = GEMM_SIZE_LITTLE;
        } else if(mGemmInfo.elh[0] >= 4096 && mGemmInfo.elhPad[2] >= 128 && mGemmInfo.elhPad[1] >= 64) {
            mGemmShapeSizeLevel = GEMM_SIZE_LARGE;
        } else if(mGemmInfo.elh[0] >= 512 && mGemmInfo.elhPad[2] >= 512 && mGemmInfo.elhPad[1] >= 512) {
            mGemmShapeSizeLevel = GEMM_SIZE_LARGE;
        }
    }

    mGpuComputeCap = runtime->compute_capability();
    //MNN_PRINT("Gpu smArch is sm_%d\n", mGpuComputeCap);
    if(mGpuComputeCap < 75) {
        MNN_PRINT("ConvInt8 not support CUDA Capability < 75\n");
        return NO_ERROR;
    }
    if(mGpuComputeCap >= 80) {
        callCutlassGemmInt8TensorCore16832(inputs, outputs);
        return NO_ERROR;
    }
    return callCutlassGemmInt8TensorCore(inputs, outputs);
}

ErrorCode ConvInt8CutlassExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    //MNN_PRINT("cuda convSingleInput onExecute in, inputsize:%d %d\n", (int)inputs.size(), workspace_size_);
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    const void *input_addr = (const void*)inputs[0]->deviceId();
    const void *filter_addr = mResource->mWeightInt8Ptr;
    const void *bias_addr = mResource->mBiasInt32Ptr;
    auto bn = backend();
    void *output_addr = (void*)outputs[0]->deviceId();

    const int sw = mIm2ColParamter.strideX;
    const int sh = mIm2ColParamter.strideY;
    const int dw = mIm2ColParamter.dilateX;
    const int dh = mIm2ColParamter.dilateY;
    const int pw = mIm2ColParamter.padX;
    const int ph = mIm2ColParamter.padY;
    const int icDiv4 = mIm2ColParamter.icDiv4;
    const int iw = mIm2ColParamter.iw;
    const int ih = mIm2ColParamter.ih;
    const int ic = input->channel();
    const int icp = UP_DIV(ic, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;

    //MNN_PRINT("%d-%d-%d-%d-%d, %d-%d\n", cpuIm2Col->icDiv4, cpuIm2Col->ih, cpuIm2Col->iw, cpuIm2Col->oh, cpuIm2Col->ow, eAlign, lAlign);
    // Im2col in Block
    for(int block_idx = 0; block_idx < mBlockNum; block_idx++) {
        if (mNeedIm2Col) {
            DivModFast lpD(mGemmInfo.elhPad[1]/INT8_PACK_NUMBER);
            DivModFast icpD(icp/INT8_PACK_NUMBER);
            DivModFast fxD(mIm2ColParamter.kernelX);
            DivModFast owD(mIm2ColParamter.ow);
            DivModFast ohD(mIm2ColParamter.oh);
        
            size_t maxCount = mGemmInfo.elh[0] * mGemmInfo.elhPad[1] / INT8_PACK_NUMBER;
            size_t block_num = runtime->blocks_num(maxCount);
            size_t block_size = runtime->threads_num();

            // [(NHW)p, KhKw(Ic)p]
            Im2Col_packC_16<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, ic, iw, ih, 
                maxCount, block_idx, icDiv4, mGemmInfo.elh[0], mGemmInfo.elh[1], (const int32_t*)input_addr, (int32_t *)mIm2ColBuffer, \
                lpD, owD, ohD, icpD, fxD);
            checkKernelErrors;
        }
    }

    if(mGpuComputeCap >= 80) {
        cutlass::Status status = mGemmInt8ClampNormalSm80();        
        cutlass_check(status);
	    return NO_ERROR;
    }
    if(mGemmShapeSizeLevel == GEMM_SIZE_NORMAL) {
        cutlass::Status status = mGemmInt8ClampNormal();
        cutlass_check(status);
    } else if(mGemmShapeSizeLevel == GEMM_SIZE_LITTLE) {
        cutlass::Status status = mGemmInt8ClampLittle();
        cutlass_check(status);
    } else {
        cutlass::Status status = mGemmInt8ClampLarge();
        cutlass_check(status);
    }
    return NO_ERROR;
}


}// namespace CUDA
}// namespace MNN
#endif