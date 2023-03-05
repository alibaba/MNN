//
//  ConvInt8CutlassExecution.cpp
//  MNN
//
//  Created by MNN on 2023/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvInt8CutlassExecution.hpp"
#include "core/TensorUtils.hpp"

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
    const int iBlock,
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
            *(AP + indexO) = (T1)0;
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
        *(AP + indexO) = (T1)0;
    }
}

template<typename T>
__global__ void WeightPackFill(const int8_t* param,
    T* output,
    const size_t maxCount,
    const int l,
    const int h,
    const int hp,
    DivModFast d_lp,
    const bool ocMajor
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int lpIndex, hpIndex;
        d_lp.divmod(index, hpIndex, lpIndex);

        int out_idx = index;
        if(ocMajor) {
            out_idx = lpIndex * hp + hpIndex;
        }

        if(lpIndex >= l || hpIndex >= h) {
            output[out_idx] = (T)0;
            continue;
        }

        output[out_idx] = param[hpIndex * l + lpIndex];
    }
}

void ConvInt8CutlassExecution::Resource::updateInputOutputScale(std::vector<float> inputQuantInfo, std::vector<float> outputQuantInfo) {
    std::call_once(flag, [&](){
        // new scales and zero points
        float inputScale = inputQuantInfo[0];
        float outputScale = outputQuantInfo[0];
        float inputZeroPoint = inputQuantInfo[1];
        float outputZeroPoint = outputQuantInfo[1];

        if (inputScale == 0.f || outputScale == 0.f) {
            return;
        }
        if (mInputScale == inputScale && mOutputScale == outputScale) {
            return;
        }
        auto scalePtr = mScaleFloatVec;
        auto biasPtr = mBiasInt32Vec;
        int size = mOutputChannelPack;
        float is = mInputScale / inputScale;
        float os = mOutputScale / outputScale;

        const int kernelNum = mInt8WeightKernelSum.size();

        // compute remains used in asymmetric quant
        std::vector<int> remainsCorrection;
        for (int i = 0; i < kernelNum; i++) {
            int temp = (int(inputZeroPoint) - mInputZeroPoint) * mInt8WeightKernelSum[i];
            remainsCorrection.emplace_back(temp);
        }

        for (int i = kernelNum; i < size; i++) {
            remainsCorrection.emplace_back(0);
        }

        for (int i = 0; i < size; i++) {
            // compute outputZeroPointFused in asymmetric quant
            int correction1 = static_cast<int32_t>(mOutputZeroPoint / scalePtr[i]);
            scalePtr[i] = scalePtr[i] * os / is;
            int correction2 = static_cast<int32_t>(outputZeroPoint / scalePtr[i]);
            int outputZeroPointFusedCorrection = correction2 - correction1;

            biasPtr[i] = biasPtr[i] - remainsCorrection[i] + outputZeroPointFusedCorrection;
            biasPtr[i] = static_cast<int32_t>(biasPtr[i] * is);
        }
        mInputScale = inputScale;
        mOutputScale = outputScale;
        mInputZeroPoint = int8_t(inputZeroPoint);
        mOutputZeroPoint = int8_t(outputZeroPoint);
        mClampMin = int8_t(outputQuantInfo[2]);
        mClampMax = int8_t(outputQuantInfo[3]);
    });
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

    // printf("resource init %p-%p\n", mScaleFloatPtr, mBiasInt32Ptr);

    //weight host->device
    const int8_t* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;

    bool res = ConvolutionCommon::getConvInt8Parameters(conv, quanCommon, filterDataPtr, weightSize, 
                                    mScaleFloatVec, 
                                    mBiasInt32Vec);
                                    // inputScale, 
                                    // outputScale, 
                                    // conv->symmetricQuan()->zeroPoint(),
                                    // conv->symmetricQuan()->outputZeroPoint());
    if(!res) {
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

    mInputZeroPoint = conv->symmetricQuan()->zeroPoint();
    mOutputZeroPoint = conv->symmetricQuan()->outputZeroPoint();
    mClampMin = conv->symmetricQuan()->clampMin();
    mClampMax = conv->symmetricQuan()->clampMax();

    auto oc = common->outputCount();

    int l = weightSize / oc;
    int h = oc;
    int lp = UP_DIV(l, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;
    int hp = UP_DIV(h, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;

    // Reorder weight
    {
        auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(int8_t));
        int8_t* cacheWeight = (int8_t*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
        runtime->memcpy(cacheWeight, filterDataPtr, weightSize * sizeof(int8_t), MNNMemcpyHostToDevice);

        mWeightInt8Tensor.reset(Tensor::createDevice<int8_t>({lp * hp}));
        bn->onAcquireBuffer(mWeightInt8Tensor.get(), Backend::STATIC);
        mWeightInt8Ptr = (void *)mWeightInt8Tensor.get()->buffer().device;

        DivModFast lpD(lp);
        int block_num = runtime->blocks_num(lp*hp);
        int block_size = runtime->threads_num();

        bool ocMajor = false;
        if(op->type() == OpType_DepthwiseConvInt8) {
            ocMajor = true;
        }
        WeightPackFill<<<block_num, block_size>>>((int8_t*)cacheWeight, (int8_t*)mWeightInt8Ptr, lp*hp, l, h, hp, lpD, ocMajor);
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
    mGemmInfo.elhPad[1] = UP_DIV(l, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;
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
        if(mGemmInfo.elh[0] < 128 || mGemmInfo.elhPad[2] < 128 || mGemmInfo.elhPad[1] < 64) { // M-N-K
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

    //printf("%d-%d-%d-%d-%d, %d-%d\n", cpuIm2Col->icDiv4, cpuIm2Col->ih, cpuIm2Col->iw, cpuIm2Col->oh, cpuIm2Col->ow, eAlign, lAlign);
    // Im2col in Block
    for(int block_idx = 0; block_idx < mBlockNum; block_idx++) {
        if (mNeedIm2Col) {
            DivModFast lpD(mGemmInfo.elhPad[1]);
            DivModFast fxyD((mIm2ColParamter.kernelX * mIm2ColParamter.kernelY));
            DivModFast fxD(mIm2ColParamter.kernelX);
            DivModFast owD(mIm2ColParamter.ow);
            DivModFast ohD(mIm2ColParamter.oh);
        
            size_t maxCount = mGemmInfo.elh[0] * mGemmInfo.elhPad[1];
            size_t block_num = runtime->blocks_num(maxCount);
            size_t block_size = runtime->threads_num();

            Im2Col_packC<<<block_num, block_size>>>(sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, 
                maxCount, block_idx, INT8_PACK_NUMBER, mGemmInfo.elh[0], mGemmInfo.elh[1], (const int8_t*)input_addr, (int8_t *)mIm2ColBuffer, \
                lpD, owD, ohD, fxyD, fxD);
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