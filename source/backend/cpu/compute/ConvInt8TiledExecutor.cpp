//
//  ConvInt8TiledExecutor.cpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvInt8TiledExecutor.hpp"
#include "ConvolutionTiledExecutor.hpp"
#include "core/Macro.h"
#include "core/BufferAllocator.hpp"

#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"

namespace MNN {

ConvInt8TiledExecutor::ConvInt8TiledExecutor(Backend* backend, const Convolution2D* convOp, std::shared_ptr<ResourceInt8> res): CPUConvolution(convOp->common(), backend), mResourceInt8(res), mMutableResource(res, backend) {
    mValid = mMutableResource.mValid;
}

ConvInt8TiledExecutor::~ConvInt8TiledExecutor() {
    // Do nothing
}

bool ConvInt8TiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    return false;
}

ErrorCode ConvInt8TiledExecutor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mMutableResource.updateInputOutputScale(TensorUtils::getQuantInfo(inputs[0]), TensorUtils::getQuantInfo(outputs[0]));
    CPUConvolution::onResize(inputs, outputs);
    ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY, static_cast<CPUBackend*>(backend())->functions(), static_cast<CPUBackend*>(backend())->int8Functions());
    return NO_ERROR;
}

void ConvInt8TiledExecutor::reorderWeight(Tensor* weight, const uint8_t* weightSrc, int SRC_UNIT, int UNIT, int ic, int oc, int kernelCount, int pack) {
    auto weightDst = weight->host<uint8_t>();
    memset(weightDst, 0, weight->size());
    if (SRC_UNIT > pack) {
        auto icDivU = UP_DIV(ic, pack);
        for (int k = 0; k < kernelCount; ++k) {
            const auto srcK = weightSrc + k;
            for (int y = 0; y < ic; ++y) {
                const int yOutSide    = y / pack;
                const int yInSide     = y % pack;
                const int yIndex      = yOutSide + k * icDivU;
                const int ySubOutSide = yIndex / (SRC_UNIT / pack);
                const int ySubInSide  = yIndex % (SRC_UNIT / pack);

                auto dstY       = weightDst + ySubOutSide * weight->stride(1) + ySubInSide * pack + yInSide;
                const auto srcY = srcK + y * kernelCount;
                for (int x = 0; x < oc; ++x) {
                    const int xOutSide = x / UNIT;
                    const int xInSide  = x % UNIT;
                    const int dstIndex = xOutSide * weight->stride(0) + xInSide * SRC_UNIT;
                    const int srcIndex = x * kernelCount * ic;
                    dstY[dstIndex]     = srcY[srcIndex];
                }
            }
        }
    } else {
        for (int k = 0; k < kernelCount; ++k) {
            auto icDivU = UP_DIV(ic, SRC_UNIT);
            const auto srcK = weightSrc + k;
            for (int y = 0; y < ic; ++y) {
                const int yOutSide    = y / SRC_UNIT;
                const int yInSide     = y % SRC_UNIT;

                auto dstY       = weightDst + (yOutSide + k * icDivU)  * weight->stride(1) + yInSide;
                const auto srcY = srcK + y * kernelCount;
                for (int x = 0; x < oc; ++x) {
                    const int xOutSide = x / UNIT;
                    const int xInSide  = x % UNIT;
                    const int dstIndex = xOutSide * weight->stride(0) + xInSide * SRC_UNIT;
                    const int srcIndex = x * kernelCount * ic;
                    dstY[dstIndex]     = srcY[srcIndex];
                }
            }
        }
    }
}

static bool _reorderWeightInside(Backend* bn, const Convolution2DCommon* common,
                                 const std::shared_ptr<Tensor>& weightOrigin,
                                 std::shared_ptr<Tensor>& weight) {
    auto core = static_cast<CPUBackend*>(bn)->int8Functions();
    auto gcore = static_cast<CPUBackend*>(bn)->functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    // reorder weight, [oc, ic, k^2] => [oc/unit, ((ic/unit)*k^2)/(src_unit/unit), unit(oc), (src_unit/unit), unit(ic)]
    int oc = common->outputCount(), ic = common->inputCount(), kernelCount = common->kernelX() * common->kernelY();
    std::vector<int> shape;
    int pack = gcore->pack;
    if (gcore->bytes == 2 && gcore->pack == 8) {
        pack = 4;
    }
    if (SRC_UNIT > pack) {
        MNN_ASSERT(SRC_UNIT % UNIT == 0);
        shape = {UP_DIV(oc, UNIT), UP_DIV(UP_DIV(ic, pack) * kernelCount, SRC_UNIT / pack), UNIT, SRC_UNIT};
    } else {
        shape = {UP_DIV(oc, UNIT), UP_DIV(ic, SRC_UNIT) * kernelCount, UNIT, SRC_UNIT};
    }

    weight.reset(Tensor::createDevice<int8_t>(shape));

    bool succ = bn->onAcquireBuffer(weight.get(), Backend::STATIC);
    if (!succ) {
        MNN_ERROR("Memory not enough");
        return false;
    }
    ConvInt8TiledExecutor::reorderWeight(weight.get(), weightOrigin->host<uint8_t>(), SRC_UNIT, UNIT, ic, oc, kernelCount, pack);
    return true;
}

static void Getfp32Info (std::shared_ptr<CPUConvolution::Resource> resource, std::shared_ptr<Tensor> weightOrigin, const Convolution2D* conv2d, std::shared_ptr<ConvolutionCommon::Int8Common> quantCommon) {
    // common parameters
    int outputCount = conv2d->common()->outputCount();
    auto core = static_cast<CPUBackend*>(resource->backend)->functions();
    int LSize = conv2d->common()->inputCount() * conv2d->common()->kernelX() * conv2d->common()->kernelY();
    int ocUp4 = ROUND_UP(outputCount, core->pack);
    
    int dequantCnt = quantCommon->alpha.size();
    if (quantCommon->asymmetric) {
        dequantCnt /= 2;
    }
    int blockNum = dequantCnt / outputCount;
    int scaleSize = blockNum * ocUp4; // pack size.
    int blockSize = LSize / blockNum;
    int originOffset = 0;
    if (quantCommon->canUseInt4) {
        originOffset = -8;
    }

    // Save weight quant scale and bias: wf=scale*wi+bias
    int bytes = 4;
    resource->mDequantize.mScaleBias.reset(Tensor::createDevice<uint8_t>({2 * scaleSize * bytes}));
    auto success = resource->backend->onAcquireBuffer(resource->mDequantize.mScaleBias.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Alloc denquant scaleBias memory error\n");
        return;
    }
    auto alphaPtr = resource->mDequantize.mScaleBias->host<float>();
    auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(alphaPtr) + scaleSize * bytes);
    ::memset(alphaPtr, 1, scaleSize * bytes);
    ::memset(biasPtr, 0, scaleSize * bytes);
    auto quanInfoPtr = quantCommon->alpha.get();
    int h = quantCommon->alpha.size();
    if (quantCommon->asymmetric) {
        for (int i = 0; i < blockNum; ++i) {
            auto dstAlpha = alphaPtr + i * ocUp4;
            auto dstBias  = biasPtr + i * ocUp4;
            for (int j = 0; j < outputCount; ++j) {
                int scaleIndex = j * blockNum + i;
                dstAlpha[j] = quanInfoPtr[2 * scaleIndex + 1];
                dstBias[j] = quanInfoPtr[2 * scaleIndex] + (float)originOffset * dstAlpha[j];
            }
        }

    } else {
        for (int i = 0; i < blockNum; ++i) {
            auto dstAlpha = alphaPtr + i * ocUp4;
            auto dstBias  = biasPtr + i * ocUp4;
            for (int j = 0; j < outputCount; ++j) {
                int scaleIndex = j * blockNum + i;
                dstAlpha[j] = quanInfoPtr[scaleIndex];
                dstBias[j] = (float)originOffset * dstAlpha[j];
            }
        }
    }
    // Save float weight kernel sum
    resource->mWeightKernelSum.reset(Tensor::createDevice<uint8_t>({bytes * ocUp4}));
    success = resource->backend->onAcquireBuffer(resource->mWeightKernelSum.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Alloc denquant mWeightKernelSum memory error\n");
        return;
    }
    auto weightKernelSum = resource->mWeightKernelSum->host<float>();
    auto realWeightData = weightOrigin->host<int8_t>();
    ::memset(weightKernelSum, 0, resource->mWeightKernelSum->size());
    for (int j = 0; j < outputCount; ++j) {
        float sum = 0.f;
        for (int k = 0; k < blockNum; ++k) {
            int scaleIndex = k + j * blockNum;
            float scale = 0;
            float bias  = 0;
            if (quantCommon->asymmetric) {
                scale = quanInfoPtr[2 * scaleIndex + 1];
                bias  = quanInfoPtr[2 * scaleIndex];
            } else {
                scale = quanInfoPtr[scaleIndex];
                bias = 0;
            }
            int tmp = 0;
            for (int i = 0; i < blockSize; ++i) {
                int l_index = k * blockSize + i;
                tmp += (int)realWeightData[j * blockNum * blockSize + l_index];
            }
            sum += (tmp * scale + blockSize * bias);
        }
        weightKernelSum[j] = sum;
    }
}

DenseConvInt8TiledExecutor::DenseConvInt8TiledExecutor(Backend* backend, const Convolution2D* convOp, std::shared_ptr<ResourceInt8> res, bool dynamicQuantExe) : ConvInt8TiledExecutor(backend, convOp, res) {
    std::shared_ptr<Tensor> weightOrigin = mResourceInt8->mWeightInt8;
    std::shared_ptr<MNN::ConvolutionCommon::Int8Common> quanCommon ;
    mDynamicQuantExe = dynamicQuantExe;
    if (dynamicQuantExe) {
        MNN_ASSERT(convOp->quanParameter() != nullptr && convOp->quanParameter()->buffer() != nullptr);
        quanCommon = ConvolutionCommon::load(convOp, backend, false, true);
        // fp32 weightKernelSum
        mResource.reset(new CPUConvolution::Resource);
        mResource->backend = backend;
        Getfp32Info(mResource, weightOrigin, convOp, quanCommon); // Call this before reorder weight.
    }
    
    mValid = _reorderWeightInside(backend, convOp->common(), weightOrigin, mResourceInt8->mWeightInt8);
    if(!mValid) {
        return;
    }
    auto core = static_cast<CPUBackend*>(backend)->int8Functions();
    auto gcore = static_cast<CPUBackend*>(backend)->functions();
    // offline quant
    if (false == dynamicQuantExe) {
        mGemmKernel = core->Int8GemmKernel;
#ifdef MNN_USE_SSE
        int actBits = convOp->symmetricQuan()->nbits();
        if (actBits <= 7) {
            mGemmKernel = core->Int8GemmKernelFast;
        }
#else
        if(convOp->symmetricQuan()->method() == QuantizeAlgo_OVERFLOW_AWARE){
            mGemmKernel = core->Int8GemmKernelFast;
        }
#endif
        mResource.reset(new CPUConvolution::Resource);
        CPUConvolution::makeResource(backend, mResource, convOp, mResourceInt8);
        return;
    }

    // dynamic quant
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    bool needPermuteInt4weight = ((UNIT == 8 && SRC_UNIT == 8 && DST_XUNIT ==10) || (UNIT == 4 && SRC_UNIT == 8 && DST_XUNIT ==20) || (UNIT == 64 && SRC_UNIT == 4 && DST_XUNIT ==4));
    mResource->mDequantize.bits = 8;
    if (quanCommon->canUseInt4) {
        mResourceInt8->mWeightAsymmetricQuant = true;
        auto weightLength = mResourceInt8->mWeightInt8->size();
        MNN_ASSERT(weightLength % 2 == 0);
        mResource->mDequantize.bits = 4;
        std::shared_ptr<MNN::Tensor> weightLow(Tensor::createDevice<uint8_t>( mResourceInt8->mWeightInt8->shape()));
        auto res = mResource->backend->onAcquireBuffer(weightLow.get(), Backend::STATIC);
        if (!res) {
            MNN_ERROR("int4 weight acquire buffer error\n");
            return ;
        }
        auto srcPtr = mResourceInt8->mWeightInt8->host<int8_t>();
        auto dstPtr = weightLow->host<uint8_t>();
        // Pack two int4-weight to one int8-weight.
        if (false == needPermuteInt4weight) {
            weightLength = UP_DIV(weightLength, 2);
            for (int i=0; i < weightLength; ++i) {
                int s0 = srcPtr[2 * i + 0];
                int s1 = srcPtr[2 * i + 1];
                int d = (s0 + 8) * 16 + (s1 + 8);
                dstPtr[i] = d;
            }
        } else {
            int permuteUnit = UNIT * SRC_UNIT;
            int halfPermuteStride = static_cast<int32_t>(permuteUnit / 2);
            for (int i = 0; i < weightLength / permuteUnit; ++i) {
                auto src0 = srcPtr + i * permuteUnit;
                auto dst0 = dstPtr + i * halfPermuteStride;
                for (int j = 0; j < halfPermuteStride; ++j) {
                    int s0 = src0[j];
                    int s1 = src0[j + halfPermuteStride];
                    int d = (s0 + 8) * 16 + (s1 + 8);
                    dst0[j] = d;
                }
            }
        }
        // Update int4 weight to mWeightInt8.
        mResourceInt8->mWeightInt8 = weightLow;
    }
    // Relu/Relu6 post parameters
    auto postPtr = getPostParameters();
    mResource->mReluThreshold.resize(2);
    mResource->mReluThreshold[0] = postPtr[2];
    mResource->mReluThreshold[1] = postPtr[3];
    if (gcore->bytes == 2) {
        gcore->MNNFp32ToLowp(mResource->mReluThreshold.data(), reinterpret_cast<int16_t*>(mResource->mReluThreshold.data()), 2);
    }
    if (mCommon->relu()) {
        mResource->mReluThreshold[0] = 0.f;
    }
    if (mCommon->relu6()) {
        mResource->mReluThreshold[0] = 0.f;
        mResource->mReluThreshold[1] = 6.f;
    }
}

DenseConvInt8TiledExecutor::DenseConvInt8TiledExecutor(Backend* backend, const Convolution2D* convOp, bool dynamicQuantExe, const DenseConvInt8TiledExecutor& exe)
    : ConvInt8TiledExecutor(backend, convOp, exe.mResourceInt8), mGemmKernel(exe.mGemmKernel), mResource(exe.mResource), mDynamicQuantExe(dynamicQuantExe) {
}

DenseConvInt8TiledExecutor::~DenseConvInt8TiledExecutor() {
    // Do nothing
}

bool DenseConvInt8TiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new DenseConvInt8TiledExecutor(bn, op->main_as_Convolution2D(), mDynamicQuantExe, *this);
    if (!exe->valid()) {
        return false;
    }
    *dst = exe;
    return true;
}

void DenseConvInt8TiledExecutor::getPackParameter(int* Unit, int* srcUnit, int* DestUnit, const CoreInt8Functions* core) {
    core->MNNGetGemmUnit(Unit, srcUnit, DestUnit);
}


ErrorCode DenseConvInt8TiledExecutor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mUseBatchQuan = (static_cast<CPUBackend*>(backend())->getRuntime()->hint().dynamicQuantOption == 1);
    mUseBatchQuan &= mCommon->kernelY() == 1 && mCommon->kernelX() == 1
        && outputs[0]->width() == inputs[0]->width() && outputs[0]->height() == inputs[0]->height()
        && mCommon->strideX() == 1 && mCommon->strideY() == 1 && mCommon->padX() == 0 && mCommon->padY() == 0
        && outputs[0]->height() == 1 && outputs[0]->width() == 1;
    mUseBatchQuan &= mDynamicQuantExe;
    mUseBatchQuan &= (inputs[0]->batch() > 1);
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    auto gcore =static_cast<CPUBackend*>(backend())->functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);

    if (mDynamicQuantExe == false) {
        mMutableResource.updateInputOutputScale(TensorUtils::getQuantInfo(inputs[0]), TensorUtils::getQuantInfo(outputs[0]));
        CPUConvolution::onResize(inputs, outputs);
        ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY, gcore, core);
        mBlockNum = 1;
    } else { // Dynamic Quant kernels
        CPUConvolution::onResize(inputs, outputs);
        // Gemm Kernel
        mGemmKernel = core->Int8GemmKernel;
        if (mResource->mDequantize.bits == 4) {
            mGemmKernel = core->Int8GemmKernel_W4;
        }
        mQuantFunc = core->MNNFloat2Int8;
        if (gcore->bytes == 2 && gcore->pack == 8) {
            mGemmKernel = core->MNNGemmInt8AddBiasScale_Unit_FP16;
            if (mResource->mDequantize.bits == 4) {
                mGemmKernel = core->MNNGemmInt8AddBiasScale_w4_Unit_FP16;
            }
            mQuantFunc = core->DynamicQuanInput_ARM82;
            mQuantAndReorderFunc = core->DynamicQuanInputAndReorder_ARM82;
            
        }
        // A axisSum kernel
        mSumByAxisLFunc = gcore->MNNSumByAxisLForMatmul_A;
        if (gcore->bytes == 2 && gcore->pack == 8) { // use fp16
            ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY, gcore, core, 4);
        } else {
            ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY, gcore, core);
        }
        int ocUp4 = ROUND_UP(outputs[0]->channel(), gcore->pack);
        int alphaSize = mResource->mDequantize.mScaleBias->size() / (4 * 2);
        mBlockNum  = alphaSize / ocUp4;
    }
    
    // input scale buffer
    int batch = inputs[0]->batch();
//    mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({mThreadNums, DST_XUNIT * mIm2ColCount * mResourceInt8->mWeightInt8->length(1) * SRC_UNIT}));
    mInputDeqScales.reset(Tensor::createDevice<int8_t>({batch * 4}));
    bool success = backend()->onAcquireBuffer(mInputDeqScales.get(), Backend::DYNAMIC);

    // Im2col info
    auto output = outputs[0];
    const int threads = static_cast<CPUBackend*>(backend())->threadNumber();
    auto planeSize = output->width() * output->height() * output->batch();
    const int L2Size = 2048;
    const int tileLimitByC = UP_DIV(L2Size, mIm2ColParamter.kernelCountUnit * SRC_UNIT);
    int tileLimit = 0;
    int outC    = output->channel();
    int outC4 = UP_DIV(outC, gcore->pack);

    if (threads < planeSize) { // Thread split by output nhw.
        tileLimit = ALIMIN(tileLimitByC, UP_DIV(planeSize, threads));
        mSplitByOc = false;
    } else {
        tileLimit = ALIMIN(tileLimitByC, planeSize);
        auto ocPerThread = UP_DIV(outC4, threads);
        auto threadNeed = UP_DIV(outC4, ocPerThread);
        if (UNIT > gcore->pack) { // AVX512:UNIT=64,pack=16
            MNN_ASSERT(UNIT % gcore->pack == 0);
            int ocDivUnit = UP_DIV(outC4 * gcore->pack, UNIT);
            ocPerThread = UP_DIV(ocDivUnit, threads);
            threadNeed  = UP_DIV(ocDivUnit, ocPerThread);
        }
        mThreadNums = ALIMIN(threads, threadNeed);
        mSplitByOc = true;

        mDivides.resize(threads+1);
        mDivides[0] = 0;
        static_cast<const CPURuntime*>(backend()->getRuntime())->computeDivideSizes(outC4, mDivides.data() + 1);
    }
    mIm2ColCount = UP_DIV(tileLimit, DST_XUNIT);
    auto DynamicDestUnit = DST_XUNIT * mIm2ColCount;
    mTileCount        = UP_DIV(planeSize, DynamicDestUnit);
    
    if (threads < planeSize) {
        mThreadNums = ALIMIN(threads, mTileCount);
        mDivides.resize(threads+1);
        mDivides[0] = 0;
        static_cast<const CPURuntime*>(backend()->getRuntime())->computeDivideSizes(mTileCount, mDivides.data() + 1);
    }
    int ocUp4 = ROUND_UP(outC, gcore->pack);
    int alphaSize = mResource->mDequantize.mScaleBias->size() / (4 * 2);

    auto bufferAlloc = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    auto blitInfoSize = ConvolutionTiledExecutor::computeBlitInfoSize(DST_XUNIT * mIm2ColCount, mIm2ColParamter.ow, mIm2ColParamter.kernelX * mIm2ColParamter.kernelY, mThreadNums);
    mBlitInfoStride = blitInfoSize.second;
    mBlitInfo = bufferAlloc->alloc(blitInfoSize.first);
    mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({mThreadNums, DST_XUNIT * mIm2ColCount * mResourceInt8->mWeightInt8->length(1) * SRC_UNIT}));
    mTempSrcSum.resize(mThreadNums * mBlockNum * DST_XUNIT * mIm2ColCount * 4); // Use 4 bytes to save kernel sum.

    success &= backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (!success || mBlitInfo.invalid()) {
        return OUT_OF_MEMORY;
    }
    if (false == mDynamicQuantExe) {
        bufferAlloc->free(mBlitInfo);
        backend()->onReleaseBuffer(mInputDeqScales.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
        return NO_ERROR;
    }

    
    int inC   = inputs[0]->channel();
    // set im2col tensor info
    mQuantInput.reset((Tensor::createDevice<int8_t>({batch, mIm2ColParamter.ih, mIm2ColParamter.iw, ROUND_UP(inC, gcore->pack)})));
    // set dynamic quant buffer
    mTempMaxMinValueBuffer.reset(Tensor::createDevice<uint8_t>({mThreadNums, 2 * gcore->bytes}));
    // set compute buffer
    mDynamicBias.reset(Tensor::createDevice<uint8_t>({ocUp4 * 4}));
    mScaleFuse.reset(Tensor::createDevice<uint8_t>({alphaSize * 4}));
    
    success &= backend()->onAcquireBuffer(mQuantInput.get(), Backend::DYNAMIC);
    success &= backend()->onAcquireBuffer(mDynamicBias.get(), Backend::DYNAMIC);
    success &= backend()->onAcquireBuffer(mTempMaxMinValueBuffer.get(), Backend::DYNAMIC);
    success &= backend()->onAcquireBuffer(mScaleFuse.get(), Backend::DYNAMIC);
    
    if (mUseBatchQuan) {
        int infobytes = 4; // use float32 to save dequant scale and quant scale.
        int size = mThreadNums * batch * gcore->bytes + 2 * batch * infobytes;
        mBatchQuantInfo.reset(Tensor::createDevice<int8_t>({size}));
        success &= backend()->onAcquireBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
    }
    if (!success) {
        return OUT_OF_MEMORY;
    }
    bufferAlloc->free(mBlitInfo);
    backend()->onReleaseBuffer(mInputDeqScales.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mQuantInput.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mDynamicBias.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempMaxMinValueBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mScaleFuse.get(), Backend::DYNAMIC);
    if (mUseBatchQuan) {
        backend()->onReleaseBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

ErrorCode DenseConvInt8TiledExecutor::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Timer kernelTimer;
    const auto input = inputs[0];
    auto output      = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    auto gcore = static_cast<CPUBackend*>(backend())->functions();

    int UNIT__, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT__, &SRC_UNIT, &DST_XUNIT);
    auto blitProc = core->MNNPackC4Int8ForMatMul_A;
    if ( mDynamicQuantExe && gcore->bytes == 2 && core->MNNPackC4Int8ForMatMul_A_ARM86FP16) {
        blitProc = core->MNNPackC4Int8ForMatMul_A_ARM86FP16;
    }
    const int plane                  = output->batch() * mIm2ColParamter.oh * mIm2ColParamter.ow;
    const int batch                  = input->batch();
    const int PackUnit               = gcore->pack;
    const int dstZStep               = plane * PackUnit;
    const int ocDiv4                 = UP_DIV(output->channel(), PackUnit);
    const int ocUp4                  = ROUND_UP(output->channel(), PackUnit);
    const auto kernelCountUnitDouble = mIm2ColParamter.kernelCountUnit;
    const auto col_buffer_unit_size  = kernelCountUnitDouble * DST_XUNIT * SRC_UNIT * sizeof(int8_t);
    const auto col_buffer_size       = col_buffer_unit_size * mIm2ColCount;
    const int dstBytes               = static_cast<CPUBackend*>(backend())->getBytes(backend(), output);
    const int alphaSize              = mResource->mDequantize.mScaleBias->size() / (4 * 2);
    const int blockL                  = kernelCountUnitDouble / mBlockNum; // source depthQuad for each block.
    float weightBytes                = 1.f;
    int weight_step_Y                = weightBytes * (UNIT__ * SRC_UNIT);
    int src_step_Y                   = DST_XUNIT * SRC_UNIT;

    auto inputDataPtr        = input->host<int8_t>();
    auto im2colPtr           = mTempIm2ColBuffer->host<int8_t>();
    const auto weightDataPtr = mResourceInt8->mWeightInt8->host<int8_t>();
    auto srcKernelSumPtr     = mTempSrcSum.data();
    auto weightDequantBias = mResource->mDequantize.mScaleBias->host<uint8_t>() + alphaSize * 4;

    auto outputDataPtr = output->host<int8_t>();
    auto biasPtr       = mMutableResource.mBiasFloat->host<uint8_t>();
    auto scalePtr      = mMutableResource.mScaleFloat->host<uint8_t>();

    auto inputZeroPoint  = mMutableResource.mInputZeroPoint;
    auto inputScalePtr = mInputDeqScales->host<uint8_t>();
    (reinterpret_cast<float*>(inputScalePtr))[0]     =  mMutableResource.mInputScale;

    auto SingleDynamicQuant = [&] () {
        const auto floatptr = input->host<float>();
        auto int8ptr        = mQuantInput->host<int8_t>();
        auto inputsize      = static_cast<CPUBackend*>(backend())->getTensorSize(inputs[0]);
        float quantscale    = 0.f;
        float dequantscale  = 0.f;
        int zeropoint       = 0;

         /* Count max and min value to compute input scale and zeropoint */
        auto maxMinValPtr = mTempMaxMinValueBuffer->host<uint8_t>();
        int threadNeed = mThreadNums;
        auto inputSizeCount = UP_DIV(inputsize, mThreadNums);
        if (inputSizeCount < 9) {
            threadNeed = 1;
            inputSizeCount = inputsize;
        } else {
            threadNeed = ALIMIN(UP_DIV(inputsize, inputSizeCount), mThreadNums);
            inputSizeCount = UP_DIV(inputsize, threadNeed);
        }
        auto findMaxMinValueFunction = [&](int tId) {
            auto perThreadWorkCount = ALIMIN(inputSizeCount, inputsize - tId * inputSizeCount);
            auto minValPtrTid = reinterpret_cast<float*>(maxMinValPtr + tId * mTempMaxMinValueBuffer->stride(0));
            auto maxValPtrTid = reinterpret_cast<float*>(maxMinValPtr + tId * mTempMaxMinValueBuffer->stride(0) + gcore->bytes);
            auto inputDataPtrTid = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(floatptr) + tId * inputSizeCount * gcore->bytes);
            gcore->MNNCountMaxMinValue(inputDataPtrTid, minValPtrTid, maxValPtrTid, perThreadWorkCount);
        };
        MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
            findMaxMinValueFunction((int)tId);
        }
        MNN_CONCURRENCY_END();
        if (threadNeed > 1) {
            gcore->MNNCountMaxMinValue(reinterpret_cast<float*>(maxMinValPtr),reinterpret_cast<float*>(maxMinValPtr), reinterpret_cast<float*>(maxMinValPtr + gcore->bytes), 2 * mThreadNums);
        }
        float maxVal = 0;
        float minVal = 0;
        if (gcore->bytes == 4) {
            maxVal = (reinterpret_cast<float*>(maxMinValPtr))[1];
            minVal = (reinterpret_cast<float*>(maxMinValPtr))[0];
        }
        if (gcore->bytes == 2) {
            std::vector<float> _mVal(2);
            gcore->MNNLowpToFp32(reinterpret_cast<int16_t*>(maxMinValPtr), _mVal.data(), 2);
            maxVal = _mVal[1];
            minVal = _mVal[0];
        }

        /* Dynamic quant */
        float range = maxVal - minVal;
        quantscale = 255.0f / range;
        dequantscale = range / 255.0f;
        zeropoint = static_cast<int32_t>(roundf(-minVal * 255.f / range) - 128.0f);
        std::vector<float>qsVec(PackUnit, quantscale);
        auto sizeDiv = UP_DIV(inputsize, PackUnit);
        int inputPlane = input->batch() * mIm2ColParamter.iw * mIm2ColParamter.ih;
        if (gcore->bytes == 2 && gcore->pack == 8 && inputPlane > 1) { // C8->C4
            mQuantAndReorderFunc(floatptr, int8ptr, inputPlane, qsVec.data(), -128, 127, (ssize_t)zeropoint, UP_DIV(input->channel(), PackUnit), 4 * inputPlane);
        } else {
            mQuantFunc(floatptr, int8ptr, sizeDiv, qsVec.data(), -128, 127, (ssize_t)zeropoint);
        }

        /* bias float */
        #ifdef MNN_USE_SSE
        int offset = 128;
    #else
        int offset = 0;
    #endif
        auto biasfp32 = mMutableResource.mResource->mOriginBias->host<float>();
        auto weightDequantScale = mResource->mDequantize.mScaleBias->host<float>();
        float zerofp32 = (zeropoint + offset) * dequantscale;

        gcore->MNNDynamicUpdateConvBiasScale(mDynamicBias->host<float>(), mScaleFuse->host<float>(), biasfp32, weightDequantScale, &dequantscale, mResource->mWeightKernelSum->host<float>(), &zerofp32, UP_DIV(output->channel(), 4), alphaSize);
        // Move step for A and B for each block computing

        inputZeroPoint = zeropoint;
        (reinterpret_cast<float*>(inputScalePtr))[0]  = dequantscale;
        biasPtr   = mDynamicBias->host<uint8_t>();
        scalePtr  = mScaleFuse->host<uint8_t>();
        inputDataPtr = int8ptr;
    };

    auto BatchDynamicQuant = [&]() {
        // Allocate input max/sum/dequant/quant buffer
        auto infobytes = 4;
        auto dequantPtr = mBatchQuantInfo->host<uint8_t>();
        auto quantPtr = dequantPtr + batch * infobytes;
        auto maxPtr = mBatchQuantInfo->host<uint8_t>() + 2 * batch * infobytes;

        // compute sum and absmax
        int icDiv4 = UP_DIV(input->channel(), PackUnit);
        int threadwork = UP_DIV(icDiv4, mThreadNums);
        int threadNeed = UP_DIV(icDiv4, threadwork);
        int threadTmp = ALIMIN(mThreadNums, threadNeed);
        threadwork = UP_DIV(icDiv4, threadTmp);
        MNN_CONCURRENCY_BEGIN(tId, threadTmp) {
            int workCount = threadwork;
            if (tId == threadTmp - 1) {
                workCount = icDiv4 - tId * threadwork;
            }
            int icIndex = tId * threadwork;
            auto inputData = reinterpret_cast<const float*>(input->host<uint8_t>() + icIndex * batch * PackUnit * gcore->bytes);
            auto batchMax = reinterpret_cast<float*>(maxPtr + tId * batch * gcore->bytes);
            gcore->MNNAbsMax(inputData, batchMax, workCount, batch, PackUnit);
        }
        MNN_CONCURRENCY_END();

        // Compute quant scale
        gcore->MNNQuantScale((float*)maxPtr, (float*)quantPtr, (float*)dequantPtr, threadTmp, batch);

        // quant
        MNN_CONCURRENCY_BEGIN(tId, threadTmp) {
            int workCount = threadwork;
            if (tId == threadTmp - 1) {
                workCount = icDiv4 - tId * threadwork;
            }
            auto icIndex    = tId * threadwork;
            auto inputData = reinterpret_cast<float*>(input->host<uint8_t>() + icIndex * batch * PackUnit * gcore->bytes);
            auto int8ptr   = mQuantInput->host<int8_t>() + icIndex * batch * PackUnit;
            auto scale_ptr = reinterpret_cast<float*>(quantPtr);
            gcore->MNNDynamicQuant(inputData, int8ptr, scale_ptr, workCount, batch, PackUnit);
        }
        MNN_CONCURRENCY_END();
        
        inputZeroPoint = 0;
        inputScalePtr     = (uint8_t*)dequantPtr;
        inputDataPtr = mQuantInput->host<int8_t>();
        biasPtr = mMutableResource.mResource->mOriginBias->host<uint8_t>();
        scalePtr = mResource->mDequantize.mScaleBias->host<uint8_t>();
    };
    ssize_t oneScale = 1;
    if (mUseBatchQuan) {
        BatchDynamicQuant();
        oneScale = 0;
    } else if (mDynamicQuantExe) {
        SingleDynamicQuant();
    } else {
        // offline quant.
    }
    
    if (mResource->mDequantize.bits == 4) {
        weightBytes   = 0.5;
        weight_step_Y *= 0.5;
    }
    
    SumByAxisParams sumParams;
    sumParams.oneScale = oneScale;
    sumParams.SRC_UNIT = SRC_UNIT;
    sumParams.blockNum = mBlockNum;
    sumParams.DST_XUNIT = DST_XUNIT;
    sumParams.col_buffer_unit_size = col_buffer_unit_size;
    sumParams.kernelCountUnitDouble = kernelCountUnitDouble;
    
    auto ThreadFunction = [&](int tId, int eStartIndex, int eEndIndex, int estep, int ocIndex) {
        auto ocDivThread = ocDiv4;
        if (mSplitByOc) { // Thread split by OC
            ocDivThread = ALIMIN(mDivides[tId + 1] - mDivides[tId], ocDiv4 - mDivides[tId]);
        }
        float* reluPtr = mResource->mReluThreshold.data();
        uint8_t* extraScale = nullptr; // input scale for batch dynamic quant.
        QuanPostTreatParameters quanParam;
        quanParam.blockNum = mBlockNum;
        if (mUseBatchQuan) {
            extraScale = inputScalePtr;
        }
#ifdef MNN_USE_SSE
        quanParam.extraBias = mResource->mWeightKernelSum->host<float>() + ocIndex;
#endif
        if (dstBytes != 1) {
            quanParam.useInt8 = 0;
            quanParam.fp32minmax = reluPtr;
        } else {
            quanParam.maxValue = mMutableResource.mClampMax;
            if (mResourceInt8->mRelu) {
                quanParam.minValue = mMutableResource.mOutputZeroPoint;
            } else {
                quanParam.minValue = mMutableResource.mClampMin;
            }
        }
        auto outputTid = outputDataPtr + ocIndex * plane * dstBytes;
        const auto biasFloatTid = reinterpret_cast<float*>(biasPtr + ocIndex * 4);
        const auto scaleFloatTid = reinterpret_cast<float*>(scalePtr + ocIndex * 4);
        const auto weightDequanBiasTid  = reinterpret_cast<float*>(weightDequantBias + ocIndex * 4);
        const auto weightPtrTid = weightDataPtr + static_cast<int32_t>(ocIndex * kernelCountUnitDouble * SRC_UNIT * weightBytes);
        if (mBlockNum == 1) {
            quanParam.biasFloat = biasFloatTid;
            quanParam.scale = scaleFloatTid;
            quanParam.weightQuanBias = weightDequanBiasTid;
        }

        auto colAddr        = im2colPtr + tId * mTempIm2ColBuffer->stride(0);
        auto srcPtr     = (int8_t const **)(mBlitInfo.ptr() + tId * mBlitInfoStride.first);
        auto el         = (int32_t *)(srcPtr + mBlitInfoStride.second);
        auto xKernelSumPtrTid = reinterpret_cast<float*>(srcKernelSumPtr + tId * mBlockNum * DST_XUNIT * mIm2ColCount * 4);

        int32_t info[6];
        info[1] = mIm2ColParamter.iw * mIm2ColParamter.ih * batch;
        info[2] = static_cast<int32_t>(col_buffer_unit_size);
        info[3] = mIm2ColParamter.strideX;
        info[5] = kernelCountUnitDouble;
        for (int tIndex = eStartIndex; tIndex < eEndIndex; tIndex += estep) {
            const int xIndexStart  = tIndex * DST_XUNIT * mIm2ColCount;
            int realDstCount = ALIMIN(plane - xIndexStart, DST_XUNIT * mIm2ColCount);
            auto ptrExtraScale = extraScale != nullptr ? (extraScale + xIndexStart * 4) : nullptr;
            auto ptrInputscale = mUseBatchQuan == true ? (inputScalePtr + xIndexStart * 4) : inputScalePtr;
            // im2col
            auto res = ConvolutionTiledExecutor::turnIm2ColToBlitInfo((const float**)srcPtr, el, xIndexStart, realDstCount, mIm2ColParamter, (const uint8_t*)inputDataPtr, 1);
            int number = res.first;
            bool needZero = res.second;
            if (needZero) {
#ifdef MNN_USE_SSE
                ::memset(colAddr, inputZeroPoint + 128, col_buffer_size);
#else
                ::memset(colAddr, inputZeroPoint, col_buffer_size);
#endif
            }
            info[0] = number;
            info[4] = realDstCount;
            if (number > 0) {
                blitProc(colAddr, srcPtr, info, el);
            }
            if (mResourceInt8->mWeightAsymmetricQuant) {
                mSumByAxisLFunc(xKernelSumPtrTid, colAddr, (float*)ptrInputscale, realDstCount, sumParams);
            }
            auto outputInTilePtr = outputTid + xIndexStart * PackUnit * dstBytes;
            auto colAddrTemp = colAddr;
            auto ptrX = xKernelSumPtrTid;
            if (mBlockNum == 1) {
                do {
                    int step = ALIMIN(DST_XUNIT, realDstCount);
                    quanParam.srcKernelSum = ptrX;
                    quanParam.extraScale = extraScale != nullptr ? (float*)ptrExtraScale : nullptr;
                    mGemmKernel(outputInTilePtr, colAddrTemp, weightPtrTid, kernelCountUnitDouble, dstZStep * dstBytes, ocDivThread, &quanParam, step);
                    ptrX += step;
                    realDstCount-=step;
                    outputInTilePtr += DST_XUNIT * PackUnit * dstBytes;
                    colAddrTemp += col_buffer_unit_size;
                    ptrExtraScale = extraScale != nullptr ? (ptrExtraScale + step * 4) : nullptr;
                } while(realDstCount > 0);
            } else { // Now offline quant do not run into.
                do {
                    int step = ALIMIN(DST_XUNIT, realDstCount);
                    quanParam.extraScale = extraScale != nullptr ? (float*)ptrExtraScale : nullptr;
                    for (int k = 0; k < mBlockNum; ++k) {
                        quanParam.biasFloat = nullptr;
                        quanParam.fp32minmax = nullptr;
                        if (k == 0) {
                            quanParam.biasFloat = (float*)biasFloatTid;
                        }
                        if (k == mBlockNum - 1) {
                            quanParam.fp32minmax = reluPtr;
                        }
                        quanParam.srcKernelSum = ptrX + k * step;
                        quanParam.weightQuanBias = weightDequanBiasTid + k * ocUp4;
                        quanParam.scale = (float*)(scaleFloatTid + k * ocUp4);

                        mGemmKernel(outputInTilePtr, colAddrTemp + k * blockL * src_step_Y, weightPtrTid + k * blockL * weight_step_Y, blockL, dstZStep * dstBytes, ocDivThread, &quanParam, step);
                    }
                    ptrX += (step * mBlockNum);
                    realDstCount-=step;
                    outputInTilePtr += DST_XUNIT * PackUnit * dstBytes;
                    colAddrTemp += col_buffer_unit_size;
                    ptrExtraScale = extraScale != nullptr ? (ptrExtraScale + step * 4) : nullptr;
                } while(realDstCount > 0);
            }
        }
    };

    if (!mSplitByOc) {
        MNN_CONCURRENCY_BEGIN(tId, mThreadNums) {
            ThreadFunction((int)tId, mDivides[tId], mDivides[tId + 1], 1, 0);
        }
        MNN_CONCURRENCY_END();
    } else {
        MNN_CONCURRENCY_BEGIN(tId, mThreadNums) {
            int ocIndex = PackUnit * mDivides[tId];
            ThreadFunction((int)tId, 0, mTileCount,1, ocIndex);
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}





} // namespace MNN
