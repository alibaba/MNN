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

ConvInt8TiledExecutor::ConvInt8TiledExecutor(Backend* backend, const Op* op): CPUConvolution(op->main_as_Convolution2D()->common(), backend) {}

ConvInt8TiledExecutor::ConvInt8TiledExecutor(Backend* backend, const Op* op, std::shared_ptr<ResourceInt8> res): CPUConvolution(op->main_as_Convolution2D()->common(), backend), mResourceInt8(res) {
    if (!res->mDynamicQuant) {
        mMutableResource.reset(new MutableResourceInt8(res, backend));
        mValid = mMutableResource->mValid;
    }
}

ConvInt8TiledExecutor::~ConvInt8TiledExecutor() {
    // Do nothing
}

bool ConvInt8TiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    return false;
}

ErrorCode ConvInt8TiledExecutor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (nullptr != mMutableResource) {
        mMutableResource->updateInputOutputScale(TensorUtils::getQuantInfo(inputs[0]), TensorUtils::getQuantInfo(outputs[0]));
    }
    CPUConvolution::onResize(inputs, outputs);
    ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY, static_cast<CPUBackend*>(backend())->functions(), static_cast<CPUBackend*>(backend())->int8Functions());
    return NO_ERROR;
}

void ConvInt8TiledExecutor::reorderWeight(Tensor* weight, const uint8_t* weightSrc, int SRC_UNIT, int UNIT, int ic, int oc, int kernelCount, int pack, int blockNum) {
    auto weightDst = weight->host<uint8_t>();
    memset(weightDst, 0, weight->size());
    int kernelCountUnit = weight->length(1);
    int blockL = kernelCountUnit / blockNum;
    int strideOutside = ROUND_UP(oc, UNIT) * SRC_UNIT * blockL;
    int strideInside   = weight->stride(0) / blockNum;
    if (SRC_UNIT > pack) { // shape = {blockNum, UP_DIV(oc, UNIT), UP_DIV(UP_DIV(ic, pack) * kernelCount, SRC_UNIT / pack) / blockNum, UNIT, SRC_UNIT};
        auto icDivU = UP_DIV(ic, pack);
        for (int k = 0; k < kernelCount; ++k) {
            const auto srcK = weightSrc + k;
            for (int y = 0; y < ic; ++y) {
                const int yOutSide    = y / pack;
                const int yInSide     = y % pack;
                const int yIndex      = yOutSide + k * icDivU;
                const int ySubOutSide = yIndex / (SRC_UNIT / pack);
                const int ySubInSide  = yIndex % (SRC_UNIT / pack);

                int blockId = ySubOutSide / blockL;
                int blockInsideId = ySubOutSide % blockL;

                auto dstY       = weightDst + blockId * strideOutside + blockInsideId * weight->stride(1) + ySubInSide * pack + yInSide;
                const auto srcY = srcK + y * kernelCount;
                for (int x = 0; x < oc; ++x) {
                    const int xOutSide = x / UNIT;
                    const int xInSide  = x % UNIT;
                    const int dstIndex = xOutSide * strideInside + xInSide * SRC_UNIT;
                    const int srcIndex = x * kernelCount * ic;
                    dstY[dstIndex]     = srcY[srcIndex];
                }
            }
        }
    } else { // shape = {blockNum, UP_DIV(oc, UNIT), UP_DIV(ic, SRC_UNIT) * kernelCount / blockNum, UNIT, SRC_UNIT};
        for (int k = 0; k < kernelCount; ++k) {
            auto icDivU = UP_DIV(ic, SRC_UNIT);
            const auto srcK = weightSrc + k;
            for (int y = 0; y < ic; ++y) {
                const int yOutSide    = y / SRC_UNIT;
                const int yInSide     = y % SRC_UNIT;

                int blockId = (yOutSide + k * icDivU) / blockL;
                int blockInsideId = (yOutSide + k * icDivU) % blockL;

                auto dstY       = weightDst + blockId * strideOutside + blockInsideId * weight->stride(1) + yInSide;
                const auto srcY = srcK + y * kernelCount;
                for (int x = 0; x < oc; ++x) {
                    const int xOutSide = x / UNIT;
                    const int xInSide  = x % UNIT;
                    const int dstIndex = xOutSide * strideInside + xInSide * SRC_UNIT;
                    const int srcIndex = x * kernelCount * ic;
                    dstY[dstIndex]     = srcY[srcIndex];
                }
            }
        }
    }
}

static bool _reorderWeightInside(Backend* bn, const Convolution2DCommon* common,
                                 const std::shared_ptr<Tensor>& weightOrigin,
                                 std::shared_ptr<Tensor>& weight, int blockNum, bool alloc) {
    MNN_ASSERT(blockNum > 0);
    auto core = static_cast<CPUBackend*>(bn)->int8Functions();
    auto gcore = static_cast<CPUBackend*>(bn)->functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    // reorder weight, [oc, ic, k^2] => [oc/unit, ((ic/unit)*k^2)/(src_unit/unit), unit(oc), (src_unit/unit), unit(ic)]
    int oc = common->outputCount(), ic = common->inputCount(), kernelCount = common->kernelX() * common->kernelY();
    std::vector<int> shape;
    int pack = gcore->pack;
    if (SRC_UNIT > pack) {
        MNN_ASSERT(SRC_UNIT % pack == 0);
        shape = {UP_DIV(oc, UNIT), UP_DIV(UP_DIV(ic, pack) * kernelCount, SRC_UNIT / pack), UNIT, SRC_UNIT};
    } else {
        shape = {UP_DIV(oc, UNIT), UP_DIV(ic, SRC_UNIT) * kernelCount, UNIT, SRC_UNIT};
    }
    bool useCachedMmap = bn->getRuntime()->hint().useCachedMmap > 1;
    if (alloc) {
        weight.reset(Tensor::createDevice<int8_t>(shape));
        bool succ = bn->onAcquireBuffer(weight.get(), Backend::STATIC);
        if (!succ) {
            MNN_ERROR("Memory not enough");
            return false;
        }
    } else {
        // useCachedMmap don't need alloc memory
        weight.reset(useCachedMmap ? Tensor::createDevice<int8_t>(shape) : Tensor::create<int8_t>(shape));
    }
    if (useCachedMmap) {
        return true;
    }
    ConvInt8TiledExecutor::reorderWeight(weight.get(), weightOrigin->host<uint8_t>(), SRC_UNIT, UNIT, ic, oc, kernelCount, pack, blockNum);
    return true;
}

static void GetResourceInt8(std::shared_ptr<CPUConvolution::ResourceInt8> resource, std::shared_ptr<ConvolutionCommon::Int8Common> quantCommon, const Convolution2D* conv2d, Backend* backend, int32_t* blocknumPtr) {
    // common parameters
    int outputCount = conv2d->common()->outputCount();
    auto core = static_cast<CPUBackend*>(backend)->functions();
    int LSize = conv2d->common()->inputCount() * conv2d->common()->kernelX() * conv2d->common()->kernelY();
    int ocUp4 = ROUND_UP(outputCount, core->pack);
    bool useCachedMmap = backend->getRuntime()->hint().useCachedMmap > 1;

    int dequantCnt = quantCommon->alpha.size();
    if (quantCommon->asymmetric) {
        dequantCnt /= 2;
    }
    int blockNum = dequantCnt / outputCount;
    blocknumPtr[0] = blockNum;
    int scaleSize = blockNum * ocUp4; // pack size.
    int blockSize = LSize / blockNum;
    int originOffset = 0;
    resource->mActBits = 8;
    if (quantCommon->canUseInt4) {
        originOffset = -8;
        resource->mActBits = 4;
    }
    // alloc memory
    resource->mOriginBias.reset(Tensor::createDevice<int32_t>({ocUp4})); // float
    auto success = backend->onAcquireBuffer(resource->mOriginBias.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Alloc bias memory error\n");
        return;
    }
    int bytes = 4;
    resource->mOriginScale.reset(Tensor::createDevice<uint8_t>({2 * scaleSize * bytes}));
    success = backend->onAcquireBuffer(resource->mOriginScale.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Alloc denquant alpha, zero memory error\n");
        return;
    }
    resource->mWeightKernelSum.reset(Tensor::createDevice<uint8_t>({bytes * ocUp4}));
    success = backend->onAcquireBuffer(resource->mWeightKernelSum.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Alloc denquant weight kernel sum memory error\n");
        return;
    }
    if (useCachedMmap) {
        return;
    }
    // Save bias
    ::memset(resource->mOriginBias->host<float>(), 0, ocUp4 * sizeof(float));
    if (conv2d->bias()) {
        ::memcpy(resource->mOriginBias->host<float>(), conv2d->bias()->data(), outputCount * sizeof(float));
    } else {
        ::memset(resource->mOriginBias->host<float>(), 0, ocUp4 * sizeof(float));
    }
    // Save weight quant alpha and zero: wf=alpha*wi+zero
    auto alphaPtr = resource->mOriginScale->host<float>();
    auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(alphaPtr) + scaleSize * bytes);
    if (outputCount % core->pack != 0) {
        ::memset(alphaPtr, 0, scaleSize * bytes);
        ::memset(biasPtr, 0, scaleSize * bytes);
    }
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
    auto weightKernelSum = resource->mWeightKernelSum->host<float>();
    auto realWeightData = quantCommon->weight.get();
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
            if (quantCommon->canUseInt4) {
                for (int i = 0; i < blockSize; ++i) {
                    int l_index = k * blockSize + i;
                    int w_idx = (j * blockNum * blockSize + l_index);
                    int w_offset = w_idx / 2;
                    int w_mask = w_idx % 2;
                    uint8_t s = realWeightData[w_offset];
                    int val = w_idx % 2 ? s & 0x0f : s >> 4;
                    tmp += (val - 8);
                }
            } else {
                for (int i = 0; i < blockSize; ++i) {
                    int l_index = k * blockSize + i;
                    tmp += (int)realWeightData[j * blockNum * blockSize + l_index];
                }
            }

            sum += (tmp * scale + blockSize * bias);
        }
        weightKernelSum[j] = sum;
    }
}

DenseConvInt8TiledExecutor::DenseConvInt8TiledExecutor(Backend* backend, const Op* op, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon) : ConvInt8TiledExecutor(backend, op) {
    auto convOp = op->main_as_Convolution2D();
    auto core = static_cast<CPUBackend*>(backend)->int8Functions();
    auto gcore = static_cast<CPUBackend*>(backend)->functions();
    mResourceInt8.reset(new CPUConvolution::ResourceInt8);
    mResourceInt8->mDynamicQuant = true;
    int blockNum = 1;
    GetResourceInt8(mResourceInt8, quanCommon, convOp, backend, &blockNum);
    mBlockNum = blockNum;
    // dynamic quant
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int pack = gcore->pack;
    auto weightLength = quanCommon->weight.size();
    int kernelCount = mCommon->kernelX() * mCommon->kernelY();
    int oc = convOp->common()->outputCount();
    int ic = convOp->common()->inputCount();
    bool directReadInt4weight = (kernelCount == 1 && ROUND_UP(oc, UNIT) == oc && ROUND_UP(ic, SRC_UNIT) == ic);
    bool useCachedMmap = backend->getRuntime()->hint().useCachedMmap > 1;
#ifdef MNN_KLEIDIAI_ENABLED
    bool half_act = gcore->bytes == 2;
    int biasSize = mResourceInt8->mOriginBias->size();
    int alphaSize = mResourceInt8->mOriginScale->size();
    bool blockwise = (biasSize * 2) != alphaSize;
    KleidiAI kai = KleidiAI::getInstance(quanCommon->asymmetric, half_act, blockwise);
    if(quanCommon->canUseInt4 && kai.canAccelerate()) {
        int n = oc;
        int k = ic;
        int packedWeightSize = kai.getRhsPackedSize(n, k);

        //Alloc packed weight tensor.
        mResourceInt8->mWeightInt8.reset(Tensor::createDevice<uint8_t>({packedWeightSize}));
        bool success = backend->onAcquireBuffer(mResourceInt8->mWeightInt8.get(), Backend::STATIC);

        if (!success) {
            MNN_ERROR("Out of static memory!\n");
            return;
        }

        //Run rhs pack.
        kai.runRhsPack(n, k, (uint8_t*)quanCommon->weight.get(),
                       mResourceInt8->mOriginScale->host<float>(),
                       mResourceInt8->mOriginBias->host<float>(),
                       mResourceInt8->mWeightInt8->host<uint8_t>(),
                       directReadInt4weight);

        return;
    }

#endif

    if (quanCommon->canUseInt4 && directReadInt4weight) {
        // int4 weight reorder
        mResourceInt8->mWeightAsymmetricQuant = true;
        // shape = {UP_DIV(oc, UNIT), UP_DIV(ic, SRC_UNIT) * kernelCount, UNIT, SRC_UNIT};
        int hU = UP_DIV(oc, UNIT);
        int lU = UP_DIV(ic, SRC_UNIT);
        int hP = UNIT;
        int lP = SRC_UNIT;

        // weight shape.
        std::vector<int32_t> shape;
        if (SRC_UNIT > pack) {
            MNN_ASSERT(SRC_UNIT % pack == 0);
            shape = {UP_DIV(oc, UNIT), UP_DIV(UP_DIV(ic, pack) * kernelCount, SRC_UNIT / pack), UNIT * SRC_UNIT / 2};
        } else {
            shape = {UP_DIV(oc, UNIT), UP_DIV(ic, SRC_UNIT) * kernelCount, UNIT * SRC_UNIT / 2};
        }
        mResourceInt8->mWeightInt8.reset(Tensor::createDevice<uint8_t>(shape));

        auto res = backend->onAcquireBuffer(mResourceInt8->mWeightInt8.get(), Backend::STATIC);
        if (!res) {
            MNN_ERROR("int4 weight acquire buffer error\n");
            return ;
        }
        if (!useCachedMmap) {
            auto srcPtr = (uint8_t*)quanCommon->weight.get();
            auto dstPtr = mResourceInt8->mWeightInt8->host<uint8_t>();
            ::memset(dstPtr, 0, mResourceInt8->mWeightInt8->size());
            // Pack two int4-weight to one int8-weight.
            int cnt = lP * hP / 4;
            int L = lU * lP;
            int blockL = lU / blockNum;
            int stride0 = (lP * hP) * hU * blockL;
            int stride1 = (lP * hP) * blockL;
            for (int i = 0; i < hU; ++i) {
                for (int j = 0; j < lU; ++j) {
                    int blockId = j / blockL;
                    int blockkInsideId = j % blockL;
                    for (int k = 0; k < cnt; ++k) {
                        int dstIndx0 = (blockId * stride0 + i * stride1 + blockkInsideId * lP * hP) / 2 + (2 * k);
                        int hpId0     = (2 * k + 1) / lP;
                        int lpId0     = (2 * k) % lP;
                        int hpId1     = (2 * (k + cnt) + 1) / lP;
                        int lpId1     = (2 * (k + cnt)) % lP;
                        int srcIndx0 = ((i * hP + hpId0) * L + (j * lP + lpId0)) / 2;
                        int srcIndx1 = ((i * hP + hpId1) * L + (j * lP + lpId1)) / 2;
                        int s0 = (srcPtr[srcIndx0] >> 4);
                        int s1 = (srcPtr[srcIndx0] & 15);
                        int s2 = (srcPtr[srcIndx1] >> 4);
                        int s3 = (srcPtr[srcIndx1] & 15);
                        int d0 = s0 * 16 + s2;
                        int d1 = s1 * 16 + s3;
                        dstPtr[dstIndx0] = d0;
                        dstPtr[dstIndx0 + 1] = d1;
                    }
                }
            }
        }
    } else {
        // std::shared_ptr<Tensor> srcWeight;
        if (quanCommon->canUseInt4) {
            mResourceInt8->mWeightAsymmetricQuant = true;
            auto srcPtr = reinterpret_cast<uint8_t*>(quanCommon->weight.get());
            std::vector<int8_t> tmpWeight(weightLength * 2, 0);
            for (int i = 0; i < weightLength; ++i) {
                int8_t s0 = (srcPtr[i] >> 4) - 8; // For int4 quant weight, +8 saved in quant buffer
                int8_t s1 = (srcPtr[i] & 0x0f) - 8;
                tmpWeight[2 * i + 0] = s0;
                tmpWeight[2 * i + 1] = s1;
            }
            std::shared_ptr<Tensor> srcWeight(Tensor::create<uint8_t>({weightLength * 2}, (void*)tmpWeight.data()));
            std::shared_ptr<Tensor> tmpWeightInt8;
            mValid = _reorderWeightInside(backend, convOp->common(), srcWeight, tmpWeightInt8, blockNum, false);
            if(!mValid) {
                return;
            }
            MNN_ASSERT(tmpWeightInt8->size() % 2 == 0);
            int leng = tmpWeightInt8->size();
            int halflen = leng / 2;
            mResourceInt8->mWeightInt8.reset(Tensor::createDevice<int8_t>({halflen}));
            bool succ = backend->onAcquireBuffer(mResourceInt8->mWeightInt8.get(), Backend::STATIC);
            if (!succ) {
                MNN_ERROR("Memory not enough");
                return;
            }
            if (!useCachedMmap) {
                auto srcint4Ptr = tmpWeightInt8->host<int8_t>();
                auto dstint4Ptr = mResourceInt8->mWeightInt8->host<uint8_t>();
                int permuteUnit = UNIT * SRC_UNIT;
                int halfPermuteStride = static_cast<int32_t>(permuteUnit / 2);
                for (int i = 0; i < leng / permuteUnit; ++i) {
                    auto src0 = srcint4Ptr + i * permuteUnit;
                    auto dst0 = dstint4Ptr + i * halfPermuteStride;
                    for (int j = 0; j < halfPermuteStride; ++j) {
                        int s0 = src0[j];
                        int s1 = src0[j + halfPermuteStride];
                        int d = (s0 + 8) * 16 + (s1 + 8);
                        dst0[j] = d;
                    }
                }
            }
        } else {
            std::shared_ptr<Tensor> srcWeight(Tensor::create<uint8_t>({weightLength}, (void*)quanCommon->weight.get()));
            mValid = _reorderWeightInside(backend, convOp->common(), srcWeight, mResourceInt8->mWeightInt8, blockNum, true);
            if(!mValid) {
                return;
            }
        }
    }

    // Relu/Relu6 post parameters
    auto postPtr = getPostParameters();
    mResourceInt8->mReluThreshold.resize(2);
    mResourceInt8->mReluThreshold[0] = postPtr[2];
    mResourceInt8->mReluThreshold[1] = postPtr[3];
    if (gcore->bytes == 2) {
        gcore->MNNFp32ToLowp(mResourceInt8->mReluThreshold.data(), reinterpret_cast<int16_t*>(mResourceInt8->mReluThreshold.data()), 2);
    }
}
static void _computeAlphaScale(Backend* backend, const Convolution2D* conv2d, std::shared_ptr<CPUConvolution::ResourceInt8> resourceInt8) {
    /* Used to compute weight quant scale and bias and weightKernelSum of type float. */
    bool quanBuffer = (conv2d->quanParameter() != nullptr && conv2d->quanParameter()->buffer() != nullptr);
    MNN_ASSERT(quanBuffer || resourceInt8);
    auto core = static_cast<CPUBackend*>(backend)->functions();
    // common parameters
    int outputCount = conv2d->common()->outputCount();
    int LSize = conv2d->common()->inputCount() * conv2d->common()->kernelX() * conv2d->common()->kernelY();
    int ocUp4 = ROUND_UP(outputCount, core->pack);
    int8_t* weightOrigin;

    // Save weight quant scale and bias: wf=scale*wi+bias
    std::shared_ptr<Tensor> scaleBias(Tensor::createDevice<uint8_t>({2 * ocUp4 * core->bytes}));
    auto success = backend->onAcquireBuffer(scaleBias.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Alloc dequant scaleBias memory error\n");
        return;
    }
    auto alphaPtr = scaleBias->host<float>();
    auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(alphaPtr) + ocUp4 * core->bytes);
    ::memset(alphaPtr, 0, 2 * ocUp4 * core->bytes);

    // Load quant scale and bias
    weightOrigin = resourceInt8->mWeightInt8->host<int8_t>();
    auto wZero = resourceInt8->mWeightQuantZero->host<int32_t>(); // has packed to outputUp4
    auto wScale = resourceInt8->mOriginScale->host<float>();
    int h = ocUp4;
    MNN_ASSERT(4 == core->bytes);
    for (int i=0; i< h; ++i) {
        alphaPtr[i] = wScale[i];
        biasPtr[i] = (-1.f) * wZero[i] * wScale[i];
    }
    resourceInt8->mOriginScale = scaleBias;

    // Compute float weightKernelSum
    resourceInt8->mWeightKernelSum.reset(Tensor::createDevice<uint8_t>({ocUp4 * 4}));
    success = backend->onAcquireBuffer(resourceInt8->mWeightKernelSum.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Alloc dequant mWeightKernelSum memory error\n");
        return;
    }
    auto weightKernelSum = resourceInt8->mWeightKernelSum->host<float>();
    for (int i = 0; i < outputCount; ++i) {
        int sum = 0;
        for (int j = 0; j < LSize; ++j) {
            sum = sum + static_cast<int>(weightOrigin[j + i * LSize]);
        }
        auto scale = alphaPtr[i];
        auto bias = biasPtr[i];
        weightKernelSum[i] = static_cast<float>(sum) * scale + LSize * bias;
    }
}

DenseConvInt8TiledExecutor::DenseConvInt8TiledExecutor(Backend* backend, const Op* op, std::shared_ptr<ResourceInt8> res) : ConvInt8TiledExecutor(backend, op, res) {
    std::shared_ptr<Tensor> weightOrigin = mResourceInt8->mWeightInt8;
    auto convOp = op->main_as_Convolution2D();
    mValid = _reorderWeightInside(backend, convOp->common(), weightOrigin, mResourceInt8->mWeightInt8, mBlockNum, true);
    if(!mValid) {
        return;
    }
    // offline quant: choose int8 gemm kernel
    auto core = static_cast<CPUBackend*>(backend)->int8Functions();
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
    _computeAlphaScale(backend, convOp, mResourceInt8);
}

DenseConvInt8TiledExecutor::DenseConvInt8TiledExecutor(Backend* backend, const Op* op, const DenseConvInt8TiledExecutor& exe)
    : ConvInt8TiledExecutor(backend, op, exe.mResourceInt8), mGemmKernel(exe.mGemmKernel) {
}

DenseConvInt8TiledExecutor::~DenseConvInt8TiledExecutor() {
    // Do nothing
}

bool DenseConvInt8TiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new DenseConvInt8TiledExecutor(bn, op, *this);
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
    mUseBatchQuan &= mResourceInt8->mDynamicQuant;
    mUseBatchQuan &= (inputs[0]->batch() > 1);
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    auto gcore =static_cast<CPUBackend*>(backend())->functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    auto output = outputs[0];
    float weightBytes = mResourceInt8->mActBits == 4 ? 0.5 : 1;


#ifdef MNN_KLEIDIAI_ENABLED
    KleidiAI& kai = KleidiAI::getInstance();
    if(mResourceInt8->mDynamicQuant && mResourceInt8->mActBits == 4 && kai.canAccelerate()) {
        int batch = inputs[0]->batch();
        int channel = inputs[0]->channel();

        int packedSize = kai.getLhsQuantedPackedSize(batch, channel);
        mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({packedSize}));
        bool success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
        if (!success) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }

        backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
        return NO_ERROR;
    }
#endif

    if (mResourceInt8->mDynamicQuant == false) {
        mMutableResource->updateInputOutputScale(TensorUtils::getQuantInfo(inputs[0]), TensorUtils::getQuantInfo(outputs[0]));
        CPUConvolution::onResize(inputs, outputs);
        ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY, gcore, core);
        mBlockNum = 1;
    } else { // Dynamic Quant kernels
        CPUConvolution::onResize(inputs, outputs);
        // Gemm Kernel
        mGemmKernel = core->Int8GemmKernel;
        if (mResourceInt8->mActBits == 4) {
            mGemmKernel = core->Int8GemmKernel_W4;
        }
        mQuantFunc = core->MNNFloat2Int8;
        if (gcore->bytes == 2 && gcore->pack == 8) {
            mGemmKernel = core->MNNGemmInt8AddBiasScale_Unit_FP16;
            if (mResourceInt8->mActBits == 4) {
                mGemmKernel = core->MNNGemmInt8AddBiasScale_w4_Unit_FP16;
            }
            mQuantFunc = core->DynamicQuanInput_ARM82;
            mQuantAndReorderFunc = core->DynamicQuanInputAndReorder_ARM82;

        }
        // A axisSum kernel
        mSumByAxisLFunc = gcore->MNNSumByAxisLForMatmul_A;
        ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY, gcore, core);

        int ocUp4 = ROUND_UP(outputs[0]->channel(), gcore->pack);
        int alphaSize = mResourceInt8->mOriginScale->size() / (sizeof(float) * 2);
        mBlockNum  = alphaSize / ocUp4;
    }

    // input scale buffer
    int batch = inputs[0]->batch();
    mInputDeqScales.reset(Tensor::createDevice<int8_t>({batch * 4}));
    bool success = backend()->onAcquireBuffer(mInputDeqScales.get(), Backend::DYNAMIC);

    // Im2col info
    const int threads = static_cast<CPUBackend*>(backend())->threadNumber();
    auto planeSize = output->width() * output->height() * output->batch();
    const int L2Size = 2048;
    const int tileLimitByC = UP_DIV(L2Size, mIm2ColParamter.kernelCountUnit * SRC_UNIT);
    int tileLimit = 0;
    int outC    = output->channel();
    int outC4 = UP_DIV(outC, gcore->pack);
    auto icDiv4KernelCount = mIm2ColParamter.kernelCountUnit;
    mSplitByOc = true;
    
    // flop and io
    float flop = gcore->bytes * planeSize * (ROUND_UP(output->channel(), gcore->pack) * icDiv4KernelCount * SRC_UNIT / 1024.0 / 1024.0 / 1024.0);
    float ios  = (((CPUBackend*)backend())->getTensorSize(outputs[0], true) + ((CPUBackend*)backend())->getTensorSize(inputs[0], true) + ((CPUBackend*)backend())->getTensorSize(mResourceInt8->mWeightInt8.get()) * weightBytes) / (1024.0 * 1024.0 * 1024.0);

    if (threads < planeSize) { // Thread split by output nhw.
        tileLimit = ALIMIN(tileLimitByC, UP_DIV(planeSize, threads));
        mIm2ColCount = UP_DIV(tileLimit, DST_XUNIT);
        auto DynamicDestUnit = DST_XUNIT * mIm2ColCount;
        mTileCount        = UP_DIV(planeSize, DynamicDestUnit);
        if (mTileCount > threads) {
            mSplitByOc = false;
        }
        
    }
    if (mSplitByOc) {
        tileLimit = ALIMIN(tileLimitByC, planeSize);
        mIm2ColCount = UP_DIV(tileLimit, DST_XUNIT);
        auto DynamicDestUnit = DST_XUNIT * mIm2ColCount;
        mTileCount        = UP_DIV(planeSize, DynamicDestUnit);
        auto ocPerThread = UP_DIV(outC4, threads);
        auto threadNeed = UP_DIV(outC4, ocPerThread);
        int totalWork = outC4;
        int part = 1;
        if (UNIT > gcore->pack) { // AVX512:UNIT=64,pack=16
            MNN_ASSERT(UNIT % gcore->pack == 0);
            int ocDivUnit = UP_DIV(outC4 * gcore->pack, UNIT);
            ocPerThread = UP_DIV(ocDivUnit, threads);
            threadNeed  = UP_DIV(ocDivUnit, ocPerThread);
            totalWork = ocDivUnit;
            part = UNIT / gcore->pack;
        }
        mThreadNums = ALIMIN(threads, threadNeed);
        mSplitByOc = true;

        mDivides.resize(threads+1);
        mDivides[0] = 0;
        static_cast<CPUBackend *>(backend())->computeDivideSizes(totalWork, mDivides.data() + 1, flop / ios);
        for (int i = 0; i < mDivides.size(); ++i) {
            mDivides[i] *= part;
        }
    }

    if (!mSplitByOc) {
        mThreadNums = ALIMIN(threads, mTileCount);
        mDivides.resize(threads+1);
        mDivides[0] = 0;
        static_cast<CPUBackend *>(backend())->computeDivideSizes(mTileCount, mDivides.data() + 1, flop / ios);
    }
    int ocUp4 = ROUND_UP(outC, gcore->pack);
    int alphaSize = mResourceInt8->mOriginScale->size() / (sizeof(float) * 2);
    int k = mThreadNums;
    int workPT = DST_XUNIT * mIm2ColCount;
    if (mSplitByOc) {
        k = 1; // Use one thread to finish im2col.
        workPT = mTileCount * DST_XUNIT * mIm2ColCount;
    }

    auto bufferAlloc = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    auto blitInfoSize = ConvolutionTiledExecutor::computeBlitInfoSize(workPT, mIm2ColParamter.ow, mIm2ColParamter.kernelX * mIm2ColParamter.kernelY, k);
    mBlitInfoStride = blitInfoSize.second;
    mBlitInfo = bufferAlloc->alloc(blitInfoSize.first);
    if (!mSplitByOc) {
        mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({threads, DST_XUNIT * mIm2ColCount * icDiv4KernelCount * SRC_UNIT}));
        mTempSrcSum.resize(threads * mBlockNum * DST_XUNIT * mIm2ColCount * 4); // Use 4 bytes to save kernel sum.
    } else {
        mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({mTileCount, DST_XUNIT * mIm2ColCount * icDiv4KernelCount * SRC_UNIT}));
        mTempSrcSum.resize(mTileCount * mBlockNum * DST_XUNIT * mIm2ColCount * 4); // Use 4 bytes to save kernel sum.
    }
    

    success &= backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (!success || mBlitInfo.invalid()) {
        return OUT_OF_MEMORY;
    }
    if (false == mResourceInt8->mDynamicQuant) {
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

#ifdef MNN_KLEIDIAI_ENABLED
    KleidiAI& kai = KleidiAI::getInstance();
    if(mResourceInt8->mDynamicQuant && mResourceInt8->mActBits == 4 && kai.canAccelerate()) {
        const size_t m = input->batch(); //lhs vector number.
        const size_t n = output->channel(); //rhs vector number.
        const size_t k = input->channel(); //vector size.

        auto lhs = input->host<uint8_t>();
        auto lhsPacked = mTempIm2ColBuffer->host<int8_t>();
        auto rhsPacked = mResourceInt8->mWeightInt8->host<uint8_t>();
        auto dst = output->host<uint8_t>();

        int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
        int threadNeed, vecPerThread;

#if !KAI_CONV_NCHW_IN_OUT
        kai.packNC4HW4ToNCHW((float *)lhs, m, k);
#endif

        //Dynamic quant pack lhs.
        if(m == 1) {
            kai.runLhsQuantPack(1, k, 1, lhs, lhsPacked);
        } else {
            vecPerThread = kai.getVecNumPerThread(m, threadNum, kai.getMr(m));
            threadNeed = m % vecPerThread == 0 ? m / vecPerThread : (m / vecPerThread + 1);
            size_t srcStride = vecPerThread * k * sizeof(float);

            auto BatchDynamicQuant = [=, &kai](int tId) {
                auto threadSrc = lhs + tId * srcStride;
                auto threadDst = lhsPacked + kai.getLhsQuantedPackedOffset(m, tId * vecPerThread, k);
                int vecNum = (tId == threadNeed - 1) ? (m - vecPerThread * tId) : vecPerThread; //Last threadN may less than vecPerThread.
                kai.runLhsQuantPack(vecNum, k, kai.getMr(m), threadSrc, threadDst);
            };

            MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
                BatchDynamicQuant((int)tId);
            }
            MNN_CONCURRENCY_END();
        }

        //Run matmul.
        vecPerThread = kai.getVecNumPerThread(n, threadNum, kai.getNStep());
        threadNeed = n % vecPerThread == 0 ? n / vecPerThread : (n / vecPerThread + 1);

        auto ThreadFunction = [=, &kai](int tId) {
            auto threadRhsPacked = rhsPacked + kai.getRhsPackedOffset(tId * vecPerThread, k);
            auto threadDst = dst + kai.getDstOffset(0, tId * vecPerThread, n);
            int vecNum = (tId == threadNeed - 1) ? (n - vecPerThread * tId) : vecPerThread; //Last threadN may less than vecPerThread.
            kai.runMatmul(m, vecNum, k, lhsPacked, threadRhsPacked, n * sizeof(float), threadDst);
        };

        MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
            ThreadFunction((int)tId);
        }
        MNN_CONCURRENCY_END();

#if !KAI_CONV_NCHW_IN_OUT
        kai.packNCHWToNC4HW4((float *)dst, m, n);
#endif

        return NO_ERROR;
    }
#endif

    int UNIT__, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT__, &SRC_UNIT, &DST_XUNIT);
    auto blitProc = core->MNNPackC4Int8ForMatMul_A;
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
    // const int alphaSize              = mResource->mDequantize.mScaleBias->size() / (4 * 2);
    const int alphaSize              = mResourceInt8->mOriginScale->size() / (sizeof(float) * 2);
    const int blockL                  = kernelCountUnitDouble / mBlockNum; // source depthQuad for each block.
    float weightBytes                = 1.f;
    int weight_step_Y                = weightBytes * (UNIT__ * SRC_UNIT);
    int src_step_Y                   = DST_XUNIT * SRC_UNIT;

    auto inputDataPtr        = input->host<int8_t>();
    auto im2colPtr           = mTempIm2ColBuffer->host<int8_t>();
    if (SRC_UNIT > PackUnit) {
        memset(im2colPtr, 0, mTempIm2ColBuffer->size());
    }
    const auto weightDataPtr = mResourceInt8->mWeightInt8->host<int8_t>();
    auto srcKernelSumPtr     = mTempSrcSum.data();
    auto weightDequantBias = mResourceInt8->mOriginScale->host<uint8_t>() + alphaSize * 4;

    auto outputDataPtr = output->host<int8_t>();
    uint8_t* biasPtr = nullptr;
    uint8_t* scalePtr = nullptr;
    int32_t inputZeroPoint = 0;
    auto inputScalePtr = mInputDeqScales->host<uint8_t>();
    if (nullptr != mMutableResource.get()) {
        biasPtr       = mMutableResource->mBiasFloat->host<uint8_t>();
        scalePtr      = mMutableResource->mScaleFloat->host<uint8_t>();
        inputZeroPoint  = mMutableResource->mInputZeroPoint;
        (reinterpret_cast<float*>(inputScalePtr))[0] =  mMutableResource->mInputScale;
    }

    auto SingleDynamicQuant = [&] () {
        const auto floatptr = input->host<float>();
        auto int8ptr        = mQuantInput->host<int8_t>();
        auto inputsize      = static_cast<CPUBackend*>(backend())->getTensorSize(inputs[0]);
        float quantscale    = 0.f;
        float dequantscale  = 0.f;
        float zeropoint       = 0;

         /* Count max and min value to compute input scale and zeropoint */
        auto maxMinValPtr = mTempMaxMinValueBuffer->host<uint8_t>();
        int threadNeed = mThreadNums;
        auto inputSizeCount = UP_DIV(inputsize, mThreadNums);
        if (true) {
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
        if (mCommon->padX() > 0 || mCommon->padY() > 0) { // Ensure "0.0f" included in range.
            if (minVal > 0.f) {
                minVal = 0.f;
            } else if (maxVal < 0.f){
                maxVal = 0.f;
            } else {
                //
            }
        }
        float range = maxVal - minVal;
        if (fabs(range) < 1e-7) {
            zeropoint = maxVal;
            quantscale = 1.0f;
            dequantscale = 1.0f;
        } else {
            quantscale = 255.0f / range;
            dequantscale = range / 255.0f;
            zeropoint = roundf(-minVal * 255.f / range) - 128.0f;
        }
        auto sizeDiv = UP_DIV(inputsize, PackUnit);

        threadNeed = mThreadNums;
        inputSizeCount = UP_DIV(sizeDiv, mThreadNums);
        if (true) {
            threadNeed = 1;
            inputSizeCount = sizeDiv;
        } else {
            threadNeed = ALIMIN(UP_DIV(sizeDiv, inputSizeCount), mThreadNums);
            inputSizeCount = UP_DIV(sizeDiv, threadNeed);
        }
        MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
            auto perThreadWorkCount = ALIMIN(inputSizeCount, sizeDiv - tId * inputSizeCount);
            auto inptr_ = (float*)(((int8_t*)floatptr) + tId * inputSizeCount * PackUnit * gcore->bytes);
            mQuantFunc(inptr_ , int8ptr + tId * inputSizeCount * PackUnit, perThreadWorkCount, &quantscale, -128, 127, &zeropoint, 0);
        }
        MNN_CONCURRENCY_END();

        /* bias float */
    #ifdef MNN_USE_SSE
        int offset = 128;
    #else
        int offset = 0;
    #endif
        auto biasfp32 = mResourceInt8->mOriginBias->host<float>();
        auto weightDequantScale = mResourceInt8->mOriginScale->host<float>();
        float zerofp32 = (zeropoint + offset) * dequantscale;

        gcore->MNNDynamicUpdateConvBiasScale(mDynamicBias->host<float>(), mScaleFuse->host<float>(), biasfp32, weightDequantScale, &dequantscale, mResourceInt8->mWeightKernelSum->host<float>(), &zerofp32, UP_DIV(output->channel(), 4), alphaSize);
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
        biasPtr = mResourceInt8->mOriginBias->host<uint8_t>();
        scalePtr = mResourceInt8->mOriginScale->host<uint8_t>();
    };
    ssize_t oneScale = 1;
    if (mUseBatchQuan) {
        BatchDynamicQuant();
        oneScale = 0;
    } else if (mResourceInt8->mDynamicQuant) {
//        MNN::Timer _t;
        SingleDynamicQuant();
//        auto time = (float)_t.durationInUs() / 1000.0f;
//        printf("time = %f \n", time);
    } else {
        // offline quant.
    }


    if (mResourceInt8->mActBits == 4) {
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
        float* reluPtr = mResourceInt8->mReluThreshold.data();
        uint8_t* extraScale = nullptr; // input scale for batch dynamic quant.
        QuanPostTreatParameters quanParam;
        quanParam.blockNum = mBlockNum;
        if (mUseBatchQuan) {
            extraScale = inputScalePtr;
        }
#ifdef MNN_USE_SSE
        quanParam.extraBias = mResourceInt8->mWeightKernelSum->host<float>() + ocIndex;
#endif
        if (dstBytes != 1) {
            quanParam.useInt8 = 0;
            quanParam.fp32minmax = reluPtr;
        } else {
            quanParam.maxValue = mMutableResource->mClampMax;
            if (mResourceInt8->mRelu) {
                quanParam.minValue = mMutableResource->mOutputZeroPoint;
            } else {
                quanParam.minValue = mMutableResource->mClampMin;
            }
        }
        auto outputTid = outputDataPtr + ocIndex * plane * dstBytes;
        const auto biasFloatTid = reinterpret_cast<float*>(biasPtr + ocIndex * 4);
        const auto scaleFloatTid = reinterpret_cast<float*>(scalePtr + ocIndex * 4);
        const auto weightDequanBiasTid  = reinterpret_cast<float*>(weightDequantBias + ocIndex * 4);
        const auto weightPtrTid = weightDataPtr + static_cast<int32_t>(ocIndex * blockL * SRC_UNIT * weightBytes);
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
                    // printf("step=%d, ocDivThread=%d\n", step, ocDivThread);
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

                        mGemmKernel(outputInTilePtr, colAddrTemp + k * blockL * step * SRC_UNIT, weightPtrTid + k * blockL * weight_step_Y * UP_DIV(output->channel(), UNIT__), blockL, dstZStep * dstBytes, ocDivThread, &quanParam, step);
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
    auto exeFunction = [&](int threads) { // Thread split by OC
        auto colAddr        = im2colPtr;
        auto srcPtr     = (int8_t const **)(mBlitInfo.ptr());
        auto el         = (int32_t *)(srcPtr + mBlitInfoStride.second);
        auto xKernelSumPtrTid = reinterpret_cast<float*>(srcKernelSumPtr);

        int32_t info[6];
        info[1] = mIm2ColParamter.iw * mIm2ColParamter.ih * batch;
        info[2] = static_cast<int32_t>(col_buffer_unit_size);
        info[3] = mIm2ColParamter.strideX;
        info[5] = kernelCountUnitDouble;
        
        float* reluPtr = mResourceInt8->mReluThreshold.data();
        // im2col
        auto res = ConvolutionTiledExecutor::turnIm2ColToBlitInfo((const float**)srcPtr, el, 0, plane, mIm2ColParamter, (const uint8_t*)inputDataPtr, 1);
        int number = res.first;
        bool needZero = res.second;
        if (needZero) {
#ifdef MNN_USE_SSE
            ::memset(colAddr, inputZeroPoint + 128, mTempIm2ColBuffer->size());
#else
            ::memset(colAddr, inputZeroPoint, mTempIm2ColBuffer->size());
#endif
        }
        info[0] = number;
        info[4] = plane;
        if (number > 0) {
            blitProc(colAddr, srcPtr, info, el);
        }
        if (mResourceInt8->mWeightAsymmetricQuant) {
            mSumByAxisLFunc(xKernelSumPtrTid, colAddr, (float*)inputScalePtr, plane, sumParams);
        }
        
        MNN_CONCURRENCY_BEGIN(tId, threads) {
            int ocIndex = PackUnit * mDivides[tId];
            auto ocDivThread = ALIMIN(mDivides[tId + 1] - mDivides[tId], ocDiv4 - mDivides[tId]);
            if (ocIndex < ocUp4) {
                QuanPostTreatParameters quanParam;
                quanParam.blockNum = mBlockNum;
                uint8_t* extraScale = nullptr; // input scale for batch dynamic quant.
                if (mUseBatchQuan) {
                    extraScale = inputScalePtr;
                }
#ifdef MNN_USE_SSE
                quanParam.extraBias = mResourceInt8->mWeightKernelSum->host<float>() + ocIndex;
#endif
                if (dstBytes != 1) {
                    quanParam.useInt8 = 0;
                    quanParam.fp32minmax = reluPtr;
                } else {
                    quanParam.maxValue = mMutableResource->mClampMax;
                    if (mResourceInt8->mRelu) {
                        quanParam.minValue = mMutableResource->mOutputZeroPoint;
                    } else {
                        quanParam.minValue = mMutableResource->mClampMin;
                    }
                }
                auto outputInTilePtr = outputDataPtr + ocIndex * plane * dstBytes;
                const auto biasFloatTid = reinterpret_cast<float*>(biasPtr + ocIndex * 4);
                const auto scaleFloatTid = reinterpret_cast<float*>(scalePtr + ocIndex * 4);
                const auto weightDequanBiasTid  = reinterpret_cast<float*>(weightDequantBias + ocIndex * 4);
                const auto weightPtrTid = weightDataPtr + static_cast<int32_t>(ocIndex * blockL * SRC_UNIT * weightBytes);
                if (mBlockNum == 1) {
                    quanParam.biasFloat = biasFloatTid;
                    quanParam.scale = scaleFloatTid;
                    quanParam.weightQuanBias = weightDequanBiasTid;
                }

                int realDstCount = plane;
                auto ptrExtraScale = extraScale != nullptr ? (extraScale) : nullptr;
                auto colAddrTemp = colAddr;
                auto ptrX = xKernelSumPtrTid;
                if (mBlockNum == 1) {
                    do {
                        int step = ALIMIN(DST_XUNIT, realDstCount);
                        quanParam.srcKernelSum = ptrX;
                        quanParam.extraScale = extraScale != nullptr ? (float*)ptrExtraScale : nullptr;
                        // printf("step=%d, ocDivThread=%d\n", step, ocDivThread);
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
                            
                            mGemmKernel(outputInTilePtr, colAddrTemp + k * blockL * step * SRC_UNIT, weightPtrTid + k * blockL * weight_step_Y * UP_DIV(output->channel(), UNIT__), blockL, dstZStep * dstBytes, ocDivThread, &quanParam, step);
                        }
                        ptrX += (step * mBlockNum);
                        realDstCount-=step;
                        outputInTilePtr += DST_XUNIT * PackUnit * dstBytes;
                        colAddrTemp += col_buffer_unit_size;
                        ptrExtraScale = extraScale != nullptr ? (ptrExtraScale + step * 4) : nullptr;
                    } while(realDstCount > 0);
                }
            }
        }
        MNN_CONCURRENCY_END();
        
    };
    const int threads = static_cast<CPUBackend*>(backend())->threadNumber();
    if (!mSplitByOc) {
        MNN_CONCURRENCY_BEGIN(tId, threads) {
                ThreadFunction((int)tId, mDivides[tId], mDivides[tId + 1], 1, 0);

        }
        MNN_CONCURRENCY_END();
    } else {
        exeFunction(threads);
    }
    return NO_ERROR;
}





} // namespace MNN
