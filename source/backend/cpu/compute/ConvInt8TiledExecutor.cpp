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

#define QUANT_INFO_BYTES 4
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

void ConvInt8TiledExecutor::reorderWeight(uint8_t* dst, const uint8_t* src, int32_t* info, int32_t initval, float* kernelsum, weightSummerFuncion summerFunc) {
    // weight shape = {blockNum, UP_DIV(oc, UNIT), kernelCount * UP_DIV(ic / blockNum, SRC_UNIT), UNIT, SRC_UNIT};
    MNN_ASSERT(dst != nullptr && src != nullptr);
    
    int blockNum = info[0];
    int oc = info[1];
    int ic = info[2];
    int kernelCount = info[3];
    int UNIT = info[4];
    int SRC_UNIT = info[5];

    int blockL  = UP_DIV(ic / blockNum, SRC_UNIT) * kernelCount;
    int stride0 = ROUND_UP(oc, UNIT) * SRC_UNIT * blockL; // weight->stride(0)
    int stride1 = blockL * SRC_UNIT * UNIT;               // weight->stride(1)
    int stride2 = UNIT * SRC_UNIT;                        // weight->stride(2)
    int weightlen = stride0 * blockNum;
    memset(dst, initval, weightlen);

    auto hU = UP_DIV(oc, UNIT);
    auto lU = UP_DIV(ic / blockNum, SRC_UNIT) * kernelCount;
    bool fast = (kernelCount == 1 && ROUND_UP(oc, UNIT) == oc && ROUND_UP(ic, SRC_UNIT) == ic);
    if (fast) {
        for (int i = 0; i < hU; ++i) {
            for (int k = 0; k < UNIT; ++k) {
                for (int bl = 0; bl < blockNum; ++bl) {
                    for (int j = 0; j < lU; ++j) {
                        int srcindex = (i * UNIT + k) * ic + bl * (lU * SRC_UNIT) + j * SRC_UNIT;
                        int dstindex = bl * stride0 + i * stride1 + j * stride2 + k * SRC_UNIT;
                        memcpy(dst + dstindex, src + srcindex, SRC_UNIT);
                    }
                }
            }
        }
    } else {
        AutoStorage<uint8_t> tmpBuffer(ic * kernelCount * ROUND_UP(oc, UNIT));
        memset(tmpBuffer.get(), 0, tmpBuffer.size());
        
        auto area = ic * kernelCount;
        // [oc, ic, k2] -> [hU, ic, k2, hP]
        for (int i = 0; i < oc; ++i) {
            auto outId = i / UNIT;
            auto inId  = i % UNIT;
            for (int j = 0; j < area; ++j) {
                tmpBuffer.get()[outId * area * UNIT + j * UNIT + inId] = src[i * area + j];
            }
        }
        // [hU, ic, (k2, hP)] -> [blocknum, hU, lU, (k2, hP), lP]
        AutoStorage<uint8_t> packedBuffer(weightlen);
        memset(packedBuffer.get(), 0, weightlen);
        area = kernelCount * UNIT;
        auto blockic = ic / blockNum;
        for (int i = 0; i < hU; ++i) {
            for (int j = 0; j < ic; ++j) {
                int bk = j / blockic;
                int blu = (j % blockic) / SRC_UNIT;
                int blp = (j % blockic) % SRC_UNIT;
                for (int k = 0; k < area; ++k) {
                    int dstindex = bk * stride0 + i * stride1 + blu * kernelCount * stride2 + k * SRC_UNIT + blp;
                    int srcindex = i * ic * area + j * area + k;
                    packedBuffer.get()[dstindex] = tmpBuffer.get()[srcindex];
                }
            }
        }
        // [(blocknum, hU), lU, k2, (hP, lP)] -> [(blocknum, hU), k2, lU, (hP, lP)]
        area = UNIT * SRC_UNIT;
        auto bklU = UP_DIV(ic, SRC_UNIT) / blockNum;
        for (int bk = 0; bk < blockNum * hU; ++bk) {
            for (int i = 0; i < kernelCount; ++i) {
                for (int j = 0; j < bklU; ++j) {
                    memcpy(dst + bk * stride1 + i * bklU * area + j * area, packedBuffer.get() + bk * stride1 + i * area + j * kernelCount * area, area);
                }
            }
        }
    } // not fast

    if (summerFunc != nullptr && kernelsum != nullptr) {
        summerFunc(kernelsum, (int8_t*)dst, blockNum * hU, blockL, UNIT, SRC_UNIT);
    }
}

void ConvInt8TiledExecutor::packWeightAndQuantInfo(int8_t* dstbuffer, const int8_t* weight, const int8_t* quantInfo, int32_t* info, int infoBytes) {
    int blockNum    = info[0];
    int ocDiv       = info[1];
    int blockL      = info[2];
    int UNIT        = info[3];
    int SRC_UNIT    = info[4];
    auto ocUp4      = info[5];
    auto src0 = weight;              // int8 weight: [blocknum, oc/hp, ic/lp*(kx*ky)/blocknum, hp, lp]
    auto src1 = quantInfo;           // dequant scale
    auto src2 = src1 + infoBytes * ocUp4 * blockNum; // dequant bias
    int stride0 = info[1] * info[2] * info[3] * info[4];
    int stride1 = info[2] * info[3] * info[4];

    for (int bl = 0; bl < blockNum; ++bl) {
        auto blockPtr = dstbuffer + bl * (stride0 + ocUp4 * 2 * infoBytes);
        for (int hU = 0; hU < ocDiv; ++hU) {
            int scaleCount = ALIMIN(ocUp4 - hU * UNIT, UNIT);
            auto hUPtr = blockPtr + hU * (stride1 + 2 * UNIT * infoBytes);
            memcpy(hUPtr, src0 + bl * stride0 + hU * stride1, stride1);
            memcpy(hUPtr + stride1, src1 + (bl * ocUp4 + hU * UNIT) * infoBytes, scaleCount * infoBytes);
            memcpy(hUPtr + stride1 + scaleCount * infoBytes, src2 + (bl * ocUp4 + hU * UNIT) * infoBytes, scaleCount * infoBytes);
        }
    }
}

static void _computeReorderQuantInfo(std::shared_ptr<CPUConvolution::ResourceInt8> resource, std::shared_ptr<ConvolutionCommon::Int8Common> quantCommon, int outputCount, int kernelCount, Backend* backend, AutoStorage<int8_t>& reorderedQuantInfo, float* ikernelSum, int ocUpHp, bool realInt4OrInt8) {
    // Only used for dynamic quant:
    // copy gemm bias
    // copy/compute real dequant bias/scale
    // dequant weight kernel sum
    auto core = static_cast<CPUBackend*>(backend)->functions();
    int ocUp4 = ROUND_UP(outputCount, core->pack);

    int blockNum = resource->mBlockNum;
    int scaleSize = blockNum * ocUp4; // pack size.
    int blockSize = kernelCount / blockNum;
    int originOffset = 0;
    if (quantCommon->canUseInt4) {
        originOffset = -8;
    }
    // Save weight quant alpha and zero: wf=alpha*wi+zero
    auto alphaPtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
    auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(alphaPtr) + scaleSize * QUANT_INFO_BYTES);
    if (outputCount % core->pack != 0) {
        ::memset(alphaPtr, 0, scaleSize * QUANT_INFO_BYTES);
        ::memset(biasPtr, 0, scaleSize * QUANT_INFO_BYTES);
    }
    auto quanInfoPtr = quantCommon->alpha.get();
    auto weightKernelSum = resource->mWeightKernelSum->host<float>();
    ::memset(weightKernelSum, 0, resource->mWeightKernelSum->size());
    // ikernelsum: [blocknum, oc]

    if (quantCommon->asymmetric) {
        for (int i = 0; i < outputCount; ++i) {
            float accum = 0.f;
            for (int j = 0; j < blockNum; ++j) {
                int index = i * blockNum + j;
                alphaPtr[j * ocUp4 + i] = quanInfoPtr[2 * index + 1];
                biasPtr[j * ocUp4 + i] = quanInfoPtr[2 * index] + (float)originOffset * quanInfoPtr[2 * index + 1];
                if (realInt4OrInt8) {
                    accum += (ikernelSum[j * ocUpHp + i] * quanInfoPtr[2 * index + 1] + blockSize * biasPtr[j * ocUp4 + i]);
                } else {
                    accum += ((ikernelSum[j * ocUpHp + i]  - blockSize * 8)* quanInfoPtr[2 * index + 1] + blockSize * quanInfoPtr[2 * index]);
                }
            }
            weightKernelSum[i] = accum;
        }
    } else {
        for (int i = 0; i < outputCount; ++i) {
            float accum = 0.f;
            for (int j = 0; j < blockNum; ++j) {
                int index = i * blockNum + j;
                alphaPtr[j * ocUp4 + i] = quanInfoPtr[index];
                biasPtr[j * ocUp4 + i] = (float)originOffset * quanInfoPtr[index];
                if (realInt4OrInt8) {
                    accum += (ikernelSum[j * ocUpHp + i] * quanInfoPtr[index] + blockSize * biasPtr[j * ocUp4 + i]);
                } else {
                    accum += ((ikernelSum[j * ocUpHp + i]  - blockSize * 8) * quanInfoPtr[index]);
                }
            }
            weightKernelSum[i] = accum;
        }
    }
    
}

DenseConvInt8TiledExecutor::DenseConvInt8TiledExecutor(Backend* backend, const Op* op, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon) : ConvInt8TiledExecutor(backend, op) {
    // convolution info
    auto convOp = op->main_as_Convolution2D();
    int kernelCount = mCommon->kernelX() * mCommon->kernelY();
    int oc = convOp->common()->outputCount();
    int ic = convOp->common()->inputCount();

    int dequantCnt = quanCommon->alphaSize;
    if (quanCommon->asymmetric) {
        dequantCnt /= 2;
    }
    int blockNum = dequantCnt / oc;

    // backend info
    auto core = static_cast<CPUBackend*>(backend)->int8Functions();
    auto gcore = static_cast<CPUBackend*>(backend)->functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int pack = gcore->pack;

    // compute info
    int ocUp4 = ROUND_UP(oc, pack);
    int lU = UP_DIV(ic / blockNum, SRC_UNIT) * kernelCount;
    int scaleSize = ocUp4 * blockNum;
    std::vector<int> shape = {blockNum, UP_DIV(oc, UNIT), lU, UNIT, SRC_UNIT};
    mResourceInt8.reset(new CPUConvolution::ResourceInt8);
    mResourceInt8->mDynamicQuant = true;
    mResourceInt8->mWeightAsymmetricQuant = quanCommon->asymmetric;
    mResourceInt8->mActBits = 8;
    mResourceInt8->mBlockNum = blockNum;
    if (quanCommon->canUseInt4) {
        shape[4] = SRC_UNIT / 2;
        mResourceInt8->mActBits = 4;
        mResourceInt8->mWeightAsymmetricQuant = true; // offset: 8 from uint8_t
    }
    // Relu/Relu6 post parameters
    auto postPtr = getPostParameters();
    mResourceInt8->mReluThreshold.resize(2);
    mResourceInt8->mReluThreshold[0] = postPtr[2];
    mResourceInt8->mReluThreshold[1] = postPtr[3];
    if (gcore->bytes == 2) {
        gcore->MNNFp32ToLowp(mResourceInt8->mReluThreshold.data(), reinterpret_cast<int16_t*>(mResourceInt8->mReluThreshold.data()), 2);
    }
    // buffer allocate
    auto quantlen = 2 * blockNum * ROUND_UP(oc, pack) * QUANT_INFO_BYTES;
    auto weightlen = shape[0] * shape[1] * shape[2] * shape[3] * shape[4];
    mResourceInt8->mWeightInt8.reset(Tensor::createDevice<uint8_t>({weightlen + quantlen}));
    mResourceInt8->mOriginBias.reset(Tensor::createDevice<int32_t>({ocUp4})); // float
    mResourceInt8->mWeightKernelSum.reset(Tensor::createDevice<uint8_t>({QUANT_INFO_BYTES * ocUp4}));

    auto res = backend->onAcquireBuffer(mResourceInt8->mOriginBias.get(), Backend::STATIC);
    res &= backend->onAcquireBuffer(mResourceInt8->mWeightKernelSum.get(), Backend::STATIC);
    res &= backend->onAcquireBuffer(mResourceInt8->mWeightInt8.get(), Backend::STATIC);

    if (!res) {
        MNN_ERROR("weight acquire buffer error\n");
        return;
    }
    bool useCachedMmap = backend->getRuntime()->hint().useCachedMmap > 1;
    if (useCachedMmap) {
        return;
    }

    bool directReadInt4weight = (kernelCount == 1 && ROUND_UP(oc, UNIT) == oc && ROUND_UP(ic, SRC_UNIT) == ic);
    // Save bias
    ::memset(mResourceInt8->mOriginBias->host<float>(), 0, ocUp4 * sizeof(float));
    if (convOp->bias()) {
        ::memcpy(mResourceInt8->mOriginBias->host<float>(), convOp->bias()->data(), oc * sizeof(float));
    }
#ifdef MNN_KLEIDIAI_ENABLED
    if(quanCommon->canUseInt4) {
        bool bFP16 = gcore->bytes == 2 ? true : false;
        bool bAsym = quanCommon->asymmetric;
        size_t blkSize = mResourceInt8->mBlockNum == 1 ? 0 : ic / mResourceInt8->mBlockNum;
        KleidiAI::AccelType accelType = KleidiAI::getQIntAccelType(4, bAsym, blkSize);

        if (!KleidiAI::mKaiInitialized) {
            KleidiAI& kai = KleidiAI::getInstance(*MNNGetCPUInfo(), bFP16, false);
        }

        KleidiAI& kai = KleidiAI::getInstance();
        if(!kai.isLoaded(accelType)) {
            kai.setLoaded(accelType);
            kai.printInfo(accelType);
        }

        if(kai.canAccelerate(accelType)) {
            AutoStorage<int8_t> reorderedQuantInfo;
            reorderedQuantInfo.reset(2 * scaleSize * QUANT_INFO_BYTES);
            if (reorderedQuantInfo.get() == nullptr) {
                MNN_ERROR("Memory not enough\n");
                return;
            }

            //Prepare scale and zero data.
            {
                int outputCount = convOp->common()->outputCount();
                int originOffset = -8;
                auto quanInfoPtr = quanCommon->alpha.get();
                auto scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
                auto zeroPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(scalePtr) + scaleSize * QUANT_INFO_BYTES);
                if (quanCommon->asymmetric) {
                    for (int i = 0; i < blockNum; ++i) {
                        auto dstScale = scalePtr + i * ocUp4;
                        auto dstZero  = zeroPtr + i * ocUp4;
                        for (int j = 0; j < outputCount; ++j) {
                            int scaleIndex = j * blockNum + i;
                            dstScale[j] = quanInfoPtr[2 * scaleIndex + 1];
                            dstZero[j] = quanInfoPtr[2 * scaleIndex] + (float)originOffset * dstScale[j];
                        }
                    }
                } else {
                    for (int i = 0; i < blockNum; ++i) {
                        auto dstScale = scalePtr + i * ocUp4;
                        auto dstZero  = zeroPtr + i * ocUp4;
                        for (int j = 0; j < outputCount; ++j) {
                            int scaleIndex = j * blockNum + i;
                            dstScale[j] = quanInfoPtr[scaleIndex];
                            dstZero[j] = (float)originOffset * dstScale[j];
                        }
                    }
                }
            }

            mAccelType = accelType;
            int n = oc;
            int k = ic;
            int packedWeightSize = kai.getRhsPackedSize(mAccelType, n, k, blkSize);

            //Alloc packed weight tensor.
            mResourceInt8->mWeightInt8.reset(Tensor::createDevice<uint8_t>({packedWeightSize}));
            bool success = backend->onAcquireBuffer(mResourceInt8->mWeightInt8.get(), Backend::STATIC);

            if (!success) {
                MNN_ERROR("Out of static memory!\n");
                return;
            }

            size_t paraNum = scaleSize;
            float *scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
            float *zeroPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + paraNum;
            float *biasPtr = mResourceInt8->mOriginBias->host<float>();
            //Reload some parameters to fit ukernels' layout.
            auto quanInfoPtr = quanCommon->alpha.get();
            if(bAsym) {
                for(int i = 0; i < paraNum; i++) {
                    zeroPtr[i] = quanInfoPtr[i * 2];
                    scalePtr[i] = quanInfoPtr[i * 2 + 1];
                }
            } else {
                if(blkSize != 0) {
                    memcpy(scalePtr, (uint8_t*)quanInfoPtr, paraNum * sizeof(float));
                }
            }

            //Run rhs pack.
            auto weightPackedData = mResourceInt8->mWeightInt8->host<uint8_t>();
            kai.runRhsPack(mAccelType, 1, n, k, blkSize, 0/*unused*/,
                           (uint8_t*)quanCommon->weight.get(),
                           (const void*)scalePtr, (const void*)zeroPtr, (const void*)biasPtr,
                           weightPackedData, directReadInt4weight);
            return;
        }
    }
#endif
    auto target = mResourceInt8;
    auto function = [shape, UNIT, SRC_UNIT, quanCommon, weightlen, scaleSize, directReadInt4weight, blockNum, ic, oc, quantlen, kernelCount, backend, convOp, gcore, target]() -> int {
        auto sh = shape;
        AutoStorage<int8_t> weightReordered(weightlen);
        AutoStorage<int8_t> reorderedQuantInfo(2 * scaleSize * QUANT_INFO_BYTES);
        if (weightReordered.get() == nullptr || reorderedQuantInfo.get() == nullptr) {
            MNN_ERROR("Memory not enough\n");
            return -1;
        }
        /* 1. reorder weight */
        if (quanCommon->canUseInt4 && directReadInt4weight) {
            auto srcPtr = (uint8_t*)quanCommon->weight.get();
            auto dstPtr = (uint8_t*)weightReordered.get();
            ::memset(dstPtr, 0, weightlen);
            AutoStorage<int8_t> kernelsum(blockNum * ROUND_UP(oc, UNIT) * QUANT_INFO_BYTES);
            if (kernelsum.get() == nullptr) {
                MNN_ERROR("Kernel sum buffer allocated error\n");
                return -1;
            }
            gcore->MNNReorderWeightInt4(dstPtr, srcPtr, sh.data(), sh.size(), (float*)kernelsum.get());
            _computeReorderQuantInfo(target, quanCommon, oc, kernelCount * ic, backend, reorderedQuantInfo, (float*)kernelsum.get(), ROUND_UP(oc, UNIT), true);
        } else {
            // std::shared_ptr<Tensor> srcWeight;
            auto weightLength = quanCommon->weight.size();
            int blocksize = ic * kernelCount / blockNum;
            int originOffset = 0;
            int32_t info[6] = {blockNum, oc, ic, kernelCount, UNIT, SRC_UNIT};
            if (quanCommon->canUseInt4) {
                originOffset = -8;
                auto srcPtr = reinterpret_cast<uint8_t*>(quanCommon->weight.get());
                std::vector<uint8_t> tmpWeight(weightLength * 2);
                for (int j = 0; j < oc; ++j) {
                    for (int k = 0; k < blockNum; ++k) {
                        for (int i = 0; i < blocksize; ++i) {
                            int index = j * blockNum * blocksize + k * blocksize + i;
                            uint8_t w_ = srcPtr[index / 2];
                            int truew = index % 2 ? (w_ & 0x0f) : (w_ >> 4);
                            tmpWeight[index] = truew;
                        }
                    }
                }
                AutoStorage<uint8_t> packedInt8weight(weightlen * 2);
                if (packedInt8weight.get() == nullptr) {
                    MNN_ERROR("Weight reorder memory not enough!\n");
                    return -1;
                }
                AutoStorage<int8_t> kernelsum(blockNum * ROUND_UP(oc, UNIT) * QUANT_INFO_BYTES);
                if (kernelsum.get() == nullptr) {
                    MNN_ERROR("Kernel sum buffer allocated error\n");
                    return -1;
                }
                reorderWeight(packedInt8weight.get(), (uint8_t*)tmpWeight.data(), info,0, (float*)kernelsum.get(), gcore->MNNSumWeightInt8);
                _computeReorderQuantInfo(target, quanCommon, oc, kernelCount * ic, backend, reorderedQuantInfo, (float*)kernelsum.get(), ROUND_UP(oc, UNIT), false);
                // pack two int4 to int8
                int leng = weightlen * 2;
                auto srcint4Ptr = (uint8_t*)packedInt8weight.get();
                auto dstint4Ptr = (uint8_t*)weightReordered.get();
                int permuteUnit = UNIT * SRC_UNIT;
                int halfPermuteStride = static_cast<int32_t>(permuteUnit / 2);
                for (int i = 0; i < leng / permuteUnit; ++i) {
                    auto src0 = srcint4Ptr + i * permuteUnit;
                    auto dst0 = dstint4Ptr + i * halfPermuteStride;
                    for (int j = 0; j < halfPermuteStride; ++j) {
                        int s0 = src0[j];
                        int s1 = src0[j + halfPermuteStride];
                        int d = (s0) * 16 + (s1);
                        dst0[j] = d;
                    }
                }
            } else {
                AutoStorage<int8_t> kernelsum(blockNum * ROUND_UP(oc, UNIT) * QUANT_INFO_BYTES);
                if (kernelsum.get() == nullptr) {
                    MNN_ERROR("Kernel sum buffer allocated error\n");
                    return -1;
                }
                reorderWeight((uint8_t*)weightReordered.get(), (uint8_t*)quanCommon->weight.get(), info, 0, (float*)kernelsum.get(), gcore->MNNSumWeightInt8);
                _computeReorderQuantInfo(target, quanCommon, oc, kernelCount * ic, backend, reorderedQuantInfo, (float*)kernelsum.get(), ROUND_UP(oc, UNIT), true);
            }
        }
        /* 2. put weight and quantInfo together */
        int32_t params[6] = {shape[0], shape[1], shape[2], shape[3], shape[4], quantlen / (2 * QUANT_INFO_BYTES * blockNum)};
        ConvInt8TiledExecutor::packWeightAndQuantInfo(target->mWeightInt8->host<int8_t>(), (int8_t*)weightReordered.get(), reorderedQuantInfo.get(), params, QUANT_INFO_BYTES);

        return 0;
    };
    static_cast<CPUBackend*>(backend)->enqueueTask(std::move(function));
}

DenseConvInt8TiledExecutor::DenseConvInt8TiledExecutor(Backend* backend, const Op* op, std::shared_ptr<ResourceInt8> res) : ConvInt8TiledExecutor(backend, op, res) {
    // offline quant
    auto convOp = op->main_as_Convolution2D();
    auto core = static_cast<CPUBackend*>(backend)->int8Functions();
    auto gcore = static_cast<CPUBackend*>(backend)->functions();
    int pack = gcore->pack;
    int ic = convOp->common()->inputCount();
    int oc = convOp->common()->outputCount();
    int kernelCount = convOp->common()->kernelX() * convOp->common()->kernelY();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int lU = UP_DIV(ic, SRC_UNIT) * kernelCount;
    
    std::vector<int> shape = {1, UP_DIV(oc, UNIT), lU, UNIT, SRC_UNIT};
    int weightlen = shape[0] * shape[1] * shape[2] * shape[3] * shape[4];
    auto quantlen = mResourceInt8->mOriginScale->size();
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
    auto resourceInt8 = mResourceInt8;
    int LSize = convOp->common()->inputCount() * convOp->common()->kernelX() * convOp->common()->kernelY();
    int ocUp4 = ROUND_UP(oc, gcore->pack);
    int bytes = gcore->bytes;
    std::shared_ptr<uint8_t> tmpWeightStorage(new uint8_t[mResourceInt8->mWeightInt8->size()], [](void* ptr) {
        delete [] (uint8_t*)ptr;
    });
    ::memcpy(tmpWeightStorage.get(), mResourceInt8->mWeightInt8->host<uint8_t>(), mResourceInt8->mWeightInt8->size());
    mResourceInt8->mWeightInt8.reset(Tensor::createDevice<uint8_t>({weightlen + quantlen}));
    auto allocSuc = backend->onAcquireBuffer(mResourceInt8->mWeightInt8.get(), Backend::STATIC);
    if (!allocSuc) {
        MNN_ERROR("Buffer alloc error!\n");
        return;
    }
    auto reorderFunction = [tmpWeightStorage, weightlen, oc, ic, kernelCount, UNIT, SRC_UNIT, shape, resourceInt8, bytes, LSize, ocUp4, quantlen]()
    {
        AutoStorage<uint8_t> weightReordered(weightlen);
        if (!weightReordered.get()) {
            MNN_ERROR("Memory not enough for quant model weight reorder\n");
            return -1;
        }
        int32_t info[6] = {1, oc, ic, kernelCount, UNIT, SRC_UNIT};
        ConvInt8TiledExecutor::reorderWeight((uint8_t*)weightReordered.get(), tmpWeightStorage.get(), info, 0);

        auto alphaPtr = resourceInt8->mOriginScale->host<float>();
        auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(alphaPtr) + ocUp4 * bytes);

        // Load quant scale and bias
        auto wZero = resourceInt8->mWeightQuantZero->host<int32_t>(); // has packed to outputUp4
        auto wScale = resourceInt8->mOriginScale->host<float>();
        int h = ocUp4;
        MNN_ASSERT(4 == bytes);
        for (int i=0; i< h; ++i) {
            biasPtr[i] = (-1.f) * wZero[i] * wScale[i];
        }
        int32_t params[6] = {shape[0], shape[1], shape[2], shape[3], shape[4], quantlen/ (2 * QUANT_INFO_BYTES * 1)};
        ConvInt8TiledExecutor::packWeightAndQuantInfo(resourceInt8->mWeightInt8->host<int8_t>(), (int8_t*)weightReordered.get(), resourceInt8->mOriginScale->host<int8_t>(), params, QUANT_INFO_BYTES);
        return 0;
    };
    static_cast<CPUBackend*>(backend)->enqueueTask(reorderFunction);
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
#ifdef MNN_KLEIDIAI_ENABLED
    exe->mAccelType = this->mAccelType;
#endif
    *dst = exe;
    return true;
}

void DenseConvInt8TiledExecutor::getPackParameter(int* Unit, int* srcUnit, int* DestUnit, const CoreInt8Functions* core) {
    core->MNNGetGemmUnit(Unit, srcUnit, DestUnit);
}


ErrorCode DenseConvInt8TiledExecutor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // default option
    mUseBatchQuan = false;
    mQuantFirst = true;
    auto option = static_cast<CPUBackend*>(backend())->getRuntime()->hint().dynamicQuantOption;
    int batch = inputs[0]->batch();
    int inC   = inputs[0]->channel();
    auto output = outputs[0];
    int inputPlane  = batch * inputs[0]->width() * inputs[0]->height();
    auto planeSize = output->width() * output->height() * output->batch();
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    auto gcore =static_cast<CPUBackend*>(backend())->functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int kernelCount = mCommon->kernelY() * mCommon->kernelX();
    bool fastway = (kernelCount == 1) && (output->width() == inputs[0]->width()) && (output->height() == inputs[0]->height()) && (mCommon->strideX() * mCommon->strideY()) == 1;
    if (inputPlane > 1) {
        mUseBatchQuan = true;
    }
    if (!fastway) { // general conv
        mQuantFirst = false;
        if (planeSize > 1) {
            mUseBatchQuan = true;
        }
        if (option == 1) { // lowest level.
            mQuantFirst = true;
            mUseBatchQuan = false;
        }
    }
    
    float weightBytes = mResourceInt8->mActBits == 4 ? 0.5 : 1;
    mBlockNum = mResourceInt8->mBlockNum;

#ifdef MNN_KLEIDIAI_ENABLED
    KleidiAI& kai = KleidiAI::getInstance();
    if(mResourceInt8->mDynamicQuant && mResourceInt8->mActBits == 4 && kai.canAccelerate(mAccelType)) {
        MNN_ASSERT(kai.isLoaded(mAccelType));
        const size_t m = inputs[0]->batch(); //lhs vector number.
        const size_t n = outputs[0]->channel(); //rhs vector number.
        const size_t k = inputs[0]->channel(); //vector size.
        const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;

        int packedSize = kai.getLhsQuantedPackedSize(mAccelType, m, k, blkSize);
        int elementSize = kai.isHalf() ? sizeof(__fp16) : sizeof(float);
        if(m > 1 && !kai.isLinear()) {
            int srcSize = m * k * elementSize;
            int dstSize = m * n * elementSize;
            int extraSize = srcSize > dstSize ? srcSize : dstSize;
            packedSize += extraSize;
        }

        //Split mTempIm2ColBuffer as two parts for linear/tile transfer:
        //Part0: Lhs_packed.
        //Part1: Lhs/Dst before transfer.
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
    bool useExtraScale = true;
    if (mResourceInt8->mDynamicQuant == false) {
        mMutableResource->updateInputOutputScale(TensorUtils::getQuantInfo(inputs[0]), TensorUtils::getQuantInfo(outputs[0]));
        if (mMutableResource->mResource->mUseConvQuan) {
            useExtraScale = false;
        }
        CPUConvolution::onResize(inputs, outputs);
        ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY, gcore, core);
        mBlockNum = 1;
        mQuantFirst = true;
        mUseBatchQuan = false;
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
    }

    // input scale buffer
    const int threads = static_cast<CPUBackend*>(backend())->threadNumber();

    // Im2col info
    int im2colBytes = 1;
    const int L2Size = 2048;
    int tileLimitByC = UP_DIV(L2Size, mIm2ColParamter.kernelCountUnit * SRC_UNIT);
    
    if (mQuantFirst == false) {
        im2colBytes = gcore->bytes;
        tileLimitByC = 1;
    }
    int ic = inputs[0]->channel();
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
    int im2colBuffSize = DST_XUNIT * mIm2ColCount * icDiv4KernelCount * SRC_UNIT;
    
    bool toReorderInputBuffer = (mResourceInt8->mDynamicQuant && mBlockNum > 1 && kernelCount > 1);
    if (!mSplitByOc) {
        mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({threads, im2colBuffSize * im2colBytes}));
        mTempSrcSum.resize(threads * mBlockNum * DST_XUNIT * mIm2ColCount * 4); // Use 4 bytes to save kernel sum.
        if (toReorderInputBuffer) {
            mInputReorderBuffer = bufferAlloc->alloc(threads * im2colBuffSize);
            if (mInputReorderBuffer.invalid()) {
                return OUT_OF_MEMORY;
            }
        }
    } else {
        mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({mTileCount, im2colBuffSize * im2colBytes}));
        mTempSrcSum.resize(mTileCount * mBlockNum * DST_XUNIT * mIm2ColCount * 4); // Use 4 bytes to save kernel sum.
        if (toReorderInputBuffer) {
            mInputReorderBuffer = bufferAlloc->alloc(mTileCount * im2colBuffSize);
            if (mInputReorderBuffer.invalid()) {
                return OUT_OF_MEMORY;
            }
        }
    }
    auto success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (useExtraScale) {
        // dequant_scale: area * sizeof(float)
        // quant_scale:   area * sizeof(float)
        // absmax:        [threads, area*core->bytes)]
        int size = DST_XUNIT * mIm2ColCount * QUANT_INFO_BYTES;
        if (mUseBatchQuan) {
            if (mQuantFirst == false) {
                size = 2 * mIm2ColCount * DST_XUNIT * QUANT_INFO_BYTES + mIm2ColCount * DST_XUNIT * gcore->bytes;
            } else {
                size = 2 * inputPlane * QUANT_INFO_BYTES + inputPlane * gcore->bytes;
            }
        }
        mBatchQuantInfo.reset(Tensor::createDevice<int8_t>({threads, size}));
        
        success &= backend()->onAcquireBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
    }

    
    if (!success || mBlitInfo.invalid()) {
        return OUT_OF_MEMORY;
    }
    if (false == mResourceInt8->mDynamicQuant) {
        bufferAlloc->free(mBlitInfo);
        backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
        if (useExtraScale) {
            backend()->onReleaseBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
        }
        return NO_ERROR;
    }

    // set im2col tensor info
    if (mQuantFirst) {
        mQuantInput.reset((Tensor::createDevice<int8_t>({batch, mIm2ColParamter.ih, mIm2ColParamter.iw, ROUND_UP(inC, gcore->pack)})));
    } else if (!mSplitByOc){
        mQuantInput.reset((Tensor::createDevice<int8_t>({threads, im2colBuffSize * 1})));
        // mIm2ColParamter.ic = inC;
    } else {
        mQuantInput.reset((Tensor::createDevice<int8_t>({mTileCount, im2colBuffSize * 1})));
    }
    success &= backend()->onAcquireBuffer(mQuantInput.get(), Backend::DYNAMIC);
    // set dynamic quant buffer
    
    // set compute buffer
    if (!mUseBatchQuan) {
        mTempMaxMinValueBuffer.reset(Tensor::createDevice<uint8_t>({threads, 2 * gcore->bytes}));
        if (mQuantFirst) {
            mDynamicBias.reset(Tensor::createDevice<uint8_t>({ocUp4 * 4}));
        } else {
            mDynamicBias.reset(Tensor::createDevice<uint8_t>({threads, ocUp4 * 4}));
        }
        success &= backend()->onAcquireBuffer(mDynamicBias.get(), Backend::DYNAMIC);
        success &= backend()->onAcquireBuffer(mTempMaxMinValueBuffer.get(), Backend::DYNAMIC);
    }
    mAccumBuffer.reset(Tensor::createDevice<int32_t>({threads, DST_XUNIT * ocUp4}));
    success &= backend()->onAcquireBuffer(mAccumBuffer.get(), Backend::DYNAMIC);

    if (!success) {
        return OUT_OF_MEMORY;
    }
    bufferAlloc->free(mBlitInfo);
    if (toReorderInputBuffer) {
        bufferAlloc->free(mInputReorderBuffer);
    }
    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mQuantInput.get(), Backend::DYNAMIC);

    if (mUseBatchQuan == false) {
        backend()->onReleaseBuffer(mDynamicBias.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mTempMaxMinValueBuffer.get(), Backend::DYNAMIC);
    }
    backend()->onReleaseBuffer(mAccumBuffer.get(), Backend::DYNAMIC);
    
    return NO_ERROR;
}

ErrorCode DenseConvInt8TiledExecutor::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    auto gcore = static_cast<CPUBackend*>(backend())->functions();

#ifdef MNN_KLEIDIAI_ENABLED
    KleidiAI& kai = KleidiAI::getInstance();
    if(mResourceInt8->mDynamicQuant && mResourceInt8->mActBits == 4 && kai.canAccelerate(mAccelType)) {
        MNN_ASSERT(kai.isLoaded(mAccelType));
        const size_t m = input->batch(); //lhs vector number.
        const size_t n = output->channel(); //rhs vector number.
        const size_t k = input->channel(); //vector size.
        const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;

        bool bHalf = kai.isHalf();
        size_t elementSize = bHalf ? sizeof(__fp16) : sizeof(float);
        size_t lhsPackedSize = kai.getLhsQuantedPackedSize(mAccelType, m, k, blkSize);

        auto lhs = input->host<uint8_t>();
        auto lhsPacked = mTempIm2ColBuffer->host<int8_t>();
        auto rhsPacked = mResourceInt8->mWeightInt8->host<uint8_t>();
        auto dst = output->host<uint8_t>();

        uint8_t *linearLhs, *linearDst;
        if(m > 1 && !kai.isLinear()) {
            linearLhs = (uint8_t *)lhsPacked + lhsPackedSize;
            linearDst = linearLhs;
        } else {
            linearLhs = lhs;
            linearDst = dst;
        }

        int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
        int threadNeed, vecPerThread;

        //Dynamic quant pack lhs.
        if(m == 1) {
            kai.runLhsQuantPack(mAccelType, 1, k, blkSize, 1, linearLhs, lhsPacked);
        } else {
            if(!kai.isLinear()) {
                if(bHalf) {
                    KleidiAIUtil::transferNC4HW4ToNCHW((__fp16 *)lhs, (__fp16 *)linearLhs, m, k);
                } else {
                    KleidiAIUtil::transferNC4HW4ToNCHW((float *)lhs, (float *)linearLhs, m, k);
                }
            }

            vecPerThread = kai.getVecNumPerThread(m, threadNum, kai.getMr(mAccelType, m));
            threadNeed = m % vecPerThread == 0 ? m / vecPerThread : (m / vecPerThread + 1);
            size_t srcStride = vecPerThread * k * elementSize;

            auto BatchDynamicQuant = [=, &kai](int tId) {
                auto threadSrc = linearLhs + tId * srcStride;
                auto threadDst = lhsPacked + kai.getLhsQuantedPackedOffset(mAccelType, m, tId * vecPerThread, k, blkSize);
                int vecNum = (tId == threadNeed - 1) ? (m - vecPerThread * tId) : vecPerThread; //Last threadN may less than vecPerThread.
                kai.runLhsQuantPack(mAccelType, vecNum, k, blkSize, kai.getMr(mAccelType, m), threadSrc, threadDst);
            };

            MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
                BatchDynamicQuant((int)tId);
            }
            MNN_CONCURRENCY_END();
        }

        //Run matmul.
        if(kai.bSupportSme2() && mAccelType == KleidiAI::AccelType::QI4_SYM_CHNLQT) {
            //SME prefer running on single thread to obtain better performance/power consumption ratio.
            threadNum = 1;
        }

        vecPerThread = kai.getVecNumPerThread(n, threadNum, kai.getNStep(mAccelType));
        threadNeed = n % vecPerThread == 0 ? n / vecPerThread : (n / vecPerThread + 1);

        auto ThreadFunction = [=, &kai](int tId) {
            auto threadRhsPacked = rhsPacked + kai.getRhsPackedOffset(mAccelType, tId * vecPerThread, k, blkSize);
            auto threadDst = linearDst + kai.getDstOffset(0, tId * vecPerThread, n, elementSize);
            int vecNum = (tId == threadNeed - 1) ? (n - vecPerThread * tId) : vecPerThread; //Last threadN may less than vecPerThread.
            float scalarMax = bHalf ? FLT16_MAX : FLT_MAX;
            kai.runMatmul(mAccelType, m, vecNum, k, blkSize, lhsPacked, threadRhsPacked, threadDst, n * elementSize, elementSize, scalarMax, -scalarMax);
        };

        MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
            ThreadFunction((int)tId);
        }
        MNN_CONCURRENCY_END();

        if(m > 1 && !kai.isLinear()) {
            if(bHalf) {
                KleidiAIUtil::transferNCHWToNC4HW4((__fp16 *)linearDst, (__fp16 *)dst, m, n);
            } else {
                KleidiAIUtil::transferNCHWToNC4HW4((float *)linearDst, (float *)dst, m, n);
            }
        }

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
    const int blockL                 = kernelCountUnitDouble / mBlockNum; // source depthQuad for each block.
    const int kxky                   = mIm2ColParamter.kernelX * mIm2ColParamter.kernelY;
    const int blocklu                = blockL / kxky;
    float weightBytes                = 1.f;
    int weight_step_Y                = weightBytes * (UNIT__ * SRC_UNIT);
    int src_step_Y                   = DST_XUNIT * SRC_UNIT;
    int inputPlane                   = batch * input->width() * input->height();

    auto im2colPtr           = mTempIm2ColBuffer->host<int8_t>();
    if (SRC_UNIT > PackUnit) {
        memset(im2colPtr, 0, mTempIm2ColBuffer->size());
    }
    const auto weightDataPtr = mResourceInt8->mWeightInt8->host<int8_t>();
    auto srcKernelSumPtr     = mTempSrcSum.data();

    auto outputDataPtr = output->host<int8_t>();
    uint8_t* biasPtr = nullptr;
    int32_t inputZeroPoint = 0;
    if (nullptr != mMutableResource.get()) {
        biasPtr       = mMutableResource->mBiasFloat->host<uint8_t>();
        inputZeroPoint  = mMutableResource->mInputZeroPoint;
        if (mBatchQuantInfo.get()) {
            float scalein = TensorUtils::getQuantInfo(inputs[0])[0];
            float scaleou = TensorUtils::getQuantInfo(outputs[0])[0];
            for (int i = 0; i < DST_XUNIT * mIm2ColCount; ++i) {
                mBatchQuantInfo->host<float>()[i] = scalein / scaleou;
            }
        }
    }

    auto SingleDynamicQuant = [&] (uint8_t* floatPtr, int32_t& inputZero, uint8_t* inputDequantScale, uint8_t* matmulBiasPtr, int inputsize, int threads, uint8_t* maxMinValPtr, int8_t* int8ptr) {
        float quantscale    = 0.f;
        float dequantscale  = 0.f;
        float zeropoint       = 0;

         /* Count max and min value to compute input scale and zeropoint */
        auto findMaxMinValueFunction = [&]() {
            auto perThreadWorkCount = inputsize;
            auto minValPtrTid = reinterpret_cast<float*>(maxMinValPtr);
            auto maxValPtrTid = reinterpret_cast<float*>(maxMinValPtr + gcore->bytes);
            auto inputDataPtrTid = reinterpret_cast<float*>(floatPtr);
            gcore->MNNCountMaxMinValue(inputDataPtrTid, minValPtrTid, maxValPtrTid, inputsize);
        };
        findMaxMinValueFunction();
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
        if (mIm2ColParamter.padX > 0 || mIm2ColParamter.padY > 0) { // Ensure "0.0f" included in range.
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
            zeropoint = (-1 * maxVal)-128;
            quantscale = 1.0f;
            dequantscale = 1.0f;
        } else {
            quantscale = 255.0f / range;
            dequantscale = range / 255.0f;
            zeropoint = roundf(-minVal * 255.f / range) - 128.0f;
        }
        auto sizeDiv = UP_DIV(inputsize, PackUnit);
        mQuantFunc((float*)floatPtr , int8ptr, sizeDiv, &quantscale, -128, 127, &zeropoint, 0);

        /* bias float */
        int offset = 0;
        auto scale_ = (float*)inputDequantScale;
        auto unitsize = mBatchQuantInfo->length(1) / QUANT_INFO_BYTES;
        for (int i = 0; i < unitsize; ++i) {
            scale_[i] = dequantscale;
        }
        auto biasfp32 = mResourceInt8->mOriginBias->host<float>();
        float zerofp32 = (zeropoint + offset) * dequantscale;

        gcore->MNNDynamicUpdateConvBiasScale((float*)matmulBiasPtr, biasfp32, mResourceInt8->mWeightKernelSum->host<float>(), &zerofp32, UP_DIV(output->channel(), 4));
        // Move step for A and B for each block computing

        inputZero = zeropoint;
        biasPtr = matmulBiasPtr;
    };

    auto BatchDynamicQuant = [&](uint8_t* floatPtr, int32_t& inputZero, uint8_t* inputDequantScale, int LU, int EP, int LP, int32_t availableThreads, int8_t* dstInt8) {
        // Allocate input max/sum/dequant/quant buffer
        auto quantPtr = inputDequantScale + EP * QUANT_INFO_BYTES;
        auto maxPtr = inputDequantScale + 2 * EP * QUANT_INFO_BYTES;
        
        // compute sum and absmax
        int divlu = UP_DIV(LU, availableThreads);
        MNN_CONCURRENCY_BEGIN (tId, availableThreads) {
            auto exeLU = ALIMIN(divlu, LU - tId * divlu);
            auto batchMax = reinterpret_cast<float*>(maxPtr + tId * EP * gcore->bytes);
            auto ptr_     = reinterpret_cast<float*>(floatPtr + tId * divlu * gcore->bytes * EP * LP);
            gcore->MNNAbsMax(ptr_, batchMax, exeLU, EP, LP);
        } MNN_CONCURRENCY_END();
        

        // Compute quant scale
        gcore->MNNQuantScale((float*)maxPtr, (float*)quantPtr, (float*)inputDequantScale, availableThreads, EP);

        // quant
        auto scale_ptr = reinterpret_cast<float*>(quantPtr);
        gcore->MNNDynamicQuant((float*)floatPtr, dstInt8, scale_ptr, LU, EP, LP);
        inputZero = 0;
    };

    ssize_t oneScale = mUseBatchQuan ? 0 : 1;
    if (mUseBatchQuan) {
        biasPtr = mResourceInt8->mOriginBias->host<uint8_t>();
    }
    int8_t* inputDataPtr = nullptr; // Matmul input.
    auto im2colSrc = input->host<uint8_t>(); // if not quant first, im2colSrc is original float input data.
    auto inputsize = UP_DIV(input->channel(), PackUnit)  * PackUnit * batch * input->height() * input->width();
    if (mQuantFirst) { // quant first, then im2col
        if (mUseBatchQuan) {
            int icDiv4 = UP_DIV(input->channel(), PackUnit);
            int exethreads = 1;
            if (icDiv4 > mThreadNums && inputPlane > 256) {
                exethreads = mThreadNums;
            }
            BatchDynamicQuant(input->host<uint8_t>(), inputZeroPoint, mBatchQuantInfo->host<uint8_t>(), icDiv4, inputPlane, PackUnit, exethreads, mQuantInput->host<int8_t>());
            inputDataPtr = mQuantInput->host<int8_t>();
        } else if (mResourceInt8->mDynamicQuant) {
            SingleDynamicQuant(input->host<uint8_t>(), inputZeroPoint, mBatchQuantInfo->host<uint8_t>(), mDynamicBias->host<uint8_t>(),  inputsize, 1, mTempMaxMinValueBuffer->host<uint8_t>(), mQuantInput->host<int8_t>());
            inputDataPtr = mQuantInput->host<int8_t>();
        } else {
            // offline quant.
            inputDataPtr = input->host<int8_t>();
        }
        im2colSrc = (uint8_t*)inputDataPtr;
    }

    if (mResourceInt8->mActBits == 4) {
        weightBytes   = 0.5;
        weight_step_Y *= 0.5;
    }
    int blockunit = ocUp4 * 2 * QUANT_INFO_BYTES + blockL * weight_step_Y * UP_DIV(output->channel(), UNIT__);
    auto inputchannel = input->channel();
    SumByAxisParams sumParams;
    sumParams.oneScale = oneScale;
    sumParams.SRC_UNIT = SRC_UNIT;
    sumParams.blockNum = mBlockNum;
    sumParams.DST_XUNIT = DST_XUNIT;
    sumParams.col_buffer_unit_size = col_buffer_unit_size;
    sumParams.kernelCountUnitDouble = kernelCountUnitDouble;
    sumParams.valid = inputchannel % SRC_UNIT;
    sumParams.kernelxy = kxky;
    sumParams.LU = UP_DIV(inputchannel, SRC_UNIT);
    
    int im2colBytes = mQuantFirst == true ? 1 : gcore->bytes;
    bool toReorderInput = (mResourceInt8->mDynamicQuant && mBlockNum > 1 && kxky > 1);

    auto tileSplitFunction = [&](int tId, int eStartIndex, int eEndIndex, int estep) {
        auto ocDivThread = ocDiv4;
        float* reluPtr = mResourceInt8->mReluThreshold.data();
        QuanPostTreatParameters quanParam;
        quanParam.blockNum = mBlockNum;
        float* accumbuff = nullptr;
        uint8_t* extraScale = nullptr;
        uint8_t* ptrExtraScale = nullptr;
        if (mBatchQuantInfo.get() && mQuantFirst) {
            extraScale = mBatchQuantInfo->host<uint8_t>();
            ptrExtraScale = extraScale;
        }
        AutoStorage<int8_t> reorderBuffer;
        if (mBlockNum > 1) {
            accumbuff = reinterpret_cast<float*>(mAccumBuffer->host<int8_t>() + tId * mAccumBuffer->stride(0) * sizeof(int32_t));
        }
        
#ifdef MNN_USE_SSE
        if (mResourceInt8->mDynamicQuant) {
            quanParam.extraBias = mResourceInt8->mWeightKernelSum->host<float>();
        }
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
        auto biasFloatTid = reinterpret_cast<float*>(biasPtr);
        auto weightPtrTid = weightDataPtr;
        if (mBlockNum == 1) {
            quanParam.biasFloat = biasFloatTid;
        }
        // auto im2colPtr           = mTempIm2ColBuffer->host<int8_t>();
        auto colAddr        = im2colPtr + tId * mTempIm2ColBuffer->stride(0);
        auto srcPtr     = (int8_t const **)(mBlitInfo.ptr() + tId * mBlitInfoStride.first);
        auto el         = (int32_t *)(srcPtr + mBlitInfoStride.second);
        auto xKernelSumPtrTid = reinterpret_cast<float*>(srcKernelSumPtr + tId * mBlockNum * DST_XUNIT * mIm2ColCount * 4);

        int32_t info[5];
        info[1] = mIm2ColParamter.iw * mIm2ColParamter.ih * batch;
        info[2] = static_cast<int32_t>(col_buffer_unit_size);
        info[3] = mIm2ColParamter.strideX;
        for (int tIndex = eStartIndex; tIndex < eEndIndex; tIndex += estep) {
            const int xIndexStart  = tIndex * DST_XUNIT * mIm2ColCount;
            auto outputInTilePtr = outputDataPtr + xIndexStart * PackUnit * dstBytes;
            int realDstCount = ALIMIN(plane - xIndexStart, DST_XUNIT * mIm2ColCount);
            ptrExtraScale = mUseBatchQuan ? (extraScale + xIndexStart * QUANT_INFO_BYTES) : extraScale;
            // im2col
            auto res = ConvolutionTiledExecutor::turnIm2ColToBlitInfo((const float**)srcPtr, el, xIndexStart, realDstCount, mIm2ColParamter, (uint8_t*)im2colSrc, im2colBytes);
            int number = res.first;
            bool needZero = res.second;
            if (needZero && mQuantFirst) {
#ifdef MNN_USE_SSE
                ::memset(colAddr, inputZeroPoint + 128, col_buffer_size);
#else
                ::memset(colAddr, inputZeroPoint, col_buffer_size);
#endif
            } else if (needZero) {
                ::memset(colAddr, 0, mTempIm2ColBuffer->stride(0));
            }
            info[0] = number;
            info[4] = realDstCount;
            int8_t* colAddrTemp = colAddr;
            if (mQuantFirst && number > 0) {
                blitProc(colAddr, srcPtr, info, el);
                colAddrTemp = colAddr;
            } else if (number > 0) {
                if (SRC_UNIT > PackUnit && !needZero) {
                    memset(colAddr, 0, mTempIm2ColBuffer->stride(0));
                }
                info[2] = realDstCount;
                gcore->MNNGeneralIm2Col((float*)colAddr, (float const**)srcPtr, info, el, SRC_UNIT, PackUnit); // colAddr: [lu, realDstCount, lp]
            }
            if (!mQuantFirst) {
                auto ptrInputscale = mBatchQuantInfo->host<uint8_t>() + tId * mBatchQuantInfo->stride(0);
                if (mUseBatchQuan) {
                    BatchDynamicQuant((uint8_t*)colAddr, inputZeroPoint, ptrInputscale, kernelCountUnitDouble, realDstCount, SRC_UNIT, 1, mQuantInput->host<int8_t>() + tId * mQuantInput->stride(0));
                } else {
                    biasFloatTid = reinterpret_cast<float*>(mDynamicBias->host<uint8_t>() + tId * mDynamicBias->stride(0));
                    auto maxMinPtr = mTempMaxMinValueBuffer->host<uint8_t>() + tId * mTempMaxMinValueBuffer->stride(0);
                    SingleDynamicQuant((uint8_t*)colAddr, inputZeroPoint, ptrInputscale, (uint8_t*)biasFloatTid, kernelCountUnitDouble*realDstCount*SRC_UNIT, 1, maxMinPtr, mQuantInput->host<int8_t>() + tId * mQuantInput->stride(0));
                    if (mBlockNum == 1) {
                        quanParam.biasFloat = biasFloatTid;
                    }
                }
                extraScale = ptrInputscale;
                ptrExtraScale = ptrInputscale;
                colAddrTemp = mQuantInput->host<int8_t>() + tId * mQuantInput->stride(0);
            }
            if (toReorderInput) {
                auto dst_ = (int8_t*)mInputReorderBuffer.ptr() + tId * col_buffer_size;
                auto eU = UP_DIV(realDstCount, DST_XUNIT);
                for (int k = 0; k < eU; ++k) {
                    int inside  = blocklu * SRC_UNIT * ALIMIN(realDstCount - k * DST_XUNIT, DST_XUNIT);
                    auto dstbuffer = dst_ + k * col_buffer_unit_size;
                    auto srcbuffer = colAddrTemp + k * col_buffer_unit_size;
                    for (int i = 0; i < mBlockNum; ++i) {
                        for (int j = 0; j < kxky; ++j) {
                            memcpy(dstbuffer + i * kxky * inside + j * inside, srcbuffer + i * inside + j * mBlockNum * inside, inside);
                        }
                    }
                }
                colAddrTemp = (int8_t*)mInputReorderBuffer.ptr() + tId * col_buffer_size;
            }
            if (mResourceInt8->mWeightAsymmetricQuant) {
                MNN_ASSERT(mBatchQuantInfo.get() && mBatchQuantInfo->host<float>());
                mSumByAxisLFunc(xKernelSumPtrTid, colAddrTemp, (float*)ptrExtraScale, realDstCount, sumParams);
            }
            auto ptrX = xKernelSumPtrTid;
            do {
                int step = ALIMIN(DST_XUNIT, realDstCount);
                quanParam.extraScale = (float*)ptrExtraScale;
                if (mBlockNum > 1) {
                    memset(accumbuff, 0, ocUp4 * 4 * DST_XUNIT);
                    quanParam.accumBuffer = accumbuff;
                }
                int8_t* saveResult = nullptr;
                for (int k = 0; k < mBlockNum; ++k) {
                    quanParam.biasFloat = nullptr;
                    quanParam.fp32minmax = nullptr;
                    if (k == 0) {
                        quanParam.biasFloat = biasFloatTid;
                    }
                    if (k == mBlockNum - 1) {
                        quanParam.fp32minmax = reluPtr;
                        saveResult = outputInTilePtr;
                    }
                    quanParam.srcKernelSum = ptrX + k * step;

                    mGemmKernel(saveResult, colAddrTemp + k * blockL * step * SRC_UNIT, weightPtrTid + k * blockunit, blockL, dstZStep * dstBytes, ocDivThread, &quanParam, step);
                }
                ptrX += (step * mBlockNum);
                realDstCount-=step;
                outputInTilePtr += DST_XUNIT * PackUnit * dstBytes;
                colAddrTemp += col_buffer_unit_size;
                ptrExtraScale = mUseBatchQuan ? (ptrExtraScale + step * QUANT_INFO_BYTES) : extraScale;
            } while(realDstCount > 0);
        }
    };
    auto ocSplitFunction = [&](int threads) { // Thread split by OC
        auto colAddr           = mTempIm2ColBuffer->host<int8_t>();
        auto srcPtr     = (int8_t const **)(mBlitInfo.ptr());
        auto el         = (int32_t *)(srcPtr + mBlitInfoStride.second);
        auto xKernelSumPtrTid = reinterpret_cast<float*>(srcKernelSumPtr);

        int32_t info[5];
        info[1] = mIm2ColParamter.iw * mIm2ColParamter.ih * batch;
        info[2] = static_cast<int32_t>(col_buffer_unit_size);
        info[3] = mIm2ColParamter.strideX;
        
        float* reluPtr = mResourceInt8->mReluThreshold.data();
        int8_t* matmulInput;
        if (mQuantFirst) { // im2col
            auto res = ConvolutionTiledExecutor::turnIm2ColToBlitInfo((const float**)srcPtr, el, 0, plane, mIm2ColParamter, (uint8_t*)im2colSrc, im2colBytes);
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
            matmulInput = colAddr;
        }
        if (false == mQuantFirst) {
            int realDstCount = plane;
            int start = 0;
            auto im2colDst = colAddr;
            auto ptrInputscale = mBatchQuantInfo->host<uint8_t>();
            auto int8Ptr = mQuantInput->host<int8_t>();
            int sizePacked = 0;
            while (realDstCount > 0) {
                int work = std::min(realDstCount, DST_XUNIT);
                sizePacked += (work * SRC_UNIT * kernelCountUnitDouble);
                auto res = ConvolutionTiledExecutor::turnIm2ColToBlitInfo((const float**)srcPtr, el, start, work, mIm2ColParamter, (uint8_t*)im2colSrc, im2colBytes);
                int number = res.first;
                bool needZero = res.second;
                if (needZero) {
                    ::memset(im2colDst, 0, col_buffer_unit_size * gcore->bytes);
                }
                info[0] = number;
                info[2] = work;
                if (number > 0) { // im2col
                    gcore->MNNGeneralIm2Col((float*)im2colDst, (float const**)srcPtr, info, el, SRC_UNIT, PackUnit); // colAddr: [lu, realDstCount, lp]
                }
                // batch quant
                if (mUseBatchQuan) {
                    BatchDynamicQuant((uint8_t*)im2colDst, inputZeroPoint, ptrInputscale, kernelCountUnitDouble, work, SRC_UNIT, 1, int8Ptr);
                    ptrInputscale += (work * sizeof(int32_t));
                    int8Ptr += col_buffer_unit_size;
                }
                realDstCount -= work;
                start += work;
                im2colDst += (col_buffer_unit_size * gcore->bytes);
            }
            if (!mUseBatchQuan) {
                SingleDynamicQuant((uint8_t*)colAddr, inputZeroPoint, ptrInputscale, mDynamicBias->host<uint8_t>(), sizePacked, 1, mTempMaxMinValueBuffer->host<uint8_t>(), mQuantInput->host<int8_t>());
            }
            matmulInput = mQuantInput->host<int8_t>();
        }
        if (toReorderInput) {
            auto dst_ = (int8_t*)mInputReorderBuffer.ptr();
            auto eU = UP_DIV(plane, DST_XUNIT);
            for (int k = 0; k < eU; ++k) {
                int inside  = blocklu * SRC_UNIT * ALIMIN(DST_XUNIT, plane - k * DST_XUNIT);
                auto dstbuffer = dst_ + k * col_buffer_unit_size;
                auto srcbuffer = matmulInput + k * col_buffer_unit_size;
                for (int i = 0; i < mBlockNum; ++i) {
                    for (int j = 0; j < kxky; ++j) {
                        memcpy(dstbuffer + i * kxky * inside + j * inside, srcbuffer + i * inside + j * mBlockNum * inside, inside);
                    }
                }
            }
            matmulInput = (int8_t*)mInputReorderBuffer.ptr();
        }

        if (mResourceInt8->mWeightAsymmetricQuant) {
            MNN_ASSERT(mBatchQuantInfo.get() && mBatchQuantInfo->host<float>());
            mSumByAxisLFunc(xKernelSumPtrTid, matmulInput, mBatchQuantInfo->host<float>(), plane, sumParams);
        }
        
        MNN_CONCURRENCY_BEGIN(tId, threads) {
            int ocIndex = PackUnit * mDivides[tId];
            auto ocDivThread = ALIMIN(mDivides[tId + 1] - mDivides[tId], ocDiv4 - mDivides[tId]);
            if (ocIndex < ocUp4) {
                auto colAddrTemp = matmulInput;
                QuanPostTreatParameters quanParam;
                quanParam.blockNum = mBlockNum;
                uint8_t* extraScale = nullptr; // input scale for batch dynamic quant.
                uint8_t* ptrExtraScale = nullptr;
                float* accumbuff = nullptr;
                if (mBatchQuantInfo.get()) {
                    extraScale = mBatchQuantInfo->host<uint8_t>();
                    ptrExtraScale = extraScale;
                }
                if (mBlockNum > 1) {
                    accumbuff = reinterpret_cast<float*>(mAccumBuffer->host<int8_t>() + tId * mAccumBuffer->stride(0) * sizeof(int32_t));
                }
#ifdef MNN_USE_SSE
                if (mResourceInt8->mDynamicQuant) {
                    quanParam.extraBias = mResourceInt8->mWeightKernelSum->host<float>() + ocIndex;
                }
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
                const auto weightPtrTid = weightDataPtr + static_cast<int32_t>(ocIndex * blockL * SRC_UNIT * weightBytes + ocIndex * 2 * QUANT_INFO_BYTES);
                int realDstCount = plane;
                auto ptrX = xKernelSumPtrTid;
                do {
                    int step = ALIMIN(DST_XUNIT, realDstCount);
                    quanParam.extraScale = (float*)ptrExtraScale;
                    if (mBlockNum > 1) {
                        memset(accumbuff, 0, ocUp4 * 4 * DST_XUNIT);
                        quanParam.accumBuffer = accumbuff;
                    }
                    int8_t* saveResult = nullptr;
                    for (int k = 0; k < mBlockNum; ++k) {
                        quanParam.biasFloat = nullptr;
                        quanParam.fp32minmax = nullptr;
                        if (k == 0) {
                            quanParam.biasFloat = (float*)biasFloatTid;
                        }
                        if (k == mBlockNum - 1) {
                            quanParam.fp32minmax = reluPtr;
                            saveResult = outputInTilePtr;
                        }
                        quanParam.srcKernelSum = ptrX + k * step;
                        mGemmKernel(saveResult, colAddrTemp + k * blockL * step * SRC_UNIT, weightPtrTid + k * blockunit, blockL, dstZStep * dstBytes, ocDivThread, &quanParam, step);
                    }
                    ptrX += (step * mBlockNum);
                    realDstCount-=step;
                    outputInTilePtr += DST_XUNIT * PackUnit * dstBytes;
                    colAddrTemp += col_buffer_unit_size;
                    ptrExtraScale = mUseBatchQuan ? (ptrExtraScale + step * QUANT_INFO_BYTES) : extraScale;
                } while(realDstCount > 0);
            }
        }
        MNN_CONCURRENCY_END();
        
    };
    const int threads = static_cast<CPUBackend*>(backend())->threadNumber();
    if (!mSplitByOc) {
        MNN_CONCURRENCY_BEGIN(tId, threads) {
            if (mDivides[tId + 1] - mDivides[tId] > 0) {
                tileSplitFunction((int)tId, mDivides[tId], mDivides[tId + 1], 1);
            }
        }
        MNN_CONCURRENCY_END();
    } else {
        ocSplitFunction(threads);
    }
    return NO_ERROR;
}





} // namespace MNN
