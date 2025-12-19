#include "ConvTMac.hpp"
#include "core/BufferAllocator.hpp"
#include "core/Concurrency.h"
#include "CommonOptFunction.h"

#define QUANT_INFO_BYTES 4

namespace MNN {
struct TMacCache {
    MemChunk featuremapTable;
    MemChunk inputscale;
    MemChunk inputSum;
};

static void _createResource(TMacResource* resource, std::shared_ptr<ConvolutionCommon::Int8Common> quantCommon, const Convolution2D* conv2d, Backend* backend, bool compute) {
    resource->mBits = quantCommon->originBits;
    // common parameters
    int outputCount = conv2d->common()->outputCount();
    auto core = static_cast<CPUBackend*>(backend)->functions();
    resource->mHp = core->tmacHp;
    int LSize = conv2d->common()->inputCount() * conv2d->common()->kernelX() * conv2d->common()->kernelY();
    int ocUp4 = ROUND_UP(outputCount, resource->mHp);
    MNN_ASSERT(LSize % 4 == 0);
    int LC4 = LSize / 4;

    int dequantCnt = quantCommon->alpha.size();
    if (quantCommon->asymmetric) {
        dequantCnt /= 2;
    }
    int blockNum = dequantCnt / outputCount;
    int scaleSize = blockNum * ocUp4; // aligned size.
    int blockSize = LSize / blockNum;
    resource->mOutputCount = outputCount;
    resource->mBlockSizeC4 = blockSize / 4;
    resource->mBlockNumber = blockNum;
    int originOffset = - (1 << (resource->mBits - 1));
    // Save bias
    resource->mOriginBias.reset(Tensor::createDevice<int32_t>({ocUp4})); // float
    auto success = backend->onAcquireBuffer(resource->mOriginBias.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Alloc bias memory error\n");
        return;
    }
    ::memset(resource->mOriginBias->host<float>(), 0, ocUp4 * sizeof(float));
    if (conv2d->bias()) {
        ::memcpy(resource->mOriginBias->host<float>(), conv2d->bias()->data(), outputCount * sizeof(float));
    }
    // Save weight quant alpha and zero: wf=alpha*wi+zero
    resource->mOriginScale.reset(Tensor::createDevice<uint8_t>({2 * scaleSize * QUANT_INFO_BYTES}));
    success = backend->onAcquireBuffer(resource->mOriginScale.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Alloc denquant alpha, zero memory error\n");
        return;
    }
    auto alphaPtr = resource->mOriginScale->host<float>();
    auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(alphaPtr) + resource->mHp * QUANT_INFO_BYTES);
    ::memset(alphaPtr, 0, 2 * scaleSize * QUANT_INFO_BYTES);
    auto quanInfoPtr = quantCommon->alpha.get();
    int h = quantCommon->alpha.size();
    if (quantCommon->asymmetric) {
        for (int i = 0; i < blockNum; ++i) {
            auto dstAlpha = alphaPtr + 2 * i * resource->mHp;
            auto dstBias  = biasPtr + 2 * i * resource->mHp;
            for (int j = 0; j < outputCount; ++j) {
                int jc = j / resource->mHp;
                int jr = j % resource->mHp;
                int scaleIndex = j * blockNum + i;
                int pos = 2 * jc*blockNum*resource->mHp + jr;
                dstAlpha[pos] = quanInfoPtr[2 * scaleIndex + 1];
                dstBias[pos] = quanInfoPtr[2 * scaleIndex] + (float)originOffset * dstAlpha[pos];
            }
        }
    } else {
        for (int i = 0; i < blockNum; ++i) {
            auto dstAlpha = alphaPtr + 2 * i * resource->mHp;
            auto dstBias  = biasPtr + 2 * i * resource->mHp;
            for (int j = 0; j < outputCount; ++j) {
                int jc = j / resource->mHp;
                int jr = j % resource->mHp;
                int scaleIndex = j * blockNum + i;
                int pos = 2 * jc*blockNum*resource->mHp + jr;
                dstAlpha[pos] = quanInfoPtr[scaleIndex];
                dstBias[pos] = (float)originOffset * dstAlpha[pos];
            }
        }
    }
    auto realWeightData = quantCommon->weight.get();
    int blC4 = blockSize / 4;
    resource->mWeightInt8.reset(Tensor::createDevice<uint8_t>({UP_DIV(outputCount, resource->mHp), resource->mBlockNumber, resource->mBits * blC4, resource->mHp / 2}));
    success = backend->onAcquireBuffer(resource->mWeightInt8.get(), Backend::STATIC);
    ::memset(resource->mWeightInt8->host<void>(), 0, resource->mWeightInt8->usize());
    if (!success) {
        MNN_ERROR("int4 weight acquire buffer error\n");
        return ;
    }
    // TODO: Support 8bit mode
    auto halfhp = resource->mHp / 2;
    for (int j = 0; j < outputCount; ++j) {
        int jC = j / resource->mHp;
        int jR = j % resource->mHp;
        auto dstZ = resource->mWeightInt8->host<uint8_t>() + (jC * resource->mBits * blockNum * blC4 * resource->mHp) / 2 + (jR % halfhp);
        int dstMul = jR < halfhp ? 1 : 16;
        // Int4 is save as v + 8, we need value as v + originOffset
        int offsetDiff = 8 + originOffset;
        for (int k = 0; k < blockNum; ++k) {
            int scaleIndex = k + j * blockNum;
            auto dstBlock = dstZ + k * resource->mBits * blC4 * resource->mHp / 2;
            if (quantCommon->canUseInt4) {
                auto srcK = realWeightData + ((j * blockNum + k) * blockSize) / 2;
                for (int i = 0; i < blC4; ++i) {
                    uint8_t s0 = srcK[2*i+0];
                    uint8_t s1 = srcK[2*i+1];
                    uint8_t s01 = (s0 & 0x0f) - offsetDiff;
                    uint8_t s00 = (s0 >> 4) - offsetDiff;
                    uint8_t s11 = (s1 & 0x0f) - offsetDiff;
                    uint8_t s10 = (s1 >> 4) - offsetDiff;
                    uint8_t res[4];
                    auto dstI = dstBlock + i * resource->mHp / 2;
                    for (int v=0; v<resource->mBits; ++v) {
                        auto a = s00 % 2;
                        auto b = s01 % 2;
                        auto c = s10 % 2;
                        auto d = s11 % 2;
                        res[v] = (a << 3) + (b << 2) + (c << 1) + d;
                        s00 /= 2;
                        s01 /= 2;
                        s10 /= 2;
                        s11 /= 2;
                    }
                    for (int b=0; b<resource->mBits; ++b) {
                        dstI[b * blC4 * resource->mHp / 2] += (dstMul * res[resource->mBits-b-1]);
                    }
                }
            } else {
                auto srcK = realWeightData + ((j * blockNum + k) * blockSize);
                auto bitOffset = (1 << (resource->mBits - 1));
                for (int i = 0; i < blC4; ++i) {
                    uint8_t s00 = (int)srcK[4*i+0] + bitOffset;
                    uint8_t s01 = (int)srcK[4*i+1] + bitOffset;
                    uint8_t s10 = (int)srcK[4*i+2] + bitOffset;
                    uint8_t s11 = (int)srcK[4*i+3] + bitOffset;
                    uint8_t res[4];
                    auto dstI = dstBlock + i * resource->mHp / 2;
                    for (int v=0; v<resource->mBits; ++v) {
                        auto a = s00 % 2;
                        auto b = s01 % 2;
                        auto c = s10 % 2;
                        auto d = s11 % 2;
                        res[v] = (a << 3) + (b << 2) + (c << 1) + d;
                        s00 /= 2;
                        s01 /= 2;
                        s10 /= 2;
                        s11 /= 2;
                    }
                    for (int b=0; b<resource->mBits; ++b) {
                        dstI[b * blC4 * resource->mHp / 2] += (dstMul * res[resource->mBits-b-1]);
                    }
                }
            }
        }
    }
}

ConvTMac::ConvTMac(const Convolution2DCommon *convOp, Backend *b) : CPUConvolution(convOp, b) {
    mCache.reset(new TMacCache);
    mParameters = getPostParameters();
}
ConvTMac::~ ConvTMac() {
    mCache = nullptr;
    mResource = nullptr;
}
bool ConvTMac::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto res = new ConvTMac(op->main_as_Convolution2D()->common(), bn);
    res->mResource = mResource;
    *dst = res;
    return true;
}
ErrorCode ConvTMac::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputSize = inputs[0]->elementSize();
    auto ic = inputs[0]->length(1);
    int eP = 1;
    auto bufferAlloc = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    auto functions = static_cast<CPUBackend*>(backend())->functions();
    auto icC4 = UP_DIV(ic, 4);
    mCache->featuremapTable = bufferAlloc->alloc(eP * static_cast<CPUBackend*>(backend())->threadNumber() * icC4 * 16 * QUANT_INFO_BYTES);
    // Compute featuremapTable
    mCache->inputSum = bufferAlloc->alloc(eP * static_cast<CPUBackend*>(backend())->threadNumber() * mResource->mBlockNumber * QUANT_INFO_BYTES);
    bufferAlloc->free(mCache->inputSum);
    bufferAlloc->free(mCache->featuremapTable);

    // functions
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    mQuantFunc = core->MNNFloat2Int8;
    return NO_ERROR;
}
ErrorCode ConvTMac::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputSize = inputs[0]->elementSize();
    auto ic = inputs[0]->length(1);
    auto oc = outputs[0]->length(1);
    auto planeSize = inputs[0]->length(0);
    for (int i=2; i<inputs[0]->dimensions(); ++i) {
        planeSize *= inputs[0]->length(i);
    }
    auto functions = static_cast<CPUBackend*>(backend())->functions();
    auto i8functions = static_cast<CPUBackend*>(backend())->int8Functions();
    auto icC4 = UP_DIV(ic, functions->pack);
    auto ocC4 = UP_DIV(oc, mResource->mHp);
    const int tableunit = 16;
    auto blC4 = mResource->mBlockSizeC4;
    auto weightPtr = mResource->mWeightInt8->host<uint8_t>();
    auto threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();
    {
        auto src = inputs[0]->host<int8_t>();
        int8_t tmpBuff[8];
        for (int x=0; x<planeSize; ++x) {
            // Compute featuremapTable and compute Input sum
            functions->MNNTMacBuildTable((int8_t*)(mCache->featuremapTable.ptr()), (int8_t*)mCache->inputSum.ptr(), src + x * functions->pack * functions->bytes, planeSize, icC4, functions->pack, blC4, mResource->mBlockNumber, tableunit);
            // Quant Input
            MNNCountMaxMinValue((float*)(mCache->featuremapTable.ptr()), (float*)(tmpBuff), (float*)(tmpBuff + QUANT_INFO_BYTES), mResource->mBlockNumber * blC4 * tableunit);
            float minVal = ((float*)tmpBuff)[0];
            float maxVal = ((float*)tmpBuff)[1];
            float range = maxVal - minVal;
            float zeropoint, quantscale, dequantscale;
            if (fabs(range) < 1e-7) {
                zeropoint = (-1 * maxVal) - 128;
                quantscale = 1.0f;
                dequantscale = 1.0f;
            } else {
                quantscale = 255.0f / range;
                dequantscale = range / 255.0f;
                zeropoint = roundf(-minVal * 255.f / range) - 128.0f;
            }
            mQuantFunc((float*)((uint8_t*)mCache->featuremapTable.ptr()), ((int8_t*)mCache->featuremapTable.ptr()), mResource->mBlockNumber * blC4 * tableunit / 4, &quantscale, -128, 127, &zeropoint, 0);
            auto dstX = outputs[0]->host<uint8_t>() + (x) * functions->pack * functions->bytes;
            float offset = zeropoint * ((1 << mResource->mBits) - 1) * mResource->mBlockSizeC4;
            auto ocThreadPart = UP_DIV(ocC4, threadNumber);
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                auto ocStart = tId * ocThreadPart;
                auto ocEnd = ALIMIN(ocStart + ocThreadPart, ocC4);
                if (ocEnd > ocStart) {
                    PlaneInfo info;
                    info.planeSize = planeSize;
                    info.offset = offset;
                    info.dequantscale = dequantscale;
                    info.maxValue = mParameters[3];
                    info.minValue = mParameters[2];
                    info.ocDiv = ocEnd - ocStart;
                    info.mWeightPtr = mResource->mWeightInt8->host<uint8_t>() + ocStart * mResource->mWeightInt8->stride(0);
                    info.mWeightScalePtr = mResource->mOriginScale->host<uint8_t>() + ocStart * mResource->mBlockNumber * 2 * mResource->mHp * QUANT_INFO_BYTES;
                    info.mBiasPtr = mResource->mOriginBias->host<uint8_t>() + ocStart * mResource->mHp * QUANT_INFO_BYTES;
                    i8functions->MNNTMacCompute((float*)(dstX + ocStart * mResource->mHp * planeSize * functions->bytes), (int8_t*)mCache->featuremapTable.ptr(), (float*)mCache->inputSum.ptr(), mResource.get(), &info);
                }
            };
            MNN_CONCURRENCY_END();

        }
    }

    return NO_ERROR;
}
ConvTMac* ConvTMac::create(Backend* backend, const Op* op, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon) {
    std::shared_ptr<TMacResource> resource(new TMacResource);
    _createResource(resource.get(), quanCommon, op->main_as_Convolution2D(), backend, true);
    auto conv = op->main_as_Convolution2D();
    MNN_ASSERT(conv->common()->inputCount() % 4 == 0);
    // Shuffle Weight

    auto exe = new ConvTMac(op->main_as_Convolution2D()->common(), backend);
    exe->mResource = resource;
    return exe;
}


};
