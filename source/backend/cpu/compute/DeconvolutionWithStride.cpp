//
//  DeconvolutionWithStride.cpp
//  MNN
//
//  Created by MNN on 2018/10/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/DeconvolutionWithStride.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "core/Concurrency.h"
#include "ConvOpt.h"
#include "core/Macro.h"
#include "math/WingoradGenerater.hpp"
#include "backend/cpu/compute/WinogradOptFunction.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

#define OPEN_WINOGRAD

using namespace MNN::Math;

namespace MNN {

static const int gDefaultUnit = 3;
static void _winograd(const DeconvolutionWithStride::ComputeUnit& unit, int threadId, int strideX, int strideY,
                      const Tensor* src, const Tensor* dst, std::map<int, std::shared_ptr<Tensor>>& sourceTransformMap,
                      std::map<int, bool>& sourceTransformed, float* cachePackBuffer, int ic, int oc) {
    int eP, lP, hP;
    MNNGetMatMulPackMode(&eP, &lP, &hP);

    auto srcUnit = unit.winogradInfo.srcUnitX;
    auto buffer  = sourceTransformMap[srcUnit];
    // We allocated the buffer with 2*numberThread
    int numberThread = buffer->length(0) / 2;
    auto dstUnit     = gDefaultUnit;
    int dc_4         = dst->length(3) / 4 / eP;
    int srcCount     = src->stride(2);
    int totalCount   = dst->stride(2);
    int ic_4         = srcCount / eP / 4;
    auto dstTotal    = dst->host<float>() + threadId * dst->stride(0);
    auto srcTotal    = src->host<float>() + threadId * src->stride(0);

    if (!sourceTransformed[srcUnit]) {
        auto A        = unit.winogradInfo.A.get();
        auto midAddr  = buffer->host<float>() + (threadId + numberThread) * buffer->stride(0);
        auto destAddr = buffer->host<float>() + (threadId)*buffer->stride(0);

        WinogradFunction::productLeft(srcTotal, A->host<float>(), midAddr, dstUnit, srcUnit, dstUnit,
                                      ic_4 * eP);
        WinogradFunction::productRight(midAddr, A->host<float>(), destAddr, srcUnit, srcUnit, dstUnit,
                                       ic_4 * eP);

        sourceTransformed[srcUnit] = true;
    }

    auto sourceAddr = buffer->host<float>() + (threadId)*buffer->stride(0);
    auto destAddr   = unit.dstBuffer->host<float>() + threadId * unit.dstBuffer->stride(0);
    int32_t info[4];
    info[0] = 1;
    info[1] = eP;
    info[2] = eP;
    info[3] = 1;
    int32_t el[4];
    el[0] = eP;
    el[1] = ic;
    el[2] = 0;
    el[3] = 0;
    size_t parameters[6];
    parameters[0] = eP * sizeof(float);
    parameters[1] = ic;
    parameters[2] = oc;
    parameters[3] = eP * 4 * sizeof(float);
    parameters[4] = 0;
    parameters[5] = 0;

    for (int i = 0; i < srcUnit * srcUnit; ++i) {
        const float* tempSourceAddr = sourceAddr + i * buffer->stride(2);
        auto tempColAddr    = destAddr + i * unit.dstBuffer->stride(1);
        auto weightAddr     = unit.weight->host<float>() + unit.weight->stride(0) * i;
        MNNPackC4ForMatMul_A(cachePackBuffer, &tempSourceAddr, info, el);
        MNNPackedMatMul(tempColAddr, cachePackBuffer,weightAddr, parameters, nullptr, nullptr, nullptr, nullptr);
    }
    auto B       = unit.winogradInfo.B.get();
    auto midAddr = unit.winogradInfo.dstTransformedBuffer->host<float>() +
                   threadId * unit.winogradInfo.dstTransformedBuffer->stride(0);
    WinogradFunction::productLeft(destAddr, B->host<float>(), midAddr, srcUnit, srcUnit, srcUnit,
                                  dc_4 * eP);
    WinogradFunction::productRight(midAddr, B->host<float>(), destAddr, srcUnit, srcUnit, srcUnit,
                                   dc_4 * eP);

    // Add to dest
    for (int fy = 0; fy < srcUnit; ++fy) {
        int sy = fy * strideY + unit.yOffset;
        for (int fx = 0; fx < srcUnit; ++fx) {
            int sx      = fx * strideX + unit.xOffset;
            auto dest   = dstTotal + sx * dst->stride(2) + sy * dst->stride(1);
            auto source = destAddr + (fx + fy * srcUnit) * totalCount;
            MNNAddC4WithStride(source, dest, 4, 4, totalCount / 4);
        }
    }
}

static void _gemmAndIm2col(const DeconvolutionWithStride::ComputeUnit& unit, int threadId, int strideX, int strideY,
                           const Tensor* src, const Tensor* dst, float* cachePackBuffer, int ic, int oc) {
    auto tempColAddr = unit.dstBuffer->host<float>() + unit.dstBuffer->stride(0) * threadId;
    int eP, lP, hP;
    MNNGetMatMulPackMode(&eP, &lP, &hP);
    int ocDiv4       = dst->length(3) / 4 / eP;
    int count        = ocDiv4 * unit.xUnit * unit.yUnit;
    auto weightAddr  = unit.weight->host<float>();
    auto dstTotal    = dst->host<float>() + threadId * dst->stride(0);
    auto srcTotal    = src->host<float>() + threadId * src->stride(0);
    int srcCount     = src->stride(2);
    int totalCount   = dst->stride(2);
    int ic_4         = srcCount / eP / 4;
    int dc_4         = ocDiv4;
    int32_t info[4];
    info[0] = 1;
    info[1] = eP;
    info[2] = eP;
    info[3] = 1;
    int32_t el[4];
    el[0] = eP;
    el[1] = ic;
    el[2] = 0;
    el[3] = 0;
    size_t parameters[6];
    parameters[0] = eP * sizeof(float);
    parameters[1] = ic;
    parameters[2] = oc;
    parameters[3] = eP * 4 * sizeof(float);
    parameters[4] = 0;
    parameters[5] = 0;

    for (int dy = 0; dy < gDefaultUnit; ++dy) {
        for (int dx = 0; dx < gDefaultUnit; ++dx) {
            const float* tempSourceAddr = srcTotal + (dx + dy * gDefaultUnit) * srcCount;
            MNNPackC4ForMatMul_A(cachePackBuffer, &tempSourceAddr, info, el);
            for (int fy = 0; fy < unit.yUnit; ++fy) {
                for (int fx = 0; fx < unit.xUnit; ++fx) {
                    auto ucolAddr = tempColAddr + dc_4 * eP * 4 * (fx + fy * unit.xUnit);
                    auto uwAddr = weightAddr + unit.weight->stride(0) * (fx + fy * unit.xUnit);
                    MNNPackedMatMul(ucolAddr, cachePackBuffer, uwAddr, parameters, nullptr, nullptr, nullptr, nullptr);
                }
            }
            // FUNC_PRINT_ALL(tempColAddr[0], f);

            for (int fy = 0; fy < unit.yUnit; ++fy) {
                for (int fx = 0; fx < unit.xUnit; ++fx) {
                    int sx      = (dx + fx) * strideX + unit.xOffset;
                    int sy      = (dy + fy) * strideY + unit.yOffset;
                    auto dest   = dstTotal + sx * dst->stride(2) + sy * dst->stride(1);
                    auto source = tempColAddr + (fx + fy * unit.xUnit) * totalCount;
                    MNNAddC4WithStride(source, dest, 4, 4, totalCount / 4);
                }
            }
        }
    }
}

DeconvolutionWithStride::DeconvolutionWithStride(const Tensor* input, const Op* convOp, Backend* b)
    : CPUDeconvolutionCommon(input, convOp, b, false) {
    auto conv2D = convOp->main_as_Convolution2D();
    MNN_ASSERT(nullptr != conv2D->bias());
    auto common     = conv2D->common();
    int outputCount = common->outputCount();
    int kx          = common->kernelX();
    int ky          = common->kernelY();
    int eP, lP, hP;
    MNNGetMatMulPackMode(&eP, &lP, &hP);

    const float* tempWeight = nullptr;
    int tempWeightSize   = 0;
    int srcCount = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, b, conv2D, &tempWeight, &tempWeightSize);
    srcCount = tempWeightSize / kx / ky / outputCount;

    int sy = common->strideY();
    int sx = common->strideX();

    for (int y = 0; y < sy; ++y) {
        if (y >= ky) {
            continue;
        }
        int subKY = 1 + (ky - y - 1) / sy;
        for (int x = 0; x < sx; ++x) {
            if (x >= kx) {
                continue;
            }
            int subKx = 1 + (kx - x - 1) / sx;
            ComputeUnit unit;
            unit.xOffset = x;
            unit.yOffset = y;
            unit.xUnit   = subKx;
            unit.yUnit   = subKY;
#ifdef OPEN_WINOGRAD
            if (subKx == subKY) {
                // Open Winograd
                int sourceUnitX = subKx + gDefaultUnit - 1;
                int sourceUnitY = subKY + gDefaultUnit - 1;

                unit.winogradInfo.open     = true;
                unit.winogradInfo.srcUnitX = sourceUnitX;
                unit.winogradInfo.srcUnitY = sourceUnitY;
                Math::WinogradGenerater generater(gDefaultUnit, subKx);

                // Transpose A, B
                auto A = generater.A();
                unit.winogradInfo.A.reset(Matrix::create(A->length(0), A->length(1)));
                Matrix::transpose(unit.winogradInfo.A.get(), A.get());

                auto B = generater.B();
                unit.winogradInfo.B.reset(Matrix::create(B->length(0), B->length(1)));
                Matrix::transpose(unit.winogradInfo.B.get(), B.get());

                unit.winogradInfo.G = generater.G();
                unit.weight.reset(Tensor::createDevice<float>(
                    std::vector<int>{sourceUnitX * sourceUnitY, UP_DIV(outputCount, hP), UP_DIV(srcCount, lP), lP * hP}));
            } else
#endif
            {
                unit.weight.reset(Tensor::createDevice<float>(
                    std::vector<int>{unit.yUnit * unit.xUnit, UP_DIV(outputCount, hP), UP_DIV(srcCount, lP), lP * hP}));
            }
            mComputeUnits.emplace_back(unit);
        }
    }
    bool res = _alloc(Backend::STATIC);
    if (!res) {
        MNN_ERROR("Not Enought Memory for DeconvolutionWithStride\n");
        mValid = false;
        return;
    }
    _extract(convOp);
    mPostParameters = getPostParameters();
}

bool DeconvolutionWithStride::_alloc(Backend::StorageType type) {
    auto b = backend();
    for (auto& unit : mComputeUnits) {
        bool success = b->onAcquireBuffer(unit.weight.get(), type);
        if (!success) {
            return false;
        }
    }

    return true;
}
void DeconvolutionWithStride::_release(Backend::StorageType type) {
    for (auto& unit : mComputeUnits) {
        backend()->onReleaseBuffer(unit.weight.get(), type);
    }
}
void DeconvolutionWithStride::_extract(const Op* convOp) {
    auto conv2D = convOp->main_as_Convolution2D();
    MNN_ASSERT(nullptr != conv2D->bias());
    auto common     = conv2D->common();
    int outputCount = common->outputCount();
    int kx          = common->kernelX();
    int ky          = common->kernelY();
    int eP, lP, hP;
    MNNGetMatMulPackMode(&eP, &lP, &hP);

    const float* tempWeight = nullptr;
    int tempWeightSize   = 0;
    int srcCount = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, backend(), conv2D, &tempWeight, &tempWeightSize);
    srcCount = tempWeightSize / kx / ky / outputCount;
    
    std::shared_ptr<Tensor> weightWrap(
        Tensor::create<float>(std::vector<int>{srcCount, outputCount, ky * kx}, (void*)tempWeight));

    int sy = common->strideY();
    int sx = common->strideX();
    for (auto& unit : mComputeUnits) {
        int y     = unit.yOffset;
        int x     = unit.xOffset;
        int subKy = unit.yUnit;
        int subKx = unit.xUnit;

        // Crop
        std::shared_ptr<Tensor> tempWeight(
            Tensor::create<float>(std::vector<int>{srcCount, outputCount, subKy, subKx}));
        for (int sz = 0; sz < srcCount; ++sz) {
            for (int oz = 0; oz < outputCount; ++oz) {
                auto dst = tempWeight->host<float>() + tempWeight->stride(0) * sz + tempWeight->stride(1) * oz;
                auto src = weightWrap->host<float>() + weightWrap->stride(0) * sz + weightWrap->stride(1) * oz;
                for (int fy = 0; fy < subKy; ++fy) {
                    auto oriFy = y + fy * sy;
                    for (int fx = 0; fx < subKx; ++fx) {
                        auto oriFx           = x + fx * sx;
                        dst[fx + fy * subKx] = src[oriFy * kx + oriFx];
                    }
                }
            }
        }

        // Winograd Transform
        if (unit.winogradInfo.open) {
            std::shared_ptr<Tensor> K(Matrix::createShape(unit.xUnit, unit.yUnit));
            std::shared_ptr<Tensor> K_Transform(
                Matrix::createShape(unit.winogradInfo.srcUnitX, unit.winogradInfo.srcUnitY));
            std::shared_ptr<Tensor> M(Matrix::create(unit.xUnit, unit.winogradInfo.srcUnitX));

            std::shared_ptr<Tensor> tempWeightDst(Tensor::create<float>(
                std::vector<int>{srcCount, outputCount, unit.winogradInfo.srcUnitX, unit.winogradInfo.srcUnitY}));

            auto G = unit.winogradInfo.G;
            std::shared_ptr<Tensor> GT(Matrix::create(G->length(0), G->length(1)));
            Matrix::transpose(GT.get(), G.get());

            for (int sz = 0; sz < srcCount; ++sz) {
                for (int oz = 0; oz < outputCount; ++oz) {
                    auto src = tempWeight->host<float>() + tempWeight->stride(0) * sz + tempWeight->stride(1) * oz;
                    auto dst =
                        tempWeightDst->host<float>() + tempWeightDst->stride(0) * sz + tempWeightDst->stride(1) * oz;
                    // M=G*K
                    K->buffer().host = (uint8_t*)(src);
                    Matrix::multi(M.get(), G.get(), K.get());

                    // K_Transform = M*GT
                    K_Transform->buffer().host = (uint8_t*)(dst);
                    Matrix::multi(K_Transform.get(), M.get(), GT.get());
                }
            }
            subKx      = unit.winogradInfo.srcUnitX;
            subKy      = unit.winogradInfo.srcUnitY;
            tempWeight = tempWeightDst;
        }

        // Reorder
        auto weighStrideK = unit.weight->stride(0);
        ::memset(unit.weight->host<float>(), 0, unit.weight->size());
        for (int sz = 0; sz < srcCount; ++sz) {
            int sz4   = sz / lP;
            int my    = sz % lP;
            auto dstS = unit.weight->host<float>() + hP * lP * sz4;
            for (int oz = 0; oz < outputCount; ++oz) {
                int oz4   = oz / hP;
                int mx    = oz % hP;
                auto dstO = dstS + unit.weight->stride(1) * oz4;
                auto src  = tempWeight->host<float>() + tempWeight->stride(0) * sz + tempWeight->stride(1) * oz;
                for (int fy = 0; fy < subKy; ++fy) {
                    for (int fx = 0; fx < subKx; ++fx) {
                        dstO[weighStrideK * (fy * subKx + fx) + my + lP * mx] = src[fy * subKx + fx];
                    }
                }
            }
        }
    }
}

DeconvolutionWithStride::~DeconvolutionWithStride() {
    _release(Backend::STATIC);
}

ErrorCode DeconvolutionWithStride::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    CPUDeconvolutionCommon::onResize(inputs, outputs);
    auto input  = inputs[0];
    auto output = outputs[0];
    auto ic     = input->channel();
    auto oc     = output->channel();

    int eP, lP, hP;
    MNNGetMatMulPackMode(&eP, &lP, &hP);
    int numThread = std::max(1, ((CPUBackend*)backend())->threadNumber());
    mSrcBuffer.reset(Tensor::createDevice<float>(
        std::vector<int>{numThread, gDefaultUnit, gDefaultUnit, eP * ALIGN_UP4(ic)}));
    int dstXUnit = (gDefaultUnit - 1) * mCommon->strideX() + (mCommon->kernelX() - 1) * mCommon->dilateX() + 1;
    int dstYUnit = (gDefaultUnit - 1) * mCommon->strideY() + (mCommon->kernelY() - 1) * mCommon->dilateY() + 1;

    mMatMulPackBuffer.reset(Tensor::createDevice<float>(std::vector<int>{numThread, eP * ALIGN_UP4(ic)}));
    mDestBuffer.reset(Tensor::createDevice<float>(
        std::vector<int>{numThread, dstYUnit, dstXUnit, eP * ALIGN_UP4(oc)}));

    bool res = backend()->onAcquireBuffer(mSrcBuffer.get(), Backend::DYNAMIC);
    res &= backend()->onAcquireBuffer(mDestBuffer.get(), Backend::DYNAMIC);
    res &= backend()->onAcquireBuffer(mMatMulPackBuffer.get(), Backend::DYNAMIC);
    mTransformedBuffer.clear();

    for (auto& unit : mComputeUnits) {
        auto kxky = unit.yUnit * unit.xUnit;
        if (!unit.winogradInfo.open) {
            unit.dstBuffer.reset(Tensor::createDevice<float>(
                std::vector<int>{numThread, UP_DIV(oc, 4) * kxky, eP, 4}));
            res &= backend()->onAcquireBuffer(unit.dstBuffer.get(), Backend::DYNAMIC);
            continue;
        }
        auto srcUnit = unit.winogradInfo.srcUnitX;
        unit.dstBuffer.reset(Tensor::createDevice<float>(
            std::vector<int>{numThread, srcUnit * srcUnit, UP_DIV(oc, 4), eP * 4}));
        res &= backend()->onAcquireBuffer(unit.dstBuffer.get(), Backend::DYNAMIC);

        unit.winogradInfo.dstTransformedBuffer.reset(Tensor::createDevice<float>(
            std::vector<int>{numThread, srcUnit * srcUnit, UP_DIV(oc, 4), eP * 4}));
        res &= backend()->onAcquireBuffer(unit.winogradInfo.dstTransformedBuffer.get(), Backend::DYNAMIC);
        if (mTransformedBuffer.find(srcUnit) == mTransformedBuffer.end()) {
            // We Need 2 buffer for transform, one for mid buffer and one for dest
            std::shared_ptr<Tensor> transformBuffer = std::shared_ptr<Tensor>(Tensor::createDevice<float>(
                std::vector<int>{2 * numThread, srcUnit, srcUnit, eP * ALIGN_UP4(ic)}));
            mTransformedBuffer[srcUnit]             = transformBuffer;
        }
    }
    for (auto& iter : mTransformedBuffer) {
        res &= backend()->onAcquireBuffer(iter.second.get(), Backend::DYNAMIC);
    }
    if (!res) {
        return OUT_OF_MEMORY;
    }
    for (auto& unit : mComputeUnits) {
        backend()->onReleaseBuffer(unit.dstBuffer.get(), Backend::DYNAMIC);
        if (unit.winogradInfo.open) {
            backend()->onReleaseBuffer(unit.winogradInfo.dstTransformedBuffer.get(), Backend::DYNAMIC);
        }
    }
    backend()->onReleaseBuffer(mSrcBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mDestBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mMatMulPackBuffer.get(), Backend::DYNAMIC);

    for (auto& iter : mTransformedBuffer) {
        backend()->onReleaseBuffer(iter.second.get(), Backend::DYNAMIC);
    }
    mStrideY = mCommon->strideY();
    mStrideX = mCommon->strideX();

    return NO_ERROR;
}

ErrorCode DeconvolutionWithStride::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    int batchSize = input->batch();
    MNN_ASSERT(batchSize == output->batch());
    int oc     = output->channel();
    int ow     = output->width();
    int oh     = output->height();
    int ocDiv4 = UP_DIV(oc, 4);
    int oZstep = ow * oh * 4 * batchSize;

    int ic     = input->channel();
    int iw     = input->width();
    int ih     = input->height();
    int icDiv4 = UP_DIV(ic, 4);
    int iZstep = iw * ih * 4 * batchSize;

    int strideX = mStrideX;
    int strideY = mStrideY;
    int eP, lP, hP;
    MNNGetMatMulPackMode(&eP, &lP, &hP);

    //        FUNC_PRINT(mPadX);
    //        FUNC_PRINT(mPadY);

    int wUnit     = UP_DIV(iw, gDefaultUnit);
    int hUnit     = UP_DIV(ih, gDefaultUnit);
    int total     = wUnit * hUnit * batchSize;
    int tileCount = UP_DIV(total, eP);
    int numThread = std::max(1, ((CPUBackend*)backend())->threadNumber());
    numThread     = std::min(numThread, tileCount);

    auto srcOrigin = input->host<float>();
    auto dstOrigin = output->host<float>();

    ::memset(mSrcBuffer->host<float>(), 0, mSrcBuffer->size());
    ::memset(dstOrigin, 0, ow * oh * ocDiv4 * 4 * batchSize * sizeof(float));
    auto threadFunction = [&](int threadId) {
        auto srcTotal = mSrcBuffer->host<float>() + threadId * mSrcBuffer->stride(0);
        auto dstTotal = mDestBuffer->host<float>() + threadId * mDestBuffer->stride(0);
        auto packBuffer = mMatMulPackBuffer->host<float>() + threadId * mMatMulPackBuffer->stride(0);
        for (int tIndex = (int)threadId; tIndex < tileCount; tIndex += numThread) {
            // Move Source to tile Source
            int xIndex = tIndex * eP;
            int xCount = std::min(eP, total - xIndex);
            {
                int destUnitStride = icDiv4 * eP * 4;
                for (int index = 0; index < xCount; ++index) {
                    int whIndex = xIndex + index;
                    int wIndex  = whIndex % wUnit;
                    int hbIndex  = whIndex / wUnit;
                    int hIndex = hbIndex % hUnit;
                    int bIndex = hbIndex / hUnit;

                    auto dstStart = srcTotal + index * 4;
                    auto sx       = wIndex * gDefaultUnit;
                    auto sy       = hIndex * gDefaultUnit;
                    auto srcStart = 4 * (sx + sy * iw) + srcOrigin + bIndex * iw * ih * 4;
                    for (int subY = 0; subY < gDefaultUnit; ++subY) {
                        for (int subX = 0; subX < gDefaultUnit; ++subX) {
                            auto dstUnit = dstStart + (subX + subY * gDefaultUnit) * destUnitStride;
                            int x        = sx + subX;
                            int y        = sy + subY;
                            if (x < 0 || x >= iw || y < 0 || y >= ih) {
#ifdef MNN_USE_NEON
                                auto zero = vdupq_n_f32(0.0f);
#endif
                                for (int z = 0; z < icDiv4; ++z) {
#ifdef MNN_USE_NEON
                                    vst1q_f32(dstUnit + 4 * eP * z, zero);
#else
                                    for (int j = 0; j < 4; ++j) {
                                        dstUnit[4 * eP * z + j] = 0;
                                    }
#endif
                                }
                                continue;
                            }
                            auto srcUnit = srcStart + (subX + subY * iw) * 4;
                            MNNCopyC4WithStride(srcUnit, dstUnit, iZstep, eP * 4, icDiv4);
                        }
                    }
                }
            }

            // Compute to tile Dest
            ::memset(dstTotal, 0, mDestBuffer->stride(0) * sizeof(float));
            std::map<int, bool> transformed;
            for (auto& iter : mTransformedBuffer) {
                transformed[iter.first] = false;
            }
            for (auto& unit : mComputeUnits) {
                if (unit.winogradInfo.open) {
                    _winograd(unit, (int)threadId, strideX, strideY, mSrcBuffer.get(), mDestBuffer.get(),
                              mTransformedBuffer, transformed, packBuffer, ic, oc);
                } else {
                    _gemmAndIm2col(unit, (int)threadId, strideX, strideY, mSrcBuffer.get(), mDestBuffer.get(), packBuffer, ic, oc);
                }
            }

            // Merge to Dest
            {
                std::unique_lock<std::mutex> __l(mLock);
                int srcUnitStride = ocDiv4 * eP * 4;
                int destXUnit     = mDestBuffer->length(2);
                int destYUnit     = mDestBuffer->length(1);
                for (int index = 0; index < xCount; ++index) {
                    int whIndex = xIndex + index;
                    int wIndex  = whIndex % wUnit;
                    int hbIndex  = whIndex / wUnit;
                    int hIndex = hbIndex % hUnit;
                    int bIndex = hbIndex / hUnit;

                    auto srcStart = dstTotal + index * 4;
                    auto sx       = wIndex * gDefaultUnit * strideX - mPadX;
                    auto sy       = hIndex * gDefaultUnit * strideY - mPadY;
                    // MNN_PRINT("%d, %d\n", sx, sy);
                    auto dstStart = dstOrigin + 4 * (sx + sy * ow) + bIndex * ow * oh * 4;
                    int yEnd      = std::min(destYUnit, oh - sy);
                    int xEnd      = std::min(destXUnit, ow - sx);
                    int xStart    = std::max(-sx, 0);
                    int yStart    = std::max(-sy, 0);

                    for (int subY = yStart; subY < yEnd; ++subY) {
                        for (int subX = xStart; subX < xEnd; ++subX) {
                            auto srcUnit = srcStart + (subX + subY * destXUnit) * srcUnitStride;
                            auto dstUnit = dstStart + (subX + subY * ow) * 4;
                            MNNAddC4WithStride(srcUnit, dstUnit, 4 * eP, oZstep, ocDiv4);
                        }
                    }
                }
            }
        }
    };

    MNN_CONCURRENCY_BEGIN(threadId, numThread) {
        threadFunction((int)threadId);
    }
    MNN_CONCURRENCY_END();
    MNNAxByClampBroadcastUnit(dstOrigin, dstOrigin, mBias->host<float>(), ow * oh * batchSize, ow * oh * 4 * batchSize, ow * oh * 4 * batchSize, ocDiv4, mPostParameters.data());

    return NO_ERROR;
}

} // namespace MNN
