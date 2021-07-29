//
//  SparseConvolutionTiledExecutor
//  MNN
//
//  Created by MNN on 2021/04/06.
//  Copyright © 2018-2021 Alibaba Group Holding Limited.
//

#include "SparseConvolutionTiledExecutor.hpp"
#include <MNN/AutoTime.hpp>
#include "backend/cpu/CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "core/Concurrency.h"
#include "ConvOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "math/Vec.hpp"
#include "core/BufferAllocator.hpp"
#include "core/MemoryFormater.h"

using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {

void SparseConvolutionTiledExecutor::initWeight(float* dest, unsigned int* NNZMap, int* dataOffsetMap,
                                                int sparseBlockOC, const float* source, float* cache, int depth,
                                                int outputCount, int kernelSize, int eP, size_t weightNNZElement,
                                                size_t weightBlockNumber, const CoreFunctions* function) {
    ConvolutionTiledExecutor::initWeight(source, cache, depth, outputCount, kernelSize, function);
    function->MNNPackForSparseMatMul_B(dest, NNZMap, dataOffsetMap, sparseBlockOC, cache, outputCount, kernelSize * depth, eP, false);
    // MNN_PRINT("\nBCSR new weight:");
    // formatMatrix(dest, {static_cast<int>(weightNNZElement)});
    // MNN_PRINT("\nBCSR weight nnzmap:");
    // formatMatrix(NNZMap, {outputCount / sparseBlockOC + outputCount % sparseBlockOC});
    // MNN_PRINT("\nBCSR weight dataOffsetMap:");
    // formatMatrix(dataOffsetMap, {static_cast<int>(weightBlockNumber + 1)});
}


SparseConvolutionTiledExecutor::SparseConvolutionTiledExecutor(const Convolution2DCommon *common, Backend* b,
                                                   const float* originWeight, size_t originWeightSize, const SparseCommon* sparseCommon,
                                                   const float* bias, size_t biasSize)
    : ConvolutionTiledExecutor(b, bias, biasSize) {

    auto outputCount = (int)biasSize;

    int eP, lP, hP;
    auto core = static_cast<CPUBackend*>(b)->functions();
    int bytes = core->bytes;
    core->MNNGetSparseMatMulPackMode(&eP, &lP, &hP);
    auto sparseBlockOC = sparseCommon->args()->LookupByKey("sparseBlockOC")->i();
    size_t weightNNZElement = sparseCommon->args()->LookupByKey("NNZElement")->i();
    size_t weightBlockNumber = sparseCommon->args()->LookupByKey("blockNumber")->i();
    hP = sparseBlockOC; // should broadcast sparseBlockOC to other caller.
    MNN_ASSERT(hP == 1 || hP == 2 || hP == 4);

    // Don't use common->inputCount for old model common->inputCount is zero
    auto lSize = originWeightSize / outputCount;
    auto srcCount = lSize / (common->kernelX() * common->kernelY());
    // MNN_PRINT("1x%d weightNNZElement%zu, weightBlockNumber:%zu\n", sparseBlockOC, weightNNZElement, weightBlockNumber);
    mResource->mWeight.reset(Tensor::createDevice<uint8_t>(
        { static_cast<int>(weightNNZElement + 1) * bytes }));   // one more element in case of weight are all zeros
    std::shared_ptr<Tensor> cache(Tensor::createDevice<uint8_t>({static_cast<int>(outputCount * lSize * sizeof(float))})); // cache must be float

    mNNZMap.reset(Tensor::createDevice<unsigned int>({outputCount / sparseBlockOC + outputCount % sparseBlockOC}));
    mDataOffsetMap.reset(Tensor::createDevice<int>({static_cast<int>(weightBlockNumber + 1)}));

    mValid = backend()->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    mValid = mValid && backend()->onAcquireBuffer(cache.get(), Backend::STATIC);
    mValid = mValid && backend()->onAcquireBuffer(mNNZMap.get(), Backend::STATIC);
    mValid = mValid && backend()->onAcquireBuffer(mDataOffsetMap.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }

    initWeight(mResource->mWeight->host<float>(), mNNZMap->host<unsigned int>(), mDataOffsetMap->host<int>(), sparseBlockOC, originWeight, cache->host<float>(), srcCount, outputCount, common->kernelX() * common->kernelY(), eP, weightNNZElement, weightBlockNumber, core);
    backend()->onReleaseBuffer(cache.get(), Backend::STATIC);
    mProxy.reset(new SparseConvolutionTiledImpl(common, sparseCommon, b));
}

SparseConvolutionTiledExecutor::SparseConvolutionTiledExecutor(std::shared_ptr<CPUConvolution::Resource> res,
                                                               std::shared_ptr<Tensor> NNZMapSharePtr,
                                                               std::shared_ptr<Tensor> dataOffsetMapSharePtr,
                                                               const Convolution2DCommon *common,
                                                               const SparseCommon* sparseCommon, Backend* b)
    :mNNZMap(NNZMapSharePtr),
    mDataOffsetMap(dataOffsetMapSharePtr),
    ConvolutionTiledExecutor(res, b) {
    mProxy.reset(new SparseConvolutionTiledImpl(common, sparseCommon, b));
}
SparseConvolutionTiledExecutor::~SparseConvolutionTiledExecutor() {
    // Do nothing
}
bool SparseConvolutionTiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {

    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new SparseConvolutionTiledExecutor(mResource, mNNZMap, mDataOffsetMap, op->main_as_Convolution2D()->common(), mProxy->mSparseCommon, bn);
    return true;
}

void SparseConvolutionTiledImpl::getPackParameter(int* eP, int* lP, int* hP, const CoreFunctions* core) {
    core->MNNGetSparseMatMulPackMode(eP, lP, hP);
    return;
}

ErrorCode SparseConvolutionTiledImpl::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                               Tensor* NNZMap, Tensor* dataOffsetMap) {
    CPUConvolution::onResize(inputs, outputs);
    auto input   = inputs[0];
    auto weight  = inputs[1];
    Tensor *bias = nullptr;
    auto core    = static_cast<CPUBackend *>(backend())->functions();
    auto sparseMatmul = mSparseBlockOC == 4 ? core->MNNPackedSparseMatMulEpx4 : core->MNNPackedSparseMatMulEpx1;
    int bytes    = core->bytes;
    int unit     = core->pack;
    auto packA   = core->MNNPackC4ForMatMul_A;
    int eP, lP, hP;
    getPackParameter(&eP, &lP, &hP, core);
    auto weightPtr     = weight->host<float>();
    auto NNZMapPtr     = NNZMap->host<unsigned int>();
    auto dataOffsetPtr = dataOffsetMap->host<int>();
    auto strideX           = mCommon->strideX();
    auto strideY           = mCommon->strideY();
    auto dilateX           = mCommon->dilateX();
    auto dilateY           = mCommon->dilateY();
    auto padY              = mPadY;
    auto padX              = mPadX;
    auto kernel_width      = mCommon->kernelX();
    auto kernel_height     = mCommon->kernelY();
    auto output      = outputs[0];
    auto batch       = output->batch();
    auto width       = output->width();
    auto height      = output->height();
    int threadNumber = ((CPUBackend *)backend())->threadNumber();
    auto src_width                = input->width();
    auto src_height               = input->height();
    auto icC4                     = UP_DIV(input->channel(), unit);
    auto ic                       = input->channel();
    auto L                        = ic * mCommon->kernelY() * mCommon->kernelX();
    if (src_width == 1 && width == 1 && height > 1) {
        /* Swap x, y*/
        width         = height;
        height        = 1;
        padX          = mPadY;
        padY          = mPadX;
        strideX       = strideY;
        strideY       = 1; /* Don't need stride */
        src_width     = src_height;
        src_height    = 1;
        dilateX       = dilateY;
        dilateY       = 1;
        kernel_width  = kernel_height;
        kernel_height = 1;
    }
    const float *biasPtr = nullptr;
    if (inputs.size() > 2) {
        bias    = inputs[2];
        biasPtr = bias->host<float>();
    }
    auto kernelSize               = mCommon->kernelX() * mCommon->kernelY();
    mTempBufferTranspose.buffer().type          = halide_type_of<uint8_t>();
    mTempBufferTranspose.buffer().dimensions    = 2;
    mTempBufferTranspose.buffer().dim[0].extent = threadNumber;
    mTempBufferTranspose.buffer().dim[1].extent = UP_DIV(L, lP) * lP * eP * bytes;
    TensorUtils::setLinearLayout(&mTempBufferTranspose);
    auto plane    = width * height * batch;
    int tileCount = UP_DIV(plane, eP);
                                              
    bool success = backend()->onAcquireBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    auto outputChannel = output->channel();
    auto oC4           = UP_DIV(outputChannel, unit);
    auto bufferAlloc   = static_cast<CPUBackend *>(backend())->getBufferAllocator();
    auto maxLine       = UP_DIV(eP, width) + 1;
    auto tempPtr = bufferAlloc->alloc(kernelSize * maxLine * threadNumber * (4 * sizeof(int32_t) + sizeof(float *)));
    if (nullptr == tempPtr.first) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    bufferAlloc->free(tempPtr);
    auto threadNumberFirst = std::min(threadNumber, tileCount);
    auto postParameters    = getPostParameters();
    mFunction.first        = threadNumberFirst;

    mFunction.second       = [=](int tId) {
        auto gemmBuffer = mTempBufferTranspose.host<uint8_t>() + mTempBufferTranspose.stride(0) * tId;
        auto srcPtr     = (float const **)((uint8_t *)tempPtr.first + tempPtr.second +
                                       tId * kernelSize * maxLine * (4 * sizeof(int32_t) + sizeof(float *)));
        auto el         = (int32_t *)(srcPtr + kernelSize * maxLine);
                                        
        int32_t info[4];
        info[1] = src_width * src_height * batch;
        info[2] = eP;
        info[3] = strideX;
        size_t parameters[6];
        parameters[0]          = eP * bytes;
        parameters[1]          = L;
        parameters[2]          = outputChannel;
        parameters[3]          = plane * unit * bytes;
        parameters[4]          = 0;
        parameters[5]          = 0;

        auto dstOrigin = output->host<uint8_t>();
        auto srcOrigin = input->host<uint8_t>();
        for (int x = (int)tId; x < tileCount; x += threadNumberFirst) {
            int start  = (int)x * eP;
            int remain = plane - start;
            int xC     = remain > eP ? eP : remain;
            /* Compute Pack position */
            int oyBegin   = start / width;
            int oxBegin   = start % width;
            int oyEnd     = (start + xC - 1) / width;
            remain        = xC;
            int number    = 0;
            bool needZero = false;
            int eStart    = 0;
            for (int oyb = oyBegin; oyb <= oyEnd; ++oyb) {
                int step    = std::min(width - oxBegin, remain);
                int oy      = oyb % height;
                int ob      = oyb / height;
                int sySta   = oy * strideY - padY;
                int kyStart = std::max(0, UP_DIV(-sySta, dilateY));
                int kyEnd   = std::min(kernel_height, UP_DIV(src_height - sySta, dilateY));
                if (kyEnd - kyStart < kernel_height) {
                    needZero = true;
                }
                auto srcStart = srcOrigin + ((ob * src_height + sySta) * src_width) * bytes * unit;
                for (int ky = kyStart; ky < kyEnd; ++ky) {
                    auto lKYOffset = ky * kernel_width * ic;
                    auto srcKy     = srcStart + ky * dilateY * src_width * bytes * unit;
                    for (int kx = 0; kx < kernel_width; ++kx) {
                        /* Compute x range:*/
                        /* 0 <= (oxBegin + x) * strideX - padX + dilateX * kx < src_width*/
                        /* 0 <= x <= step*/
                        int end = std::min(
                            step, (src_width - oxBegin * strideX - dilateX * kx + padX + strideX - 1) / strideX);
                        int sta = std::max(0, UP_DIV((padX - oxBegin * strideX - dilateX * kx), strideX));
                        if (end - sta < step) {
                            needZero = true;
                        }
                        if (end > sta) {
                            auto lOffset = lKYOffset + (kx * ic);
                            auto srcKx   = srcKy + ((oxBegin + sta) * strideX + dilateX * kx - padX) * bytes * unit;
                            srcPtr[number]     = (const float *)srcKx;
                            el[4 * number + 0] = end - sta;
                            el[4 * number + 1] = ic;
                            el[4 * number + 2] = eStart + sta;
                            el[4 * number + 3] = lOffset;
                            number++;
                        }
                    }
                }
                oxBegin = 0;
                remain -= step;
                eStart += step;
            }
            info[0] = number;
            if (needZero || lP != 1) {
                ::memset(gemmBuffer, 0, mTempBufferTranspose.stride(0));
            }
            if (number > 0) {
                packA((float *)gemmBuffer, srcPtr, info, el);
            }
            sparseMatmul((float*)(dstOrigin + start * unit * bytes), (float*)gemmBuffer, weightPtr, xC, parameters, postParameters.data(), biasPtr, NNZMapPtr, dataOffsetPtr);
        }
    };
    return NO_ERROR;
}


} // namespace MNN
