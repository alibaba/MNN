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
#include "core/CommonCompute.hpp"

using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {

/*
    source: source matrix is h x l
    transpose: if false, export compressed matrix as h x l, other export as l x h.
 */

static int _fillIndex(int32_t* targetIndexes, uint32_t begin, uint32_t end, const uint32_t* indexes, uint32_t indexSize, int indexStart) {
    int mid = -1;
    int current = -1;
    for (int i=indexStart; i<indexSize; ++i) {
        if (indexes[i] >= begin) {
            mid = i;
            current = indexes[i];
            break;
        }
    }
    uint32_t number = end - begin;
    for (uint32_t i=0; i<number; ++i) {
        targetIndexes[i] = -1;
    }
    auto offset = current - begin;
    do {
        if (current < begin || current >= end) {
            break;
        }
        targetIndexes[current - begin] = mid;
        mid++;
        if (mid >= indexSize) {
            break;
        }
        current = indexes[mid];
    } while (true);
    return mid;
}

static void MNNGetOptimalBlockShape(size_t& weightNNZElement, size_t& weightBlockNumber, const uint32_t* indexes, uint32_t indexSize, int sparseBlockOC, size_t h, size_t l) {
    size_t nnzBlock = 0;
    size_t nnzTail = 0;
    int ocEven = (h / sparseBlockOC) * sparseBlockOC;
    std::vector<int32_t> tempIndexes(sparseBlockOC * l);
    size_t ioc = 0;
    int offset = 0;
    for (; ioc < ocEven; ioc += sparseBlockOC) {
        offset = _fillIndex(tempIndexes.data(), ioc * l, (ioc+sparseBlockOC) * l, indexes, indexSize, offset);
        for (size_t i = 0; i < l; i++) {
            bool allZero = true;
            for (int u=0; u<sparseBlockOC; ++u) {
                if (tempIndexes[u*l + i] >= 0) {
                    allZero = false;
                    break;
                }
            }
            if (!allZero) {
                nnzBlock++;
            }
        }
    }
    for (; ioc < h; ioc++) {
        offset = _fillIndex(tempIndexes.data(), ioc * l, (ioc+1) * l, indexes, indexSize, offset);
        for (size_t i = 0; i < l; i++) {
            if (tempIndexes[i] >= 0) {
                nnzTail++;
            }
        }
    }
    weightNNZElement = nnzBlock * sparseBlockOC + nnzTail;
    weightBlockNumber = nnzBlock + nnzTail;
    return;
}
static void MNNPackForSparseMatMul_B(float* dest, unsigned int* NNZMap, int* dataOffsetMap, int sparseBlockOC, const float* source, const uint32_t* indexes, uint32_t indexSize, size_t h, size_t ic, size_t kernelSize, const int eP) {
    // 1. in convolution, source B layout is OC x (KH * KW * IC),
    //    the dest layout of weight is BCSC(block compressed sparse colum) format, which is OC(!=0) x (KH*KW*IC!=0), as a canceled result, just do BCSR, transpose should be false.
    // 2. in ordinary sparse MatMul, transpose is corresponding to BCSR or BCSC
    auto l = ic * kernelSize;

    int columOffset = 0;
    int i = 0;
    std::vector<int32_t> tempIndexes(sparseBlockOC * l);
    int offset = 0;
    for (; i + sparseBlockOC <= h; i += sparseBlockOC) {
        *NNZMap = 0;
        offset = _fillIndex(tempIndexes.data(), i * l, (i+sparseBlockOC) * l, indexes, indexSize, offset);
        // Origin weight is oc, ic, kernelSize, new weight order is oc, kernelsize, ic
        for (int x=0; x<kernelSize; ++x) {
            for (int y=0; y<ic; ++y) {
                auto j = y * kernelSize + x;
                bool allZero = true;
                for (int u=0; u<sparseBlockOC; ++u) {
                    if (tempIndexes[u*l + j] >= 0) {
                        allZero = false;
                        break;
                    }
                }
                if (!allZero) {
                    for (int ioc = 0; ioc < sparseBlockOC; ioc++) {
                        auto index = tempIndexes[ioc*l + j];
                        if (index >= 0) {
                            *dest = source[index];
                        } else {
                            *dest = 0.0f;
                        }
                        dest++;
                    }
                    *NNZMap = *NNZMap + 1;
                    *dataOffsetMap = columOffset;
                    dataOffsetMap++;
                    columOffset = 0;
                }
                columOffset += eP;
            }
        }
        NNZMap++;
        columOffset -= l * eP;
    }

    for (; i < h; i++) {
        *NNZMap = 0;
        offset = _fillIndex(tempIndexes.data(), i * l, (i+1) * l, indexes, indexSize, offset);
        for (int x=0; x<kernelSize; ++x) {
            for (int y=0; y<ic; ++y) {
                auto j = y * kernelSize + x;
                auto index = tempIndexes[j];
                if (index >= 0) {
                    *dest = source[index];
                    dest++;
                    *NNZMap = *NNZMap + 1;
                    *dataOffsetMap = columOffset;
                    dataOffsetMap++;
                    columOffset = 0;
                }
                columOffset += eP;
            }
        }
        NNZMap++;
        columOffset -= l * eP;
    }

    *dataOffsetMap = columOffset; //
    return;
}
void SparseConvolutionTiledExecutor::initWeight(float* dest, unsigned int* NNZMap, int* dataOffsetMap,
                                                int sparseBlockOC, const float* source, const uint32_t* indexes, uint32_t indexSize, int depth,
                                                int outputCount, int kernelSize, int eP, size_t weightNNZElement,
                                                size_t weightBlockNumber, const CoreFunctions* function) {
    MNNPackForSparseMatMul_B(dest, NNZMap, dataOffsetMap, sparseBlockOC, source, indexes, indexSize, outputCount, depth, kernelSize, eP);

    // MNN_PRINT("\nBCSR origin weight:");
    // formatMatrix(source, {outputCount, kernelSize * depth});
    // MNN_PRINT("\nBCSR new weight:");
    // formatMatrix(dest, {static_cast<int>(weightNNZElement)});
    // MNN_PRINT("\nBCSR weight nnzmap:");
    // formatMatrix(NNZMap, {outputCount / sparseBlockOC + outputCount % sparseBlockOC});
    // MNN_PRINT("\nBCSR weight dataOffsetMap:");
    // formatMatrix(dataOffsetMap, {static_cast<int>(weightBlockNumber + 1)});
}


SparseConvolutionTiledExecutor::SparseConvolutionTiledExecutor(const Convolution2DCommon *common, Backend* b,
                                                               const IDSTQuan* weight, const SparseCommon* sparseCommon,
                                                   const float* bias, size_t biasSize)
    : ConvolutionTiledExecutor(b, bias, biasSize) {

    auto outputCount = (int)biasSize;
    // Don't use common->inputCount for old model common->inputCount is zero
    auto lSize = weight->weightSize() / outputCount;
    auto srcCount = lSize / (common->kernelX() * common->kernelY());

    int eP, lP, hP;
    auto core = static_cast<CPUBackend*>(b)->functions();
    int bytes = core->bytes;
    core->MNNGetSparseMatMulPackMode(&eP, &lP, &hP);
    auto sparseBlockOC = sparseCommon->args()->LookupByKey("sparseBlockOC")->i();
    size_t weightNNZElement = sparseCommon->args()->LookupByKey("NNZElement")->i();
    size_t weightBlockNumber = sparseCommon->args()->LookupByKey("blockNumber")->i();

    int optimalSparseBlockOC = sparseBlockOC;
    MNNPackedSparseMatMul packedSparseMatmul = nullptr;
    core->MNNAdjustOptimalSparseKernel(optimalSparseBlockOC, packedSparseMatmul);

    if (optimalSparseBlockOC != sparseBlockOC) {
        size_t optimalWeightNNZElement = weightNNZElement;
        size_t optimalWeightBlockNumber = weightBlockNumber;
        MNNGetOptimalBlockShape(optimalWeightNNZElement, optimalWeightBlockNumber, weight->index()->data(), weight->index()->size(), optimalSparseBlockOC, outputCount, lSize);
        MNN_ASSERT(sparseBlockOC == 1 || sparseBlockOC == 2 || sparseBlockOC == 4 || sparseBlockOC == 8);
        // MNN_PRINT("caution: sparsity changed!!!\nsparseBlockOC:%d -> %d weightNNZElement:%zu -> %zu, weightBlockNumber:%zu -> %zu, outputCount:%d, divide:%d, tail:%d\n",
        //     sparseBlockOC, optimalSparseBlockOC, weightNNZElement, optimalWeightNNZElement,  weightBlockNumber, optimalWeightBlockNumber, outputCount, outputCount / optimalSparseBlockOC, outputCount % optimalSparseBlockOC);
        sparseBlockOC = optimalSparseBlockOC;
        weightNNZElement = optimalWeightNNZElement;
        weightBlockNumber = optimalWeightBlockNumber;
    }

    mSparseIndexData.reset(new SparseIndexData(sparseBlockOC, weightNNZElement, weightBlockNumber, backend()));

    mResource->mWeight.reset(Tensor::createDevice<uint8_t>(
        { static_cast<int>(weightNNZElement + 1) * bytes }));   // one more element in case of weight are all zeros

    mSparseIndexData->mNNZMap.reset(Tensor::createDevice<unsigned int>({outputCount / sparseBlockOC + outputCount % sparseBlockOC}));
    mSparseIndexData->mDataOffsetMap.reset(Tensor::createDevice<int>({static_cast<int>(weightBlockNumber + 1)}));

    mValid = backend()->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    mValid = mValid && backend()->onAcquireBuffer(mSparseIndexData->mNNZMap.get(), Backend::STATIC);
    mValid = mValid && backend()->onAcquireBuffer(mSparseIndexData->mDataOffsetMap.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }

    initWeight(mResource->mWeight->host<float>(), mSparseIndexData->mNNZMap->host<unsigned int>(), mSparseIndexData->mDataOffsetMap->host<int>(), sparseBlockOC, weight->alpha()->data(), weight->index()->data(), weight->index()->size(), srcCount, outputCount, common->kernelX() * common->kernelY(), eP, weightNNZElement, weightBlockNumber, core);
    mProxy.reset(new SparseConvolutionTiledImpl(common, packedSparseMatmul, sparseBlockOC, b));
}

SparseConvolutionTiledExecutor::SparseConvolutionTiledExecutor(std::shared_ptr<CPUConvolution::Resource> res,
                                                               std::shared_ptr<SparseIndexData> sparseIndexData,
                                                               const Convolution2DCommon *common,
                                                               CoreFunctions::MNNPackedSparseMatMul packedSparseMatmul,
                                                               int sparseBlockOC, Backend* b)
    :mSparseIndexData(sparseIndexData),
    ConvolutionTiledExecutor(res, b) {
    mProxy.reset(new SparseConvolutionTiledImpl(common, packedSparseMatmul, sparseBlockOC, b));
}
SparseConvolutionTiledExecutor::~SparseConvolutionTiledExecutor() {


}
bool SparseConvolutionTiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new SparseConvolutionTiledExecutor(mResource, mSparseIndexData, op->main_as_Convolution2D()->common(), mProxy->mPackedSparseMatmul, mProxy->mSparseBlockOC, bn);
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
    ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParameters, mCommon, input, outputs[0], mPadX, mPadY, core, nullptr);
    auto sparseMatmul = mPackedSparseMatmul;
    int bytes    = core->bytes;
    int unit     = core->pack;
    auto packA   = core->MNNPackC4ForMatMul_A;
    if (core->matmulBytes != 0) {
        // Use origin packC4
        packA = MNNGetCoreFunctions()->MNNPackC4ForMatMul_A;
    }
    int eP, lP, hP;
    getPackParameter(&eP, &lP, &hP, core);
    auto weightPtr     = weight->host<float>();
    auto NNZMapPtr     = NNZMap->host<unsigned int>();
    auto dataOffsetPtr = dataOffsetMap->host<int>();
    auto output      = outputs[0];
    auto batch       = output->batch();
    int threadNumber = ((CPUBackend *)backend())->threadNumber();
    auto icC4                     = UP_DIV(input->channel(), unit);
    auto ic                       = input->channel();
    auto L                        = ic * mCommon->kernelY() * mCommon->kernelX();
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
    auto plane    = mIm2ColParameters.ow * mIm2ColParameters.oh * batch;
    int tileCount = UP_DIV(plane, eP);

    bool success = backend()->onAcquireBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    auto outputChannel = output->channel();
    auto oC4           = UP_DIV(outputChannel, unit);
    auto bufferAlloc   = static_cast<CPUBackend *>(backend())->getBufferAllocator();
    auto maxLine       = UP_DIV(eP, mIm2ColParameters.ow) + 1;
    auto tempPtr = bufferAlloc->alloc(ConvolutionTiledExecutor::computeBlitInfoSize(eP, mIm2ColParameters.ow, mIm2ColParameters.kernelX * mIm2ColParameters.kernelY, threadNumber).first);
    if (tempPtr.invalid()) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    bufferAlloc->free(tempPtr);
    auto threadNumberFirst = std::min(threadNumber, tileCount);
    auto postParameters    = getPostParameters();
    mFunction.first        = threadNumberFirst;

    mFunction.second       = [=](int tId) {
        auto gemmBuffer = mTempBufferTranspose.host<uint8_t>() + mTempBufferTranspose.stride(0) * tId;
        auto srcPtr     = (float const **)(tempPtr.ptr() + tId * kernelSize * maxLine * (4 * sizeof(int32_t) + sizeof(float *)));
        auto el         = (int32_t *)(srcPtr + kernelSize * maxLine);

        int32_t info[4];
        info[1] = mIm2ColParameters.iw * mIm2ColParameters.ih * batch;
        info[2] = eP;
        info[3] = mIm2ColParameters.strideX;
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
            auto res = ConvolutionTiledExecutor::turnIm2ColToBlitInfo(srcPtr, el, start, xC, mIm2ColParameters, srcOrigin, bytes);
            auto number = res.first;
            auto needZero = res.second;

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
