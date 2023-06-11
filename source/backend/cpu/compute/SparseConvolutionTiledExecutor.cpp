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
#include "common/MemoryFormater.h"
#include "common/CommonCompute.hpp"

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
    MNN_ASSERT(weightNNZElement > 0);
    MNN_ASSERT(weightBlockNumber > 0);

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
    auto sparseMatmul = mPackedSparseMatmul;
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

    // MNN_PRINT("sparse convoluton: n:%d, ih:%d, iw:%d, ic:%d, oh:%d, ow:%d, oc:%d, kh:%d, kw:%d, plane:%d, tileCount:%d, ePack:%d, pack:%d, mSparseBlockOC:%d, bytes:%d\n",
    //     batch, src_height, src_width, ic, height, width, outputChannel, mCommon->kernelX(), mCommon->kernelY(), plane, tileCount, eP, unit, mSparseBlockOC, bytes);

    mFunction.second       = [=](int tId) {
        Timer kernelTimer;
        uint64_t durationMul = 0;
        uint64_t packATime = 0;
        uint64_t macs = 0;

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
            // MNN_PRINT("inputdata matrix tile:");
            // formatMatrix((float*)gemmBuffer, {UP_DIV(xC, eP), L, eP});
            //  MNN_PRINT("PackedSparseMatMul packNumber:%d, eP:%d, eSize:%d, l:%zu, h:%zu, cStride:%zu, aStride:%zu\n",
            //     number, eP, xC, parameters[1], parameters[2], parameters[3] / bytes, eP * parameters[1]);
            // kernelTimer.reset();
            sparseMatmul((float*)(dstOrigin + start * unit * bytes), (float*)gemmBuffer, weightPtr, xC, parameters, postParameters.data(), biasPtr, NNZMapPtr, dataOffsetPtr);
            // MNN_PRINT("spmm sparseMatmul tile:\n");
            // formatMatrix((float*)(dstOrigin + start * unit * bytes), {UP_DIV(outputChannel, unit), xC, unit});

            // durationMul = kernelTimer.durationInUs();
            // macs = 2 * xC * unit * L * oC4; // bias
            // double gflops = double(macs) / 1000 / durationMul;
            // MNN_PRINT("sparse equal peak: %f GFLOPS. time %llu us, left mat:%d KB, right mat:%d KB\n", gflops, durationMul,  (xC * L * bytes)/1024, (L * mSparseBlockOC * bytes)/1024);

            // durationMul += kernelTimer.durationInUs();
            // macs += 2 * xC * unit * L * oC4; // bias

        }
        // double gflops = double(macs) / 1000 / durationMul;
        // MNN_PRINT("sparse equal peak: %f GFLOPS. time %llu us\n", gflops, durationMul);

    };
    return NO_ERROR;
}


} // namespace MNN
