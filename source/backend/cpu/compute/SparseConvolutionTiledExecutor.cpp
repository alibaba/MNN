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

#define GENERATE_FUNCTOR()                                                         \
    auto sparseMatmul =                                                            \
        mSparseBlockOC == 4 ? core->MNNPackedSparseMatMulEpx4 : core->MNNPackedSparseMatMulEpx1;

#define GENERATE_WEIGHT()                              \
    auto weightPtr     = weight->host<float>();        \
    auto NNZMapPtr     = NNZMap->host<unsigned int>(); \
    auto dataOffsetPtr = dataOffsetMap->host<int>();

#define GENERATE_MM()                                                                                              \
    /*MNN_PRINT("inputdata matrix tile:"); */                                                                      \
    /*formatMatrix((float*)gemmBuffer, {UP_DIV(xC, eP), L, eP});*/                                                 \
    /* SPMM */                                                                                                     \
    /*MNN_PRINT("PackedSparseMatMul packNumber:%d, eP:%d, eSize:%d, l:%zu, h:%zu, cStride:%zu, aStride:%zu\n",*/   \
    /*number, eP, xC, parameters[1], parameters[2], parameters[3] / bytes, eP * parameters[1]);*/                  \
    /*Timer kernelTimer;*/                                                                                         \
    /*for (int multi = 0; multi < 1000; multi++) {*/                                                               \
    sparseMatmul((float*)(dstOrigin + start * unit * bytes), (float*)gemmBuffer, weightPtr, xC, parameters.data(), \
                 postParameters.data(), biasPtr, NNZMapPtr, dataOffsetPtr);                                        \
    /*}*/                                                                                                          \
    /*MNN_PRINT("cost time: %lu us\n", kernelTimer.durationInUs());*/                                              \
    /*MNN_PRINT("spmm sparseMatmul tile:\n");*/                                                                    \
    /*formatMatrix((float*)(dstOrigin + start * unit * bytes), {UP_DIV(outputChannel, 4), xC, 4});*/

ErrorCode SparseConvolutionTiledImpl::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                               Tensor* NNZMap, Tensor* dataOffsetMap) {
    GENERATE_RESIZE();
}

#undef GENERATE_FUNCTOR
#undef GENERATE_WEIGHT
#undef GENERATE_MM


} // namespace MNN
