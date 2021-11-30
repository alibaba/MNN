//
//  SparseConvInt8TiledExecutor.hpp
//  MNN
//
//  Created by MNN on 2021/6/09.
//  Copyright © 2018 - 2021, Alibaba Group Holding Limited


#include "SparseConvInt8TiledExecutor.hpp"
#include "core/Macro.h"

#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "common/MemoryFormater.h"
#include "MNN/AutoTime.hpp"
#include <math.h>
#ifdef MNN_USE_SSE
extern "C" {
void MNNInt8ToUInt8(void* ptr, int count);
}
#endif
namespace MNN {

bool SparseConvInt8TiledExecutor::reorderWeight(Backend* b, const Convolution2DCommon* common,
                          const std::shared_ptr<Tensor>& weightOrigin,
                          std::shared_ptr<Tensor>& weight, const SparseCommon* sparseCommon) {

    int eP, lP, hP;
    auto core = static_cast<CPUBackend*>(b)->int8Functions();
    core->MNNGetSparseQuantMatMulPackMode(&eP, &lP, &hP);

    int oc = common->outputCount(), ic = common->inputCount(), kernelCount = common->kernelX() * common->kernelY();
    auto sparseBlockOC = sparseCommon->args()->LookupByKey("sparseBlockOC")->i();
    size_t weightNNZElement = sparseCommon->args()->LookupByKey("NNZElement")->i();
    size_t weightBlockNumber = sparseCommon->args()->LookupByKey("blockNumber")->i();

    // MNN_PRINT("1x%d weightNNZElement%zu, weightBlockNumber:%zu\n", sparseBlockOC, weightNNZElement, weightBlockNumber);
    weight.reset(Tensor::createDevice<uint8_t>({ static_cast<int>(weightNNZElement + 1)}));   // one more element in case of weight are all zeros
    mNNZMap.reset(Tensor::createDevice<unsigned int>({oc / sparseBlockOC + oc % sparseBlockOC}));
    mDataOffsetMap.reset(Tensor::createDevice<int>({static_cast<int>(weightBlockNumber + 1)}));

    mValid = backend()->onAcquireBuffer(weight.get(), Backend::STATIC);
    mValid = mValid && backend()->onAcquireBuffer(mNNZMap.get(), Backend::STATIC);
    mValid = mValid && backend()->onAcquireBuffer(mDataOffsetMap.get(), Backend::STATIC);
    if(!mValid) {
        MNN_PRINT("in: %s, out of memory!\n", __FUNCTION__);
        return false;
    }
    // MNN_PRINT("oc:%d, sparseBlockOC:%d,\n", oc, sparseBlockOC);
    core->MNNPackForSparseQuantMatMul_B(weight->host<int8_t>(), mNNZMap->host<unsigned int>(),
                                       mDataOffsetMap->host<int>(), sparseBlockOC, weightOrigin->host<int8_t>(), oc, kernelCount, ic, eP);

    // MNN_PRINT("\nBCSR int8 weight:");
    // formatMatrix(weight->host<int8_t>(), {static_cast<int>(weightNNZElement)});
    // MNN_PRINT("\nBCSR int8 weight nnzmap:");
    // formatMatrix(mNNZMap->host<unsigned int>(), {oc / sparseBlockOC + oc % sparseBlockOC});
    // MNN_PRINT("\nBCSR int8 weight dataOffsetMap:");
    // formatMatrix(mDataOffsetMap->host<int>(), {static_cast<int>(weightBlockNumber + 1)});

    return true;
}

SparseConvInt8TiledExecutor::SparseConvInt8TiledExecutor(Backend* backend, const Convolution2D* convOp, std::shared_ptr<ResourceInt8> res) : ConvInt8TiledExecutor(backend, convOp, res) {

    std::shared_ptr<Tensor> weightOrigin;
    weightOrigin.swap(mResource->mWeightInt8);
    const SparseCommon* sparseCommon = convOp->sparseParameter();
    mValid = reorderWeight(backend, convOp->common(), weightOrigin, mResource->mWeightInt8, sparseCommon);
    if(!mValid) {
        return;
    }

    // choose int8 sparse gemm kernel
    auto sparseBlockOC = sparseCommon->args()->LookupByKey("sparseBlockOC")->i();
    auto core = static_cast<CPUBackend*>(backend)->int8Functions();
    mSparseQuantMatMulKernel = sparseBlockOC == 4 ? core->MNNPackedSparseQuantMatMulEpx4 : core->MNNPackedSparseQuantMatMulEpx1;

}

SparseConvInt8TiledExecutor::SparseConvInt8TiledExecutor(Backend* backend, const Convolution2DCommon* common,
                                                         const SparseConvInt8TiledExecutor& exe)
    : ConvInt8TiledExecutor(backend, common, exe),
      mNNZMap(exe.mNNZMap),
      mDataOffsetMap(exe.mDataOffsetMap),
      mSparseBlockOC(exe.mSparseBlockOC),
      mSparseQuantMatMulKernel(exe.mSparseQuantMatMulKernel) {
}

SparseConvInt8TiledExecutor::~SparseConvInt8TiledExecutor() {
    // Do nothing
}

bool SparseConvInt8TiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new SparseConvInt8TiledExecutor(bn, op->main_as_Convolution2D()->common(), *this);
    if (!exe->valid()) {
        return false;
    }
    *dst = exe;
    return true;
}

void SparseConvInt8TiledExecutor::getPackParameter(int* Unit, int* SrcUnit, int* DestUnit, const CoreInt8Functions* core) {
    core->MNNGetSparseQuantMatMulPackMode(DestUnit, Unit, SrcUnit);
}

ErrorCode SparseConvInt8TiledExecutor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {

    // Timer kernelTimer;

    ConvInt8TiledExecutor::onResize(inputs, outputs);

    int eP, lP, hP;
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    getPackParameter(&lP, &hP, &eP, core);
    int lSize = mIm2ColParamter.icDiv4 * mIm2ColParamter.packCUnit * mCommon->kernelX() * mCommon->kernelY();

    mIm2ColParamter.destICStride = mIm2ColParamter.icDiv4 * mIm2ColParamter.packCUnit * eP;

    mSparseQuantParam.eP = eP;
    mSparseQuantParam.l = lSize;
    mSparseQuantParam.h = mCommon->outputCount();
    mSparseQuantParam.aStride = eP * mSparseQuantParam.l;
    mSparseQuantParam.cStride = outputs[0]->batch() * outputs[0]->height() * outputs[0]->width() * static_cast<CPUBackend*>(backend())->functions()->pack;

    mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({mThreadNums, eP, UP_DIV(lSize, lP) * lP}));
    bool success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);

    // MNN_PRINT("sparse conv2d int8 resize: cost time: %llu us\n", kernelTimer.durationInUs());
    return NO_ERROR;
}

ErrorCode SparseConvInt8TiledExecutor::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Timer kernelTimer;
    const auto input = inputs[0];
    auto output      = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();

    int PackUnit = static_cast<CPUBackend*>(backend())->functions()->pack;
    auto sparseQuantIm2col = core->MNNSparseQuantIm2col;
    const int outputPlaneLen = output->height() * output->width();
    const int inputPlaneLen = input->width() * input->height();

    const int batch = input->batch();
    const int ocDivPack = UP_DIV(output->channel(), PackUnit);

    const auto inputDataPtr = input->host<int8_t>();
    const auto weightDataPtr = mResource->mWeightInt8->host<int8_t>();
    const auto NNZMapPtr     = mNNZMap->host<unsigned int>();
    const auto dataOffsetPtr = mDataOffsetMap->host<int>();
    auto im2colPtr           = mTempIm2ColBuffer->host<int8_t>();
    auto outputDataPtr       = output->host<int8_t>();
    QuanPostTreatParameters quanParam;
    quanParam.bias = mResource->mBiasInt32->host<int32_t>();
    quanParam.scale = mResource->mScaleFloat->host<float>();
    quanParam.maxValue = mResource->mClampMax;
    if (mResource->mRelu) {
        quanParam.minValue = mResource->mOutputZeroPoint;
    } else {
        quanParam.minValue = mResource->mClampMin;
    }
    // MNN_PRINT("outputPlaneLen: %d, reduce l:%zu, minValue:%d, maxValue:%d, mTileCount:%d\n", outputPlaneLen, mSparseQuantParam.l, quanParam.minValue, quanParam.maxValue, mTileCount);
    auto threadFunction = [&](int tId) {
        auto colAddr        = im2colPtr + tId * mTempIm2ColBuffer->stride(0);
        for (int bIndex = 0; bIndex < batch; ++bIndex) {
            const auto srcPtr = inputDataPtr + bIndex * PackUnit * inputPlaneLen;
            auto dstPtr       = outputDataPtr + bIndex * PackUnit * outputPlaneLen;

            for (int tIndex = tId; tIndex < mTileCount; tIndex += mThreadNums) {
                SparseQuantMatMulParam sparseQuantParam = mSparseQuantParam;
                const int xIndexStart  = tIndex * sparseQuantParam.eP;
                const int realDstCount = ALIMIN(outputPlaneLen - xIndexStart, sparseQuantParam.eP);
                sparseQuantParam.eSize = realDstCount;
                // im2col
                sparseQuantIm2col(colAddr, srcPtr, mResource->mInputZeroPoint, &mIm2ColParamter, (size_t*)&sparseQuantParam, xIndexStart);
                // MNN_PRINT("batch:%d, realDstCount:%d, InputZeroPoint:%d, inputdata matrix im2col:\n", bIndex, realDstCount, mResource->mInputZeroPoint);
                // formatMatrix(colAddr, {static_cast<int>(UP_DIV(realDstCount, sparseQuantParam.eP)), static_cast<int>(sparseQuantParam.l), static_cast<int>(sparseQuantParam.eP)});

#ifdef MNN_USE_SSE
                const int col_buffer_size = sparseQuantParam.aStride * sizeof(int8_t);
                MNNInt8ToUInt8(colAddr, col_buffer_size);
#endif
                auto outputInTilePtr = dstPtr + xIndexStart * PackUnit;
                // MNN_PRINT("bIndex:%d, offset:%zu, spmm sparseMatmul tile:\n", bIndex, outputInTilePtr - outputDataPtr);
                mSparseQuantMatMulKernel(outputInTilePtr, colAddr, weightDataPtr, (size_t*)&sparseQuantParam, &quanParam, NNZMapPtr, dataOffsetPtr);
                // formatMatrix(outputInTilePtr, {static_cast<int>(UP_DIV(sparseQuantParam.h, PackUnit)), realDstCount, PackUnit});
            }
        }
    };
    MNN_CONCURRENCY_BEGIN(tId, mThreadNums) {
        threadFunction((int)tId);
    }
    MNN_CONCURRENCY_END();
    // MNN_PRINT("sparse conv2d int8 execute: cost time: %llu us\n", kernelTimer.durationInUs());
    return NO_ERROR;
}

} // namespace MNN
