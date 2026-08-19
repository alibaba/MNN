//
//  MNNSpacemitIme2ConvInt8Executor.cpp
//  MNN
//
//  Created by MNN on 2026/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNSpacemitIme2ConvInt8Executor.hpp"

#include <algorithm>
#include <atomic>
#include <vector>

#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/riscv/rvv/MNNRvvFastPathUtils.hpp"
#include "core/Macro.h"

extern "C" void* MNNSpacemitIme2CreateLinearResource();
extern "C" void MNNSpacemitIme2DestroyLinearResource(void* context);
extern "C" int MNNSpacemitIme2BindLinearWeight(void* context, const int8_t* weight);
extern void MNNSpacemitIme2PrepackW4Weight(void* context, const int8_t* weight, size_t srcDepthQuad,
                                           size_t dstDepthQuad, size_t blockNum);
extern "C" size_t MNNSpacemitIme2PackedAHpBytes(size_t srcDepthQuad, size_t blockNum, size_t realCount);
extern "C" int MNNSpacemitIme2PackFloatAHpStridedRowsRangeDynamicQuant(uint8_t* dst, float* srcKernelSum,
                                                                       float* inputScale, float* quantScale,
                                                                       const float* src, size_t srcDepthQuad,
                                                                       size_t blockNum, size_t srcRows, size_t rowBegin,
                                                                       size_t rowEnd);
extern "C" int MNNSpacemitIme2LinearPackedAGemm(int8_t* dst, const uint8_t* packedA, const int8_t* weight,
                                                size_t srcDepthQuad, size_t dstStep, size_t dstDepthQuad,
                                                const QuanPostTreatParameters* post, size_t realCount, int threadCount,
                                                void* context);
extern "C" int MNNSpacemitIme2LinearFloatHpDecode(int8_t* dst, const float* src, const int8_t* weight,
                                                  size_t srcDepthQuad, size_t dstStep, size_t dstDepthQuad,
                                                  const QuanPostTreatParameters* post, void* context);

namespace MNN {

static constexpr size_t kSpacemitIme2MinOutputChannels = 512;
static constexpr size_t kSpacemitIme2MinKBlocks = 2;
static constexpr int kWeightOnlineReorder = 8;

class SpacemitIme2ConvInt8Executor::LinearResource {
public:
    LinearResource() : mContext(MNNSpacemitIme2CreateLinearResource()) {}

    ~LinearResource() {
        if (mContext != nullptr) {
            MNNSpacemitIme2DestroyLinearResource(mContext);
        }
    }

    bool valid() const { return mContext != nullptr && !mFailed.load(std::memory_order_acquire); }

    bool ready() const { return valid() && mPrepared.load(std::memory_order_acquire); }

    void fail() { mFailed.store(true, std::memory_order_release); }

    bool prepare(const int8_t* weight, size_t srcDepthQuad, size_t dstDepthQuad, size_t blockNum) {
        if (mContext == nullptr || weight == nullptr || MNNSpacemitIme2BindLinearWeight(mContext, weight) == 0) {
            fail();
            return false;
        }
        MNNSpacemitIme2PrepackW4Weight(mContext, weight, srcDepthQuad, dstDepthQuad, blockNum);
        mPrepared.store(true, std::memory_order_release);
        return true;
    }

    void* context() const { return mContext; }

private:
    void* mContext = nullptr;
    std::atomic<bool> mPrepared{false};
    std::atomic<bool> mFailed{false};
};

SpacemitIme2ConvInt8Executor::SpacemitIme2ConvInt8Executor(Backend* backend, const Op* op,
                                                           std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon,
                                                           bool isDynamicQuant)
    : SpacemitIme2ConvInt8Executor(backend, op, std::move(quanCommon), isDynamicQuant,
                                   std::make_shared<LinearResource>()) {}

SpacemitIme2ConvInt8Executor::SpacemitIme2ConvInt8Executor(Backend* backend, const Op* op,
                                                           std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon,
                                                           bool isDynamicQuant,
                                                           std::shared_ptr<LinearResource> linearResource)
    : DenseConvInt8TiledExecutor(
          backend, op, std::move(quanCommon), isDynamicQuant,
          [linearResource](bool weightReady, const std::shared_ptr<CPUConvolution::ResourceInt8>& resource,
                           const WeightReorderInfo& info) {
              const bool supportedLayout = weightReady && resource != nullptr && resource->mWeightBits == 4 &&
                                           info.ocBranch == 0 && info.unit == GEMM_INT8_UNIT &&
                                           info.packedSrcUnit == GEMM_INT8_SRC_UNIT / 2;
              if (!supportedLayout || resource->mWeightInt8 == nullptr ||
                  resource->mWeightInt8->host<int8_t>() == nullptr || !linearResource->valid()) {
                  linearResource->fail();
                  return;
              }
              linearResource->prepare(resource->mWeightInt8->host<int8_t>(), info.srcDepthQuad, info.dstDepthQuad,
                                      info.blockNum);
          }),
      mLinearResource(std::move(linearResource)) {}

SpacemitIme2ConvInt8Executor::SpacemitIme2ConvInt8Executor(Backend* backend, const Op* op,
                                                           const SpacemitIme2ConvInt8Executor& exe)
    : DenseConvInt8TiledExecutor(backend, op, exe), mLinearResource(exe.mLinearResource) {}

SpacemitIme2ConvInt8Executor::~SpacemitIme2ConvInt8Executor() {}

DenseConvInt8TiledExecutor* SpacemitIme2ConvInt8Executor::createClone(Backend* bn, const Op* op) const {
    return new SpacemitIme2ConvInt8Executor(bn, op, *this);
}

bool SpacemitIme2ConvInt8Executor::onSetupLinearFastPath(const std::vector<Tensor*>& inputs,
                                                         const std::vector<Tensor*>& outputs) {
    mUsePrefill = false;
    mDecodeBiasChecked = false;
    mDecodeBiasAllZero = false;
    mLinear1x1 = mIm2ColParamter.kernelX == 1 && mIm2ColParamter.kernelY == 1 && mIm2ColParamter.padX == 0 &&
                 mIm2ColParamter.padY == 0 && mIm2ColParamter.strideX == 1 && mIm2ColParamter.strideY == 1 &&
                 mIm2ColParamter.dilateX == 1 && mIm2ColParamter.dilateY == 1 &&
                 outputs[0]->width() == inputs[0]->width() && outputs[0]->height() == inputs[0]->height();
    if (mLinearResource == nullptr || !mLinearResource->valid()) {
        return false;
    }

    auto gcore = static_cast<CPUBackend*>(backend())->functions();
    const int threadCount = static_cast<CPUBackend*>(backend())->threadNumber();
    const int dynamicQuantOption =
        static_cast<CPUBackend*>(backend())->getRuntime()->hint().dynamicQuantOption % kWeightOnlineReorder;
    const int inputBlockNum = dynamicQuantOption == 2 ? mBlockNum : 1;
    const int inputPlane = inputs[0]->batch() * inputs[0]->width() * inputs[0]->height();
    const int inputChannels = inputs[0]->channel();
    const size_t realCount = outputs[0]->batch() * outputs[0]->width() * outputs[0]->height();
    const size_t srcDepthQuad = mBlockNum > 0 ? mIm2ColParamter.kernelCountUnit / mBlockNum : 0;
    const size_t dstDepthQuad = UP_DIV(outputs[0]->channel(), gcore->pack);
    const int srcUnit = mGemmUnits[1];
    const int dstXUnit = mGemmUnits[2];
    const int dstBytes = static_cast<CPUBackend*>(backend())->getBytes(backend(), outputs[0]);

    if (mResourceInt8->mWeightBits != 4 || !mResourceInt8->mDynamicQuant || !mUseBatchQuan || !mIm2ColBasedInt8 ||
        mMixedKernel || mOnlineReorderWeightSme || m4BitPtq || !mLinear1x1 || dynamicQuantOption == 2 ||
        inputBlockNum != 1 || realCount <= static_cast<size_t>(dstXUnit) ||
        realCount != static_cast<size_t>(inputPlane) || dstBytes != 4 || gcore->bytes != 4 || gcore->pack != 4 ||
        srcDepthQuad == 0 || srcDepthQuad % 2 != 0 || dstDepthQuad == 0 || dstDepthQuad % 8 != 0 || threadCount <= 0 ||
        mBlockNum <= 0 || srcUnit != 16 ||
        mIm2ColParamter.kernelCountUnit != static_cast<int>(srcDepthQuad) * mBlockNum ||
        mIm2ColParamter.kernelCountUnit * srcUnit != inputChannels) {
        return false;
    }
    const size_t countN = dstDepthQuad * static_cast<size_t>(gcore->pack);
    const size_t kBlocks = (srcDepthQuad / 2) * static_cast<size_t>(mBlockNum);
    mUsePrefill = countN >= kSpacemitIme2MinOutputChannels && kBlocks >= kSpacemitIme2MinKBlocks && kBlocks % 8 == 0;
    return mUsePrefill;
}

ErrorCode SpacemitIme2ConvInt8Executor::onExecute(const std::vector<Tensor*>& inputs,
                                                  const std::vector<Tensor*>& outputs) {
    if (tryExecuteFast(inputs, outputs)) {
        return NO_ERROR;
    }
    return DenseConvInt8TiledExecutor::onExecute(inputs, outputs);
}

bool SpacemitIme2ConvInt8Executor::tryExecuteFast(const std::vector<Tensor*>& inputs,
                                                  const std::vector<Tensor*>& outputs) {
    if (mLinearResource == nullptr || !mLinearResource->ready()) {
        return false;
    }
    const auto input = inputs[0];
    auto output = outputs[0];
    auto gcore = static_cast<CPUBackend*>(backend())->functions();
    const int unit = mGemmUnits[0];
    const int srcUnit = mGemmUnits[1];
    const int batch = input->batch();
    const int plane = output->batch() * mIm2ColParamter.oh * mIm2ColParamter.ow;
    const int inputPlane = batch * input->width() * input->height();
    const int inputChannels = input->channel();
    const int outputChannels = output->channel();
    const int packUnit = gcore->pack;
    const int dstBytes = static_cast<CPUBackend*>(backend())->getBytes(backend(), output);
    const int kernelCountUnit = mIm2ColParamter.kernelCountUnit;
    const size_t srcDepthQuad = kernelCountUnit / mBlockNum;
    const size_t dstDepthQuad = UP_DIV(outputChannels, packUnit);
    const size_t dstStep = plane * packUnit * dstBytes;
    auto dst = output->host<int8_t>();
    auto src = input->host<float>();
    auto weight = mResourceInt8->mWeightInt8->host<int8_t>();
    auto bias = mResourceInt8->mOriginBias->host<float>();

    if (mResourceInt8->mWeightBits == 4 && mResourceInt8->mDynamicQuant && !mUseBatchQuan && mInputBlockNum == 1 &&
        mIm2ColBasedInt8 && !mMixedKernel && !mOnlineReorderWeightSme && !m4BitPtq && mLinear1x1 && batch == 1 &&
        plane == 1 && inputPlane == 1 && dstBytes == 4 && gcore->bytes == 4 && packUnit == 4 && unit == 4 &&
        srcUnit == 16 && mBlockNum > 0 && srcDepthQuad > 0 && srcDepthQuad % 2 == 0 &&
        kernelCountUnit == static_cast<int>(srcDepthQuad) * mBlockNum && kernelCountUnit * srcUnit == inputChannels &&
        inputChannels % 256 == 0 && outputChannels > 0 && outputChannels % 32 == 0 &&
        dstDepthQuad * static_cast<size_t>(packUnit) == outputChannels && dst != nullptr && src != nullptr &&
        weight != nullptr && bias != nullptr) {
        if (!mDecodeBiasChecked) {
            mDecodeBiasAllZero = true;
            for (int i = 0; i < outputChannels; ++i) {
                if (bias[i] != 0.0f) {
                    mDecodeBiasAllZero = false;
                    break;
                }
            }
            mDecodeBiasChecked = true;
        }
        QuanPostTreatParameters post = {};
        post.blockNum = mBlockNum;
        post.biasFloat = mDecodeBiasAllZero ? nullptr : bias;
        post.useInt8 = 0;
        post.fp32minmax = mResourceInt8->mReluThreshold.data();
        if (MNNSpacemitIme2LinearFloatHpDecode(dst, src, weight, srcDepthQuad, dstStep, dstDepthQuad, &post,
                                               mLinearResource->context()) != 0) {
            return true;
        }
    }

    if (!mUsePrefill || dst == nullptr || src == nullptr || weight == nullptr || mBatchQuantInfo == nullptr ||
        mBatchQuantInfo->host<float>() == nullptr || mQScaleZero.ptr() == nullptr) {
        return false;
    }
    const size_t realCount = plane;
    const size_t packedBytes = MNNSpacemitIme2PackedAHpBytes(srcDepthQuad, mBlockNum, realCount);
    if (packedBytes == 0) {
        return false;
    }

    std::vector<uint8_t> packedA(packedBytes);
    std::vector<float> srcKernelSums(static_cast<size_t>(mBlockNum) * realCount);
    std::atomic<int> packedOk{1};
    auto inputScale = mBatchQuantInfo->host<float>();
    auto quantScale = reinterpret_cast<float*>(mQScaleZero.ptr());
    const int threadCount = static_cast<CPUBackend*>(backend())->threadNumber();
    const int rowBlockCount = UP_DIV(static_cast<int>(realCount), 4);
    const int packWorkers = ALIMIN(threadCount, rowBlockCount);
    MNNRvvFastPathParallelFor(backend(), packWorkers, [&](int workerId) {
        const int blocksPerWorker = UP_DIV(rowBlockCount, packWorkers);
        const int blockBegin = workerId * blocksPerWorker;
        const int blockEnd = ALIMIN(rowBlockCount, blockBegin + blocksPerWorker);
        const size_t rowBegin = static_cast<size_t>(blockBegin) * 4;
        const size_t rowEnd = ALIMIN(realCount, static_cast<size_t>(blockEnd) * 4);
        if (rowBegin < rowEnd) {
            const int ok = MNNSpacemitIme2PackFloatAHpStridedRowsRangeDynamicQuant(
                packedA.data(), srcKernelSums.data(), inputScale, quantScale, src, srcDepthQuad, mBlockNum, realCount,
                rowBegin, rowEnd);
            if (ok == 0) {
                packedOk.store(0, std::memory_order_relaxed);
            }
        }
    });
    if (packedOk.load(std::memory_order_relaxed) == 0) {
        return false;
    }

    QuanPostTreatParameters post = {};
    post.blockNum = mBlockNum;
    post.weightKernelSum = mResourceInt8->mWeightKernelSum->host<float>();
    post.biasFloat = bias;
    int32_t indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    post.indices = indices;
    post.useInt8 = 0;
    post.fp32minmax = mResourceInt8->mReluThreshold.data();
    post.inputScale = inputScale;
    post.inputBias = nullptr;
    post.srcKernelSum = srcKernelSums.data();
    post.accumBuffer = nullptr;
    return MNNSpacemitIme2LinearPackedAGemm(dst, packedA.data(), weight, srcDepthQuad, dstStep, dstDepthQuad, &post,
                                            realCount, threadCount, mLinearResource->context()) != 0;
}

Execution* MNNSpacemitIme2CreateInt8GemmExecution(Backend* backend, const Op* op,
                                                  std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon,
                                                  bool isDynamicQuant) {
#ifndef MNN_LOW_MEMORY
    return nullptr;
#else
    if (backend == nullptr || op == nullptr || quanCommon == nullptr || !isDynamicQuant || !quanCommon->canUseInt4) {
        return nullptr;
    }
    auto convolution = op->main_as_Convolution2D();
    if (convolution == nullptr || convolution->common() == nullptr) {
        return nullptr;
    }
    auto common = convolution->common();
    if (common->kernelX() != 1 || common->kernelY() != 1 || common->strideX() != 1 || common->strideY() != 1 ||
        common->dilateX() != 1 || common->dilateY() != 1) {
        return nullptr;
    }
    auto cpuBackend = static_cast<CPUBackend*>(backend);
    auto core = cpuBackend->functions();
    int unit = 0;
    int srcUnit = 0;
    int dstXUnit = 0;
    cpuBackend->int8GemmFunctions()->MNNGetGemmUnit(&unit, &srcUnit, &dstXUnit);
    if (core->bytes != 4 || core->pack != 4 || unit != GEMM_INT8_UNIT || srcUnit != GEMM_INT8_SRC_UNIT ||
        dstXUnit <= 0) {
        return nullptr;
    }
    return new SpacemitIme2ConvInt8Executor(backend, op, std::move(quanCommon), isDynamicQuant);
#endif
}

} // namespace MNN
