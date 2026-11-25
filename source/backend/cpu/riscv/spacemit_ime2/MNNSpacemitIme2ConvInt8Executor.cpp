//
//  MNNSpacemitIme2ConvInt8Executor.cpp
//  MNN
//
//  Created by MNN on 2026/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNSpacemitIme2ConvInt8Executor.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <atomic>
#include <vector>

#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/riscv/common/MNNRiscvFastPathUtils.hpp"
#include "backend/cpu/riscv/rvv/RVVBackend.hpp"
#include "core/Macro.h"

#include "MNNSpacemitIme2Half.hpp"

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
extern "C" int MNNSpacemitIme2LinearTiedBitDecode(int8_t* dst, const float* src, const int8_t* weight,
                                                  size_t srcDepthQuad, size_t dstStep, size_t dstDepthQuad,
                                                  const QuanPostTreatParameters* post, void* context);
extern "C" int MNNSpacemitIme2LinearPackedADecode(int8_t* dst, const uint8_t* packedA, const int8_t* weight,
                                                  size_t srcDepthQuad, size_t dstStep, size_t dstDepthQuad,
                                                  const QuanPostTreatParameters* post, void* context);
extern "C" int MNNSpacemitIme2SetDecodePackCache(int active);
extern "C" int MNNSpacemitIme2LinearFloatHpDecodeFusedGateUp(
    float* dst, uint8_t* outPackedA, const float* src, const int8_t* wGate, const float* biasGate,
    const float* clampGate, void* ctxGate, const int8_t* wUp, const float* biasUp, const float* clampUp, void* ctxUp,
    size_t srcDepthQuad, size_t dstDepthQuad, size_t blockNum);
struct MNNSpacemitIme2DecodeProjDesc {
    float* dst;
    const int8_t* weight;
    const float* bias;
    const float* clamp;
    void* context;
    size_t dstDepthQuad;
    size_t realCols;
};
extern "C" int MNNSpacemitIme2LinearFloatHpDecodeMultiProj(const float* src, size_t srcDepthQuad, size_t blockNum,
                                                           const MNNSpacemitIme2DecodeProjDesc* projs,
                                                           size_t projCount);

namespace MNN {

static constexpr size_t kSpacemitIme2MinOutputChannels = 512;
static constexpr size_t kSpacemitIme2MinKBlocks = 2;
static constexpr int kWeightOnlineReorder = 8;

namespace {
struct PrefillActivationPack {
    const float* parentInput = nullptr;
    const float* source = nullptr;
    size_t depthQuad = 0;
    size_t blocks = 0;
    size_t rows = 0;
    std::vector<uint8_t> data;
    std::vector<float> scales;
    std::vector<float> sums;

    bool prepare(Backend* backend, const float* src, size_t srcDepthQuad, size_t blockNum, size_t realCount) {
        if (source == src && depthQuad == srcDepthQuad && blocks == blockNum && rows == realCount) {
            return true;
        }
        source = nullptr;
        const size_t packedBytes = MNNSpacemitIme2PackedAHpBytes(srcDepthQuad, blockNum, realCount);
        if (packedBytes == 0) {
            return false;
        }
        data.resize(packedBytes);
        scales.resize(2 * realCount);
        sums.resize(blockNum * realCount);
        const int rowTiles = UP_DIV(static_cast<int>(realCount), 4);
        const int workers = std::min(static_cast<CPUBackend*>(backend)->threadNumber(), rowTiles);
        std::atomic<bool> packed{true};
        MNNRiscvFastPathParallelFor(backend, workers, [&](int worker) {
            const int perWorker = UP_DIV(rowTiles, workers);
            const size_t start = static_cast<size_t>(worker) * perWorker * 4;
            const size_t end = std::min(realCount, start + perWorker * 4);
            if (start < end && MNNSpacemitIme2PackFloatAHpStridedRowsRangeDynamicQuant(
                    data.data(), sums.data(), scales.data(), scales.data() + realCount, src,
                    srcDepthQuad, blockNum, realCount, start, end) == 0) {
                packed.store(false, std::memory_order_relaxed);
            }
        });
        if (!packed.load(std::memory_order_relaxed)) {
            return false;
        }
        source = src;
        depthQuad = srcDepthQuad;
        blocks = blockNum;
        rows = realCount;
        return true;
    }
};

thread_local PrefillActivationPack* gFusedProjectionPack = nullptr;
} // namespace

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
namespace {
struct DecodeGateUpCapture {
    int stage = 0; // 0 = ready for gate, 1 = gate captured, 2 = gate+up captured, -1 = disabled
    Execution* gateExe = nullptr;
    Execution* upExe = nullptr;
    Tensor* srcTensor = nullptr;
    Tensor* gateTensor = nullptr;
    Tensor* upTensor = nullptr;
    const float* src = nullptr;
    const int8_t* wGate = nullptr;
    const float* biasGate = nullptr;
    const float* clampGate = nullptr;
    void* ctxGate = nullptr;
    const void* gateDst = nullptr;
    const int8_t* wUp = nullptr;
    const float* biasUp = nullptr;
    const float* clampUp = nullptr;
    void* ctxUp = nullptr;
    const void* upDst = nullptr;
    size_t srcDepthQuad = 0, dstDepthQuad = 0, blockNum = 0, cols = 0;
    bool allowHandoff = false;
};
thread_local DecodeGateUpCapture* gDecodeGateUp = nullptr;

struct DecodeMultiProjCapture {
    static constexpr size_t kMaxProjs = 8;
    const float* src = nullptr;
    size_t blockNum = 0;
    size_t count = 0;
    MNNSpacemitIme2DecodeProjDesc descs[kMaxProjs];
    Execution* exes[kMaxProjs];
    Tensor* srcTensors[kMaxProjs];
    Tensor* dstTensors[kMaxProjs];
};
thread_local DecodeMultiProjCapture* gDecodeMultiProj = nullptr;

#if defined(__riscv_zvfh)
struct FusedGateUpCapture {
    int stage = 0; // 0 = ready for gate, 1 = gate captured, 2 = gate+up captured, -1 = disabled
    Execution* gateExe = nullptr;
    Execution* upExe = nullptr;
    Tensor* srcTensor = nullptr;
    Tensor* gateTensor = nullptr;
    Tensor* upTensor = nullptr;
    const float* src = nullptr;
    const uint8_t* weightGate = nullptr;
    const float* biasGate = nullptr;
    const float* clampGate = nullptr;
    const void* gateDst = nullptr;
    const uint8_t* weightUp = nullptr;
    const float* biasUp = nullptr;
    const float* clampUp = nullptr;
    const void* upDst = nullptr;
    size_t rows = 0, depth = 0, cols = 0, workers = 0;
    void* workspace = nullptr;
    bool allowHandoff = false;
};
thread_local FusedGateUpCapture* gFusedGateUp = nullptr;
#endif

class Ime2FusedProjection final : public Execution {
public:
    Ime2FusedProjection(Execution* inner, bool isGateUp)
        : Execution(inner->backend()), mInner(inner), mIsGateUp(isGateUp) {
        mValid = inner->valid();
        mNeedAllocIO = inner->needAllocIO();
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        return mInner->onResize(inputs, outputs);
    }
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        return runScoped(inputs, outputs, false);
    }
    ErrorCode onExecuteWithCallBack(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        return runScoped(inputs, outputs, true);
    }
    bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (!dst) {
            return mInner->onClone(bn, op, nullptr);
        }
        Execution* copy = nullptr;
        if (!mInner->onClone(bn, op, &copy) || !copy) {
            return false;
        }
        // The generic child clones itself directly without re-entering RVVBackend::onCreate.
        *dst = new Ime2FusedProjection(copy, mIsGateUp);
        return true;
    }
private:
    ErrorCode runScoped(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, bool withCallBack) {
        if (inputs[0]->length(0) == 1) {
            const int prevCache = MNNSpacemitIme2SetDecodePackCache(1);
            DecodeGateUpCapture decGateUp;
            decGateUp.allowHandoff = !withCallBack;
            DecodeMultiProjCapture decMulti;
            auto prevDecGateUp = gDecodeGateUp;
            auto prevDecMulti = gDecodeMultiProj;
            if (mIsGateUp) {
                gDecodeGateUp = &decGateUp;
            } else {
                gDecodeMultiProj = &decMulti;
            }
            auto result = withCallBack ? mInner->onExecuteWithCallBack(inputs, outputs)
                                       : mInner->onExecute(inputs, outputs);
            gDecodeGateUp = prevDecGateUp;
            gDecodeMultiProj = prevDecMulti;
            if (result == NO_ERROR && decMulti.count > 0) {
                const bool fusedOk =
                    decMulti.count >= 2 &&
                    MNNSpacemitIme2LinearFloatHpDecodeMultiProj(decMulti.src, 4, decMulti.blockNum, decMulti.descs,
                                                                decMulti.count) != 0;
                if (!fusedOk) {
                    for (size_t i = 0; i < decMulti.count; ++i) {
                        result = decMulti.exes[i]->onExecute({decMulti.srcTensors[i]}, {decMulti.dstTensors[i]});
                        if (result != NO_ERROR) {
                            break;
                        }
                    }
                }
            }
            MNNSpacemitIme2SetDecodePackCache(prevCache);
            return result;
        }
        if (inputs[0]->length(0) <= 4) {
            return withCallBack ? mInner->onExecuteWithCallBack(inputs, outputs)
                                : mInner->onExecute(inputs, outputs);
        }
        // Children share the input pack only within this projection. Its address
        // can be reused by the allocator on the next execution.
        PrefillActivationPack activation;
        activation.parentInput = inputs[0]->host<float>();
        auto previous = gFusedProjectionPack;
        gFusedProjectionPack = &activation;
#if defined(__riscv_zvfh)
        MNNIme2ActivationCache halfCache;
        auto prevHalfCache = MNNIme2SwapActivationCache(&halfCache);
        FusedGateUpCapture gateUp;
        gateUp.allowHandoff = !withCallBack;
        auto prevGateUp = gFusedGateUp;
        if (mIsGateUp) {
            gFusedGateUp = &gateUp;
        }
#endif
        auto result = withCallBack ? mInner->onExecuteWithCallBack(inputs, outputs)
                                   : mInner->onExecute(inputs, outputs);
#if defined(__riscv_zvfh)
        gFusedGateUp = prevGateUp;
        MNNIme2SwapActivationCache(prevHalfCache);
#endif
        gFusedProjectionPack = previous;
        return result;
    }
    std::unique_ptr<Execution> mInner;
    bool mIsGateUp = false;
};
} // namespace

Execution* MNNSpacemitIme2WrapFusedProjection(Execution* inner, bool isGateUp) {
    return new Ime2FusedProjection(inner, isGateUp);
}

bool MNNSpacemitIme2TryExecuteFusedGateUp(const Backend* backend, const void* gateDst, const void* upDst,
                                         Tensor* outTensor) {
    float* out = outTensor ? outTensor->host<float>() : nullptr;
    if (auto* dcap = gDecodeGateUp) {
        if (dcap->stage == 1) {
            dcap->stage = -1;
            if (dcap->gateExe && dcap->srcTensor && dcap->gateTensor) {
                dcap->gateExe->onExecute({dcap->srcTensor}, {dcap->gateTensor});
            }
            return false;
        }
        if (dcap->stage != 2 || dcap->gateDst != gateDst || dcap->upDst != upDst || !out) {
            return false;
        }
        dcap->stage = -1;
        uint8_t* outPackedA = nullptr;
        RvvHandoffContext* handoff = nullptr;
        if (dcap->allowHandoff && dcap->cols % 1024 == 0) {
            handoff = MNNRvvGetHandoffContext();
            if (handoff && handoff->backend == backend && handoff->activeGateUp.empty() &&
                handoff->gateUpToDownProj.count(outTensor) != 0) {
                thread_local std::vector<uint8_t> decodePackedASlab;
                const size_t bytes = (dcap->cols / 256) * (size_t(256) + 17 * sizeof(uint16_t));
                if (decodePackedASlab.size() < bytes) {
                    decodePackedASlab.resize(bytes);
                }
                outPackedA = decodePackedASlab.data();
            }
        }
        if (MNNSpacemitIme2LinearFloatHpDecodeFusedGateUp(
                out, outPackedA, dcap->src, dcap->wGate, dcap->biasGate, dcap->clampGate, dcap->ctxGate, dcap->wUp,
                dcap->biasUp, dcap->clampUp, dcap->ctxUp, dcap->srcDepthQuad, dcap->dstDepthQuad,
                dcap->blockNum) != 0) {
            if (outPackedA != nullptr && handoff != nullptr) {
                auto& info = handoff->activeGateUp[outTensor];
                info.ptr = out;
                info.packedA = outPackedA;
                info.rows = 1;
                info.depth = dcap->cols;
            }
            return true;
        }
        if (dcap->gateExe && dcap->upExe && dcap->srcTensor && dcap->gateTensor && dcap->upTensor) {
            dcap->gateExe->onExecute({dcap->srcTensor}, {dcap->gateTensor});
            dcap->upExe->onExecute({dcap->srcTensor}, {dcap->upTensor});
        }
        return false;
    }
#if defined(__riscv_zvfh)
    auto* cap = gFusedGateUp;
    if (!cap) {
        return false;
    }
    if (cap->stage == 1) {
        cap->stage = -1;
        if (cap->gateExe && cap->srcTensor && cap->gateTensor) {
            cap->gateExe->onExecute({cap->srcTensor}, {cap->gateTensor});
        }
        return false;
    }
    if (cap->stage != 2 || cap->gateDst != gateDst || cap->upDst != upDst || !out) {
        return false;
    }
    cap->stage = -1;
    if (cap->allowHandoff && cap->cols % 512 == 0) {
        if (auto* handoff = MNNRvvGetHandoffContext()) {
            if (handoff->backend == backend && handoff->activeGateUp.empty() &&
                handoff->gateUpToDownProj.count(outTensor) != 0) {
                if (const void* tiledFp16 = MNNIme2HalfFusedGateUpToFp16(
                        cap->src, cap->biasGate, cap->clampGate, cap->weightGate, cap->biasUp, cap->clampUp,
                        cap->weightUp, cap->rows, cap->depth, cap->cols, cap->workers, cap->workspace)) {
                    auto& info = handoff->activeGateUp[outTensor];
                    info.ptr = out;
                    info.packedA = tiledFp16;
                    info.rows = cap->rows;
                    info.depth = cap->cols;
                    return true;
                }
            }
        }
    }
    if (MNNIme2HalfFusedGateUp(cap->src, out, cap->biasGate, cap->clampGate, cap->weightGate, cap->biasUp,
                               cap->clampUp, cap->weightUp, cap->rows, cap->depth, cap->cols, cap->workers,
                               cap->workspace)) {
        return true;
    }
    // Fallback on range overflow: materialize gate and up normally so RvvMulSilu proceeds safely.
    if (cap->gateExe && cap->upExe && cap->srcTensor && cap->gateTensor && cap->upTensor) {
        cap->gateExe->onExecute({cap->srcTensor}, {cap->gateTensor});
        cap->upExe->onExecute({cap->srcTensor}, {cap->upTensor});
    }
    return false;
#else
    (void)backend;
    (void)gateDst;
    (void)upDst;
    (void)outTensor;
    return false;
#endif
}
#endif

class SpacemitIme2ConvInt8Executor::LinearResource {
public:
    explicit LinearResource(bool asymmetric)
        : mContext(MNNSpacemitIme2CreateLinearResource()), mAllowNarrow(asymmetric) {}

    ~LinearResource() {
        if (mContext != nullptr) {
            MNNSpacemitIme2DestroyLinearResource(mContext);
        }
    }

    bool valid() const { return mContext != nullptr && !mFailed.load(std::memory_order_acquire); }

    bool ready() const { return valid() && mPrepared.load(std::memory_order_acquire); }

    void fail() { mFailed.store(true, std::memory_order_release); }

    bool prepare(const int8_t* weight, size_t srcDepthQuad, size_t dstDepthQuad, size_t blockNum,
                 const float* bias) {
        if (mAllowNarrow && srcDepthQuad == 4 && dstDepthQuad == 4 && blockNum == 32) {
            return prepareNarrow(weight, bias);
        }
        if (mContext == nullptr || weight == nullptr || MNNSpacemitIme2BindLinearWeight(mContext, weight) == 0) {
            fail();
            return false;
        }
        // Block-64 projections select the prefill or decode layout at execution time.
        if (srcDepthQuad != 4) {
            MNNSpacemitIme2PrepackW4Weight(mContext, weight, srcDepthQuad, dstDepthQuad, blockNum);
        }
        mPrepared.store(true, std::memory_order_release);
        return true;
    }

    void* context() const { return mContext; }
    bool allowNarrow() const { return mAllowNarrow; }
    const int8_t* narrowWeight() const { return mNarrowWeight.empty() ? nullptr : mNarrowWeight.data(); }
    const float* narrowBias() const { return mNarrowBias.data(); }

private:
    bool prepareNarrow(const int8_t* weight, const float* bias) {
        // One immutable packed weight is shared by all shape-specific Execution clones.
        std::lock_guard<std::mutex> lock(mNarrowPrepareMutex);
        if (ready()) {
            if (mNarrowSource != weight) {
                fail();
                return false;
            }
            return true;
        }
        if (mContext == nullptr || weight == nullptr || bias == nullptr) {
            fail();
            return false;
        }
        // Generic W4 is [C4, block, K/16 * 32 bytes + scale[4] + offset[4]].
        // K2048/block64 gives 32 blocks and 160 bytes per block. Copy the real
        // four C4 groups, then pad four more groups with zero q/scale/offset.
        constexpr size_t realBytes = 4 * 32 * (4 * 32 + 2 * 4 * sizeof(float));
        mNarrowWeight.assign(2 * realBytes, 0);
        std::memcpy(mNarrowWeight.data(), weight, realBytes);
        mNarrowBias.fill(0.0f);
        std::memcpy(mNarrowBias.data(), bias, 16 * sizeof(float));
        if (MNNSpacemitIme2BindLinearWeight(mContext, mNarrowWeight.data()) == 0) {
            fail();
            return false;
        }
        mNarrowSource = weight;
        mPrepared.store(true, std::memory_order_release);
        return true;
    }

    void* mContext = nullptr;
    const bool mAllowNarrow;
    std::mutex mNarrowPrepareMutex;
    const int8_t* mNarrowSource = nullptr;
    std::vector<int8_t> mNarrowWeight;
    std::array<float, 32> mNarrowBias;
    std::atomic<bool> mPrepared{false};
    std::atomic<bool> mFailed{false};
};

SpacemitIme2ConvInt8Executor::SpacemitIme2ConvInt8Executor(Backend* backend, const Op* op,
                                                           std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon,
                                                           bool isDynamicQuant)
    : SpacemitIme2ConvInt8Executor(backend, op, quanCommon, isDynamicQuant,
                                   std::make_shared<LinearResource>(quanCommon && quanCommon->asymmetric)) {}

SpacemitIme2ConvInt8Executor::SpacemitIme2ConvInt8Executor(Backend* backend, const Op* op,
                                                           std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon,
                                                           bool isDynamicQuant,
                                                           std::shared_ptr<LinearResource> linearResource)
    : DenseConvInt8TiledExecutor(
          backend, op, std::move(quanCommon), isDynamicQuant,
          [linearResource](bool weightReady, const std::shared_ptr<CPUConvolution::ResourceInt8>& resource,
                           const WeightReorderInfo& info) {
              const bool supportedLayout = weightReady && resource != nullptr && resource->mWeightBits == 4 &&
                                           !resource->mWeightQuantInfoMode &&
                                           info.ocBranch == 0 && info.unit == GEMM_INT8_UNIT &&
                                           info.packedSrcUnit == GEMM_INT8_SRC_UNIT / 2;
              if (!supportedLayout || resource->mWeightInt8 == nullptr ||
                  resource->mWeightInt8->host<int8_t>() == nullptr || !linearResource->valid()) {
                  linearResource->fail();
                  return;
              }
              linearResource->prepare(resource->mWeightInt8->host<int8_t>(), info.srcDepthQuad, info.dstDepthQuad,
                                      info.blockNum, resource->mOriginBias == nullptr ? nullptr :
                                          resource->mOriginBias->host<float>());
          }),
      mLinearResource(std::move(linearResource)),
      mIsLmHead(op != nullptr && op->name() != nullptr &&
                std::strcmp(op->name()->c_str(), "/lm/lm_head/Linear") == 0) {}

SpacemitIme2ConvInt8Executor::SpacemitIme2ConvInt8Executor(Backend* backend, const Op* op,
                                                           const SpacemitIme2ConvInt8Executor& exe)
    : DenseConvInt8TiledExecutor(backend, op, exe),
      mLinearResource(exe.mLinearResource),
      mIsLmHead(exe.mIsLmHead) {}

SpacemitIme2ConvInt8Executor::~SpacemitIme2ConvInt8Executor() {}

DenseConvInt8TiledExecutor* SpacemitIme2ConvInt8Executor::createClone(Backend* bn, const Op* op) const {
    return new SpacemitIme2ConvInt8Executor(bn, op, *this);
}

bool SpacemitIme2ConvInt8Executor::onSetupLinearFastPath(const std::vector<Tensor*>& inputs,
                                                         const std::vector<Tensor*>& outputs) {
    mHalfWorkspace = {};
    mUsePrefill = false;
    mUseNarrowPrefill = false;
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
    const bool narrow = mLinearResource->allowNarrow() && inputChannels == 2048 &&
                        outputs[0]->channel() == 16 && srcDepthQuad == 4 && mBlockNum == 32;

    if (mResourceInt8->mWeightBits == 4 && mResourceInt8->mDynamicQuant && !mUseBatchQuan && mIm2ColBasedInt8 &&
        !mMixedKernel && !mOnlineReorderWeightSme && !m4BitPtq && mLinear1x1 && dynamicQuantOption != 2 &&
        inputBlockNum == 1 && realCount == 1 && inputPlane == 1 && dstBytes == 4 && gcore->bytes == 4 &&
        gcore->pack == 4 && srcUnit == 16 && srcDepthQuad == 4 && mBlockNum > 0 && mBlockNum % 4 == 0 &&
        mIm2ColParamter.kernelCountUnit == static_cast<int>(srcDepthQuad) * mBlockNum &&
        mIm2ColParamter.kernelCountUnit * srcUnit == inputChannels && inputChannels % 1024 == 0 &&
        outputs[0]->channel() > 0 && outputs[0]->channel() % 32 == 0) {
        MNNRvvRegisterHalfConv(backend(), this);
    }
    if (mResourceInt8->mWeightBits != 4 || !mResourceInt8->mDynamicQuant || !mUseBatchQuan || !mIm2ColBasedInt8 ||
        mMixedKernel || mOnlineReorderWeightSme || m4BitPtq || !mLinear1x1 || dynamicQuantOption == 2 ||
        inputBlockNum != 1 || realCount <= static_cast<size_t>(dstXUnit) ||
        realCount != static_cast<size_t>(inputPlane) || dstBytes != 4 || gcore->bytes != 4 || gcore->pack != 4 ||
        srcDepthQuad == 0 || srcDepthQuad % 2 != 0 || dstDepthQuad == 0 ||
        (dstDepthQuad % 8 != 0 && !narrow) || threadCount <= 0 ||
        mBlockNum <= 0 || srcUnit != 16 ||
        mIm2ColParamter.kernelCountUnit != static_cast<int>(srcDepthQuad) * mBlockNum ||
        mIm2ColParamter.kernelCountUnit * srcUnit != inputChannels) {
        return false;
    }
    if (narrow) {
        // Shape-dependent scratch belongs to the Execution, not the shared immutable resource.
        const size_t packedBytes = MNNSpacemitIme2PackedAHpBytes(4, 32, realCount);
        if (packedBytes == 0) {
            return false;
        }
        mNarrowOutput.resize(32 * realCount);
        mUseNarrowPrefill = true;
    } else {
        const size_t countN = dstDepthQuad * static_cast<size_t>(gcore->pack);
        const size_t kBlocks = (srcDepthQuad / 2) * static_cast<size_t>(mBlockNum);
        mUsePrefill = countN >= kSpacemitIme2MinOutputChannels && kBlocks >= kSpacemitIme2MinKBlocks && kBlocks % 8 == 0;
    }
#if defined(__riscv_zvfh)
    auto cpu = static_cast<CPUBackend*>(backend());
    if ((narrow || (mUsePrefill && outputs[0]->channel() >= 256)) && srcDepthQuad == 4 &&
        MNNIme2::halfPrefillEnabled() && cpu->precisionMode() == BackendConfig::Precision_Low) {
        const auto bytes = MNNIme2HalfProjectionBytes(realCount, inputChannels, narrow ? 32 : outputs[0]->channel(), threadCount);
        if (bytes) {
            auto alloc = cpu->getBufferAllocator();
            mHalfWorkspace = alloc->alloc(bytes);
            // Split-K and reduction complete within this child execution.
            if (!mHalfWorkspace.invalid()) {
                alloc->free(mHalfWorkspace);
                if (!narrow && mUsePrefill && inputChannels % 512 == 0) {
                    MNNRvvRegisterHalfConv(backend(), this);
                }
            }
        }
    }
#endif
    return mUseNarrowPrefill || mUsePrefill;
}

ErrorCode SpacemitIme2ConvInt8Executor::onExecute(const std::vector<Tensor*>& inputs,
                                                  const std::vector<Tensor*>& outputs) {
    if (tryExecuteFast(inputs, outputs)) {
        return NO_ERROR;
    }
    return DenseConvInt8TiledExecutor::onExecute(inputs, outputs);
}

bool SpacemitIme2ConvInt8Executor::tryExecuteNarrowPrefill(const std::vector<Tensor*>& inputs,
                                                           const std::vector<Tensor*>& outputs) {
    auto output = outputs[0];
    const size_t realCount = output->batch() * output->width() * output->height();
    const auto weight = mLinearResource->narrowWeight();
    const auto src = inputs[0]->host<float>();
    auto dst = output->host<float>();
    if (weight == nullptr || src == nullptr || dst == nullptr || mBatchQuantInfo == nullptr ||
        mBatchQuantInfo->host<float>() == nullptr || mQScaleZero.ptr() == nullptr ||
        mNarrowOutput.size() != 32 * realCount) {
        return false;
    }
#if defined(__riscv_zvfh)
    if (!mHalfWorkspace.invalid()) {
        auto weights = MNNIme2GetBlock64Weight(mLinearResource->context(), weight, 2048, 32);
        const int threads = static_cast<CPUBackend*>(backend())->threadNumber();
        if (MNNIme2HalfProjection(src, mNarrowOutput.data(), mLinearResource->narrowBias(),
                                  mResourceInt8->mReluThreshold.data(), weights, realCount, 2048, 32, threads, mHalfWorkspace.ptr())) {
            std::memcpy(dst, mNarrowOutput.data(), 16 * realCount * sizeof(float));
            return true;
        }
    }
#endif
    PrefillActivationPack local;
    auto& packed = gFusedProjectionPack && gFusedProjectionPack->parentInput == src ? *gFusedProjectionPack : local;
    if (!packed.prepare(backend(), src, 4, 32, realCount)) {
        return false;
    }
    const int threadCount = static_cast<CPUBackend*>(backend())->threadNumber();
    QuanPostTreatParameters post = {};
    post.blockNum = 32;
    post.useInt8 = 0;
    post.biasFloat = mLinearResource->narrowBias();
    post.fp32minmax = mResourceInt8->mReluThreshold.data();
    post.inputScale = packed.scales.data();
    post.srcKernelSum = packed.sums.data();
    // inputBias is null: no weightKernelSum correction or per-channel index array is used.
    const size_t dstStep = realCount * 4 * sizeof(float);
    if (MNNSpacemitIme2LinearPackedAGemm(reinterpret_cast<int8_t*>(mNarrowOutput.data()), packed.data.data(),
            weight, 4, dstStep, 8, &post, realCount, threadCount, mLinearResource->context()) == 0) {
        return false;
    }
    // C4 layout is [channel/4, row, lane]. The four real channel groups form
    // one contiguous prefix; padded channels never touch the real output tensor.
    std::memcpy(dst, mNarrowOutput.data(), realCount * 16 * sizeof(float));
    return true;
}

bool SpacemitIme2ConvInt8Executor::tryExecuteFast(const std::vector<Tensor*>& inputs,
                                                  const std::vector<Tensor*>& outputs) {
    if (mLinearResource == nullptr || !mLinearResource->ready()) {
        return false;
    }
    if (mUseNarrowPrefill) {
        return tryExecuteNarrowPrefill(inputs, outputs);
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
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
        if (auto* dcap = gDecodeGateUp) {
            if (srcDepthQuad == 4 && mBlockNum % 4 == 0) {
                if (dcap->stage == 0) {
                    dcap->stage = 1;
                    dcap->gateExe = this;
                    dcap->srcTensor = inputs[0];
                    dcap->gateTensor = outputs[0];
                    dcap->src = src;
                    dcap->wGate = weight;
                    dcap->biasGate = post.biasFloat;
                    dcap->clampGate = post.fp32minmax;
                    dcap->ctxGate = mLinearResource->context();
                    dcap->gateDst = dst;
                    dcap->srcDepthQuad = srcDepthQuad;
                    dcap->dstDepthQuad = dstDepthQuad;
                    dcap->blockNum = mBlockNum;
                    dcap->cols = outputChannels;
                    return true;
                }
                if (dcap->stage == 1 && dcap->src == src && dcap->srcDepthQuad == srcDepthQuad &&
                    dcap->dstDepthQuad == dstDepthQuad && dcap->blockNum == static_cast<size_t>(mBlockNum)) {
                    dcap->stage = 2;
                    dcap->upExe = this;
                    dcap->upTensor = outputs[0];
                    dcap->wUp = weight;
                    dcap->biasUp = post.biasFloat;
                    dcap->clampUp = post.fp32minmax;
                    dcap->ctxUp = mLinearResource->context();
                    dcap->upDst = dst;
                    return true;
                }
                if (dcap->stage == 1) {
                    dcap->stage = -1;
                    if (dcap->gateExe && dcap->srcTensor && dcap->gateTensor) {
                        dcap->gateExe->onExecute({dcap->srcTensor}, {dcap->gateTensor});
                    }
                }
            }
        }
        if (auto* dm = gDecodeMultiProj) {
            if (srcDepthQuad == 4 && mBlockNum % 4 == 0 && dm->count < DecodeMultiProjCapture::kMaxProjs &&
                (dm->count == 0 || (dm->src == src && dm->blockNum == static_cast<size_t>(mBlockNum)))) {
                dm->src = src;
                dm->blockNum = mBlockNum;
                const size_t idx = dm->count++;
                dm->descs[idx] = {reinterpret_cast<float*>(dst), weight, post.biasFloat, post.fp32minmax,
                                  mLinearResource->context(), dstDepthQuad, static_cast<size_t>(outputChannels)};
                dm->exes[idx] = this;
                dm->srcTensors[idx] = inputs[0];
                dm->dstTensors[idx] = outputs[0];
                return true;
            }
        }
        if (auto* handoff = MNNRvvGetHandoffContext()) {
            if (handoff->backend == backend()) {
                auto it = handoff->activeGateUp.find(inputs[0]);
                if (it != handoff->activeGateUp.end()) {
                    const auto info = it->second;
                    handoff->activeGateUp.erase(it);
                    if (info.rows == 1 && info.depth == static_cast<size_t>(inputChannels) && info.ptr == src &&
                        info.packedA != nullptr &&
                        MNNSpacemitIme2LinearPackedADecode(dst, static_cast<const uint8_t*>(info.packedA), weight,
                                                           srcDepthQuad, dstStep, dstDepthQuad, &post,
                                                           mLinearResource->context()) != 0) {
                        return true;
                    }
                }
            }
        }
#endif
        if (mIsLmHead && MNNIme2::tiedBitEnabled() &&
            MNNSpacemitIme2LinearTiedBitDecode(dst, src, weight, srcDepthQuad, dstStep, dstDepthQuad, &post,
                                               mLinearResource->context()) != 0) {
            return true;
        }
        if (MNNSpacemitIme2LinearFloatHpDecode(dst, src, weight, srcDepthQuad, dstStep, dstDepthQuad, &post,
                                               mLinearResource->context()) != 0) {
            return true;
        }
    }
    if (mResourceInt8->mWeightBits == 4 && mResourceInt8->mDynamicQuant && !mUseBatchQuan && mInputBlockNum == 1 &&
        mIm2ColBasedInt8 && !mMixedKernel && !mOnlineReorderWeightSme && !m4BitPtq && mLinear1x1 && batch == 1 &&
        plane == 1 && inputPlane == 1 && dstBytes == 4 && gcore->bytes == 4 && packUnit == 4 && unit == 4 &&
        srcUnit == 16 && inputChannels == 2048 && outputChannels == 16 && srcDepthQuad == 4 && mBlockNum == 32 &&
        mLinearResource->narrowWeight() != nullptr && dst != nullptr && src != nullptr && bias != nullptr) {
        if (!mDecodeBiasChecked) {
            mDecodeBiasAllZero = true;
            for (int i = 0; i < 16; ++i) {
                if (bias[i] != 0.0f) {
                    mDecodeBiasAllZero = false;
                    break;
                }
            }
            mDecodeBiasChecked = true;
        }
        QuanPostTreatParameters post = {};
        post.blockNum = 32;
        post.biasFloat = mDecodeBiasAllZero ? nullptr : mLinearResource->narrowBias();
        post.useInt8 = 0;
        post.fp32minmax = mResourceInt8->mReluThreshold.data();
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
        if (auto* dm = gDecodeMultiProj) {
            if (dm->count < DecodeMultiProjCapture::kMaxProjs &&
                (dm->count == 0 || (dm->src == src && dm->blockNum == 32))) {
                dm->src = src;
                dm->blockNum = 32;
                const size_t idx = dm->count++;
                dm->descs[idx] = {reinterpret_cast<float*>(dst), mLinearResource->narrowWeight(), post.biasFloat,
                                  post.fp32minmax, mLinearResource->context(), 8, 16};
                dm->exes[idx] = this;
                dm->srcTensors[idx] = inputs[0];
                dm->dstTensors[idx] = outputs[0];
                return true;
            }
        }
#endif
        alignas(64) float tmpOut[32];
        if (MNNSpacemitIme2LinearFloatHpDecode(reinterpret_cast<int8_t*>(tmpOut), src, mLinearResource->narrowWeight(),
                                               4, GEMM_INT8_UNIT * sizeof(float), 8, &post,
                                               mLinearResource->context()) != 0) {
            std::memcpy(dst, tmpOut, 16 * sizeof(float));
            return true;
        }
    }

    if (!mUsePrefill || dst == nullptr || src == nullptr || weight == nullptr || mBatchQuantInfo == nullptr ||
        mBatchQuantInfo->host<float>() == nullptr || mQScaleZero.ptr() == nullptr) {
        return false;
    }
    const size_t realCount = plane;
#if defined(__riscv_zvfh)
    const auto cpu = static_cast<CPUBackend*>(backend());
    const int workers = cpu->threadNumber();
    if (!mHalfWorkspace.invalid()) {
        auto weights = MNNIme2GetBlock64Weight(mLinearResource->context(), weight, inputChannels, outputChannels);
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
        if (auto* cap = gFusedGateUp) {
            if (cap->stage == 0 && weights) {
                cap->stage = 1;
                cap->gateExe = this;
                cap->srcTensor = inputs[0];
                cap->gateTensor = outputs[0];
                cap->src = src;
                cap->weightGate = weights;
                cap->biasGate = bias;
                cap->clampGate = mResourceInt8->mReluThreshold.data();
                cap->gateDst = dst;
                cap->rows = realCount;
                cap->depth = inputChannels;
                cap->cols = outputChannels;
                cap->workers = workers;
                cap->workspace = mHalfWorkspace.ptr();
                return true;
            }
            if (cap->stage == 1 && weights && cap->src == src && cap->rows == realCount &&
                cap->depth == static_cast<size_t>(inputChannels) &&
                cap->cols == static_cast<size_t>(outputChannels) &&
                cap->workers == static_cast<size_t>(workers)) {
                cap->stage = 2;
                cap->upExe = this;
                cap->upTensor = outputs[0];
                cap->weightUp = weights;
                cap->biasUp = bias;
                cap->clampUp = mResourceInt8->mReluThreshold.data();
                cap->upDst = dst;
                return true;
            }
            if (cap->stage == 1) {
                // Unexpected shape/weight mismatch: flush deferred gate projection before continuing.
                cap->stage = -1;
                if (cap->gateExe && cap->srcTensor && cap->gateTensor) {
                    cap->gateExe->onExecute({cap->srcTensor}, {cap->gateTensor});
                }
            }
        }
#endif
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
        if (auto* handoff = MNNRvvGetHandoffContext()) {
            if (handoff->backend == backend()) {
                auto it = handoff->activeGateUp.find(inputs[0]);
                if (it != handoff->activeGateUp.end()) {
                    const auto info = it->second;
                    handoff->activeGateUp.erase(it);
                    if (info.ptr == src && info.rows == realCount &&
                        info.depth == static_cast<size_t>(inputChannels) && info.packedA) {
                        if (MNNIme2HalfProjectionFromFp16(info.packedA, reinterpret_cast<float*>(dst), bias,
                                                          mResourceInt8->mReluThreshold.data(), weights, realCount,
                                                          inputChannels, outputChannels, workers, mHalfWorkspace.ptr())) {
                            return true;
                        }
                        // Fallback if GEMM overflowed: restore FP32 NC4HW4 input before generic path.
                        MNNIme2ConvertTiledFp16ToC4(info.packedA, inputs[0]->host<float>(), realCount, inputChannels);
                    }
                }
            }
        }
#endif
        if (MNNIme2HalfProjection(src, reinterpret_cast<float*>(dst), bias, mResourceInt8->mReluThreshold.data(),
                                  weights, realCount, inputChannels, outputChannels, workers, mHalfWorkspace.ptr())) {
            return true;
        }
    }
#endif
    PrefillActivationPack local;
    auto& packed = gFusedProjectionPack && gFusedProjectionPack->parentInput == src ? *gFusedProjectionPack : local;
    if (!packed.prepare(backend(), src, srcDepthQuad, mBlockNum, realCount)) {
        return false;
    }

    const int threadCount = static_cast<CPUBackend*>(backend())->threadNumber();

    QuanPostTreatParameters post = {};
    post.blockNum = mBlockNum;
    post.weightKernelSum = mResourceInt8->mWeightKernelSum->host<float>();
    post.biasFloat = bias;
    int32_t indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    post.indices = indices;
    post.useInt8 = 0;
    post.fp32minmax = mResourceInt8->mReluThreshold.data();
    post.inputScale = packed.scales.data();
    post.inputBias = nullptr;
    post.srcKernelSum = packed.sums.data();
    post.accumBuffer = nullptr;
    return MNNSpacemitIme2LinearPackedAGemm(dst, packed.data.data(), weight, srcDepthQuad, dstStep, dstDepthQuad, &post,
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
