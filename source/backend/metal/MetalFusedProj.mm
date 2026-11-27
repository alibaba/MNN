//
//  MetalFusedProj.mm
//  MNN
//
//  Metal execution for the export-time fused projection op (FusedLinear).
//  Owns the grouped Decode GEMV and wide Prefill dispatches for compatible packed projections.
//  Otherwise dispatches the Conv1x1, MUL_SILU and LayerNorm children as needed.
//

#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MetalExecution.hpp"
#ifdef MNN_LOW_MEMORY
#import "backend/metal/MetalConv1x1WQuant.hpp"
#import "backend/metal/MetalEnv.hpp"
#import "ConvSimdGroupShader.hpp"
#endif
#import "backend/metal/MetalLayerNorm.hpp"
#import "core/FusedProjCommon.hpp"
#import "MNN_generated.h"
#import "core/TensorUtils.hpp"
#import "core/Macro.h"
#include <algorithm>
#include <array>
#include <cstring>

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

// Standalone row scale for FusedLinearParam.row_mul, used only on routes whose
// kernel has no epilogue fold (for example, the prefill GEMM tiles).
// In the Metal C4 layout, each channel quad stores all tokens consecutively.
// Each thread updates one vector in place; its token index is index % tokens.
static const char* const gFusedProjRowScale = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void row_scale(device T4* output [[buffer(0)]], const device T4* scale [[buffer(1)]],
                      constant int2& shape [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index >= uint(shape.y)) return;
    output[index] = T4(float4(output[index]) * float(scale[index % uint(shape.x)].x));
}
)metal";

#ifdef MNN_LOW_MEMORY
// kMetalSimdGroupWidth and the GEMV geometry rules come from MetalConv1x1Decode.hpp,
// shared with the Conv1x1 decode path so the two cannot drift apart.

// The decode split-K bodies (oc8Nsg4SplitK2, oc16Nsg8SplitK2 and the wide gate/up siluSplitKCount) are
// calibrated on FP16 storage, but nothing in them is FP16-specific: in
// conv1x1_gemv_g4m1_2sg_wquant_sg the input/output/bias/dequant operands are ftype4, the
// packed weights follow the stored bit width, and the split-K reduction and LN staging run
// in FLOAT4 (MNN_METAL_FLOAT32_COMPUTER is always 1). FP32 storage already runs this source
// through the plain and oc16Nsg4SplitK1 bodies, so the wider variants were never blocked by type -
// only by a profile that had not been measured.
// Selection is therefore the only thing this widens, and only where it was measured:
// Conv1x1GpuTier::decodeFp32Validated. loadDecodePipeline still takes the real fp16 flag, so
// ftype, cache key, buffer layout, device tier, shape/alignment gates, thread counts, the
// shared-memory resource check and every fallback stay as they were; a body whose pipeline
// fails to compile still falls back to the unfused single-projection path.
static inline bool decodeSplitKPrecisionOk(const Conv1x1DeviceProfile& profile, bool fp16Storage) {
    return fp16Storage || profile.tier.decodeFp32Validated;
}

// oc16Nsg8SplitK2 needs ~80% lane-group occupancy and a K sweep (or weight size) that
// repays the reduction barrier.
static bool qkvOc16Sg8K2Admissible(int blockCount, int inputDepthQuad, int totalOutputChannels) {
    if (blockCount % 2 != 0) {
        return false;
    }
    const int lanesPerBlock = gemvLanesPerBlock(inputDepthQuad, blockCount, /*splitK=*/true);
    if (lanesPerBlock <= 0) {
        return false;
    }
    const int blocksOwned    = blockCount / 2;
    const int blocksInFlight = kMetalSimdGroupWidth / lanesPerBlock;
    const int passes         = UP_DIV(blocksOwned, blocksInFlight);
    if (blocksOwned * 5 < blocksInFlight * passes * 4) {
        return false;
    }
    constexpr int kMinSplitKQuadsPerSG = 256;
    if (inputDepthQuad / 2 >= kMinSplitKQuadsPerSG) {
        return true;
    }
    constexpr int kShortKSplitMaxTotalOutputChannels = 4096;
    return totalOutputChannels <= kShortKSplitMaxTotalOutputChannels;
}

// SwiGLU K-split factor on top of oc16Nsg2SplitK1: picks the widest factor within
// kMaxPairs whose lanes cover a pair's blocks in one pass, else the narrowest.
static int gateUpSiluSplitK(int blockCount, int quadsPerBlock) {
    constexpr int kMaxThreadgroupThreads = 1024;
    constexpr int kMaxPairs              = 6;
    const int legacyLanes = std::max(quadsPerBlock / 4, 1);
    const int minOwned    = std::max(2 * kMetalSimdGroupWidth / std::max(quadsPerBlock, 1), 1);
    const int maxOwned    = kMetalSimdGroupWidth / legacyLanes;
    int widest    = 0;
    int narrowest = 0;
    for (int factor = 2; factor * 2 * kMetalSimdGroupWidth <= kMaxThreadgroupThreads; ++factor) {
        if (blockCount % factor != 0) {
            continue;
        }
        const int blocksOwned = blockCount / factor;
        if (blocksOwned < minOwned || blocksOwned > maxOwned) {
            continue;
        }
        if (narrowest == 0) {
            narrowest = factor;
        }
        if (factor <= kMaxPairs) {
            widest = factor;
        }
    }
    return widest > 0 ? widest : narrowest;
}
#endif

class MetalFusedProj : public MetalExecution, public MetalBackend::FusedProjFusionHost {
public:
    MetalFusedProj(Backend *backend, const MNN::Op *op, bool eager = true) : MetalExecution(backend) {
        auto param  = op->main_as_FusedLinearParam();
        mIsGateUp   = param->act_silu_mul();
        mHasLn      = param->has_ln() && param->ln() != nullptr;
        mRowMul     = param->row_mul();
        mNumConvs   = (int)param->convs()->size();
        mNumProjOut = mIsGateUp ? 1 : mNumConvs;
        mParam      = param;
        mFusedOp    = op;
        if (!eager) {
            return;
        }
        // Create the member convs eagerly so cloned sessions can share their
        // weights via onClone before the first onResize.
        auto metalBackend = static_cast<MetalBackend *>(backend);
        for (int i = 0; i < mNumConvs; ++i) {
            auto conv   = mParam->convs()->GetAs<Convolution2D>(i);
            auto convOp = _wrapConv(conv, mChildFormat);
            // The conv creator inspects the input tensor for dispatch; feed
            // shaped dummies (weights load from the op, not the tensors).
            std::shared_ptr<Tensor> dummyIn(
                Tensor::createDevice({1, conv->common()->inputCount(), 1, 1}, halide_type_of<float>()));
            std::shared_ptr<Tensor> dummyOut(
                Tensor::createDevice({1, conv->common()->outputCount(), 1, 1}, halide_type_of<float>()));
            TensorUtils::getDescribe(dummyIn.get())->dimensionFormat  = mChildFormat;
            TensorUtils::getDescribe(dummyOut.get())->dimensionFormat = mChildFormat;
            Execution *exe = metalBackend->onCreate({dummyIn.get()}, {dummyOut.get()}, convOp);
            if (exe == nullptr || !exe->valid()) {
                delete exe;
                mValid = false;
                return;
            }
            mConvs.emplace_back(static_cast<MetalExecution *>(exe));
        }
    }
    virtual ~MetalFusedProj() = default;

    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (!mValid) {
            return false;
        }
        if (nullptr == dst) {
            // Capability check: cloneable once every child (weight) exists.
            return (int)mConvs.size() == mNumConvs;
        }
        if ((int)mConvs.size() != mNumConvs) {
            return false;
        }
        // Share conv weights through each child's onClone; the lightweight
        // mMulSilu / mLn children are re-created lazily on first onResize.
        std::unique_ptr<MetalFusedProj> clone(new MetalFusedProj(bn, op, /*eager=*/false));
        for (int i = 0; i < mNumConvs; ++i) {
            const Op* convOp = clone->_wrapConv(clone->mParam->convs()->GetAs<Convolution2D>(i), mChildFormat);
            Execution* childClone = nullptr;
            if (!mConvs[i]->onClone(bn, convOp, &childClone) || nullptr == childClone ||
                !childClone->valid()) {
                delete childClone;
                return false;
            }
            clone->mConvs.emplace_back(static_cast<MetalExecution *>(childClone));
        }
        *dst = clone.release();
        MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, *dst);
        return true;
    }

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto backend = static_cast<MetalBackend *>(this->backend());
#ifdef MNN_LOW_MEMORY
        mDecode = DecodePlan();
        mMultiProjPrefill = MultiProjPrefillPlan();
        mGroupStamp = 0;
#endif
        mPrefillDualFused = mPrefillSiluFused = false;
        Tensor *hidden = mHasLn ? inputs[1] : inputs[0];
        auto fmt = TensorUtils::getDescribe(hidden)->dimensionFormat;
        mChildFormat = fmt;
        if (mRowMul) {
            if (!_createChildren(backend, inputs, outputs, hidden, fmt)) {
                return NOT_SUPPORT;
            }
            mConvs[0]->requestRowScale(inputs[1]);
            auto err = mConvs[0]->onResize({hidden}, {outputs[0]});
            if (err != NO_ERROR || mConvs[0]->rowScaleFused()) {
                return err;
            }
            err = _prepareRowScale(backend, outputs[0]);
            if (err != NO_ERROR) {
                return err;
            }
            const int tokens = outputs[0]->length(0);
            const int quads = UP_DIV(outputs[0]->length(1), 4);
            if (tokens != mRowScaleTokens || quads != mRowScaleQuads || inputs[1] != mRowScaleInput) {
                ++mResizeGeneration;
            }
            mRowScaleTokens = tokens;
            mRowScaleQuads = quads;
            mRowScaleInput = inputs[1];
            return NO_ERROR;
        }
        if (mHasLn) {
            if (_ensureIntermediate(mNormalized, hidden, hidden->length(1), fmt)) {
                mResizeGeneration++;
            }
            if (!backend->onAcquireBuffer(mNormalized.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }
        auto releaseNormalized = [&]() {
            if (mHasLn) {
                backend->onReleaseBuffer(mNormalized.get(), Backend::DYNAMIC);
            }
        };
        if (mIsGateUp) {
            int oc = outputs[0]->length(1);
            if (_ensureIntermediate(mGate, hidden, oc, fmt)) {
                mResizeGeneration++;
            }
            if (_ensureIntermediate(mUp, hidden, oc, fmt)) {
                mResizeGeneration++;
            }
        }
        bool hasGateBuf = false;
        bool hasUpBuf = false;
        auto releaseIntermediates = [&]() {
            if (hasUpBuf) {
                backend->onReleaseBuffer(mUp.get(), Backend::DYNAMIC);
                hasUpBuf = false;
            }
            if (hasGateBuf) {
                backend->onReleaseBuffer(mGate.get(), Backend::DYNAMIC);
                hasGateBuf = false;
            }
            releaseNormalized();
        };
        Tensor *projInput = mHasLn ? mNormalized.get() : hidden;
        if ((int)mConvs.size() != mNumConvs || (mIsGateUp && !mMulSilu) || (mHasLn && !mLn)) {
            if (!_createChildren(backend, inputs, outputs, projInput, fmt)) {
                MNN_ERROR("MetalFusedProj: failed to create sub-executions\n");
                releaseIntermediates();
                return NOT_SUPPORT;
            }
        }
        ErrorCode err = NO_ERROR;
        if (mHasLn) {
            err = mLn->onResize({inputs[0], inputs[1]}, {outputs[mNumProjOut], mNormalized.get()});
        }
        if (err == NO_ERROR) {
            if (mIsGateUp) {
#ifdef MNN_LOW_MEMORY
                auto *upWq   = _asWQuant(mConvs[1].get());
                auto *gateWq = _asWQuant(mConvs[0].get());
                // Probe GateUpDual before acquiring mGate/mUp so a dual-fused prefill
                // does not inflate peak dynamic memory with unused intermediates.
                if (upWq != nullptr && gateWq != nullptr && hidden->length(0) >= 2 &&
                    upWq->quantLayoutCompatible(gateWq)) {
                    upWq->requestPrefillSiluMul(nullptr);
                    upWq->requestPrefillGateUpDual(gateWq);
                    err = mConvs[1]->onResize({projInput}, {outputs[0]});
                    mPrefillDualFused = err == NO_ERROR && upWq->prefillGateUpDualFused();
                }
                if (err == NO_ERROR && !mPrefillDualFused) {
                    if (!backend->onAcquireBuffer(mGate.get(), Backend::DYNAMIC)) {
                        releaseIntermediates();
                        return OUT_OF_MEMORY;
                    }
                    hasGateBuf = true;
                    err = mConvs[0]->onResize({projInput}, {mGate.get()});
                    if (err == NO_ERROR && upWq != nullptr && hidden->length(0) >= 2) {
                        // Offer the up projection the group's MUL_SILU directly into outputs[0];
                        // mGate stays acquired while outputs[0] is live to prevent aliasing.
                        upWq->requestPrefillSiluMul(mGate.get());
                        upWq->requestPrefillGateUpDual(nullptr);
                        err = mConvs[1]->onResize({projInput}, {outputs[0]});
                        mPrefillSiluFused = err == NO_ERROR && upWq->prefillSiluMulFused();
                    }
                    if (err == NO_ERROR && !mPrefillSiluFused) {
                        if (!backend->onAcquireBuffer(mUp.get(), Backend::DYNAMIC)) {
                            releaseIntermediates();
                            return OUT_OF_MEMORY;
                        }
                        hasUpBuf = true;
                        if (upWq != nullptr) {
                            upWq->requestPrefillSiluMul(nullptr);
                            upWq->requestPrefillGateUpDual(nullptr);
                        }
                        err = mConvs[1]->onResize({projInput}, {mUp.get()});
                    }
                }
#else
                // No quant residency: the group always runs two projections + MUL_SILU.
                if (!backend->onAcquireBuffer(mGate.get(), Backend::DYNAMIC)) {
                    releaseIntermediates();
                    return OUT_OF_MEMORY;
                }
                hasGateBuf = true;
                if (!backend->onAcquireBuffer(mUp.get(), Backend::DYNAMIC)) {
                    releaseIntermediates();
                    return OUT_OF_MEMORY;
                }
                hasUpBuf = true;
                err = mConvs[0]->onResize({projInput}, {mGate.get()});
                if (err == NO_ERROR) {
                    err = mConvs[1]->onResize({projInput}, {mUp.get()});
                }
#endif
                if (err == NO_ERROR && !mPrefillDualFused && !mPrefillSiluFused) {
                    err = mMulSilu->onResize({mUp.get(), mGate.get()}, {outputs[0]});
                }
            } else {
                for (int i = 0; i < mNumConvs && err == NO_ERROR; ++i) {
                    err = mConvs[i]->onResize({projInput}, {outputs[i]});
                }
            }
        }
        releaseIntermediates();
#ifdef MNN_LOW_MEMORY
        if (err == NO_ERROR) {
            // Registered on every successful resize: the backend clears the
            // registry in onResizeBegin.
            mProjOutputs.assign(outputs.begin(), outputs.begin() + mNumProjOut);
            mProjInput     = projInput;
            mLnResidualIn  = mHasLn ? inputs[0] : nullptr;
            mLnHiddenIn    = mHasLn ? inputs[1] : nullptr;
            mLnResidualOut = mHasLn ? outputs[mNumProjOut] : nullptr;
            backend->registerFusedProj(this);
        }
#endif
        return err;
    }

    // Run after allocator compute(), when the planned buffer ranges are known.
    virtual void setupFusion() override {
#ifdef MNN_LOW_MEMORY
        if (mPrefillDualFused || mPrefillSiluFused) {
            return;
        }
        auto metal = static_cast<MetalBackend*>(backend());
        if (!mIsGateUp) {
            MultiProjPrefillPlan candidate;
            if (prepareMultiProjPrefill(metal, candidate)) {
                mMultiProjPrefill = std::move(candidate);
                mGroupStamp = multiProjPrefillStamp();
                return;
            }
            if (mNumConvs < 3) {
                return;
            }
        }
        int reduceRows = 1;
        bool decodeOk = true;
        for (int i = 0; i < mNumConvs; ++i) {
            auto* conv = _asWQuant(mConvs[i].get());
            if (!conv) {
                return;
            }
            mDecodeMembers[i] = conv->decodeMember();
            if (!mDecodeMembers[i]) {
                decodeOk = false;
                break;
            }
        }
        if (!decodeOk) {
            for (int i = 0; i < mNumConvs; ++i) {
                auto* conv = _asWQuant(mConvs[i].get());
                int memberRows = 0;
                mDecodeMembers[i] = conv->shortPrefillMember(&memberRows);
                if (!mDecodeMembers[i] || (i > 0 && memberRows != reduceRows)) {
                    return;
                }
                reduceRows = memberRows;
            }
        }
        DecodePlan candidate;
        candidate.reduceRows = reduceRows;
        if (mIsGateUp) {
            if (!_asWQuant(mConvs[0].get())->quantLayoutCompatible(_asWQuant(mConvs[1].get())) ||
                !prepareGateUp(metal, candidate)) {
                return;
            }
        } else if (!prepareQKV(metal, candidate)) {
            return;
        }
        // A rejected LN fold does not invalidate the projection plan.
        if (reduceRows == 1) {
            tryFoldLN(metal, candidate);
        }
        logDecodePlan(candidate);
        mDecode = std::move(candidate);
        mGroupStamp = decodeStamp();
#endif
    }

    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                          id<MTLComputeCommandEncoder> encoder) override {
        const auto steps = dispatchSteps();
        mRecordedGeneration = mResizeGeneration;
        mRecordedSteps = steps;
#ifdef MNN_LOW_MEMORY
        mRecordedGroupStamp = mGroupStamp;
#endif
        auto runChild = [&](MetalExecution* child, const std::vector<Tensor*>& in,
                            const std::vector<Tensor*>& out) {
#if MNN_METAL_OP_PROFILE
            // Children may flush/replace the encoder during profiling.
            mEncodeStatus = child->onExecute(in, out);
            return mEncodeStatus == NO_ERROR;
#else
            child->onEncode(in, out, encoder);
            return true;
#endif
        };
        if ((steps & RunLn) && !runChild(mLn.get(), {inputs[0], inputs[1]},
                                        {outputs[mNumProjOut], mNormalized.get()})) {
            return;
        }
        Tensor* projInput = (steps & RunLn) ? mNormalized.get() : inputs[mHasLn ? 1 : 0];
#ifdef MNN_LOW_MEMORY
        if (steps & RunMultiProjPrefill) {
#if MNN_METAL_OP_PROFILE
            auto metal = static_cast<MetalBackend*>(backend());
            encoder = metal->encoder_for_net();
            metal->profileMarkOp(this);
            metal->setProfileSubtag("multi_proj_prefill");
#endif
            encodeMultiProjPrefill(projInput, outputs, encoder);
            return;
        }
        if (steps & RunDecodeGroup) {
#if MNN_METAL_OP_PROFILE
            auto metal = static_cast<MetalBackend*>(backend());
            encoder = metal->encoder_for_net();
            metal->profileMarkOp(this);
            metal->setProfileSubtag(mDecode.kind == DecodeKind::QKV ? "qkv_fused" : "gate_up_fused");
#endif
            encodeDecode(projInput, inputs, outputs, encoder);
#if MNN_METAL_OP_PROFILE
            // Otherwise the following MUL_SILU would inherit the group's encoder
            // or re-label its pending profile sample/command buffer.
            if (steps & RunMulSilu) {
                metal->profileOpEncoded();
                if (!metal->profileUseCounters()) {
                    metal->flushEncoder();
                    metal->commit_net();
                }
            }
#endif
        }
#endif
        if ((steps & RunGate) && !runChild(mConvs[0].get(), {projInput}, {mGate.get()})) {
            return;
        }
        if ((steps & RunUp) && !runChild(mConvs[1].get(), {projInput},
                                        {(steps & UpWritesResult) ? outputs[0] : mUp.get()})) {
            return;
        }
        if (steps & RunProjections) {
            for (int i = 0; i < mNumConvs; ++i) {
                if (!runChild(mConvs[i].get(), {projInput}, {outputs[i]})) {
                    return;
                }
            }
        }
        if (steps & RunRowScale) {
#if MNN_METAL_OP_PROFILE
            auto metal = static_cast<MetalBackend*>(backend());
            metal->profileMarkOp(this);
            encoder = metal->profileNextSubpass("row_scale");
#else
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
#endif
            _encodeRowScale(outputs[0], inputs[1], encoder);
        }
        if (steps & RunMulSilu) {
            runChild(mMulSilu.get(), {mUp.get(), mGate.get()}, {outputs[0]});
        }
    }

    virtual bool onReplayUpdate(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs) override {
        const auto steps = dispatchSteps();
        // Check before MetalReplay dereferences any annotated internal tensors.
        if (mRecordedGeneration != mResizeGeneration || mRecordedSteps != steps) {
            return false;
        }
#ifdef MNN_LOW_MEMORY
        if (mRecordedGroupStamp != mGroupStamp) {
            return false;
        }
#endif
        if ((steps & RunLn) && !mLn->replayCheckForOwner(inputs, {outputs[mNumProjOut], mNormalized.get()})) {
            return false;
        }
        const std::vector<Tensor*> childIn{(steps & RunLn) ? mNormalized.get() : inputs[mHasLn ? 1 : 0]};
        if ((steps & RunGate) && !mConvs[0]->replayCheckForOwner(childIn, {mGate.get()})) {
            return false;
        }
        if ((steps & RunUp) && !mConvs[1]->replayCheckForOwner(childIn,
                {(steps & UpWritesResult) ? outputs[0] : mUp.get()})) {
            return false;
        }
        if (steps & RunProjections) {
            for (int i = 0; i < mNumConvs; ++i) {
                if (!mConvs[i]->replayCheckForOwner(childIn, {outputs[i]})) {
                    return false;
                }
            }
        }
        return !(steps & RunMulSilu) ||
            mMulSilu->replayCheckForOwner({mUp.get(), mGate.get()}, {outputs[0]});
    }

private:
    ErrorCode _prepareRowScale(MetalBackend* backend, const Tensor* output) {
        if (mRowScalePipeline == nil) {
            const bool half = backend->useFp16InsteadFp32();
            std::vector<std::string> keys = {"fused_proj_row_scale", half ? "half" : "float"};
            mRowScalePipeline = backend->runtime()->findPipeline(keys);
            if (mRowScalePipeline == nil) {
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                options.preprocessorMacros = @{@"T4": half ? @"half4" : @"float4"};
                mRowScalePipeline =
                    backend->makeComputePipelineWithSourceOption(gFusedProjRowScale, "row_scale", options);
                if (mRowScalePipeline != nil) {
                    backend->runtime()->insertPipeline(keys, mRowScalePipeline);
                }
            }
            if (mRowScalePipeline == nil) {
                return NOT_SUPPORT;
            }
        }
        if (mRowScaleShape == nil) {
            mRowScaleShape = backend->getConstBuffer(2 * sizeof(int));
        }
        if (mRowScaleShape == nil) {
            return OUT_OF_MEMORY;
        }
        const int tokens = output->length(0);
        const int count = tokens * UP_DIV(output->length(1), 4);
        auto shape = static_cast<int*>(mRowScaleShape.contents);
        shape[0] = tokens;
        shape[1] = count;
        const int threads = std::min<int>(128, mRowScalePipeline.maxTotalThreadsPerThreadgroup);
        mRowScaleThreads = {MTLSizeMake(UP_DIV(count, threads), 1, 1), MTLSizeMake(threads, 1, 1)};
        return NO_ERROR;
    }

    void _encodeRowScale(const Tensor* output, const Tensor* scale, id<MTLComputeCommandEncoder> encoder) {
        [encoder setComputePipelineState:mRowScalePipeline];
        MetalBackend::setTensor(output, encoder, 0);
        MetalBackend::setTensor(scale, encoder, 1);
        [encoder setBuffer:mRowScaleShape offset:0 atIndex:2];
        [encoder dispatchThreadgroups:mRowScaleThreads.first threadsPerThreadgroup:mRowScaleThreads.second];
    }

    enum DispatchStep {
        RunLn = 1 << 0, RunDecodeGroup = 1 << 1, RunGate = 1 << 2, RunUp = 1 << 3,
        RunProjections = 1 << 4, UpWritesResult = 1 << 5, RunMulSilu = 1 << 6,
        RunRowScale = 1 << 7, RunMultiProjPrefill = 1 << 8,
    };
    unsigned dispatchSteps() const {
        if (mRowMul) {
            return RunProjections | (mConvs[0]->rowScaleFused() ? 0u : RunRowScale);
        }
        const unsigned ln = mHasLn ? static_cast<unsigned>(RunLn) : 0u;
#ifdef MNN_LOW_MEMORY
        if (mMultiProjPrefill.pipeline != nil) {
            return ln | RunMultiProjPrefill;
        }
        if (mDecode.kind != DecodeKind::None) {
            unsigned steps = RunDecodeGroup;
            if (!mDecode.gamma) {
                steps |= ln;
            }
            if (mDecode.kind == DecodeKind::GateUp) {
                steps |= RunMulSilu;
            }
            return steps;
        }
#endif
        if (!mIsGateUp) {
            return ln | RunProjections;
        }
        if (mPrefillDualFused) {
            return ln | RunUp | UpWritesResult;
        }
        return ln | RunGate | RunUp | (mPrefillSiluFused ? UpWritesResult : RunMulSilu);
    }

#ifdef MNN_LOW_MEMORY
    static MetalConv1x1WQuant* _asWQuant(MetalExecution* exe) {
        return exe ? exe->asWQuant() : nullptr;
    }

    struct MultiProjPrefillPlan {
        struct Segment {
            int firstTile;
            int outputQuads;
            float scaleCoef;
        };
        static_assert(sizeof(Segment) == 12, "FqMultiProjSegment ABI");
        std::array<Segment, 4> segments = {};
        std::array<MetalConv1x1WQuant::QuantMember, 4> members;
        id<MTLComputePipelineState> pipeline = nil;
        MTLSize grid = {}, threads = {};
    };

    bool prepareMultiProjPrefill(MetalBackend* metal, MultiProjPrefillPlan& plan) {
        if (mNumConvs < 2 || mNumConvs > 4 || mProjInput->dimensions() != 4 ||
            mProjInput->length(2) != 1 || mProjInput->length(3) != 1) {
            return false;
        }
        const int area = mProjInput->length(0);
        if (area < 2) {
            return false;
        }
        const auto& kernel = conv1x1Kernel(Conv1x1Kernel::FusedQuantStageGemm);
        const auto& first = plan.members[0];
        int tileM = 0;
        int tiles = 0;
        for (int i = 0; i < mNumConvs; ++i) {
            auto* conv = _asWQuant(mConvs[i].get());
            if (conv == nullptr) {
                return false;
            }
            const int memberTileM = conv->multiProjPrefillTileM();
            if (memberTileM <= 0 || (i > 0 && memberTileM != tileM)) {
                return false;
            }
            tileM = memberTileM;
            auto& member = plan.members[i];
            member = conv->multiProjPrefillMember();
            auto* out = mProjOutputs[i];
            if (!member || member.outputChannels <= 0 || member.outputChannels % 4 != 0 ||
                member.bits != first.bits || member.blocks != first.blocks ||
                member.activation != first.activation || member.inputChannels != first.inputChannels ||
                member.inputQuads != first.inputQuads || projectionRangesOverlap(metal, out, mProjInput)) {
                return false;
            }
            for (int j = 0; j < i; ++j) {
                if (projectionRangesOverlap(metal, out, mProjOutputs[j])) {
                    return false;
                }
            }
            plan.segments[i] = {tiles, member.outputQuads, member.scaleCoef};
            tiles += UP_DIV(member.outputChannels, kernel.tileN);
        }
        const int gridX = UP_DIV(area, tileM);
        if (area <= 32 && gridX * tiles < 16) {
            // Very narrow short-sequence groups benefit more from per-member split-K.
            return false;
        }
        // Empty slots are outside the grid; the encoder still binds valid buffers.
        for (int i = mNumConvs; i < 4; ++i) {
            plan.segments[i].firstTile = tiles;
        }
        auto pipeline = _asWQuant(mConvs[0].get())->multiProjPrefillPipeline();
        auto ctx = (__bridge MNNMetalContext*)metal->context();
        if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup < kernel.threads ||
            pipeline.staticThreadgroupMemoryLength > ctx.device.maxThreadgroupMemoryLength) {
            return false;
        }
        plan.pipeline = pipeline;
        plan.grid = MTLSizeMake(gridX, tiles, 1);
        plan.threads = MTLSizeMake(kernel.threads, 1, 1);
        return true;
    }

    void encodeMultiProjPrefill(const Tensor* input, const std::vector<Tensor*>& outputs,
                                id<MTLComputeCommandEncoder> encoder) const {
        [encoder setComputePipelineState:mMultiProjPrefill.pipeline];
        MetalBackend::setTensor(input, encoder, 0);
        const auto& first = mMultiProjPrefill.members[0];
        [encoder setBuffer:first.params offset:0 atIndex:2];
        MetalBackend::setTensor(first.weight, encoder, 6);
        for (int i = 0; i < 4; ++i) {
            const int p = i < mNumConvs ? i : 0;
            const auto& member = mMultiProjPrefill.members[p];
            const int binding = 7 + (i - 1) * 4;
            MetalBackend::setTensor(outputs[p], encoder, i == 0 ? 1 : binding);
            MetalBackend::setTensor(member.weight, encoder, i == 0 ? 3 : binding + 1);
            MetalBackend::setTensor(member.bias, encoder, i == 0 ? 4 : binding + 2);
            MetalBackend::setTensor(member.dequant, encoder, i == 0 ? 5 : binding + 3);
        }
        [encoder setBytes:mMultiProjPrefill.segments.data() length:sizeof(mMultiProjPrefill.segments) atIndex:19];
        [encoder dispatchThreadgroups:mMultiProjPrefill.grid threadsPerThreadgroup:mMultiProjPrefill.threads];
    }

    uint64_t multiProjPrefillStamp() const {
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
        mix(mMultiProjPrefill.grid.width);
        mix(mMultiProjPrefill.grid.height);
        mix(reinterpret_cast<uintptr_t>((__bridge void*)mMultiProjPrefill.pipeline));
        mix(reinterpret_cast<uintptr_t>((__bridge void*)mMultiProjPrefill.members[0].params));
        for (const auto& segment : mMultiProjPrefill.segments) {
            uint32_t scaleBits;
            memcpy(&scaleBits, &segment.scaleCoef, sizeof(scaleBits));
            mix(segment.firstTile);
            mix(segment.outputQuads);
            mix(scaleBits);
        }
        return h;
    }

    enum class DecodeKind { None, GateUp, GateUpSilu, QKV };
    enum class DecodeBody { Plain, Oc16Nsg2SplitK1, Oc16Nsg4SplitK1, Oc8Nsg4SplitK2, Oc16Nsg8SplitK2 };
    struct DecodePlan {
        DecodeKind kind = DecodeKind::None;
        DecodeBody body = DecodeBody::Plain;
        int reduceRows = 1;
        int siluSplitKCount = 0;
        int lanesPerBlock = 0;
        bool packedGrid = false;
        MTLSize grid = {};
        MTLSize threads = {};
        id<MTLComputePipelineState> pipeline = nil;
        const Tensor* gamma = nullptr;
        id<MTLBuffer> mergedWeights = nil;
        // Up to four weights followed by their dequant tables; only used for QKV.
        std::array<size_t, 8> mergedOffsets = {};
    };

    // Select once; pipeline macros, cache keys and dispatch all consume the
    // same body. LN may add staging but must not change the selected grid.
    id<MTLComputePipelineState> loadDecodePipeline(MetalBackend* metal, const DecodePlan& plan,
                                                   int lnQuads = 0) const {
        const bool fp16 = metal->useFp16InsteadFp32();
        const auto bits = std::to_string(mDecodeMembers[0].bits);
        const bool shortPrefill = plan.reduceRows > 1;
        const auto& body = conv1x1Kernel(Conv1x1Kernel::GemvG4m1_2sg);
        const std::string entryStr = shortPrefill ? conv1x1GemvG4mName(plan.reduceRows) : std::string(body.entry);
        const char* entry = entryStr.c_str();
        const PrefillSource sourceKind = shortPrefill ? kConv1x1GemvG4mSource : body.source;
        std::vector<std::string> keys = {fp16 ? "half4" : "float4", "MNN_METAL_FLOAT32_COMPUTER"};
        if (fp16) {
            keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
        }
        keys.emplace_back("conv1x1_wquant_" + bits);
        keys.emplace_back("conv1x1_wquant_sg_reduce");
        keys.emplace_back(entry);
        auto macros = conv1x1Macros(fp16);
        [macros setValue:@"1" forKey:@(("W_QUANT_" + bits).c_str())];
        const bool qkv = plan.kind == DecodeKind::QKV;
        const bool silu = plan.kind == DecodeKind::GateUpSilu;
        keys.emplace_back(qkv ? "QKV_FUSED" : "GATE_UP_FUSED");
        [macros setValue:@"1" forKey:(qkv ? @"QKV_FUSED" : @"GATE_UP_FUSED")];
        if (qkv && mNumConvs == 4) {
            keys.emplace_back("QKV_FUSED_4PROJ");
            [macros setValue:@"1" forKey:@"QKV_FUSED_4PROJ"];
        }
        if (plan.packedGrid) {
            [macros setValue:@"1" forKey:@"QKV_PACKED_GRID"];
            if (lnQuads) {
                keys.emplace_back("QKV_PACKED_GRID");
            }
        }
        if (silu) {
            [macros setValue:@"1" forKey:@"GATE_UP_SILU"];
            if (lnQuads) {
                keys.emplace_back("GATE_UP_SILU");
            }
        }
        if (lnQuads) {
            keys.emplace_back("LN_FUSED");
            [macros setValue:@"1" forKey:@"LN_FUSED"];
            if (plan.body == DecodeBody::Oc16Nsg8SplitK2 && lnQuads <= 256) {
                keys.emplace_back("LN_STAGE_" + std::to_string(lnQuads));
                [macros setValue:@"1" forKey:@"LN_STAGE"];
                [macros setValue:@(lnQuads).stringValue forKey:@"LN_STAGE_QUADS"];
            }
        }
        switch (plan.body) {
            case DecodeBody::Plain:
                break;
            case DecodeBody::Oc16Nsg2SplitK1:
                keys.emplace_back("oc16Nsg2SplitK1");
                [macros setValue:@"1" forKey:@"GEMV_SG_OC_QUADS_2"];
                break;
            case DecodeBody::Oc16Nsg4SplitK1:
                keys.emplace_back("oc16Nsg4SplitK1");
                [macros setValue:@"4" forKey:@"GEMV_QUADS_PER_TG"];
                break;
            case DecodeBody::Oc8Nsg4SplitK2:
            case DecodeBody::Oc16Nsg8SplitK2: {
                const bool oc16 = plan.body == DecodeBody::Oc16Nsg8SplitK2;
                keys.emplace_back(oc16 ? "oc16Nsg8SplitK2" : "oc8Nsg4SplitK2");
                [macros setValue:(oc16 ? @"4" : @"2") forKey:@"GEMV_QUADS_PER_TG"];
                [macros setValue:@"2" forKey:@"GEMV_SPLIT_K"];
                if (plan.lanesPerBlock > 0) {
                    keys.emplace_back("lanes" + std::to_string(plan.lanesPerBlock));
                    [macros setValue:@(plan.lanesPerBlock).stringValue forKey:@"GEMV_LANES_PER_BLOCK"];
                }
                break;
            }
        }
        if (plan.siluSplitKCount) {
            keys.emplace_back("oc16Nsg2SplitK1_siluSplitK" + std::to_string(plan.siluSplitKCount));
            [macros setValue:@(plan.siluSplitKCount).stringValue forKey:@"GEMV_SG_OC_QUADS_2_KSPLIT"];
        }
        if (!lnQuads && silu) {
            keys.emplace_back("GATE_UP_SILU");
        }
        if (!lnQuads && plan.packedGrid) {
            keys.emplace_back("QKV_PACKED_GRID");
        }
        auto rt = metal->runtime();
        auto pipeline = rt->findPipeline(keys);
        if (nil == pipeline && !rt->pipelineCompileFailed(keys)) {
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.preprocessorMacros = macros;
            const std::string source = conv1x1ShaderSource(sourceKind);
            // All group variants have a prepared unfused path to fall back to.
            pipeline = metal->makeComputePipelineWithSourceOption(source.c_str(), entry, options, /*optional=*/true);
            rt->insertPipeline(keys, pipeline);
        }
        auto context = (__bridge MNNMetalContext*)metal->context();
        if (nil == pipeline || pipeline.maxTotalThreadsPerThreadgroup < plan.threads.width ||
            pipeline.staticThreadgroupMemoryLength > context.device.maxThreadgroupMemoryLength) {
            return nil;
        }
        return pipeline;
    }

    static bool projectionRangesOverlap(const MetalBackend* metal, const Tensor* a, const Tensor* b) {
        // A zero-copy Raster's own binding may not have been refreshed yet:
        // Pipeline::_bindBufferViews runs after the resize-end callbacks.
        auto storage = [](const Tensor* tensor) -> const Tensor* {
            auto* root = TensorUtils::getDescribe(tensor)->storageRoot;
            return root ? root : tensor;
        };
        return metal->tensorsOverlap(storage(a), storage(b));
    }

    bool prepareGateUp(MetalBackend* metal, DecodePlan& plan) {
        const auto& gate = mDecodeMembers[0];
        const bool shortPrefill = plan.reduceRows > 1;
        const bool legacy = metal->runtime()->gpuInfo().gen < 17;
        const int quadsPerBlock = UP_DIV(gate.inputQuads, gate.blocks);
        const auto profile = conv1x1DeviceProfile(metal->runtime()->gpuInfo());
        auto build = [&](bool silu) {
            plan.kind = silu ? DecodeKind::GateUpSilu : DecodeKind::GateUp;
            if (shortPrefill) {
                plan.body = DecodeBody::Plain;
                plan.siluSplitKCount = 0;
                plan.grid = MTLSizeMake(UP_DIV(gate.outputQuads, 2), 1, silu ? 1 : 2);
                plan.threads = MTLSizeMake(kConv1x1GemvG4mThreads, 1, 1);
                plan.pipeline = loadDecodePipeline(metal, plan);
                return plan.pipeline != nil;
            }
            plan.body = legacy || silu ? DecodeBody::Oc16Nsg2SplitK1 : DecodeBody::Plain;
            if (!legacy && !silu && quadsPerBlock >= 32 && gate.outputQuads % 4 == 0) {
                plan.body = DecodeBody::Oc16Nsg4SplitK1;
            }
            plan.siluSplitKCount = 0;
            if (silu && profile.supportsGateUpSplitK &&
                decodeSplitKPrecisionOk(profile, metal->useFp16InsteadFp32()) &&
                gate.outputChannels % 8 == 0 &&
                (gate.bits == 2 || gate.bits == 3 || gate.bits == 4 || gate.bits == 8) &&
                gate.inputChannels % gate.blocks == 0) {
                plan.siluSplitKCount = gateUpSiluSplitK(gate.blocks, quadsPerBlock);
            }
            if (!plan.siluSplitKCount && silu && gate.outputQuads % 2 == 0 && gate.blocks >= 2 &&
                quadsPerBlock >= profile.gateUpSplitKQuadsFloor && (gate.blocks & (gate.blocks - 1)) == 0) {
                plan.siluSplitKCount = 2;
            }
            const int quadsPerTG = silu || plan.body == DecodeBody::Plain ? 2 : 4;
            const int threads = plan.siluSplitKCount ? 64 * plan.siluSplitKCount
                : (plan.body == DecodeBody::Oc16Nsg4SplitK1 ? 128 : 64);
            plan.grid = MTLSizeMake(UP_DIV(gate.outputQuads, quadsPerTG), 1, silu ? 1 : 2);
            plan.threads = MTLSizeMake(threads, 1, 1);
            plan.pipeline = loadDecodePipeline(metal, plan);
            while (plan.pipeline == nil && silu && plan.siluSplitKCount > 0) {
                // Degrade split-K factor before dropping in-kernel SiLU altogether.
                plan.siluSplitKCount = (plan.siluSplitKCount > 2 && gate.blocks % 2 == 0) ? 2 : 0;
                plan.threads = MTLSizeMake(plan.siluSplitKCount ? 64 * plan.siluSplitKCount : 64, 1, 1);
                plan.pipeline = loadDecodePipeline(metal, plan);
            }
            return plan.pipeline != nil;
        };
        // The in-kernel SiLU writes while other threadgroups still read the
        // input. On overlap (or an unavailable pipeline), try separate SiLU.
        const bool silu = !projectionRangesOverlap(metal, mProjOutputs[0], mProjInput) && build(true);
        if ((!silu && !build(false)) || !ensureDecodeSegments(metal)) {
            return false;
        }
        static_cast<float*>(mDecodeSegments.contents)[0] = mDecodeMembers[1].scaleCoef;
        return true;
    }

    bool prepareQKV(MetalBackend* metal, DecodePlan& plan) {
        const auto& q = mDecodeMembers[0];
        const bool shortPrefill = plan.reduceRows > 1;
        const int count = mNumConvs;
        int outputChannels = 0;
        bool aligned2 = true, aligned4 = true;
        for (int i = 0; i < count; ++i) {
            const auto& m = mDecodeMembers[i];
            // OC is carried per member in the segment table; only the shared
            // input/block layout, bit width and activation must agree.
            auto* out = mProjOutputs[i];
            if (projectionRangesOverlap(metal, out, mProjInput) ||
                m.bits != q.bits || m.blocks != q.blocks || m.activation != q.activation ||
                m.inputChannels != q.inputChannels || m.inputQuads != q.inputQuads) {
                return false;
            }
            for (int j = 0; j < i; ++j) {
                if (projectionRangesOverlap(metal, out, mProjOutputs[j])) {
                    return false;
                }
            }
            outputChannels += m.outputChannels;
            aligned2 = aligned2 && m.outputQuads % 2 == 0;
            aligned4 = aligned4 && m.outputQuads % 4 == 0;
        }
        plan.kind = DecodeKind::QKV;
        int quadsPerTG = 2;
        int simds = 2;
        if (!shortPrefill) {
            const auto& gpu = metal->runtime()->gpuInfo();
            const auto profile = conv1x1DeviceProfile(gpu);
            const bool evenBlocks = q.blocks % 2 == 0;
            // One admission boolean for both split-K bodies: bit width plus precision.
            const bool splitKWq = decodeSplitKPrecisionOk(profile, metal->useFp16InsteadFp32()) &&
                (q.bits == 2 || q.bits == 3 || q.bits == 4 || q.bits == 8);
            const bool wholeBlocks = q.inputChannels % q.blocks == 0;
            const bool allowOc8 = profile.tier.hasOc8Nsg4SplitK2Budget && !profile.tier.prefersOc16Nsg8SplitK2 &&
                splitKWq && evenBlocks && wholeBlocks;
            const bool allowOc16 = profile.tier.prefersOc16Nsg8SplitK2 && splitKWq && wholeBlocks;
            const bool highBwOc16 = profile.tier.highBandwidth && splitKWq && wholeBlocks &&
                qkvOc16Sg8K2Admissible(q.blocks, q.inputQuads, outputChannels);
            if (evenBlocks && aligned4 && ((allowOc16 && !allowOc8) || highBwOc16)) {
                plan.body = DecodeBody::Oc16Nsg8SplitK2;
            } else if (gpu.gen < 17) {
                plan.body = DecodeBody::Oc16Nsg2SplitK1;
            } else if (!allowOc8 && aligned4 && UP_DIV(q.inputQuads, q.blocks) >= 32) {
                plan.body = DecodeBody::Oc16Nsg4SplitK1;
            } else if (evenBlocks && aligned2 && (allowOc8 || q.blocks <= MetalEnv::get().qkvSplitKMaxBlocks)) {
                plan.body = DecodeBody::Oc8Nsg4SplitK2;
            }
            const bool splitK = plan.body == DecodeBody::Oc16Nsg8SplitK2 || plan.body == DecodeBody::Oc8Nsg4SplitK2;
            plan.lanesPerBlock = splitK ? gemvLanesPerBlock(q.inputQuads, q.blocks, true) : 0;
            quadsPerTG = plan.body == DecodeBody::Plain || plan.body == DecodeBody::Oc8Nsg4SplitK2 ? 2 : 4;
            simds = plan.body == DecodeBody::Oc16Nsg2SplitK1 ? 2 : quadsPerTG * (splitK ? 2 : 1);
        }
        int packedGridX = 0, maxGridX = 0;
        std::array<int, 4> gridX = {};
        for (int i = 0; i < count; ++i) {
            gridX[i] = UP_DIV(mDecodeMembers[i].outputQuads, quadsPerTG);
            packedGridX += gridX[i];
            maxGridX = std::max(maxGridX, gridX[i]);
        }
        plan.packedGrid = packedGridX < count * maxGridX;
        plan.grid = MTLSizeMake(plan.packedGrid ? packedGridX : maxGridX, 1, plan.packedGrid ? 1 : count);
        plan.threads = MTLSizeMake(32 * simds, 1, 1);
        plan.pipeline = loadDecodePipeline(metal, plan);
        if (nil == plan.pipeline || !ensureDecodeSegments(metal)) {
            return false;
        }
        if ((q.bits == 2 || q.bits == 3 || q.bits == 4 || q.bits == 8) && (count == 3 || count == 4)) {
            prepareQKVWeights(metal, plan); // optional: the pipeline also accepts separate buffers
        }
        const bool p4 = count == 4;
        const float seg[] = {
            mDecodeMembers[1].scaleCoef, mDecodeMembers[2].scaleCoef,
            float(mDecodeMembers[1].outputQuads), float(mDecodeMembers[2].outputQuads),
            p4 ? mDecodeMembers[3].scaleCoef : 0.0f, p4 ? float(mDecodeMembers[3].outputQuads) : 0.0f,
            float(gridX[0]), float(gridX[0] + gridX[1]), float(gridX[0] + gridX[1] + gridX[2]),
        };
        static_assert(sizeof(seg) == kQKVSegFloats * sizeof(float), "QKV segment layout");
        memcpy(mDecodeSegments.contents, seg, sizeof(seg));
        return true;
    }

    void prepareQKVWeights(MetalBackend* metal, DecodePlan& plan) {
        const int count = mNumConvs;
        const int numParts = count * 2;
        std::array<const Tensor*, 8> parts = {};
        for (int i = 0; i < count; ++i) {
            parts[i] = mDecodeMembers[i].weight;
            parts[count + i] = mDecodeMembers[i].dequant;
        }
        size_t total = 0;
        std::array<size_t, 8> bytes = {};
        plan.mergedOffsets.fill(0);
        for (int i = 0; i < numParts; ++i) {
            bytes[i] = TensorUtils::getRawSize(parts[i]) * parts[i]->getType().bytes();
            total = (total + 15) & ~size_t(15);
            plan.mergedOffsets[i] = total;
            total += bytes[i];
        }
        std::array<std::pair<id<MTLBuffer>, size_t>, 8> srcs;
        for (int i = 0; i < numParts; ++i) {
            srcs[i] = MetalBackend::getBuffer(parts[i]);
            if (nil == srcs[i].first || nullptr == srcs[i].first.contents ||
                srcs[i].second > srcs[i].first.length || bytes[i] > srcs[i].first.length - srcs[i].second) {
                return;
            }
        }
        if (nil == mQKVMergeStorage || mQKVMergeStorage.length < total) {
            auto ctx = (__bridge MNNMetalContext*)metal->context();
            auto buffer = [ctx.device newBufferWithLength:total options:MTLResourceStorageModeShared];
            if (nil == buffer) {
                return;
            }
            mQKVMergeStorage = buffer;
        }
        for (int i = 0; i < numParts; ++i) {
            memcpy(static_cast<uint8_t*>(mQKVMergeStorage.contents) + plan.mergedOffsets[i],
                   static_cast<const uint8_t*>(srcs[i].first.contents) + srcs[i].second, bytes[i]);
        }
        plan.mergedWeights = mQKVMergeStorage;
    }

    void tryFoldLN(MetalBackend* metal, DecodePlan& plan) {
        if (!mHasLn || !mLn->isNC4HW4() || !mLn->isRMSNormWithGammaBeta() ||
            mLnHiddenIn->channel() % 4 != 0) {
            return;
        }
        // Other threadgroups can still be reading either LN input. Keep LN
        // separate on overlap instead of replacing the allocator's bindings.
        auto overlapsLnInputs = [&](const Tensor* out) {
            return projectionRangesOverlap(metal, out, mLnHiddenIn) || projectionRangesOverlap(metal, out, mLnResidualIn);
        };
        if (overlapsLnInputs(mLnResidualOut)) {
            return;
        }
        auto unsafeDestination = [&](const Tensor* out) {
            return overlapsLnInputs(out) || projectionRangesOverlap(metal, out, mLnResidualOut);
        };
        if (plan.kind == DecodeKind::GateUp) {
            if (unsafeDestination(mGate.get()) || unsafeDestination(mUp.get())) {
                return;
            }
        } else if (std::any_of(mProjOutputs.begin(), mProjOutputs.end(), unsafeDestination)) {
            return;
        }
        auto pipeline = loadDecodePipeline(metal, plan, UP_DIV(mLnHiddenIn->channel(), 4));
        if (nil == pipeline) {
            return;
        }
        if (nil == mDecodeEps) {
            mDecodeEps = metal->getConstBuffer(sizeof(float));
            if (nil == mDecodeEps) {
                return;
            }
        }
        *static_cast<float*>(mDecodeEps.contents) = mLn->getEps();
        plan.gamma = mLn->getGamma().get();
        plan.pipeline = pipeline;
    }

    static constexpr int kQKVSegFloats = 9;
    bool ensureDecodeSegments(MetalBackend* metal) {
        if (nil == mDecodeSegments) {
            mDecodeSegments = metal->getConstBuffer(kQKVSegFloats * sizeof(float));
        }
        return mDecodeSegments != nil;
    }

    void encodeDecode(const Tensor* projInput, const std::vector<Tensor*>& inputs,
                      const std::vector<Tensor*>& outputs, id<MTLComputeCommandEncoder> encoder) const {
        const auto& plan = mDecode;
        const bool qkv = plan.kind == DecodeKind::QKV;
        [encoder setComputePipelineState:plan.pipeline];
        MetalBackend::setTensor(plan.gamma ? inputs[1] : projInput, encoder, 0);
        MetalBackend::setTensor(plan.kind == DecodeKind::GateUp ? mGate.get() : outputs[0], encoder, 1);
        [encoder setBuffer:mDecodeMembers[0].params offset:0 atIndex:2];
        const int outputSlot[] = {1, 6, 10, 15};
        const int weightSlot[] = {3, 7, 11, 16};
        const int biasSlot[] = {4, 8, 12, 17};
        const int dequantSlot[] = {5, 9, 13, 18};
        for (int i = 0; i < mNumConvs; ++i) {
            const auto& m = mDecodeMembers[i];
            if (i > 0) {
                MetalBackend::setTensor(qkv ? outputs[i] : mUp.get(), encoder, outputSlot[i]);
            }
            MetalBackend::setTensor(m.bias, encoder, biasSlot[i]);
            if (plan.mergedWeights != nil) {
                [encoder setBuffer:plan.mergedWeights offset:plan.mergedOffsets[i] atIndex:weightSlot[i]];
                [encoder setBuffer:plan.mergedWeights offset:plan.mergedOffsets[mNumConvs + i] atIndex:dequantSlot[i]];
            } else {
                MetalBackend::setTensor(m.weight, encoder, weightSlot[i]);
                MetalBackend::setTensor(m.dequant, encoder, dequantSlot[i]);
            }
        }
        [encoder setBuffer:mDecodeSegments offset:0 atIndex:14];
        if (plan.gamma) {
            MetalBackend::setTensor(inputs[0], encoder, 20);
            MetalBackend::setTensor(plan.gamma, encoder, 21);
            MetalBackend::setTensor(outputs[mNumProjOut], encoder, 22);
            [encoder setBuffer:mDecodeEps offset:0 atIndex:23];
        }
        [encoder dispatchThreadgroups:plan.grid threadsPerThreadgroup:plan.threads];
    }

    uint64_t decodeStamp() const {
        // The proxy validates every tensor-backed binding. Check the pipeline,
        // grid and raw buffers here: those have no tensor annotation.
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
        auto ptr = [&](const void* p) { mix(reinterpret_cast<uintptr_t>(p)); };
        const auto& p = mDecode;
        mix(static_cast<unsigned>(p.kind));
        mix(p.grid.width); mix(p.grid.height); mix(p.grid.depth);
        mix(p.threads.width); mix(p.threads.height); mix(p.threads.depth);
        ptr((__bridge void*)p.pipeline);
        ptr((__bridge void*)mDecodeMembers[0].params);
        ptr((__bridge void*)mDecodeSegments);
        ptr(p.gamma);
        if (p.gamma) {
            ptr((__bridge void*)mDecodeEps);
        }
        if (p.mergedWeights != nil) {
            ptr((__bridge void*)p.mergedWeights);
            for (auto off : p.mergedOffsets) {
                mix(off);
            }
        }
        return h;
    }

    // Routing trace, not a switch: reuses MNN_METAL_PIPELINE_INFO=1 to print the decode
    // variant this shape actually took. A timing comparison is only attributable once both
    // sides show a different body/thread count; identical signatures mean that shape never
    // exercised the difference.
    void logDecodePlan(const DecodePlan& plan) const {
        if (!MetalEnv::get().pipelineInfo) {
            return;
        }
        static const char* const kKindNames[] = {"none", "gateUp", "gateUpSilu", "qkv"};
        static const char* const kBodyNames[] = {"plain", "oc16Nsg2SplitK1", "oc16Nsg4SplitK1", "oc8Nsg4SplitK2", "oc16Nsg8SplitK2"};
        static_assert(sizeof(kKindNames) / sizeof(kKindNames[0]) == 4, "DecodeKind names");
        static_assert(sizeof(kBodyNames) / sizeof(kBodyNames[0]) == 5, "DecodeBody names");
        const auto& head = mDecodeMembers[0];
        MNN_PRINT("[DecodePlan] kind=%s body=%s rows=%d siluSplitK=%d lanes=%d ln=%d packed=%d grid=%zux%zux%zu threads=%zu bits=%d oc=%d ic=%d blocks=%d\n",
                  kKindNames[static_cast<int>(plan.kind)], kBodyNames[static_cast<int>(plan.body)],
                  plan.reduceRows, plan.siluSplitKCount, plan.lanesPerBlock, plan.gamma ? 1 : 0, plan.packedGrid ? 1 : 0,
                  plan.grid.width, plan.grid.height, plan.grid.depth, plan.threads.width,
                  head.bits, head.outputChannels, head.inputChannels, head.blocks);
    }
#endif

    // Keep the Tensor object when its description is unchanged so recorded
    // encode-replay bindings never dangle. Returns true if newly allocated.
    bool _ensureIntermediate(std::shared_ptr<Tensor> &t, const Tensor *like, int channel, MNN_DATA_FORMAT fmt) {
        auto shape = like->shape();
        if (shape.size() >= 2) {
            shape[1] = channel;
        }
        if (t != nullptr && t->shape() == shape && t->getType() == like->getType() &&
            t->getDimensionType() == like->getDimensionType() &&
            TensorUtils::getDescribe(t.get())->dimensionFormat == fmt) {
            return false;
        }
        t.reset(Tensor::createDevice(shape, like->getType(), like->getDimensionType()));
        TensorUtils::getDescribe(t.get())->dimensionFormat = fmt;
        return true;
    }
    const Op *_wrapConv(const Convolution2D *conv, MNN_DATA_FORMAT fmt) {
        std::unique_ptr<flatbuffers::FlatBufferBuilder> builder(new flatbuffers::FlatBufferBuilder(1024));
        std::unique_ptr<Convolution2DT> convT(conv->UnPack());
        auto convOffset = Convolution2D::Pack(*builder, convT.get());
        // Sub-objects must be finished before the parent table is started.
        flatbuffers::Offset<flatbuffers::String> externalPath = 0;
        if (mFusedOp->externalPath() != nullptr) {
            externalPath = builder->CreateString(mFusedOp->externalPath()->str());
        }
        OpBuilder opB(*builder);
        opB.add_type(OpType_Convolution);
        opB.add_main(convOffset.Union());
        opB.add_main_type(OpParameter_Convolution2D);
        opB.add_defaultDimentionFormat(fmt);
        if (!externalPath.IsNull()) {
            opB.add_externalPath(externalPath);
        }
        builder->Finish(opB.Finish());
        auto op = flatbuffers::GetRoot<Op>(builder->GetBufferPointer());
        mBuilders.emplace_back(std::move(builder));
        return op;
    }
    const Op *_wrapMulSilu(MNN_DATA_FORMAT fmt) {
        std::unique_ptr<flatbuffers::FlatBufferBuilder> builder(new flatbuffers::FlatBufferBuilder(256));
        BinaryOpBuilder binaryB(*builder);
        binaryB.add_opType(BinaryOpOperation_MUL_SILU);
        auto mainOffset = binaryB.Finish().Union();
        OpBuilder opB(*builder);
        opB.add_type(OpType_BinaryOp);
        opB.add_main(mainOffset);
        opB.add_main_type(OpParameter_BinaryOp);
        opB.add_defaultDimentionFormat(fmt);
        builder->Finish(opB.Finish());
        auto op = flatbuffers::GetRoot<Op>(builder->GetBufferPointer());
        mBuilders.emplace_back(std::move(builder));
        return op;
    }
    const Op *_wrapLn(const LayerNorm *ln, MNN_DATA_FORMAT fmt) {
        std::unique_ptr<flatbuffers::FlatBufferBuilder> builder(new flatbuffers::FlatBufferBuilder(1024));
        std::unique_ptr<LayerNormT> lnT(ln->UnPack());
        auto lnOffset = LayerNorm::Pack(*builder, lnT.get());
        OpBuilder opB(*builder);
        opB.add_type(OpType_LayerNorm);
        opB.add_main(lnOffset.Union());
        opB.add_main_type(OpParameter_LayerNorm);
        opB.add_defaultDimentionFormat(fmt);
        builder->Finish(opB.Finish());
        auto op = flatbuffers::GetRoot<Op>(builder->GetBufferPointer());
        mBuilders.emplace_back(std::move(builder));
        return op;
    }
    bool _createChildren(MetalBackend *backend, const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs, Tensor *projInput, MNN_DATA_FORMAT fmt) {
        if ((int)mConvs.size() != mNumConvs) {
            mConvs.clear();
            for (int i = 0; i < mNumConvs; ++i) {
                auto convOp = _wrapConv(mParam->convs()->GetAs<Convolution2D>(i), fmt);
                Tensor *childOut = mIsGateUp ? (i == 0 ? mGate.get() : mUp.get()) : outputs[i];
                Execution *exe = backend->onCreate({projInput}, {childOut}, convOp);
                if (exe == nullptr || !exe->valid()) {
                    delete exe;
                    mConvs.clear();
                    return false;
                }
                mConvs.emplace_back(static_cast<MetalExecution *>(exe));
            }
        }
        if (mIsGateUp && !mMulSilu) {
            // MUL_SILU convention: in0 = up, in1 = gate (out = in0 * silu(in1)).
            Execution *exe = backend->onCreate({mUp.get(), mGate.get()}, {outputs[0]}, _wrapMulSilu(fmt));
            if (exe == nullptr) {
                return false;
            }
            mMulSilu.reset(static_cast<MetalExecution *>(exe));
        }
        if (mHasLn && !mLn) {
            // Binary RMSNorm convention: in [residual, hidden], out [sum, normalized].
            Execution *exe = backend->onCreate({inputs[0], inputs[1]}, {outputs[mNumProjOut], mNormalized.get()},
                                               _wrapLn(mParam->ln(), fmt));
            if (exe == nullptr) {
                return false;
            }
            mLn.reset(static_cast<MetalLayerNorm *>(exe));
        }
        return true;
    }

    const FusedLinearParam *mParam = nullptr;
    const Op *mFusedOp = nullptr;
    std::vector<std::unique_ptr<flatbuffers::FlatBufferBuilder>> mBuilders;
    std::vector<std::unique_ptr<MetalExecution>> mConvs;
    std::unique_ptr<MetalExecution> mMulSilu;
    std::unique_ptr<MetalLayerNorm> mLn;
    std::shared_ptr<Tensor> mNormalized;
    std::shared_ptr<Tensor> mGate;
    std::shared_ptr<Tensor> mUp;
    // Bumped whenever one of the intermediates above is re-allocated; compared
    // against the value stamped by onEncode to invalidate stale recordings.
    int mResizeGeneration   = 0;
    int mRecordedGeneration = -1;
    unsigned mRecordedSteps = 0;
#ifdef MNN_LOW_MEMORY
    DecodePlan mDecode;
    MultiProjPrefillPlan mMultiProjPrefill;
    std::array<MetalConv1x1WQuant::QuantMember, 4> mDecodeMembers;
    // Reuse backing storage across resizes. The installed plan decides which
    // optional resources (LN epsilon or merged weights) the dispatch binds.
    id<MTLBuffer> mDecodeSegments = nil;
    id<MTLBuffer> mDecodeEps = nil;
    id<MTLBuffer> mQKVMergeStorage = nil;
    uint64_t mGroupStamp = 0;
    uint64_t mRecordedGroupStamp = 0;
    // Captured in onResize for setupFusion (which runs later, in onResizeEnd).
    std::vector<Tensor *> mProjOutputs;
    const Tensor *mProjInput     = nullptr;
    const Tensor *mLnResidualIn  = nullptr;
    const Tensor *mLnHiddenIn    = nullptr;
    Tensor *mLnResidualOut       = nullptr;
#endif
    id<MTLComputePipelineState> mRowScalePipeline = nil;
    id<MTLBuffer> mRowScaleShape = nil;
    std::pair<MTLSize, MTLSize> mRowScaleThreads = {};
    const Tensor* mRowScaleInput = nullptr;
    int mRowScaleTokens = -1;
    int mRowScaleQuads = -1;
    bool mRowMul = false;
    bool mIsGateUp   = false;
    bool mPrefillSiluFused = false;   // up projection's prefill epilogue emits it instead
    bool mPrefillDualFused = false;   // one dispatch computes both tiles, silu-mul in registers
    bool mHasLn      = false;
    int mNumConvs    = 0;
    int mNumProjOut  = 0;
    MNN_DATA_FORMAT mChildFormat = MNN_DATA_FORMAT_NC4HW4;
};

class MetalFusedProjCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend,
                                const std::vector<Tensor *> &outputs) const override {
        // Must match GeometryFusedProj::_keepWhole exactly: an op the geometry
        // keeps whole but this refuses fails session creation with the member
        // weights already stripped.
        if (!FusedProjCommon::nativeEnvelopeOk(op, inputs.size(), outputs.size(), /*allowRowMul=*/true) ||
            !FusedProjCommon::allMembersAre1x1(op)) {
            return nullptr;
        }
        return new MetalFusedProj(backend, op);
    }
};
REGISTER_METAL_OP_TRANSFORMER_CREATOR(MetalFusedProjCreator, OpType_FusedLinear);

} // namespace MNN
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
