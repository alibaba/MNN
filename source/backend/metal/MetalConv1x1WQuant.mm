//
//  MetalConv1x1WQuant.mm
//  MNN
//

#import "backend/metal/MetalConv1x1WQuant.hpp"
#import "backend/metal/MetalEnv.hpp"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MetalSharedGather.hpp"
#import "ConvSimdGroupShader.hpp"

#if MNN_METAL_ENABLED && defined(MNN_LOW_MEMORY)

#include <mutex>
#include <set>
#include <tuple>

#if MNN_METAL_OP_PROFILE
#define CONV1X1_SET_TAG(name) mProfileTag = (name)
#else
#define CONV1X1_SET_TAG(name) do {} while(0)
#endif

namespace MNN {

MetalConv1x1WQuant::MetalConv1x1WQuant(Backend *backend, const MNN::Op *op) : MetalConvolutionCommon(backend, op, nullptr) {
    loadWeight(op, true);
}

MetalConv1x1WQuant::MetalConv1x1WQuant(Backend *backend, const MNN::Op *op,
                                       std::shared_ptr<MNN::Tensor> weight,
                                       std::shared_ptr<MNN::Tensor> bias,
                                       std::shared_ptr<MNN::Tensor> dequantScale,
                                       int dequantBits, float scaleCoef)
    : MetalConvolutionCommon(backend, op, bias) {
    mWeight = weight;
    mBias = bias;
    mDequantScaleBias = dequantScale;
    mDequantBits = dequantBits;
    mScaleCoef = scaleCoef;
}

bool MetalConv1x1WQuant::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    if (op->type() == OpType_GatherV2) {
        auto dequantScale = getDequantScale();
        if (!dequantScale ||
            (mDequantBits != 2 && mDequantBits != 3 && mDequantBits != 4 && mDequantBits != 8)) {
            return false;
        }
        auto conv2D = mOp->main_as_Convolution2D();
        int oc = conv2D->common()->outputCount();
        *dst = new MetalSharedGather(bn, oc, mWeight, dequantScale, mDequantBits, mScaleCoef);
        MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, *dst);
        return true;
    }
    *dst = new MetalConv1x1WQuant(bn, op, mWeight, mBias, mDequantScaleBias,
                                  mDequantBits, mScaleCoef);
    MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, *dst);
    return true;
}

Conv1x1DeviceProfile conv1x1DeviceProfile(const MetalGpuInfo& gpu) {
    Conv1x1DieClass die;
    switch (gpu.tier) {
        case MetalGpuInfo::TierBase:
            // Includes phones: the 'p' die reports TierBase, so generation alone separates it
            // from an M-series chip of the same class.
            die = Conv1x1DieClass::Base;
            break;
        case MetalGpuInfo::TierPro:
            die = Conv1x1DieClass::Pro;
            break;
        case MetalGpuInfo::TierMax:
            die = Conv1x1DieClass::Max;
            break;
        case MetalGpuInfo::TierUltra:
            die = Conv1x1DieClass::Ultra;
            break;
        default:
            die = Conv1x1DieClass::Unknown;
            break;
    }
    return conv1x1ProfileFrom(die, gpu.gen);
}

bool MetalConv1x1WQuant::quantLayoutCompatible(const MetalConv1x1WQuant* peer) const {
    if (peer == nullptr || peer == this) {
        return false;
    }
    if (mDequantBits != peer->mDequantBits || mActivationType != peer->mActivationType) {
        return false;
    }
    if (mWeight == nullptr || peer->mWeight == nullptr ||
        mBias == nullptr || peer->mBias == nullptr ||
        mDequantScaleBias == nullptr || peer->mDequantScaleBias == nullptr) {
        return false;
    }
    const auto* selfCommon = mOp->main_as_Convolution2D()->common();
    const auto* peerCommon = peer->mOp->main_as_Convolution2D()->common();
    if (selfCommon->inputCount() != peerCommon->inputCount() ||
        selfCommon->outputCount() != peerCommon->outputCount()) {
        return false;
    }
    if (mConstBuffer != nil && peer->mConstBuffer != nil) {
        if (mBlockCount != peer->mBlockCount) {
            return false;
        }
        const auto* self  = (const Param*)mConstBuffer.contents;
        const auto* other = (const Param*)peer->mConstBuffer.contents;
        if (self->inputDepthQuad != other->inputDepthQuad ||
            self->outputDepthQuad != other->outputDepthQuad ||
            self->outputChannel != other->outputChannel) {
            return false;
        }
    }
    // Packed layout as actually loaded: byte extent of the packed weight and of
    // the per-block scale table. Matching outputCount + scale table extent implies
    // matching blockCount even before peer->onResize() has run.
    return mWeight->usize() == peer->mWeight->usize() &&
           mDequantScaleBias->usize() == peer->mDequantScaleBias->usize();
}

MetalConv1x1WQuant::QuantMember MetalConv1x1WQuant::decodeMember() const {
    return mDecodeFusionEligible && mFusedRowScale == nullptr ? quantMember() : QuantMember();
}

MetalConv1x1WQuant::QuantMember MetalConv1x1WQuant::shortPrefillMember(int* reduceRows) const {
    if (!mPrepared || mFusedRowScale != nullptr ||
        mPrefillPlan.kernel != PrefillKernel::SgReduceG4mN ||
        mPrefillPlan.pieceCount != 1 || mPrefillPlan.protectPieceTail ||
        mPrefillPlan.reduceRowsPerPiece < 2 || mPrefillPlan.reduceRowsPerPiece > 4) {
        return {};
    }
    if (reduceRows != nullptr) {
        *reduceRows = mPrefillPlan.reduceRowsPerPiece;
    }
    return quantMember();
}

MetalConv1x1WQuant::QuantMember MetalConv1x1WQuant::multiProjPrefillMember() const {
    return multiProjPrefillTileM() > 0 && mPrefillPlan.fusion == PrefillFusion::None ? quantMember() : QuantMember();
}

int MetalConv1x1WQuant::multiProjPrefillTileM() const {
    switch (mPrefillPlan.kernel) {
        case PrefillKernel::TensorInKernel8x64:  return 8;
        case PrefillKernel::TensorInKernel16x64: return 16;
        case PrefillKernel::TensorInKernel32x64: return 32;
        case PrefillKernel::TensorInKernel64x64: return 64;
        default: return 0;
    }
}

MetalConv1x1WQuant::QuantMember MetalConv1x1WQuant::quantMember() const {
    if (!mPrepared || mConstBuffer == nil || mBlockCount <= 0 ||
        !mWeight || !mBias || !mDequantScaleBias) {
        return {};
    }
    const auto* param = static_cast<const Param*>(mConstBuffer.contents);
    QuantMember member;
    member.weight = mWeight.get();
    member.bias = mBias.get();
    member.dequant = mDequantScaleBias.get();
    member.params = mConstBuffer;
    member.bits = mDequantBits;
    member.blocks = mBlockCount;
    member.activation = mActivationType;
    member.inputChannels = mOp->main_as_Convolution2D()->common()->inputCount();
    member.inputQuads = param->inputDepthQuad;
    member.outputChannels = param->outputChannel;
    member.outputQuads = param->outputDepthQuad;
    member.scaleCoef = mScaleCoef;
    return member;
}

ErrorCode MetalConv1x1WQuant::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto context = (__bridge MNNMetalContext*)mtbn->context();
    auto previousPipeline = mMain.pipeline;
    auto clearPreparation = [&]() {
        mPrepared = false;
        mFusedRowScale = nullptr;
        mPrefillPlan = QuantPrefillPlan();
        mMain.pipeline = nil;
        mDequant.pipeline = nil;
        mPack.pipeline = nil;
        mReducer.pipeline = nil;
        mMain.threads = mDequant.threads = mReducer.threads = mPack.threads =
            std::make_pair(MTLSizeMake(0, 0, 0), MTLSizeMake(0, 0, 0));
        mDecodeFusionEligible = false;
        CONV1X1_SET_TAG("");
    };
    auto ready = [&](id<MTLComputePipelineState> pipeline, MTLSize threads) {
        const NSUInteger count = threads.width * threads.height * threads.depth;
        return pipeline != nil && count > 0 && pipeline.maxTotalThreadsPerThreadgroup >= count &&
               pipeline.staticThreadgroupMemoryLength <= context.device.maxThreadgroupMemoryLength;
    };
    const bool tensorSupported = mtbn->isSupportTensorApi();
    for (int attempt = 0; attempt < 2; ++attempt) {
        clearPreparation();
        const bool allowTensorApi = tensorSupported && attempt == 0;
        QuantPrefillPlan candidate;
        auto code = resizeImpl(inputs, outputs, allowTensorApi, candidate);
        if (code == NO_ERROR &&
            (!ready(mMain.pipeline, mMain.threads.second) ||
             (candidate.weightPreparation() != WeightPreparation::None &&
              !ready(mDequant.pipeline, mDequant.threads.second)) ||
             (candidate.weightPreparation() == WeightPreparation::RowDevice &&
              !ready(mPack.pipeline, mPack.threads.second)) ||
             (candidate.kPartitions > 1 && !ready(mReducer.pipeline, mReducer.threads.second)))) {
            code = NOT_SUPPORT;
        }
        if (code == NO_ERROR) {
            mPrefillPlan = candidate;
            mPrepared = true;
            if (attempt != 0 || mMain.pipeline != previousPipeline) {
                mReplayEvents.clear();
                mReplayState = 0;
                mReplayStable = 0;
            }
            return NO_ERROR;
        }
        const bool retry = code == NOT_SUPPORT && allowTensorApi && candidate.usesTensorApi();
        clearPreparation();
        invalidateReplay();
        mRecordedStructure = 0;
        if (!retry) {
            MNN_ERROR("Metal Conv1x1 pipeline preparation failed: %d.\n", code);
            return code;
        }
        MNN_PRINT("Metal Conv1x1 Tensor pipeline unavailable; retrying non-Tensor path.\n");
    }
    return NOT_SUPPORT;
}

ErrorCode MetalConv1x1WQuant::resizeImpl(const std::vector<Tensor *> &inputs,
                                         const std::vector<Tensor *> &outputs, bool allowTensorApi,
                                         QuantPrefillPlan& candidate) {
    MetalConvolutionCommon::onResize(inputs, outputs);

    // prepare
    // For C4NHW4 format, NHW can be fuse to W
    auto input = inputs[0];
    auto output = outputs[0];
    int is = input->batch();
    for (int i=2; i<input->dimensions(); ++i) {
        is *= input->length(i);
    }
    int ic  = input->channel();
    int icDiv4  = UP_DIV(input->channel(), 4);
    int ow  = is;
    int oh  = 1;
    int os  = ow;
    int ob  = 1;
    auto oc  = output->channel();
    auto ocDiv4  = UP_DIV(output->channel(), 4);
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto dequantScale = getDequantScale();
    int blockCount = 1;
    if (dequantScale) {
        int bytes = sizeof(float);
        if(backend->useFp16InsteadFp32()) {
            bytes = sizeof(__fp16);
        }
        blockCount = (int)(dequantScale->usize() / bytes / ocDiv4 / 2 / 4);
    }
    // Allocate once and rewrite in place: this buffer is bound raw, so encode-replay
    // cannot see a swap and a live recording would keep emitting the previous Param.
    if (nil == mConstBuffer) {
        mConstBuffer = backend->getConstBuffer(sizeof(Param));
    }
    auto param = (Param *)mConstBuffer.contents;
    param->inputSize = is;
    param->inputDepthQuad = icDiv4;
    param->outputWidth = ow;
    param->outputHeight = oh;
    param->outputSize = os;
    param->outputDepthQuad = ocDiv4;
    param->outputChannel = oc;
    param->batch = ob;
    param->blockCount = blockCount;
    param->activation = mActivationType;
    param->scaleCoef = mScaleCoef;
    mBlockCount = blockCount;
    const int area = ob * ow * oh;
    auto rt = backend->runtime();
    mDeviceProfile = conv1x1DeviceProfile(rt->gpuInfo());
    if (area == 1) {
        return prepareDecodeConv(inputs, outputs);
    }
    if (!dequantScale) {
        return NOT_SUPPORT;
    }

    PrefillCaps caps;
    caps.allowTensorApi = allowTensorApi;
    caps.hasSgMatrix = rt->supportSimdGroupMatrix();
    caps.hasSgReduce = rt->supportSimdGroupReduce();
    caps.lowMemoryKernelsAvailable = true;
    caps.gpuGen = rt->gpuInfo().gen;
    caps.maxThreadgroupMemoryBytes = context.device.maxThreadgroupMemoryLength;
    QuantPrefillShape shape;
    shape.area = area;
    shape.ic = ic;
    shape.oc = oc;
    shape.blockCount = blockCount;
    shape.dequantBits = mDequantBits;
    shape.fp16Storage = backend->useFp16InsteadFp32();
    shape.siluOffer = nullptr != mPrefillSilu.gateOffer;
    // An incompatible peer loses only the dual; the read-only gate offer above
    // still lets the SiLU epilogue fold, else the group runs two projections.
    shape.dualOffer = quantLayoutCompatible(mPrefillSilu.dualPeerOffer);
    shape.gateUpDualMinArea = MetalEnv::get().fusedQ4GateUpDualMinArea;
    candidate = selectQuantPrefill(caps, shape);
    return prepareQuantPrefill(candidate, shape, inputs, outputs);
}

static const char* prefillSourceName(PrefillSource source) {
    switch (source) {
        case PrefillSource::None: return "None";
        case PrefillSource::WfpMatrix: return "WfpMatrix";
        case PrefillSource::WqMatrix: return "WqMatrix";
        case PrefillSource::WqReduce: return "WqReduce";
        case PrefillSource::Precompiled: return "Precompiled";
    }
    return "Unknown";
}

// Serialize deduplication across runtimes; never take the lock on a cache hit or
// a successful build. Keep the original NOT_SUPPORT/fallback behavior.
static void reportUnbuiltPipeline(const char* where, const PrefillPipeline& stage) {
    static std::mutex mutex;
    static std::set<std::tuple<PrefillSource, std::string, std::string>> reported;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!reported.emplace(stage.source, stage.entry, where).second) {
            return;
        }
    }
    MNN_ERROR("MNN::Metal: conv1x1 %s failed for '%s' (source=%s)\n", where,
              stage.entry.c_str(), prefillSourceName(stage.source));
}

static id<MTLComputePipelineState> loadPrefillPipeline(MetalBackend* backend, const PrefillPipeline& stage) {
    auto context = (__bridge MNNMetalContext*)backend->context();
    if (stage.source == PrefillSource::Precompiled) {
        auto pipeline = [context pipelineWithName:@(stage.entry.c_str()) fp16:stage.precompiledFp16];
        if (pipeline == nil) {
            reportUnbuiltPipeline("precompiled", stage);
        }
        return pipeline;
    }
    auto rt = backend->runtime();
    auto pipeline = rt->findPipeline(stage.keys);
    if (pipeline != nil) {
        return pipeline;
    }
    const auto source = conv1x1ShaderSource(stage.source);
    if (source.empty()) {
        reportUnbuiltPipeline("unmapped source", stage);
        return nil;
    }
    auto macros = [NSMutableDictionary dictionaryWithCapacity:stage.macros.size()];
    for (const auto& macro : stage.macros) {
        [macros setValue:@(macro.second.c_str()) forKey:@(macro.first.c_str())];
    }
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.preprocessorMacros = macros;
    pipeline = backend->makeComputePipelineWithSourceOption(
        source.c_str(), stage.entry.c_str(), options, stage.optionalCompilation);
    if (pipeline != nil) {
        rt->insertPipeline(stage.keys, pipeline);
    } else {
        reportUnbuiltPipeline("compile", stage);
    }
    return pipeline;
}

// The group holds its own launch geometry; reuse this member's Tensor prefill specialization.
id<MTLComputePipelineState> MetalConv1x1WQuant::multiProjPrefillPipeline() const {
    auto metal = static_cast<MetalBackend*>(backend());
    const auto* param = static_cast<const Param*>(mConstBuffer.contents);
    QuantPrefillShape shape;
    shape.area = param->inputSize * param->batch;
    shape.ic = mOp->main_as_Convolution2D()->common()->inputCount();
    shape.oc = param->outputChannel;
    shape.blockCount = mBlockCount;
    shape.dequantBits = mDequantBits;
    shape.fp16Storage = metal->useFp16InsteadFp32();
    QuantPrefillPlan multiProjPlan = mPrefillPlan;
    multiProjPlan.kPartitions = 1;
    multiProjPlan.tileN = 0;
    if (!shape.fp16Storage) {
        multiProjPlan.loopK64 = false;
        multiProjPlan.doubleBuf = false;
    }
    auto description = describeQuantPrefill(multiProjPlan, shape);
    description.main.specialize("FQ_MULTI_PROJ_OUTPUT");
    return loadPrefillPipeline(metal, description.main);
}

ErrorCode MetalConv1x1WQuant::buildPrefillStage(PrefillStage& stage, const PrefillPipeline& description,
                                                MetalBackend* backend,
                                                const std::pair<MTLSize, MTLSize>* explicitThreads) {
    stage.pipeline = loadPrefillPipeline(backend, description);
    if (stage.pipeline == nil) {
        return NOT_SUPPORT;
    }
    if (explicitThreads != nullptr) {
        stage.threads = *explicitThreads;
        return NO_ERROR;
    }
    const auto grid = MTLSizeMake(description.grid.x, description.grid.y, description.grid.z);
    if (description.finalization == PrefillFinalization::ComputeBestGroupAndLocal) {
        auto context = (__bridge MNNMetalContext*)backend->context();
        stage.threads = [context computeBestGroupAndLocal:stage.pipeline threads:grid];
    } else {
        stage.threads = std::make_pair(grid, MTLSizeMake(description.threads.x, description.threads.y,
                                                         description.threads.z));
    }
    return NO_ERROR;
}

// Reservations only, after every pipeline of the plan is known to build: a pipeline that
// cannot compile must not hit an allocation failure first. Only reservations actually
// acquired here are rolled back.
ErrorCode MetalConv1x1WQuant::acquirePrefillScratch(const PrefillDescription& description, MetalBackend* backend) {
    if (description.tempWeightBytes > 0) {
        const int tempSize = (int)description.tempWeightBytes;
        // Replay retains Tensor pointers; only a size change replaces the object.
        if (!mTempWeight || mTempWeight->elementSize() != tempSize) {
            mTempWeight.reset(Tensor::createDevice<uint8_t>({tempSize}));
            ++mScratchGeneration;
        }
        if (!backend->onAcquireBuffer(mTempWeight.get(), Backend::DYNAMIC)) {
            return OUT_OF_MEMORY;
        }
        if (description.packedInputBytes > 0) {
            const int inputBytes = (int)description.packedInputBytes;
            if (!mPackedInput || mPackedInput->elementSize() != inputBytes) {
                mPackedInput.reset(Tensor::createDevice<uint8_t>({inputBytes}));
                ++mScratchGeneration;
            }
            // Both allocations are live during the main dispatch.
            if (!backend->onAcquireBuffer(mPackedInput.get(), Backend::DYNAMIC)) {
                backend->onReleaseBuffer(mTempWeight.get(), Backend::DYNAMIC);
                return OUT_OF_MEMORY;
            }
            backend->onReleaseBuffer(mPackedInput.get(), Backend::DYNAMIC);
        }
        backend->onReleaseBuffer(mTempWeight.get(), Backend::DYNAMIC);
    }
    if (description.reducer.source != PrefillSource::None) {
        const int partialBytes = (int)description.partialBytes;
        if (!mKsplitPartial || mKsplitPartial->elementSize() != partialBytes) {
            mKsplitPartial.reset(Tensor::createDevice<uint8_t>({partialBytes}));
            ++mScratchGeneration;
        }
        if (!backend->onAcquireBuffer(mKsplitPartial.get(), Backend::DYNAMIC)) {
            return OUT_OF_MEMORY;
        }
        backend->onReleaseBuffer(mKsplitPartial.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

ErrorCode MetalConv1x1WQuant::prepareQuantPrefill(const QuantPrefillPlan& plan, const QuantPrefillShape& shape,
                                                  const std::vector<Tensor *> &inputs,
                                                  const std::vector<Tensor*>& outputs) {
    if (plan.kernel == PrefillKernel::Unsupported) {
        return NOT_SUPPORT;
    }
    auto description = describeQuantPrefill(plan, shape);
    if (mRowScaleOffer != nullptr && (plan.kernel == PrefillKernel::SgReduceG4mN ||
                                      plan.kernel == PrefillKernel::SgReduceG8)) {
        description.main.specialize("CONV_ROW_MUL");
        mFusedRowScale = mRowScaleOffer;
    }
    auto backend = static_cast<MetalBackend*>(this->backend());
    auto context = (__bridge MNNMetalContext*)backend->context();
    auto ready = [&](const PrefillStage& stage) {
        const NSUInteger count = stage.threads.second.width * stage.threads.second.height *
                                 stage.threads.second.depth;
        return stage.pipeline != nil && count > 0 &&
               stage.pipeline.maxTotalThreadsPerThreadgroup >= count &&
               stage.pipeline.staticThreadgroupMemoryLength <= context.device.maxThreadgroupMemoryLength;
    };
    auto build = [&](PrefillStage& stage, const PrefillPipeline& desc,
                     const std::pair<MTLSize, MTLSize>* geometry) {
        if (buildPrefillStage(stage, desc, backend, geometry) != NO_ERROR) {
            return NOT_SUPPORT;
        }
        return ready(stage) ? NO_ERROR : NOT_SUPPORT;
    };
    if (description.pack.source != PrefillSource::None) {
        // Historical launch formula, computed once from the actual input; ready, encode
        // and Replay all consume this saved dispatch.
        const auto threads = description.pack.threads.x;
        const size_t packQuads = (size_t)inputs[0]->elementSize() / 4;
        const auto packThreads = std::make_pair(MTLSizeMake((packQuads + threads - 1) / threads, 1, 1),
                                                MTLSizeMake(threads, 1, 1));
        auto code = build(mPack, description.pack, &packThreads);
        if (code != NO_ERROR) {
            return code;
        }
    }
    if (description.dequant.source != PrefillSource::None) {
        auto code = build(mDequant, description.dequant, nullptr);
        if (code != NO_ERROR) {
            return code;
        }
    }
    if (plan.fusion == PrefillFusion::GateUpDual) {
        if (mPrefillSilu.dualCoef == nil) {
            mPrefillSilu.dualCoef = backend->getConstBuffer(sizeof(float));
        }
        ((float*)mPrefillSilu.dualCoef.contents)[0] = mPrefillSilu.dualPeerOffer->scaleCoef();
    }
    if (description.main.finalization == PrefillFinalization::GetGridAndThreadgroup) {
        auto code = prepareQuantScalar(description.main, inputs, outputs);
        if (code != NO_ERROR) {
            return code;
        }
    } else if (buildPrefillStage(mMain, description.main, backend, nullptr) != NO_ERROR) {
        return NOT_SUPPORT;
    }
    // The main dispatch is validated on both branches, the scalar one included, exactly
    // as before the split into build helpers.
    if (!ready(mMain)) {
        return NOT_SUPPORT;
    }
    if (description.reducer.source != PrefillSource::None) {
        auto code = build(mReducer, description.reducer, nullptr);
        if (code != NO_ERROR) {
            return code;
        }
    }
    auto code = acquirePrefillScratch(description, backend);
    if (code != NO_ERROR) {
        return code;
    }
    CONV1X1_SET_TAG(description.profileTag);
    return NO_ERROR;
}

ErrorCode MetalConv1x1WQuant::prepareQuantScalar(const PrefillPipeline& stage,
                                                 const std::vector<Tensor*>& inputs,
                                                 const std::vector<Tensor*>& outputs) {
    auto backend = static_cast<MetalBackend*>(this->backend());
    auto context = (__bridge MNNMetalContext*)backend->context();
    auto rt = const_cast<MetalRuntime*>(backend->runtime());
    mMain.pipeline = loadPrefillPipeline(backend, stage);
    if (mMain.pipeline == nil) {
        return NOT_SUPPORT;
    }
    auto input = inputs[0];
    auto output = outputs[0];
    auto dequantScale = getDequantScale();
    NSArray *arr = [NSArray arrayWithObjects:
        ((MetalRuntimeAllocator::MetalBufferAlloc*)input->deviceId())->getBuffer(),
        ((MetalRuntimeAllocator::MetalBufferAlloc*)output->deviceId())->getBuffer(),
        mConstBuffer, ((MetalRuntimeAllocator::MetalBufferAlloc*)mWeight->deviceId())->getBuffer(),
        ((MetalRuntimeAllocator::MetalBufferAlloc*)mBias->deviceId())->getBuffer(),
        ((MetalRuntimeAllocator::MetalBufferAlloc*)dequantScale->deviceId())->getBuffer(), nil];
    int buffer_offset[] = {
        TensorUtils::getDescribeOrigin(input)->offset,
        TensorUtils::getDescribeOrigin(output)->offset,
        0,
        TensorUtils::getDescribeOrigin(mWeight.get())->offset,
        TensorUtils::getDescribeOrigin(mBias.get())->offset,
        TensorUtils::getDescribeOrigin(dequantScale.get())->offset,
        0};
    auto ret = [context getGridAndThreadgroup:mMain.pipeline gid:MTLSizeMake(stage.grid.x, stage.grid.y, stage.grid.z)
        loop:10 buffer:arr runtime:rt shaderName:stage.entry offsets:buffer_offset queue:backend->queue()];
    mMain.threads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
    CONV1X1_SET_TAG(stage.entry);
    return NO_ERROR;
}

ErrorCode MetalConv1x1WQuant::prepareDecodeConv(const std::vector<Tensor*>& inputs,
                                                const std::vector<Tensor*>& outputs) {
    if (!getDequantScale()) {
        return NOT_SUPPORT;
    }
    auto backend = static_cast<MetalBackend*>(this->backend());
    auto rt = backend->runtime();
    const auto param = (const Param*)mConstBuffer.contents;
    const int oc = param->outputChannel;
    const int ocDiv4 = param->outputDepthQuad;
    const int icDiv4 = param->inputDepthQuad;
    const int blockCount = mBlockCount;
    const auto& profile = mDeviceProfile;
    const bool fp16 = backend->useFp16InsteadFp32();
    auto dic = conv1x1Macros(fp16);
    std::vector<std::string> baseKeys = {fp16 ? "half4" : "float4", "MNN_METAL_FLOAT32_COMPUTER"};
    if (mDequantBits != 2 && mDequantBits != 3 && mDequantBits != 4 && mDequantBits != 8) {
        MNN_ERROR("metal conv weight quant not support %d bits yet!\n", mDequantBits);
        return NOT_SUPPORT;
    }
    const std::string quant = "W_QUANT_" + std::to_string(mDequantBits);
    [dic setValue:@"1" forKey:@(quant.c_str())];
    baseKeys.emplace_back("conv1x1_wquant_" + std::to_string(mDequantBits));
    if (rt->supportSimdGroupReduce()) {
        auto context = (__bridge MNNMetalContext*)backend->context();
        auto decodePipelineReady = [&](id<MTLComputePipelineState> pipeline, NSUInteger threads) {
            return pipeline != nil && pipeline.maxTotalThreadsPerThreadgroup >= threads &&
                   pipeline.staticThreadgroupMemoryLength <= context.device.maxThreadgroupMemoryLength;
        };
        baseKeys.emplace_back("conv1x1_wquant_sg_reduce");
        const bool g16LmHeadRoute = oc > 16384 && ocDiv4 % 2 == 0 && !(oc % 8 == 0 && blockCount % 2 == 0);
        if (g16LmHeadRoute) {
            const auto& body = conv1x1Kernel(Conv1x1Kernel::GemvG16);
            auto keys = baseKeys;
            keys.emplace_back(body.entry);
            mMain.pipeline = loadConv1x1Pipeline(backend, keys, conv1x1ShaderSource(body.source), body.entry, dic);
            if (!decodePipelineReady(mMain.pipeline, body.threads)) {
                mMain.pipeline = nil;
                return NOT_SUPPORT;
            }
            CONV1X1_SET_TAG(keys.back());
            mMain.threads = std::make_pair(MTLSizeMake(UP_DIV(oc, body.tileN), 1, 1), MTLSizeMake(body.threads, 1, 1));
        } else {
            const int rawIc = mOp->main_as_Convolution2D()->common()->inputCount();
            // Raw channels must divide evenly into blocks to keep the C4 tail out of tuned decode.
            const bool wqEvenBlock = (fp16 || profile.tier.decodeFp32Validated) &&
                blockCount > 0 && rawIc % blockCount == 0;
            DecodeCaps caps;
            caps.tier = profile.tier;
            caps.tgBudget = profile.tgBudget;
            caps.wqEvenBlockFp16 = wqEvenBlock;
            DecodeShape shape;
            shape.oc = oc;
            shape.icDiv4 = icDiv4;
            shape.blockCount = blockCount;
            shape.dequantBits = mDequantBits;
            const DecodeGemvPlan plan = selectDecodeGemv(caps, shape);
            const auto& body = conv1x1Kernel(Conv1x1Kernel::GemvG4m1_2sg);
            const auto source = conv1x1ShaderSource(body.source);
            auto keys = baseKeys;
            if (mRowScaleOffer != nullptr) {
                [dic setValue:@"1" forKey:@"CONV_ROW_MUL"];
                keys.emplace_back("CONV_ROW_MUL");
                mFusedRowScale = mRowScaleOffer;
            }
            keys.emplace_back(body.entry);
            auto buildPlain = [&]() -> bool {
                auto plainKeys = keys;
                NSMutableDictionary* plainDic = [dic mutableCopy];
                const int plainLpb = gemvPlainLanesPerBlock(icDiv4, blockCount, mDequantBits);
                if (plainLpb > 0) {
                    plainKeys.emplace_back("lanes" + std::to_string(plainLpb));
                    [plainDic setValue:@(std::to_string(plainLpb).c_str()) forKey:@"GEMV_LANES_PER_BLOCK"];
                }
                mMain.threads = std::make_pair(MTLSizeMake(UP_DIV(oc, 8), 1, 1), MTLSizeMake(64, 1, 1));
                mMain.pipeline = loadConv1x1Pipeline(backend, plainKeys, source, body.entry, plainDic);
                if (!decodePipelineReady(mMain.pipeline, 64)) {
                    mMain.pipeline = nil;
                    return false;
                }
                CONV1X1_SET_TAG(plainKeys.back());
                return true;
            };
            if (plan.variant == DecodeGemvVariant::Oc8Nsg4SplitK2) {
                auto skKeys = keys;
                skKeys.emplace_back("oc8Nsg4SplitK2");
                if (plan.lanesPerBlock > 0) {
                    skKeys.emplace_back("lanes" + std::to_string(plan.lanesPerBlock));
                }
                NSMutableDictionary *skDic = [dic mutableCopy];
                [skDic setValue:@"2" forKey:@"GEMV_QUADS_PER_TG"];
                [skDic setValue:@"2" forKey:@"GEMV_SPLIT_K"];
                if (plan.lanesPerBlock > 0) {
                    [skDic setValue:@(std::to_string(plan.lanesPerBlock).c_str()) forKey:@"GEMV_LANES_PER_BLOCK"];
                }
                mMain.threads = std::make_pair(MTLSizeMake(UP_DIV(oc, 8), 1, 1), MTLSizeMake(128, 1, 1));
                mMain.pipeline = loadConv1x1Pipeline(backend, skKeys, source, body.entry, skDic);
                if (decodePipelineReady(mMain.pipeline, 128)) {
                    CONV1X1_SET_TAG("splitk2_gemv_g4m1_2sg_wquant_sg");
                } else if (!buildPlain()) {
                    return NOT_SUPPORT;
                }
            } else {
                const bool useOc16Sg4K1 = plan.variant == DecodeGemvVariant::Oc16Nsg4SplitK1;
                auto vKeys = keys;
                if (useOc16Sg4K1) {
                    vKeys.emplace_back("oc16Nsg4SplitK1");
                }
                if (plan.lanesPerBlock > 0) {
                    vKeys.emplace_back("lanes" + std::to_string(plan.lanesPerBlock));
                }
                NSMutableDictionary *dic4 = [dic mutableCopy];
                if (useOc16Sg4K1) {
                    [dic4 setValue:@"4" forKey:@"GEMV_QUADS_PER_TG"];
                }
                if (plan.lanesPerBlock > 0) {
                    [dic4 setValue:@(std::to_string(plan.lanesPerBlock).c_str()) forKey:@"GEMV_LANES_PER_BLOCK"];
                }
                const NSUInteger threads = 32 * plan.tgQuads;
                mMain.threads = std::make_pair(MTLSizeMake(UP_DIV(oc, 4 * plan.tgQuads), 1, 1),
                                          MTLSizeMake(threads, 1, 1));
                mMain.pipeline = loadConv1x1Pipeline(backend, vKeys, source, body.entry, dic4);
                if (decodePipelineReady(mMain.pipeline, threads)) {
                    CONV1X1_SET_TAG(vKeys.back());
                } else if (!useOc16Sg4K1 || !buildPlain()) {
                    mMain.pipeline = nil;
                    return NOT_SUPPORT;
                }
            }
            // FusedLinear can use this member in a group GEMV.
            mDecodeFusionEligible = true;
        }
        return NO_ERROR;
    }
    if (mDequantBits != 4 && mDequantBits != 8) {
        MNN_ERROR("metal W_QUANT_%d conv1x1 requires simdgroup reduce support!\n", mDequantBits);
        return NOT_SUPPORT;
    }
    PrefillPipeline scalar;
    scalar.setKernel(mDequantBits == 4 ? Conv1x1Kernel::ScalarW4 : Conv1x1Kernel::ScalarW8);
    scalar.precompiledFp16 = fp16;
    scalar.grid = PrefillSize(1, ocDiv4, 1);
    scalar.finalization = PrefillFinalization::GetGridAndThreadgroup;
    return prepareQuantScalar(scalar, inputs, outputs);
}

uint64_t MetalConv1x1WQuant::replayStructure(const std::vector<Tensor*>& inputs) const {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    auto mixPtr = [&mix](const void* p) { mix((uint64_t)(uintptr_t)p); };
    auto mixPipeline = [&mixPtr](id<MTLComputePipelineState> p) { mixPtr((__bridge void*)p); };
    auto mixBuffer = [&mixPtr](id<MTLBuffer> b) { mixPtr((__bridge void*)b); };
    auto mixSize = [&mix](MTLSize s) {
        mix((uint64_t)s.width);
        mix((uint64_t)s.height);
        mix((uint64_t)s.depth);
    };
    auto mixGrid = [&mixSize](const std::pair<MTLSize, MTLSize>& g) {
        mixSize(g.first);
        mixSize(g.second);
    };

    mixPipeline(mMain.pipeline);
    mixPipeline(mDequant.pipeline);
    mixPipeline(mPack.pipeline);
    mixPipeline(mReducer.pipeline);
    mixGrid(mMain.threads);
    mixGrid(mDequant.threads);
    mixGrid(mReducer.threads);
    mix((uint64_t)mPrepared);
    mix((uint64_t)mPrefillPlan.kernel);
    mix((uint64_t)mPrefillPlan.tileN);
    mix((uint64_t)mPrefillPlan.weightPreparation());
    mix((uint64_t)mPrefillPlan.kPartitions);
    mix((uint64_t)mPrefillPlan.fusion);
    mix((uint64_t)mPrefillPlan.doubleBuf);
    mix((uint64_t)mPrefillPlan.reduceRowsPerPiece);
    mix((uint64_t)mPrefillPlan.pieceCount);
    mix((uint64_t)mPrefillPlan.protectPieceTail);
    mix((uint64_t)mPrefillPlan.loopK64);
    mix((uint64_t)mScratchGeneration);
    mixBuffer(mConstBuffer);
    mixPtr(mFusedRowScale);
    // The device-tensor pack grid comes from the input size, not from mMain.threads.
    mix(inputs.empty() ? 0ull : (uint64_t)inputs[0]->elementSize());

    mixPtr(mPrefillSilu.gateOffer);
    mixPtr(mPrefillSilu.dualPeerOffer);
    mixBuffer(mPrefillSilu.dualCoef);

    return h;
}

bool MetalConv1x1WQuant::onReplayUpdate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return mPrepared && replayStructure(inputs) == mRecordedStructure;
}

#if MNN_METAL_OP_PROFILE
MetalConv1x1WQuant::RouteFacts MetalConv1x1WQuant::routeFacts() const {
    auto bytes = [](const std::shared_ptr<Tensor>& t) { return nullptr == t ? (size_t)0 : t->elementSize(); };
    RouteFacts f;
    f.dequantBits = mDequantBits;
    f.blockCount = mBlockCount;
    f.scratchGeneration = mScratchGeneration;
    f.hasDequantMetadata = nullptr != mDequantScaleBias;
    f.prepared = mPrepared;
    f.kernel = mPrefillPlan.kernel;
    f.weightPreparation = mPrefillPlan.weightPreparation();
    f.usesTensorApi = mPrefillPlan.usesTensorApi();
    f.fusion = mPrefillPlan.fusion;
    f.tempWeightBytes = mPrepared && f.weightPreparation != WeightPreparation::None ? bytes(mTempWeight) : 0;
    f.packedInputBytes = mPrepared && f.weightPreparation == WeightPreparation::RowDevice ? bytes(mPackedInput) : 0;
    f.ksplitPartialBytes = mPrepared && mPrefillPlan.kPartitions > 1 ? bytes(mKsplitPartial) : 0;
    f.packPipeline = mPack.pipeline;
    f.packThreads = mPack.threads;
    f.ksplitReducePipeline = mReducer.pipeline;
    f.ksplitReduceThreads = mReducer.threads;
    f.profileName = mDeviceProfile.name;
    f.profileSource = (int)mDeviceProfile.source;
    f.profileTgBudget = mDeviceProfile.tgBudget;
    return f;
}
#endif

void MetalConv1x1WQuant::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    if (!mPrepared) {
        mEncodeStatus = NOT_SUPPORT;
        return;
    }
    mRecordedStructure = replayStructure(inputs);
#if MNN_METAL_OP_PROFILE
    {
        std::string subtag = mProfileTag;
        if (mPrefillPlan.usesGemmBindings()) {
            subtag = "outdeq+" + subtag;
        }
        static_cast<MetalBackend *>(backend())->setProfileSubtag(subtag);
    }
#endif

    auto input = inputs[0];
    auto output = outputs[0];
    auto bindRowScale = [&]() {
        if (mFusedRowScale != nullptr) {
            MetalBackend::setTensor(mFusedRowScale, encoder, 24);
        }
    };

    if (mPrefillPlan.kernel != PrefillKernel::Unsupported) {
        const auto weightPreparation = mPrefillPlan.weightPreparation();
        const bool tensorInKernel = mPrefillPlan.usesTensorApi() && mPrefillPlan.dequantInKernel();

#if MNN_METAL_OP_PROFILE
        // Split the two sub-passes into independent command buffers so each shows up
        // as its own profile row.
        if (weightPreparation != WeightPreparation::None) {
            static_cast<MetalBackend*>(backend())->setProfileSubtag("outdeq_wdq");
        }
#endif
        // pre dequant weight pipeline (legacy outer-dequant path)
        if (weightPreparation != WeightPreparation::None) {
            [encoder setComputePipelineState:mDequant.pipeline];
            MetalBackend::setTensor(mWeight.get(), encoder, 0);
            MetalBackend::setTensor(mTempWeight.get(), encoder, 1);
            [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
            MetalBackend::setTensor(getDequantScale().get(), encoder, 3);
            [encoder dispatchThreadgroups:mDequant.threads.first threadsPerThreadgroup:mDequant.threads.second];
#if MNN_METAL_OP_PROFILE
            {
                auto* mtbn = static_cast<MetalBackend*>(backend());
                encoder = mtbn->profileNextSubpass(std::string("outdeq_gemm_") + mProfileTag);
            }
#endif
        }
#if MNN_METAL_OP_PROFILE
        if (tensorInKernel) {
            static_cast<MetalBackend*>(backend())->setProfileSubtag(std::string("fused_gemm_") + mProfileTag);
        }
#endif
        if (weightPreparation == WeightPreparation::RowDevice) {
            [encoder setComputePipelineState:mPack.pipeline];
            MetalBackend::setTensor(input, encoder, 0);
            MetalBackend::setTensor(mPackedInput.get(), encoder, 1);
            [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
            [encoder dispatchThreadgroups:mPack.threads.first threadsPerThreadgroup:mPack.threads.second];
        }
        // convolution pipeline
        {
            [encoder setComputePipelineState:mMain.pipeline];
            MetalBackend::setTensor(input, encoder, 0);
            MetalBackend::setTensor(output, encoder, 1);
            [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
            if (weightPreparation == WeightPreparation::RowDevice) {
                MetalBackend::setTensor(mPackedInput.get(), encoder, 0);
            }
            if (mPrefillPlan.kPartitions > 1) {
                // Two-pass: the ksplit GEMM writes fp32 partials, the reduce pass sums
                // them into the real output.
                MetalBackend::setTensor(mKsplitPartial.get(), encoder, 1);
                MetalBackend::setTensor(mWeight.get(), encoder, 3);
                MetalBackend::setTensor(mBias.get(), encoder, 4);
                MetalBackend::setTensor(mDequantScaleBias.get(), encoder, 5);
                MetalBackend::setTensor(mWeight.get(), encoder, 6);
                [encoder dispatchThreadgroups:mMain.threads.first threadsPerThreadgroup:mMain.threads.second];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [encoder setComputePipelineState:mReducer.pipeline];
                MetalBackend::setTensor(mKsplitPartial.get(), encoder, 0);
                MetalBackend::setTensor(output, encoder, 1);
                MetalBackend::setTensor(mBias.get(), encoder, 2);
                [encoder setBuffer:mConstBuffer offset:0 atIndex:3];
                [encoder dispatchThreadgroups:mReducer.threads.first threadsPerThreadgroup:mReducer.threads.second];
                return;
            }
            MetalBackend::setTensor(weightPreparation == WeightPreparation::None ? mWeight.get() : mTempWeight.get(),
                                    encoder, 3);
            MetalBackend::setTensor(mBias.get(), encoder, 4);
            MetalBackend::setTensor(getDequantScale().get(), encoder, 5);
            if (tensorInKernel) {
                // Only Tensor in-kernel GEMM declares the unused weight alias at buffer(6).
                MetalBackend::setTensor(mWeight.get(), encoder, 6);
                if (mPrefillPlan.fusion == PrefillFusion::GateUpDual) {
                    // Peer (gate) operands mirror 3/4/5 at 7/8/9, with its own coefficient at 10.
                    MetalBackend::setTensor(mPrefillSilu.dualPeerOffer->getWeight().get(), encoder, 7);
                    MetalBackend::setTensor(mPrefillSilu.dualPeerOffer->getBias().get(), encoder, 8);
                    MetalBackend::setTensor(mPrefillSilu.dualPeerOffer->getDequantScale().get(), encoder, 9);
                    [encoder setBuffer:mPrefillSilu.dualCoef offset:0 atIndex:10];
                } else if (mPrefillPlan.fusion == PrefillFusion::SiluMul) {
                    MetalBackend::setTensor(mPrefillSilu.gateOffer, encoder, 7);
                }
            }
            bindRowScale();
            [encoder dispatchThreadgroups:mMain.threads.first threadsPerThreadgroup:mMain.threads.second];
        }
    } else {
        [encoder setComputePipelineState:mMain.pipeline];
        MetalBackend::setTensor(input, encoder, 0);
        MetalBackend::setTensor(output, encoder, 1);
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        MetalBackend::setTensor(mWeight.get(), encoder, 3);
        MetalBackend::setTensor(mBias.get(), encoder, 4);
        MetalBackend::setTensor(getDequantScale().get(), encoder, 5);
        bindRowScale();
        [encoder dispatchThreadgroups:mMain.threads.first threadsPerThreadgroup:mMain.threads.second];
    }
}
} // namespace MNN
#endif /* MNN_METAL_ENABLED && MNN_LOW_MEMORY */