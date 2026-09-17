//
//  FusedProjBufExecution.cpp
//  MNN
//
//  OpenCL (buffer mode) execution for the export-time fused projection op
//  (OpType_FusedLinear). Both flavours land here: act_silu_mul (gate/up, two
//  convs joined by MUL_SILU) and QKV (three or four convs writing straight to
//  the group outputs).
//
//  Why a container at all: decode dispatches up to five kernels per group
//  (binary add, layernorm, the projection GEMVs, MUL_SILU) and rounds the
//  intermediates through DRAM. Keeping the op whole is what lets a later change
//  collapse those.
//
//  At decode the container collapses the projections into one GEMV dispatch
//  (fused_proj_gemv_buf.cl) that reads the member convs' packed weights in
//  place: gate/up additionally keep the MUL_SILU epilogue in registers, so
//  neither projection is written to DRAM. See _fusedGemvUsable for the envelope.
//
//  Outside it — prefill, non-int4 members, channel counts with leaves — the
//  container drives the member child executions one by one, which is
//  byte-for-byte the work the geometry decomposition would have emitted. That
//  also has to stay the permanent fallback: an OpenCL execution cannot decline
//  at onResize, since backend selection already happened back at onCreate.
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "backend/opencl/execution/buffer/FusedProjBufExecution.hpp"
#include "core/FusedProjCommon.hpp"
#ifdef MNN_LOW_MEMORY
#include "backend/opencl/execution/buffer/ConvBufLowMemoryExecution.hpp"
#endif

namespace MNN {
namespace OpenCL {

// One dispatch spans members of different output widths; 64 measured best on Adreno.
static int _fusedGemvWgs(uint32_t maxWorkGroupSize) {
    int wgs = 64;
    while (wgs > 8 && (uint32_t)wgs > maxWorkGroupSize) {
        wgs /= 2;
    }
    return wgs;
}

static std::shared_ptr<Tensor> _makeLike(const Tensor *like, int channel) {
    auto shape = like->shape();
    if (shape.size() >= 2) {
        shape[1] = channel;
    }
    std::shared_ptr<Tensor> t(Tensor::createDevice(shape, like->getType(), like->getDimensionType()));
    TensorUtils::getDescribe(t.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
    return t;
}

FusedProjBufExecution::FusedProjBufExecution(const std::vector<Tensor *> &inputs,
                                             const std::vector<Tensor *> &outputs, const MNN::Op *op,
                                             Backend *backend)
    : CommonExecution(backend, op) {
    mParam      = op->main_as_FusedLinearParam();
    mIsGateUp   = mParam->act_silu_mul();
    mHasLn      = mParam->has_ln() && mParam->ln() != nullptr;
    mNumConvs   = (int)mParam->convs()->size();
    mNumProjOut = mIsGateUp ? 1 : mNumConvs;
    mSubOps.reset(new FusedProjSubOps);
    const auto fmt = op->defaultDimentionFormat();
    mSubOps->convs.resize(mNumConvs);
    for (int i = 0; i < mNumConvs; ++i) {
        mSubOps->convs[i] =
            FusedProjCommon::makeConvOp(mParam->convs()->GetAs<Convolution2D>(i), fmt, op->externalPath());
    }
    if (mIsGateUp) {
        mSubOps->mulSilu = FusedProjCommon::makeMulSiluOp(fmt);
    }
    if (mHasLn) {
        mSubOps->layerNorm = FusedProjCommon::makeLayerNormOp(mParam->ln(), fmt);
    }
    if (!_createConvs(backend)) {
        mValid = false;
    }
}

FusedProjBufExecution::FusedProjBufExecution(std::shared_ptr<FusedProjSubOps> subOps, const MNN::Op *op,
                                             Backend *backend)
    : CommonExecution(backend, op) {
    mParam      = op->main_as_FusedLinearParam();
    mIsGateUp   = mParam->act_silu_mul();
    mHasLn      = mParam->has_ln() && mParam->ln() != nullptr;
    mNumConvs   = (int)mParam->convs()->size();
    mNumProjOut = mIsGateUp ? 1 : mNumConvs;
    mSubOps     = subOps;
}

// Create the member convs — and thus load the folded weights — before the first
// onResize. Module::clone shares weights through each child's own onClone, so
// lazily created children would leave nothing to share and every cloned session
// would load a second full copy of every folded weight.
bool FusedProjBufExecution::_createConvs(Backend *backend) {
    mConvs.resize(mNumConvs);
    for (int i = 0; i < mNumConvs; ++i) {
        auto conv = mParam->convs()->GetAs<Convolution2D>(i);
        // The conv creator inspects the tensors for dispatch selection; feed
        // shaped dummies (weights come from the op, not the tensors).
        std::shared_ptr<Tensor> dummyIn(
            Tensor::createDevice<float>({1, conv->common()->inputCount(), 1, 1}));
        std::shared_ptr<Tensor> dummyOut(
            Tensor::createDevice<float>({1, conv->common()->outputCount(), 1, 1}));
        TensorUtils::getDescribe(dummyIn.get())->dimensionFormat  = MNN_DATA_FORMAT_NC4HW4;
        TensorUtils::getDescribe(dummyOut.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        Execution *exe =
            backend->onCreate({dummyIn.get()}, {dummyOut.get()}, FusedProjCommon::opOf(mSubOps->convs[i]));
        if (exe == nullptr) {
            return false;
        }
        mConvs[i].reset(exe);
    }
    return true;
}

bool FusedProjBufExecution::_createRest(Backend *backend, const std::vector<Tensor *> &inputs,
                                        const std::vector<Tensor *> &outputs) {
    // The fused GEMV applies the SiLU-mul in registers, so no child is needed.
    if (mIsGateUp && !mUseFusedGemv && !mMulSilu) {
        // MUL_SILU: out = in0 * silu(in1), so in0 = up and in1 = gate.
        Execution *exe = backend->onCreate({mUp.get(), mGate.get()}, {outputs[0]},
                                           FusedProjCommon::opOf(mSubOps->mulSilu));
        if (exe == nullptr) {
            return false;
        }
        mMulSilu.reset(exe);
    }
    if (mHasLn && !mLn) {
        // Binary RMSNorm: in [residual, hidden], out [residual_out, normalized].
        Execution *exe = backend->onCreate({inputs[0], inputs[1]}, {outputs[mNumProjOut], mNormalized.get()},
                                           FusedProjCommon::opOf(mSubOps->layerNorm));
        if (exe == nullptr) {
            return false;
        }
        mLn.reset(exe);
    }
    return true;
}

bool FusedProjBufExecution::onClone(Backend *bn, const Op *op, Execution **dst) {
    if (!mValid) {
        return false;
    }
    if ((int)mConvs.size() != mNumConvs) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    // Share the member conv weights through each child's own onClone, exactly
    // as a graph-level conv would; mMulSilu / mLn carry no bulk weights and are
    // re-created lazily on the clone's first onResize.
    std::unique_ptr<FusedProjBufExecution> clone(new FusedProjBufExecution(mSubOps, op, bn));
    clone->mConvs.resize(mNumConvs);
    for (int i = 0; i < mNumConvs; ++i) {
        Execution *childClone = nullptr;
        if (!mConvs[i]->onClone(bn, FusedProjCommon::opOf(mSubOps->convs[i]), &childClone) ||
            nullptr == childClone) {
            return false;
        }
        clone->mConvs[i].reset(childClone);
    }
    *dst = clone.release();
    return true;
}

// One dispatch for the whole group needs every member to be an int4 conv1x1 the
// member kernel would have run as a GEMV, with no channel leaves, and all of
// them packed the same way (the fused kernel binds one set of build options).
// Anything else keeps the per-member children.
bool FusedProjBufExecution::_fusedGemvUsable(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mConvRes.clear();
#ifndef MNN_LOW_MEMORY
    return false;
#else
    if ((int)mConvs.size() != mNumConvs) {
        return false;
    }
    // The fused GEMV kernel uses compile-time #if NUM_CONV>N guards for members
    // 2..4; anything outside that range needs a kernel change.
    if (mNumConvs < 2 || mNumConvs > 4) {
        return false;
    }
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    // The children inherit from ConvBufCommonExecution (which exposes
    // getConvResource()) exactly when ConvolutionBufCreator took its low-memory
    // branch: memory mode Low, an int-quantized non-scaleInt weight, one input.
    if (openCLBackend->getMemory() != BackendConfig::Memory_Low) {
        return false;
    }
    Tensor* hidden = mHasLn ? inputs[1] : inputs[0];
    auto hiddenShape = tensorShapeFormat(hidden);
    // Decode only: the kernel walks input and output as one flat channel vector.
    if (hiddenShape[0] * hiddenShape[1] * hiddenShape[2] != 1) {
        return false;
    }
    for (int i = 0; i < mNumConvs; ++i) {
        auto quan = mParam->convs()->GetAs<Convolution2D>(i)->quanParameter();
        if (nullptr == quan || quan->has_scaleInt()) {
            return false;
        }
        if (1 != quan->type() && 2 != quan->type() && 4 != quan->type()) {
            return false;
        }
        // Safe: the memory-mode-Low guard above guarantees ConvolutionBufCreator
        // produced ConvBufLowMemoryExecution instances; getConvResource() is
        // inherited from ConvBufCommonExecution.
        auto res = static_cast<ConvBufLowMemoryExecution*>(mConvs[i].get())->getConvResource();
        if (nullptr == res || !res->mConv1x1Opt || 4 != res->mNumQuantBit) {
            return false;
        }
        if (res->mRelu || res->mRelu6 || res->mPrelu) {
            return false;
        }
        if (nullptr == res->mDequantScaleOffsetBuffer || nullptr == res->mBias) {
            return false;
        }
        if (nullptr == (res->mUseImage ? (void*)res->mKernelImage.get() : (void*)res->mKernelBuffer.get())) {
            return false;
        }
        // No OUTPUT_CHANNEL_LEAVES / INPUT_CHANNEL_LEAVES handling in the kernel.
        if (0 != (res->mOutputChannel % 8) || 0 != (res->mInputChannel % 4)) {
            return false;
        }
        if (res->mBlockSize <= 0 || 0 != (res->mInputChannel % res->mBlockSize)) {
            return false;
        }
        if (res->mInputChannel != hiddenShape[3]) {
            return false;
        }
        if (i > 0) {
            const auto& first = mConvRes[0];
            if (res->mInputChannel != first->mInputChannel || res->mBlockSize != first->mBlockSize ||
                res->mUseImage != first->mUseImage ||
                res->mBuildOptions.count("-DASYMMETRIC") != first->mBuildOptions.count("-DASYMMETRIC")) {
                return false;
            }
        }
        mConvRes.emplace_back(res);
    }
    // gate/up share the tile, so they must share the output width too.
    if (mIsGateUp && mConvRes[0]->mOutputChannel != mConvRes[1]->mOutputChannel) {
        return false;
    }
    return true;
#endif
}

ErrorCode FusedProjBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifndef MNN_LOW_MEMORY
    return NOT_SUPPORT;
#else
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime = openCLBackend->getOpenCLRuntime();
    Tensor* projInput = mHasLn ? mNormalized.get() : inputs[0];
    const auto& first = mConvRes[0];
    const int srcChannelC4 = UP_DIV(first->mInputChannel, 4);

    // Kernel parameters are passed as cl_int4 / cl_float4, so the arrays are
    // fixed at 4 — the maximum NUM_CONV the kernel supports (guarded by the
    // mNumConvs range check in _fusedGemvUsable).  Unused slots keep their
    // identity-safe defaults (0 tiles / 1.0 coef).
    static constexpr int kMaxConvs = 4;
    int totalTiles = 0;
    int ocTiles[kMaxConvs] = {0, 0, 0, 0};
    int dstChannelC4[kMaxConvs] = {1, 1, 1, 1};
    float coef[kMaxConvs] = {1.0f, 1.0f, 1.0f, 1.0f};
    for (int i = 0; i < mNumConvs; ++i) {
        ocTiles[i] = UP_DIV(mConvRes[i]->mOutputChannel, 8);
        dstChannelC4[i] = UP_DIV(mConvRes[i]->mOutputChannel, 4);
        coef[i] = mConvRes[i]->mCoef;
        totalTiles += ocTiles[i];
    }
    // gate/up: both members share one tile, and the SiLU-mul happens in registers.
    if (mIsGateUp) {
        totalTiles = ocTiles[0];
    }

    std::set<std::string> buildOptions;
    const int wgs = _fusedGemvWgs(std::min(runtime->getMaxWorkItemSizes()[0], (uint32_t)runtime->MaxWorkGroupSize()));
    buildOptions.emplace("-DWGS=" + std::to_string(wgs));
    buildOptions.emplace("-DNUM_CONV=" + std::to_string(mNumConvs));
    if (mIsGateUp) {
        buildOptions.emplace("-DFUSE_SILU_MUL");
    }
    if (first->mBuildOptions.count("-DASYMMETRIC") > 0) {
        buildOptions.emplace("-DASYMMETRIC");
    }
    if (first->mUseImage) {
        buildOptions.emplace("-DUSE_IMAGE");
    }

    mUnits.resize(1);
    auto& unit = mUnits[0];
    unit.kernel =
        runtime->buildKernel("fused_proj_gemv_buf", "fused_proj_gemv_buf", buildOptions, openCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL(unit.kernel);
    mFusedGws = {(uint32_t)wgs, (uint32_t)totalTiles};
    mFusedLws = {(uint32_t)wgs, 1};

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mFusedGws[0]);
    ret |= unit.kernel->get().setArg(idx++, mFusedGws[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(projInput));
    for (int i = 0; i < mNumConvs; ++i) {
        if (mConvRes[i]->mUseImage) {
            ret |= unit.kernel->get().setArg(idx++, *mConvRes[i]->mKernelImage.get());
        } else {
            ret |= unit.kernel->get().setArg(idx++, *mConvRes[i]->mKernelBuffer.get());
        }
    }
    for (int i = 0; i < mNumConvs; ++i) {
        ret |= unit.kernel->get().setArg(idx++, *mConvRes[i]->mDequantScaleOffsetBuffer.get());
    }
    for (int i = 0; i < mNumConvs; ++i) {
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvRes[i]->mBias.get()));
    }
    // gate/up writes the single group output; the QKV flavour writes one per member.
    const int numOut = mIsGateUp ? 1 : mNumConvs;
    for (int i = 0; i < numOut; ++i) {
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(outputs[i]));
    }
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(srcChannelC4));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(first->mInputChannel / first->mBlockSize));
    {
        cl_int4 dst = {dstChannelC4[0], dstChannelC4[1], dstChannelC4[2], dstChannelC4[3]};
        cl_int4 tiles = {ocTiles[0], ocTiles[1], ocTiles[2], ocTiles[3]};
        cl_float4 cf = {coef[0], coef[1], coef[2], coef[3]};
        ret |= unit.kernel->get().setArg(idx++, dst);
        ret |= unit.kernel->get().setArg(idx++, tiles);
        ret |= unit.kernel->get().setArg(idx++, cf);
    }
    MNN_CHECK_CL_SUCCESS(ret, "setArg fused_proj_gemv_buf");

    openCLBackend->recordKernel2d(unit.kernel, mFusedGws, mFusedLws);
    unit.globalWorkSize = {mFusedGws[0], mFusedGws[1]};
    unit.localWorkSize = {mFusedLws[0], mFusedLws[1]};
    return NO_ERROR;
#endif
}

ErrorCode FusedProjBufExecution::onResize(const std::vector<Tensor *> &inputs,
                                          const std::vector<Tensor *> &outputs) {
    auto openCLBackend = static_cast<OpenCLBackend *>(backend());
    Tensor *hidden = mHasLn ? inputs[1] : inputs[0];
    // Backend::DYNAMIC, not DYNAMIC_IN_EXECUTION: the latter parks an
    // OpenCLBufferNode* in deviceId, which only openCLDeferBuffer can read, and
    // every child execution here reaches for openCLBuffer.
    mUseFusedGemv = _fusedGemvUsable(inputs, outputs);
    if (mHasLn) {
        mNormalized = _makeLike(hidden, hidden->length(1));
        OPENCL_CHECK_ALLOC(openCLBackend->onAcquireBuffer(mNormalized.get(), Backend::DYNAMIC));
    }
    if (mIsGateUp && !mUseFusedGemv) {
        // QKV convs write straight to the group outputs; only the gate/up
        // flavour needs the two projection results staged for MUL_SILU.
        const int oc = outputs[0]->length(1);
        mGate = _makeLike(hidden, oc);
        mUp   = _makeLike(hidden, oc);
        OPENCL_CHECK_ALLOC(openCLBackend->onAcquireBuffer(mGate.get(), Backend::DYNAMIC));
        OPENCL_CHECK_ALLOC(openCLBackend->onAcquireBuffer(mUp.get(), Backend::DYNAMIC));
    }
    if (!_createRest(openCLBackend, inputs, outputs)) {
        MNN_ERROR("FusedProjBufExecution: failed to create sub-executions\n");
        return NOT_SUPPORT;
    }
    ErrorCode err;
    if (mUseFusedGemv) {
        // The binary RMSNorm prologue stays a child; the projections collapse
        // into the single GEMV that onEncode records.
        err = NO_ERROR;
        if (mHasLn) {
            err = mLn->onResize({inputs[0], inputs[1]}, {outputs[mNumProjOut], mNormalized.get()});
        }
        if (NO_ERROR == err) {
            err = CommonExecution::onResize(inputs, outputs);
        }
    } else {
        err = _resize(inputs, outputs);
    }
    if (mGate) {
        openCLBackend->onReleaseBuffer(mGate.get(), Backend::DYNAMIC);
        openCLBackend->onReleaseBuffer(mUp.get(), Backend::DYNAMIC);
    }
    if (mNormalized) {
        openCLBackend->onReleaseBuffer(mNormalized.get(), Backend::DYNAMIC);
    }
    return err;
}

// Run the member ops as separate dispatches — the unfused graph, driven from
// here instead of from the geometry decomposition.
ErrorCode FusedProjBufExecution::_resize(const std::vector<Tensor *> &inputs,
                                         const std::vector<Tensor *> &outputs) {
    Tensor *projInput = inputs[0];
    if (mHasLn) {
        auto err = mLn->onResize({inputs[0], inputs[1]}, {outputs[mNumProjOut], mNormalized.get()});
        if (err != NO_ERROR) {
            return err;
        }
        projInput = mNormalized.get();
    }
    if (!mIsGateUp) {
        for (int i = 0; i < mNumConvs; ++i) {
            auto err = mConvs[i]->onResize({projInput}, {outputs[i]});
            if (err != NO_ERROR) {
                return err;
            }
        }
        return NO_ERROR;
    }
    auto err = mConvs[0]->onResize({projInput}, {mGate.get()});
    if (err != NO_ERROR) {
        return err;
    }
    err = mConvs[1]->onResize({projInput}, {mUp.get()});
    if (err != NO_ERROR) {
        return err;
    }
    return mMulSilu->onResize({mUp.get(), mGate.get()}, {outputs[0]});
}

ErrorCode FusedProjBufExecution::onExecute(const std::vector<Tensor *> &inputs,
                                           const std::vector<Tensor *> &outputs) {
    // In-order queue, so the members' data dependencies need no extra barrier.
    if (mUseFusedGemv) {
        if (mHasLn) {
            auto err = mLn->onExecute({inputs[0], inputs[1]}, {outputs[mNumProjOut], mNormalized.get()});
            if (err != NO_ERROR) {
                return err;
            }
        }
        return CommonExecution::onExecute(inputs, outputs);
    }
    Tensor *projInput = inputs[0];
    if (mHasLn) {
        auto err = mLn->onExecute({inputs[0], inputs[1]}, {outputs[mNumProjOut], mNormalized.get()});
        if (err != NO_ERROR) {
            return err;
        }
        projInput = mNormalized.get();
    }
    if (!mIsGateUp) {
        for (int i = 0; i < mNumConvs; ++i) {
            auto err = mConvs[i]->onExecute({projInput}, {outputs[i]});
            if (err != NO_ERROR) {
                return err;
            }
        }
        return NO_ERROR;
    }
    auto err = mConvs[0]->onExecute({projInput}, {mGate.get()});
    if (err != NO_ERROR) {
        return err;
    }
    err = mConvs[1]->onExecute({projInput}, {mUp.get()});
    if (err != NO_ERROR) {
        return err;
    }
    return mMulSilu->onExecute({mUp.get(), mGate.get()}, {outputs[0]});
}

class FusedProjBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~FusedProjBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (FusedProjCommon::openCLDisabled()) {
            return nullptr;
        }
        // Must match GeometryFusedProj::_keepWhole exactly: an op the geometry
        // keeps whole but this refuses would fail session creation.
        if (!FusedProjCommon::nativeEnvelopeOk(op, inputs.size(), outputs.size())) {
            return nullptr;
        }
        // The member conv executions would have set this on the real tensors in
        // the decomposed graph; keep the packing decision identical.
        for (auto t : inputs) {
            TensorUtils::setTensorSupportPack(t, false);
        }
        for (auto t : outputs) {
            TensorUtils::setTensorSupportPack(t, false);
        }
        OPENCL_CREATOR_CHECK(new FusedProjBufExecution(inputs, outputs, op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(FusedProjBufCreator, OpType_FusedLinear, BUFFER);

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
