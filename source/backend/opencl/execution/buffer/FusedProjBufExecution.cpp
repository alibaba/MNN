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
//  Today the container drives the member child executions one by one, which is
//  byte-for-byte the work the geometry decomposition would have emitted. That
//  also has to stay the permanent fallback: an OpenCL execution cannot decline
//  at onResize, since backend selection already happened back at onCreate.
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "backend/opencl/execution/buffer/FusedProjBufExecution.hpp"
#include "core/FusedProjCommon.hpp"

namespace MNN {
namespace OpenCL {

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
    if (mIsGateUp && !mMulSilu) {
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

ErrorCode FusedProjBufExecution::onResize(const std::vector<Tensor *> &inputs,
                                          const std::vector<Tensor *> &outputs) {
    auto openCLBackend = static_cast<OpenCLBackend *>(backend());
    Tensor *hidden = mHasLn ? inputs[1] : inputs[0];
    // Backend::DYNAMIC, not DYNAMIC_IN_EXECUTION: the latter parks an
    // OpenCLBufferNode* in deviceId, which only openCLDeferBuffer can read, and
    // every child execution here reaches for openCLBuffer.
    if (mHasLn) {
        mNormalized = _makeLike(hidden, hidden->length(1));
        OPENCL_CHECK_ALLOC(openCLBackend->onAcquireBuffer(mNormalized.get(), Backend::DYNAMIC));
    }
    if (mIsGateUp) {
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
    ErrorCode err = _resize(inputs, outputs);
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
