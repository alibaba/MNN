//
//  FusedProjExecution.cu
//  MNN
//
//  CUDA composite execution for the fused projection op (OpType_FusedLinear).
//  Both flavours land here: act_silu_mul (gate/up, two convs joined by
//  MUL_SILU) and QKV (three or four convs writing straight to the group
//  outputs). The children are created through the regular per-op creators, so
//  the arithmetic is identical to the geometry decomposition; the single
//  in-order stream serializes their data dependencies.
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "FusedProjExecution.hpp"
#include "core/FusedProjCommon.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace CUDA {

static std::shared_ptr<Tensor> _makeLike(const Tensor* like, int channel) {
    auto shape = like->shape();
    if (shape.size() >= 2) {
        shape[1] = channel;
    }
    std::shared_ptr<Tensor> t(Tensor::createDevice(shape, like->getType(), like->getDimensionType()));
    TensorUtils::getDescribe(t.get())->dimensionFormat = TensorUtils::getDescribe(like)->dimensionFormat;
    return t;
}

FusedProjExecution::FusedProjExecution(const MNN::Op* op, Backend* backend) : Execution(backend) {
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

FusedProjExecution::FusedProjExecution(std::shared_ptr<FusedProjSubOps> subOps, const MNN::Op* op, Backend* backend)
    : Execution(backend) {
    mParam      = op->main_as_FusedLinearParam();
    mIsGateUp   = mParam->act_silu_mul();
    mHasLn      = mParam->has_ln() && mParam->ln() != nullptr;
    mNumConvs   = (int)mParam->convs()->size();
    mNumProjOut = mIsGateUp ? 1 : mNumConvs;
    mSubOps     = subOps;
}

// Create the member convs — and thus load the folded weights — before the first
// onResize, so clones can share them through each child's own onClone.
bool FusedProjExecution::_createConvs(Backend* backend) {
    mConvs.resize(mNumConvs);
    for (int i = 0; i < mNumConvs; ++i) {
        auto conv = mParam->convs()->GetAs<Convolution2D>(i);
        // The conv creator inspects the tensors for dispatch selection; feed
        // shaped dummies (weights come from the op, not the tensors).
        std::shared_ptr<Tensor> dummyIn(Tensor::createDevice<float>({1, conv->common()->inputCount(), 1, 1}));
        std::shared_ptr<Tensor> dummyOut(Tensor::createDevice<float>({1, conv->common()->outputCount(), 1, 1}));
        TensorUtils::getDescribe(dummyIn.get())->dimensionFormat  = MNN_DATA_FORMAT_NC4HW4;
        TensorUtils::getDescribe(dummyOut.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        Execution* exe =
            backend->onCreate({dummyIn.get()}, {dummyOut.get()}, FusedProjCommon::opOf(mSubOps->convs[i]));
        if (exe == nullptr) {
            return false;
        }
        mConvs[i].reset(exe);
    }
    return true;
}

bool FusedProjExecution::_createRest(Backend* backend, const std::vector<Tensor*>& inputs,
                                     const std::vector<Tensor*>& outputs) {
    if (mIsGateUp && !mMulSilu) {
        // MUL_SILU: out = in0 * silu(in1), so in0 = up and in1 = gate.
        Execution* exe =
            backend->onCreate({mUp.get(), mGate.get()}, {outputs[0]}, FusedProjCommon::opOf(mSubOps->mulSilu));
        if (exe == nullptr) {
            return false;
        }
        mMulSilu.reset(exe);
    }
    if (mHasLn && !mLn) {
        // Binary RMSNorm: in [residual, hidden], out [residual_out, normalized].
        Execution* exe = backend->onCreate({inputs[0], inputs[1]}, {outputs[mNumProjOut], mNormalized.get()},
                                           FusedProjCommon::opOf(mSubOps->layerNorm));
        if (exe == nullptr) {
            return false;
        }
        mLn.reset(exe);
    }
    return true;
}

bool FusedProjExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid || (int)mConvs.size() != mNumConvs) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    // Share the member conv weights through each child's own onClone, exactly
    // as a graph-level conv would; mMulSilu / mLn carry no bulk weights and are
    // re-created lazily on the clone's first onResize.
    std::unique_ptr<FusedProjExecution> clone(new FusedProjExecution(mSubOps, op, bn));
    clone->mConvs.resize(mNumConvs);
    for (int i = 0; i < mNumConvs; ++i) {
        Execution* childClone = nullptr;
        if (!mConvs[i]->onClone(bn, FusedProjCommon::opOf(mSubOps->convs[i]), &childClone) ||
            nullptr == childClone) {
            return false;
        }
        clone->mConvs[i].reset(childClone);
    }
    *dst = clone.release();
    return true;
}

ErrorCode FusedProjExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto bn        = backend();
    Tensor* hidden = mHasLn ? inputs[1] : inputs[0];
    if (mHasLn) {
        mNormalized = _makeLike(hidden, hidden->length(1));
        if (!bn->onAcquireBuffer(mNormalized.get(), Backend::DYNAMIC)) {
            return OUT_OF_MEMORY;
        }
    }
    if (mIsGateUp) {
        // QKV convs write straight to the group outputs; only the gate/up
        // flavour needs the two projection results staged for MUL_SILU.
        const int oc = outputs[0]->length(1);
        mGate        = _makeLike(hidden, oc);
        mUp          = _makeLike(hidden, oc);
        if (!bn->onAcquireBuffer(mGate.get(), Backend::DYNAMIC) ||
            !bn->onAcquireBuffer(mUp.get(), Backend::DYNAMIC)) {
            return OUT_OF_MEMORY;
        }
    }
    if (!_createRest(bn, inputs, outputs)) {
        MNN_ERROR("FusedProjExecution: failed to create sub-executions\n");
        return NOT_SUPPORT;
    }
    ErrorCode err = _resize(inputs, outputs);
    if (mGate) {
        bn->onReleaseBuffer(mGate.get(), Backend::DYNAMIC);
        bn->onReleaseBuffer(mUp.get(), Backend::DYNAMIC);
    }
    if (mNormalized) {
        bn->onReleaseBuffer(mNormalized.get(), Backend::DYNAMIC);
    }
    return err;
}

ErrorCode FusedProjExecution::_resize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    Tensor* projInput = inputs[0];
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

ErrorCode FusedProjExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Single in-order stream, so the members' data dependencies need no extra
    // synchronization.
    Tensor* projInput = inputs[0];
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

class FusedProjCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        // Must match GeometryFusedProj::_keepWhole exactly: an op the geometry
        // keeps whole but this refuses would fail session creation.
        if (!FusedProjCommon::compositeEnvelopeOk(op, inputs.size(), outputs.size())) {
            return nullptr;
        }
        auto exe = new FusedProjExecution(op, backend);
        if (!exe->valid()) {
            delete exe;
            return nullptr;
        }
        return exe;
    }
};

static CUDACreatorRegister<FusedProjCreator> __FusedProjExecution(OpType_FusedLinear);

} // namespace CUDA
} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
