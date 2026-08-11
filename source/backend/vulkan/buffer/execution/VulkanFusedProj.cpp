//
//  VulkanFusedProj.cpp
//  MNN
//
//  Vulkan (buffer variant) composite execution for the fused projection op
//  (OpType_FusedLinear). Both flavours land here: act_silu_mul (gate/up, two
//  convs joined by MUL_SILU) and QKV (three or four convs writing straight to
//  the group outputs).
//
//  The member child encoders come from the regular per-op creators
//  (VulkanBackend::getCreator), so the arithmetic is identical to the geometry
//  decomposition; child data dependencies inside this op are serialized with
//  explicit buffer barriers, exactly as the multi-dispatch executions
//  (VulkanAttention etc.) do internally.
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "VulkanFusedProj.hpp"
#include "core/FusedProjCommon.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

static std::shared_ptr<Tensor> _makeLike(const Tensor* like, int channel) {
    auto shape = like->shape();
    if (shape.size() >= 2) {
        shape[1] = channel;
    }
    std::shared_ptr<Tensor> t(Tensor::createDevice(shape, like->getType(), like->getDimensionType()));
    TensorUtils::getDescribe(t.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
    return t;
}

VulkanFusedProj::VulkanFusedProj(const Op* op, Backend* backend) : VulkanBasicExecution(backend) {
    mParam      = op->main_as_FusedLinearParam();
    mIsGateUp   = mParam->act_silu_mul();
    mHasLn      = mParam->has_ln() && mParam->ln() != nullptr;
    mNumConvs   = (int)mParam->convs()->size();
    mNumProjOut = mIsGateUp ? 1 : mNumConvs;
    const auto fmt = op->defaultDimentionFormat();
    mConvOps.resize(mNumConvs);
    for (int i = 0; i < mNumConvs; ++i) {
        mConvOps[i] = FusedProjCommon::makeConvOp(mParam->convs()->GetAs<Convolution2D>(i), fmt, op->externalPath());
    }
    if (mIsGateUp) {
        mMulSiluOp = FusedProjCommon::makeMulSiluOp(fmt);
    }
    if (mHasLn) {
        mLayerNormOp = FusedProjCommon::makeLayerNormOp(mParam->ln(), fmt);
    }
    if (!_createConvs(backend)) {
        mValid = false;
    }
}

// Create the member convs — and thus load the folded weights — up front, so the
// weights are uploaded exactly once per session.
bool VulkanFusedProj::_createConvs(Backend* backend) {
    auto creator = VulkanBackend::getCreator(OpType_Convolution);
    if (creator == nullptr) {
        return false;
    }
    mConvs.resize(mNumConvs);
    for (int i = 0; i < mNumConvs; ++i) {
        auto conv = mParam->convs()->GetAs<Convolution2D>(i);
        // The conv creator inspects the tensors for dispatch selection; feed
        // shaped dummies (weights come from the op, not the tensors).
        std::shared_ptr<Tensor> dummyIn(Tensor::createDevice<float>({1, conv->common()->inputCount(), 1, 1}));
        std::shared_ptr<Tensor> dummyOut(Tensor::createDevice<float>({1, conv->common()->outputCount(), 1, 1}));
        TensorUtils::getDescribe(dummyIn.get())->dimensionFormat  = MNN_DATA_FORMAT_NC4HW4;
        TensorUtils::getDescribe(dummyOut.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        auto exe = creator->onCreate({dummyIn.get()}, {dummyOut.get()}, FusedProjCommon::opOf(mConvOps[i]), backend);
        if (exe == nullptr) {
            return false;
        }
        mConvs[i].reset(exe);
    }
    return true;
}

// The MUL_SILU / binary RMSNorm children need the real tensors, so they are
// created lazily on the first onEncode.
bool VulkanFusedProj::_createRest(Backend* backend, const std::vector<Tensor*>& inputs,
                                  const std::vector<Tensor*>& outputs) {
    if (mIsGateUp && !mMulSilu) {
        auto creator = VulkanBackend::getCreator(OpType_BinaryOp);
        if (creator == nullptr) {
            return false;
        }
        // MUL_SILU: out = in0 * silu(in1), so in0 = up and in1 = gate.
        auto exe = creator->onCreate({mUp.get(), mGate.get()}, {outputs[0]}, FusedProjCommon::opOf(mMulSiluOp),
                                     backend);
        if (exe == nullptr) {
            return false;
        }
        mMulSilu.reset(exe);
    }
    if (mHasLn && !mLn) {
        auto creator = VulkanBackend::getCreator(OpType_LayerNorm);
        if (creator == nullptr) {
            return false;
        }
        // Binary RMSNorm: in [residual, hidden], out [residual_out, normalized].
        auto exe = creator->onCreate({inputs[0], inputs[1]}, {outputs[mNumProjOut], mNormalized.get()},
                                     FusedProjCommon::opOf(mLayerNormOp), backend);
        if (exe == nullptr) {
            return false;
        }
        mLn.reset(exe);
    }
    return true;
}

ErrorCode VulkanFusedProj::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const VulkanCommandPool::Buffer* cmdBuffer) {
    if (!mValid) {
        return NOT_SUPPORT;
    }
    auto vkBn      = static_cast<VulkanBackend*>(backend());
    Tensor* hidden = mHasLn ? inputs[1] : inputs[0];
    // Workspace tensors are re-acquired every onEncode and released at the end
    // of the call: the descriptor sets capture (VkBuffer, offset) during the
    // children's onEncode, and the pool-owned VkBuffer stays alive past the
    // release; barriers below serialize the writes against the reads.
    if (mHasLn) {
        mNormalized = _makeLike(hidden, hidden->length(1));
        if (!vkBn->onAcquireBuffer(mNormalized.get(), Backend::DYNAMIC)) {
            return OUT_OF_MEMORY;
        }
    }
    if (mIsGateUp) {
        // QKV convs write straight to the group outputs; only the gate/up
        // flavour needs the two projection results staged for MUL_SILU.
        const int oc = outputs[0]->length(1);
        mGate        = _makeLike(hidden, oc);
        mUp          = _makeLike(hidden, oc);
        if (!vkBn->onAcquireBuffer(mGate.get(), Backend::DYNAMIC) ||
            !vkBn->onAcquireBuffer(mUp.get(), Backend::DYNAMIC)) {
            return OUT_OF_MEMORY;
        }
    }
    if (!_createRest(vkBn, inputs, outputs)) {
        MNN_ERROR("VulkanFusedProj: failed to create sub-executions\n");
        return NOT_SUPPORT;
    }
    auto barrier = [&](const Tensor* t) {
        auto buf = vkBn->getTensorBuffer(t);
        cmdBuffer->barrierSource(buf.first->buffer(), buf.second, vkBn->getTensorSize(t));
    };

    Tensor* projInput = inputs[0];
    ErrorCode err     = NO_ERROR;
    if (mHasLn) {
        err = mLn->onEncode({inputs[0], inputs[1]}, {outputs[mNumProjOut], mNormalized.get()}, cmdBuffer);
        if (err != NO_ERROR) {
            return err;
        }
        barrier(mNormalized.get());
        projInput = mNormalized.get();
    }
    if (!mIsGateUp) {
        for (int i = 0; i < mNumConvs; ++i) {
            err = mConvs[i]->onEncode({projInput}, {outputs[i]}, cmdBuffer);
            if (err != NO_ERROR) {
                return err;
            }
        }
    } else {
        err = mConvs[0]->onEncode({projInput}, {mGate.get()}, cmdBuffer);
        if (err != NO_ERROR) {
            return err;
        }
        err = mConvs[1]->onEncode({projInput}, {mUp.get()}, cmdBuffer);
        if (err != NO_ERROR) {
            return err;
        }
        barrier(mGate.get());
        barrier(mUp.get());
        err = mMulSilu->onEncode({mUp.get(), mGate.get()}, {outputs[0]}, cmdBuffer);
        if (err != NO_ERROR) {
            return err;
        }
    }
    if (mGate) {
        vkBn->onReleaseBuffer(mGate.get(), Backend::DYNAMIC);
        vkBn->onReleaseBuffer(mUp.get(), Backend::DYNAMIC);
    }
    if (mNormalized) {
        vkBn->onReleaseBuffer(mNormalized.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

class VulkanFusedProjCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           const MNN::Op* op, Backend* backend) const override {
        // Must match GeometryFusedProj::_keepWhole exactly: an op the geometry
        // keeps whole but this refuses would fail session creation.
        if (!FusedProjCommon::compositeEnvelopeOk(op, inputs.size(), outputs.size())) {
            return nullptr;
        }
        auto exe = new VulkanFusedProj(op, backend);
        if (!exe->valid()) {
            delete exe;
            return nullptr;
        }
        return exe;
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_FusedLinear, new VulkanFusedProjCreator);
    return true;
}();

} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
