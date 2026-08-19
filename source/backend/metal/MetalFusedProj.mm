//
//  MetalFusedProj.mm
//  MNN
//
//  Metal execution for the export-time fused projection op (FusedLinear =
//  307): a group of projections sharing one input. The execution owns one
//  Conv1x1 child per param conv, plus a MUL_SILU binary child (act_silu_mul)
//  and a binary-RMSNorm LayerNorm child (has_ln).
//
//  Fusion is decided by the EXPORTED GRAPH, not by runtime pattern matching:
//  setupFusion() drives the children's setupQKVFusion / setupLNFusion directly
//  in member order, so the group is exactly what the exporter emitted. The
//  backend calls it from onResizeEnd because the setup re-homes follower
//  outputs to STATIC, which only sticks after the allocator's compute().
//
//  Decode GEMV is the only shape the fused kernels support (see
//  MetalConvolution1x1's mIs2sgDecode conditions); any other shape, including
//  all of prefill, keeps every child dispatching its own pipeline, which is
//  exactly the unfused graph.
//

#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MetalExecution.hpp"
#import "backend/metal/MetalConvolution1x1.hpp"
#import "backend/metal/MetalLayerNorm.hpp"
#import "core/FusedProjCommon.hpp"
#import "MNN_generated.h"
#import "core/TensorUtils.hpp"
#import "core/Macro.h"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

class MetalFusedProj : public MetalExecution, public MetalBackend::FusedProjFusionHost {
public:
    MetalFusedProj(Backend *backend, const MNN::Op *op, bool eager = true) : MetalExecution(backend) {
        auto param  = op->main_as_FusedLinearParam();
        mIsGateUp   = param->act_silu_mul();
        mHasLn      = param->has_ln() && param->ln() != nullptr;
        mNumConvs   = (int)param->convs()->size();
        mNumProjOut = mIsGateUp ? 1 : mNumConvs;
        mParam      = param;
        mFusedOp    = op;
        if (!eager) {
            return;
        }
        // Create the member convs (and thus load the folded weights) right
        // away: the pre-arrange / Session::clone machinery shares weights via
        // onClone at module-clone time — before the first onResize — so lazy
        // children would leave nothing to share and every cloned session
        // would load a full second copy of every folded weight.
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
            mConvs.emplace_back(static_cast<MetalConvolution1x1 *>(exe));
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
        // Share the member conv weights through each child's own onClone
        // (same as graph-level MetalConvolution1x1), otherwise the cloned
        // session would load a full second copy of every folded weight.
        // mMulSilu / mLn carry no bulk weights and are re-created lazily on
        // the clone's first onResize.
        std::unique_ptr<MetalFusedProj> clone(new MetalFusedProj(bn, op, /*eager=*/false));
        for (int i = 0; i < mNumConvs; ++i) {
            const Op* convOp = clone->_wrapConv(clone->mParam->convs()->GetAs<Convolution2D>(i), mChildFormat);
            Execution* childClone = nullptr;
            if (!mConvs[i]->onClone(bn, convOp, &childClone) || nullptr == childClone ||
                !childClone->valid()) {
                delete childClone;
                return false;
            }
            clone->mConvs.emplace_back(static_cast<MetalConvolution1x1 *>(childClone));
        }
        *dst = clone.release();
        return true;
    }

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto backend = static_cast<MetalBackend *>(this->backend());
        Tensor *hidden = mHasLn ? inputs[1] : inputs[0];
        auto fmt = TensorUtils::getDescribe(hidden)->dimensionFormat;
        mChildFormat = fmt;
        if (mHasLn) {
            if (_ensureIntermediate(mNormalized, hidden, hidden->length(1), fmt)) {
                mResizeGeneration++;
            }
            backend->onAcquireBuffer(mNormalized.get(), Backend::DYNAMIC);
        }
        if (mIsGateUp) {
            int oc = outputs[0]->length(1);
            if (_ensureIntermediate(mGate, hidden, oc, fmt)) {
                mResizeGeneration++;
            }
            if (_ensureIntermediate(mUp, hidden, oc, fmt)) {
                mResizeGeneration++;
            }
            backend->onAcquireBuffer(mGate.get(), Backend::DYNAMIC);
            backend->onAcquireBuffer(mUp.get(), Backend::DYNAMIC);
        }
        Tensor *projInput = mHasLn ? mNormalized.get() : hidden;
        if ((int)mConvs.size() != mNumConvs || (mIsGateUp && !mMulSilu) || (mHasLn && !mLn)) {
            if (!_createChildren(backend, inputs, outputs, projInput, fmt)) {
                MNN_ERROR("MetalFusedProj: failed to create sub-executions\n");
                return NOT_SUPPORT;
            }
        }
        ErrorCode err = NO_ERROR;
        if (mHasLn) {
            err = mLn->onResize({inputs[0], inputs[1]}, {outputs[mNumProjOut], mNormalized.get()});
        }
        if (err == NO_ERROR) {
            if (mIsGateUp) {
                err = mConvs[0]->onResize({projInput}, {mGate.get()});
                if (err == NO_ERROR) {
                    err = mConvs[1]->onResize({projInput}, {mUp.get()});
                }
                if (err == NO_ERROR) {
                    err = mMulSilu->onResize({mUp.get(), mGate.get()}, {outputs[0]});
                }
            } else {
                for (int i = 0; i < mNumConvs && err == NO_ERROR; ++i) {
                    err = mConvs[i]->onResize({projInput}, {outputs[i]});
                }
            }
        }
        if (mNormalized) {
            backend->onReleaseBuffer(mNormalized.get(), Backend::DYNAMIC);
        }
        if (mGate) {
            backend->onReleaseBuffer(mGate.get(), Backend::DYNAMIC);
        }
        if (mUp) {
            backend->onReleaseBuffer(mUp.get(), Backend::DYNAMIC);
        }
        if (err == NO_ERROR) {
            // Remember what setupFusion needs, then ask the backend to call us
            // back after the allocator's compute(). Registered every resize:
            // the backend clears its registry in onResizeBegin.
            mProjOutputs.assign(outputs.begin(), outputs.begin() + mNumProjOut);
            mLnResidualIn  = mHasLn ? inputs[0] : nullptr;
            mLnHiddenIn    = mHasLn ? inputs[1] : nullptr;
            mLnResidualOut = mHasLn ? outputs[mNumProjOut] : nullptr;
            backend->registerFusedProj(this);
        }
        return err;
    }

    // Establish the fused dispatch from the exported member order. Called by
    // MetalBackend::onResizeEnd after the allocator has assigned addresses,
    // which the STATIC re-homes below depend on.
    virtual void setupFusion() override {
        auto backend = static_cast<MetalBackend *>(this->backend());
        // The fused kernels only exist on the decode-GEMV pipeline; anything
        // else (all of prefill included) stays as per-member dispatches.
        for (auto &conv : mConvs) {
            if (!conv->is2sgDecodePipeline()) {
                return;
            }
        }
        // The grouping comes straight from the exported member order:
        // act_silu_mul is (gate, up); otherwise q/k/v plus an optional fourth
        // projection (a gated variant's output gate, or linear attention's
        // qkv/z/b/a).
        bool projFused = false;
        if (mIsGateUp) {
            if (mConvs.size() == 2 && !MetalEnv::get().gateUpFusionDisabled) {
                projFused = mConvs[0]->setupGateUpFusion(mConvs[1].get(), mUp.get());
            }
        } else if (mConvs.size() >= 3 && !MetalEnv::get().qkvFusionDisabled) {
            if (mConvs.size() == 3) {
                projFused = mConvs[0]->setupQKVFusion(mConvs[1].get(), mProjOutputs[1], mConvs[2].get(),
                                                      mProjOutputs[2]);
            } else {
                projFused = mConvs[0]->setupQKVFusion(mConvs[1].get(), mProjOutputs[1], mConvs[2].get(),
                                                      mProjOutputs[2], mConvs[3].get(), mProjOutputs[3]);
            }
        }
        // The LN fold suppresses the LayerNorm's own dispatch, so mNormalized is
        // never written. That is only sound when the projection fusion succeeded
        // and the leader's single dispatch covers the whole group; if it failed
        // (env-disabled, pipeline miss, quant-layout mismatch, re-home failure),
        // the other convs still dispatch themselves reading mNormalized and the
        // LayerNorm must stay a separate dispatch.
        // lnFusionDisabled is the opt-in two-stage path: the LayerNorm child
        // writes residual_out + normalized once, then the fused projections
        // consume normalized. It helped Qwen3.5-2B decode on Mac, but not on
        // iPad M5, so the portable default remains the single folded dispatch.
        if (!projFused || !mHasLn || !mLn || MetalEnv::get().lnFusionDisabled) {
            return;
        }
        if (!mLn->isNC4HW4() || !mLn->isRMSNormWithGammaBeta()) {
            return;
        }
        // Folding the LayerNorm into the projection dispatch turns its residual
        // read and the projections' writes into one kernel. The allocator may
        // have aliased a projection output onto the residual input, which was
        // legal while the LayerNorm was a separate earlier dispatch but now
        // races. Re-home any such output to STATIC, which the dynamic pool
        // never reuses; if that fails, skip the fold rather than risk it.
        std::vector<Tensor *> written = mProjOutputs;
        if (mIsGateUp) {
            // The gate/up leader writes both halves plus the SiLU-mul result.
            written = {mGate.get(), mUp.get(), mProjOutputs[0]};
        }
        for (auto *out : written) {
            if (out == nullptr) {
                continue;
            }
            if (!backend->tensorsOverlap(out, mLnHiddenIn) &&
                !backend->tensorsOverlap(out, mLnResidualIn)) {
                continue;
            }
            if (!backend->onAcquireBuffer(out, Backend::STATIC)) {
                return;
            }
        }
        if (backend->tensorsOverlap(mLnResidualOut, mLnHiddenIn) ||
            backend->tensorsOverlap(mLnResidualOut, mLnResidualIn)) {
            if (!backend->onAcquireBuffer(mLnResidualOut, Backend::STATIC)) {
                return;
            }
        }
        if (mConvs[0]->setupLNFusion(mLnHiddenIn, mLnResidualIn, mLnResidualOut, mLn->getGamma(),
                                     mLn->getEps())) {
            mLn->setFused();
        }
    }

    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                          id<MTLComputeCommandEncoder> encoder) override {
        mRecordedGeneration = mResizeGeneration;
        Tensor *projInput = inputs[0];
#if MNN_METAL_OP_PROFILE
        // Profile builds swap the encoder mid-op per subpass; drive each child
        // through its own onExecute so per-op profile state stays consistent.
        if (mHasLn) {
            mLn->onExecute({inputs[0], inputs[1]}, {outputs[mNumProjOut], mNormalized.get()});
            projInput = mNormalized.get();
        }
        if (mIsGateUp) {
            mConvs[0]->onExecute({projInput}, {mGate.get()});
            mConvs[1]->onExecute({projInput}, {mUp.get()});
            mMulSilu->onExecute({mUp.get(), mGate.get()}, {outputs[0]});
            return;
        }
        for (int i = 0; i < mNumConvs; ++i) {
            mConvs[i]->onExecute({projInput}, {outputs[i]});
        }
        return;
#endif
        if (mHasLn) {
            mLn->onEncode({inputs[0], inputs[1]}, {outputs[mNumProjOut], mNormalized.get()}, encoder);
            projInput = mNormalized.get();
        }
        if (mIsGateUp) {
            // Once paired, the gate child (leader) dispatches both projections
            // and the up child encodes nothing.
            mConvs[0]->onEncode({projInput}, {mGate.get()}, encoder);
            mConvs[1]->onEncode({projInput}, {mUp.get()}, encoder);
            mMulSilu->onEncode({mUp.get(), mGate.get()}, {outputs[0]}, encoder);
            return;
        }
        // QKV fusion: children[0] is the leader; the followers' onEncode
        // early-return.
        for (int i = 0; i < mNumConvs; ++i) {
            mConvs[i]->onEncode({projInput}, {outputs[i]}, encoder);
        }
    }

    // The children annotate mNormalized/mGate/mUp into the recording, so a
    // recording taken before those Tensors were re-allocated holds dangling
    // pointers. Bail out whenever that happened; the normal encode re-records.
    virtual bool onReplayUpdate(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs) override {
        return mRecordedGeneration == mResizeGeneration;
    }

private:
    // Keep the Tensor object when its description is unchanged: recorded
    // encode-replay bindings hold raw Tensor* to these intermediates, so
    // re-creating them every resize would leave the recording dangling.
    // Returns true when a new Tensor was allocated.
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
                mConvs.emplace_back(static_cast<MetalConvolution1x1 *>(exe));
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
    std::vector<std::unique_ptr<MetalConvolution1x1>> mConvs;
    std::unique_ptr<MetalExecution> mMulSilu;
    std::unique_ptr<MetalLayerNorm> mLn;
    std::shared_ptr<Tensor> mNormalized;
    std::shared_ptr<Tensor> mGate;
    std::shared_ptr<Tensor> mUp;
    // Bumped whenever one of the intermediates above is re-allocated; compared
    // against the value stamped by onEncode to invalidate stale recordings.
    int mResizeGeneration   = 0;
    int mRecordedGeneration = -1;
    // Captured in onResize for setupFusion (which runs later, in onResizeEnd).
    std::vector<Tensor *> mProjOutputs;
    const Tensor *mLnResidualIn  = nullptr;
    const Tensor *mLnHiddenIn    = nullptr;
    Tensor *mLnResidualOut       = nullptr;
    bool mIsGateUp   = false;
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
        if (!FusedProjCommon::nativeEnvelopeOk(op, inputs.size(), outputs.size()) ||
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
