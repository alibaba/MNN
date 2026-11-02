//
//  MetalFusedProj.mm
//  MNN
//
//  Metal execution for the export-time fused projection op (FusedLinear).
//  Owns one Conv1x1 child per param conv plus optional MUL_SILU and LayerNorm
//  children. The group is driven by the exported member order and only the
//  decode GEMV shape actually fuses; other shapes fall back to per-child
//  dispatch.
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
                    // Offer the up projection the group's MUL_SILU; the child
                    // accepts only on the M64 prefill tile. Acquiring mGate/mUp
                    // while outputs[0] is live keeps the gate tile unaliased.
                    mConvs[1]->requestPrefillSiluMul(mGate.get());
                    // Better still: let the up projection carry the gate tile in
                    // a second accumulator, so the gate tensor is never written.
                    // Only one of the two offers can be taken; the child decides.
                    mConvs[1]->requestPrefillGateUpDual(mConvs[0].get());
                    err = mConvs[1]->onResize({projInput}, {mUp.get()});
                    mPrefillDualFused = err == NO_ERROR && mConvs[1]->prefillGateUpDualFused();
                    mPrefillSiluFused = err == NO_ERROR && mConvs[1]->prefillSiluMulFused();
                }
                if (err == NO_ERROR && !mPrefillDualFused) {
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
            mProjInput     = mHasLn ? nullptr : inputs[0];
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
        // Every resize re-decides the fusion, and the early returns below leave
        // the group unfused, so clear this before any of them can fire.
        mSiluFused = false;
        // The fused kernels only exist on the decode-GEMV pipeline; anything
        // else (all of prefill included) stays as per-member dispatches.
        for (auto &conv : mConvs) {
            if (!conv->is2sgDecodePipeline()) {
                return;
            }
        }
        // Grouping matches the exported member order: (gate, up) for
        // act_silu_mul, else q/k/v plus an optional fourth projection.
        bool projFused = false;
        if (mIsGateUp) {
            if (mConvs.size() == 2) {
                // Offer the group's output so the leader may absorb MUL_SILU;
                // if accepted, re-home the input to STATIC (the allocator may
                // have placed it under the leader's output). Fall back to a
                // separate MUL_SILU dispatch if that fails.
                const Tensor *siluOut = mProjOutputs[0];
                if (!mHasLn && backend->tensorsOverlap(siluOut, mProjInput) &&
                    !backend->onAcquireBuffer(mProjOutputs[0], Backend::STATIC)) {
                    siluOut = nullptr;
                }
                projFused = mConvs[0]->setupGateUpFusion(mConvs[1].get(), mUp.get(), siluOut);
                mSiluFused = projFused && mConvs[0]->gateUpSiluFused();
            }
        } else if (mConvs.size() >= 3) {
            if (mConvs.size() == 3) {
                projFused = mConvs[0]->setupQKVFusion(mProjOutputs[0], mConvs[1].get(), mProjOutputs[1],
                                                      mConvs[2].get(), mProjOutputs[2]);
            } else {
                projFused = mConvs[0]->setupQKVFusion(mProjOutputs[0], mConvs[1].get(), mProjOutputs[1],
                                                      mConvs[2].get(), mProjOutputs[2], mConvs[3].get(),
                                                      mProjOutputs[3]);
            }
        }
        // Only a fully fused projection group can skip materializing mNormalized.
        if (!projFused || !mHasLn || !mLn) {
            return;
        }
        if (!mLn->isNC4HW4() || !mLn->isRMSNormWithGammaBeta()) {
            return;
        }
        // Fold LayerNorm into the projection dispatch. Re-home any projection
        // output aliased onto the residual input to STATIC (safe for the
        // single-dispatch fold); skip the fold if that fails.
        std::vector<Tensor *> written = mProjOutputs;
        if (mIsGateUp) {
            // The gate/up leader writes both halves plus the SiLU-mul result --
            // unless the epilogue absorbed the SiLU-mul, in which case the halves
            // stay in registers and only the result reaches memory.
            written = mSiluFused ? std::vector<Tensor *>{mProjOutputs[0]}
                    : mPrefillDualFused
                          ? std::vector<Tensor *>{mProjOutputs[0]}
                    : mPrefillSiluFused
                          ? std::vector<Tensor *>{mGate.get(), mProjOutputs[0]}
                          : std::vector<Tensor *>{mGate.get(), mUp.get(), mProjOutputs[0]};
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
            if (!mPrefillDualFused) {
                mConvs[0]->onExecute({projInput}, {mGate.get()});
            }
            mConvs[1]->onExecute({projInput}, {(mPrefillSiluFused || mPrefillDualFused) ? outputs[0] : mUp.get()});
            if (!mSiluFused && !mPrefillSiluFused && !mPrefillDualFused) {
                mMulSilu->onExecute({mUp.get(), mGate.get()}, {outputs[0]});
            }
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
            // Decode: the gate child dispatches both projections.
            // Prefill dual: the up child dispatches both tiles instead.
            if (!mPrefillDualFused) {
                mConvs[0]->onEncode({projInput}, {mGate.get()}, encoder);
            }
            // With the prefill fold the up projection writes the group output
            // itself, reading the gate tile the previous dispatch just wrote.
            mConvs[1]->onEncode({projInput}, {(mPrefillSiluFused || mPrefillDualFused) ? outputs[0] : mUp.get()}, encoder);
            if (!mSiluFused && !mPrefillSiluFused && !mPrefillDualFused) {
                mMulSilu->onEncode({mUp.get(), mGate.get()}, {outputs[0]}, encoder);
            }
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
    const Tensor *mProjInput     = nullptr;   // only when there is no LN member
    const Tensor *mLnResidualIn  = nullptr;
    const Tensor *mLnHiddenIn    = nullptr;
    Tensor *mLnResidualOut       = nullptr;
    bool mIsGateUp   = false;
    bool mSiluFused  = false;   // leader's epilogue emits up * silu(gate)
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
