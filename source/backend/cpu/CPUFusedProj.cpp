//
//  CPUFusedProj.cpp
//  MNN
//
//  CPU composite execution for the fused projection op (OpType_FusedLinear).
//  The members are child executions from the regular per-op creators, so the
//  arithmetic matches the geometry decomposition; keeping the op whole lets
//  StaticModule's rearrange / base clone share the member weights through
//  each child's own onClone.
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPUIndexedLinear.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Execution.hpp"
#include "core/FusedProjCommon.hpp"
#include "core/IndexedLinearCommon.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

struct CPUFusedProjSubOps {
    std::vector<std::shared_ptr<BufferStorage>> convs;
    std::shared_ptr<BufferStorage> mulSilu;
    std::shared_ptr<BufferStorage> layerNorm;
};

class CPUFusedProj : public Execution {
public:
    CPUFusedProj(const Op* op, Backend* backend) : Execution(backend) {
        auto param = op->main_as_FusedLinearParam();
        _initParam(param);
        mSubOps.reset(new CPUFusedProjSubOps);
        const auto fmt = op->defaultDimentionFormat();
        mSubOps->convs.resize(mNumConvs);
        for (int i = 0; i < mNumConvs; ++i) {
            mSubOps->convs[i] =
                FusedProjCommon::makeConvOp(param->convs()->GetAs<Convolution2D>(i), fmt, op->externalPath());
        }
        if (mIsGateUp) {
            mSubOps->mulSilu = FusedProjCommon::makeMulSiluOp(fmt);
        }
        if (mHasLn) {
            mSubOps->layerNorm = FusedProjCommon::makeLayerNormOp(param->ln(), fmt);
        }
        int experts, ic, oc;
        auto cpu = static_cast<CPUBackend*>(backend);
        const auto& hint = backend->getRuntime()->hint();
        // A backend that captures the gate/up children to fuse them itself (e.g. SpacemiT
        // IME2) must see them as two projections; the merged op would bypass it.
        if (mIsGateUp && !cpu->functions()->supportFusedGateUpCapture &&
            cpu->precisionMode() != BackendConfig::Precision_Low_BF16 &&
            (hint.useCachedMmap == 0 || hint.weightMemoryPath.empty()) &&
            IndexedLinearCommon::dimensions(op, experts, ic, oc) && ic % cpu->functions()->pack == 0 &&
            oc / 2 % cpu->functions()->pack == 0) {
            std::shared_ptr<Execution> fused(new CPUIndexedLinear(backend, op));
            if (fused->valid()) {
                mGateUp = std::move(fused);
                if (mHasLn) {
                    mLn.reset(backend->onCreate({}, {}, FusedProjCommon::opOf(mSubOps->layerNorm)));
                    mValid = mLn != nullptr;
                }
                return;
            }
        }
        mValid = _createWeightedChildren(backend);
    }
    CPUFusedProj(std::shared_ptr<CPUFusedProjSubOps> subOps, const Op* op, Backend* backend) : Execution(backend) {
        _initParam(op->main_as_FusedLinearParam());
        mSubOps = subOps;
    }
    virtual ~CPUFusedProj() = default;

    bool valid() const { return mValid; }

    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (!mValid || (!mGateUp && (int)mConvs.size() != mNumConvs) || (mHasLn && !mLn)) {
            return false;
        }
        // preRearrangeWeights strips the member weights on a positive probe,
        // so the answer must reflect whether every child can really clone.
        if (nullptr == dst) {
            if (mGateUp && !mGateUp->onClone(bn, op, nullptr)) {
                return false;
            }
            for (int i = 0; i < (int)mConvs.size(); ++i) {
                if (!mConvs[i]->onClone(bn, FusedProjCommon::opOf(mSubOps->convs[i]), nullptr)) {
                    return false;
                }
            }
            return !mHasLn || mLn->onClone(bn, FusedProjCommon::opOf(mSubOps->layerNorm), nullptr);
        }
        // mMulSilu carries no weights; the clone re-creates it on its first
        // onResize. LN is cloned, not re-created from op: with a reused mmap
        // weight cache the op's gamma/beta are external and empty.
        std::unique_ptr<CPUFusedProj> clone(new CPUFusedProj(mSubOps, op, bn));
        if (mGateUp) {
            Execution* child = nullptr;
            if (!mGateUp->onClone(bn, op, &child) || child == nullptr) {
                return false;
            }
            clone->mGateUp.reset(child);
        }
        clone->mConvs.resize(mConvs.size());
        for (int i = 0; i < (int)mConvs.size(); ++i) {
            if (!_cloneChild(mConvs[i], bn, mSubOps->convs[i], clone->mConvs[i])) {
                return false;
            }
        }
        if (mHasLn && !_cloneChild(mLn, bn, mSubOps->layerNorm, clone->mLn)) {
            return false;
        }
        *dst = clone.release();
        return true;
    }

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto bn = backend();
        Tensor* hidden = mHasLn ? inputs[1] : inputs[0];
        mUseGateUp = mGateUp != nullptr && hidden->dimensions() == 4 && outputs[0]->dimensions() == 4 &&
            hidden->length(2) == 1 && hidden->length(3) == 1 &&
            outputs[0]->length(2) == 1 && outputs[0]->length(3) == 1 &&
            hidden->getType() == halide_type_of<float>() && outputs[0]->getType() == halide_type_of<float>() &&
            !TensorUtils::getDescribe(hidden)->quantAttr && !TensorUtils::getDescribe(outputs[0])->quantAttr &&
            TensorUtils::getDescribe(hidden)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
            TensorUtils::getDescribe(outputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
        if (!mUseGateUp && mConvs.empty() && !_createWeightedChildren(bn)) {
            return NOT_SUPPORT;
        }
        if (mHasLn) {
            mNormalized = _makeLike(hidden, hidden->length(1));
            if (!bn->onAcquireBuffer(mNormalized.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }
        if (mIsGateUp && !mUseGateUp) {
            const int oc = outputs[0]->length(1);
            mGate = _makeLike(hidden, oc);
            mUp = _makeLike(hidden, oc);
            if (!bn->onAcquireBuffer(mGate.get(), Backend::DYNAMIC) ||
                !bn->onAcquireBuffer(mUp.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
        }
        if (!_createMulSilu(bn, outputs)) {
            MNN_ERROR("CPUFusedProj: failed to create sub-executions\n");
            return NOT_SUPPORT;
        }
        ErrorCode err = _forEachChild(inputs, outputs,
                                      [](Execution* exe, const std::vector<Tensor*>& in,
                                         const std::vector<Tensor*>& out) { return exe->onResize(in, out); });
        if (mGate && !mUseGateUp) {
            bn->onReleaseBuffer(mGate.get(), Backend::DYNAMIC);
            bn->onReleaseBuffer(mUp.get(), Backend::DYNAMIC);
        }
        if (mNormalized) {
            bn->onReleaseBuffer(mNormalized.get(), Backend::DYNAMIC);
        }
        return err;
    }

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        return _forEachChild(inputs, outputs,
                             [](Execution* exe, const std::vector<Tensor*>& in, const std::vector<Tensor*>& out) {
                                 return exe->onExecute(in, out);
                             });
    }

private:
    void _initParam(const FusedLinearParam* param) {
        mIsGateUp = param->act_silu_mul();
        mHasLn = param->has_ln() && param->ln() != nullptr;
        mNumConvs = (int)param->convs()->size();
        mNumProjOut = mIsGateUp ? 1 : mNumConvs;
    }

    static std::shared_ptr<Tensor> _makeLike(const Tensor* like, int channel) {
        auto shape = like->shape();
        if (shape.size() >= 2) {
            shape[1] = channel;
        }
        std::shared_ptr<Tensor> t(Tensor::createDevice(shape, like->getType(), like->getDimensionType()));
        TensorUtils::getDescribe(t.get())->dimensionFormat = TensorUtils::getDescribe(like)->dimensionFormat;
        return t;
    }

    // Create the weight-carrying children at creation, before the first
    // onResize: a clone taken right after creation already shares them, and
    // with the mmap weight cache their STATIC buffers must come from the pool
    // before onClearBuffer seals it.
    bool _createWeightedChildren(Backend* backend) {
        mConvs.resize(mNumConvs);
        for (int i = 0; i < mNumConvs; ++i) {
            auto conv = FusedProjCommon::opOf(mSubOps->convs[i])->main_as_Convolution2D();
            // The conv creator only inspects the tensors for dispatch selection.
            std::shared_ptr<Tensor> dummyIn(Tensor::createDevice<float>({1, conv->common()->inputCount(), 1, 1}));
            std::shared_ptr<Tensor> dummyOut(Tensor::createDevice<float>({1, conv->common()->outputCount(), 1, 1}));
            TensorUtils::getDescribe(dummyIn.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            TensorUtils::getDescribe(dummyOut.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            Execution* exe =
                backend->onCreate({dummyIn.get()}, {dummyOut.get()}, FusedProjCommon::opOf(mSubOps->convs[i]));
            if (exe == nullptr || !exe->valid()) {
                delete exe;
                return false;
            }
            mConvs[i].reset(exe);
        }
        if (mHasLn && !mLn) {
            // The LayerNorm creator reads only the op.
            mLn.reset(backend->onCreate({}, {}, FusedProjCommon::opOf(mSubOps->layerNorm)));
            if (mLn == nullptr) {
                return false;
            }
        }
        return true;
    }

    static bool _cloneChild(const std::shared_ptr<Execution>& src, Backend* bn,
                            const std::shared_ptr<BufferStorage>& op, std::shared_ptr<Execution>& dst) {
        Execution* exe = nullptr;
        if (!src->onClone(bn, FusedProjCommon::opOf(op), &exe) || nullptr == exe) {
            return false;
        }
        dst.reset(exe);
        return true;
    }

    bool _createMulSilu(Backend* backend, const std::vector<Tensor*>& outputs) {
        if (mIsGateUp && !mUseGateUp && !mMulSilu) {
            Execution* exe =
                backend->onCreate({mUp.get(), mGate.get()}, {outputs[0]}, FusedProjCommon::opOf(mSubOps->mulSilu));
            if (exe == nullptr) {
                return false;
            }
            mMulSilu.reset(exe);
        }
        return true;
    }

    template <typename F>
    ErrorCode _forEachChild(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, F&& run) {
        Tensor* projInput = inputs[0];
        if (mHasLn) {
            auto err = run(mLn.get(), {inputs[0], inputs[1]}, {outputs[mNumProjOut], mNormalized.get()});
            if (err != NO_ERROR) {
                return err;
            }
            projInput = mNormalized.get();
        }
        if (mUseGateUp) {
            return run(mGateUp.get(), {projInput}, {outputs[0]});
        }
        if (!mIsGateUp) {
            for (int i = 0; i < mNumConvs; ++i) {
                auto err = run(mConvs[i].get(), {projInput}, {outputs[i]});
                if (err != NO_ERROR) {
                    return err;
                }
            }
            return NO_ERROR;
        }
        auto err = run(mConvs[0].get(), {projInput}, {mGate.get()});
        if (err != NO_ERROR) {
            return err;
        }
        err = run(mConvs[1].get(), {projInput}, {mUp.get()});
        if (err != NO_ERROR) {
            return err;
        }
        return run(mMulSilu.get(), {mUp.get(), mGate.get()}, {outputs[0]});
    }

    bool mIsGateUp = false;
    bool mHasLn = false;
    int mNumConvs = 0;
    int mNumProjOut = 0;
    bool mValid = true;
    bool mUseGateUp = false;
    std::shared_ptr<CPUFusedProjSubOps> mSubOps;
    std::vector<std::shared_ptr<Execution>> mConvs;
    std::shared_ptr<Execution> mMulSilu;
    std::shared_ptr<Execution> mGateUp;
    std::shared_ptr<Execution> mLn;
    std::shared_ptr<Tensor> mGate;
    std::shared_ptr<Tensor> mUp;
    std::shared_ptr<Tensor> mNormalized;
};

class CPUFusedProjCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        // Must match GeometryFusedProj::_keepWhole: an op kept whole but
        // refused here falls to the backup backend or fails session creation.
        if (!FusedProjCommon::nativeEnvelopeOk(op, inputs.size(), outputs.size())) {
            return nullptr;
        }
        auto exe = new CPUFusedProj(op, backend);
        if (!exe->valid()) {
            delete exe;
            return nullptr;
        }
        return exe;
    }
};

REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUFusedProjCreator, OpType_FusedLinear);

} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
