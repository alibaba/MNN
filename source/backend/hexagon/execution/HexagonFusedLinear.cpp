//
//  HexagonFusedLinear.cpp
//  MNN
//
//  See HexagonFusedLinear.hpp. Both flavours of OpType_FusedLinear land here: act_silu_mul
//  (gate/up, two convs joined by MUL_SILU) and QKV (three or four convs writing straight to the
//  group outputs), each optionally preceded by the binary RMSNorm.
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "HexagonFusedLinear.hpp"

#include "HexagonBinary.hpp"
#include "HexagonLayerNorm.hpp"
#include "HexagonTMac.hpp"
#include "backend/hexagon/backend/HexagonBackend.hpp"
#include "backend/hexagon/backend/HexagonRuntime.hpp"
#include "core/FusedProjCommon.hpp"
#include "core/TensorUtils.hpp"
#include "htp_command.h"

namespace MNN {

static std::shared_ptr<Tensor> _makeLike(const Tensor* like, int channel) {
    auto shape = like->shape();
    if (shape.size() >= 2) {
        shape[1] = channel;
    }
    std::shared_ptr<Tensor> t(Tensor::createDevice(shape, like->getType(), like->getDimensionType()));
    TensorUtils::getDescribe(t.get())->dimensionFormat = TensorUtils::getDescribe(like)->dimensionFormat;
    return t;
}

HexagonFusedLinear::HexagonFusedLinear(Backend* backend, std::shared_ptr<Shared> shared)
    : Execution(backend), mShared(std::move(shared)) {}

HexagonFusedLinear* HexagonFusedLinear::create(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                                               const std::vector<Tensor*>& outputs) {
    auto param = op->main_as_FusedLinearParam();
    // Same predicate as GeometryFusedProj::_keepWhole: an op kept whole that this refused would fail
    // session creation. Every member then has a regular Hexagon execution, so creation succeeds.
    if (param == nullptr || !FusedProjCommon::compositeEnvelopeOk(op, inputs.size(), outputs.size())) {
        return nullptr;
    }
    const auto fmt = op->defaultDimentionFormat();
    std::shared_ptr<Shared> shared(new Shared);
    shared->gateUp = param->act_silu_mul();
    shared->hasLn = param->has_ln() && param->ln() != nullptr;
    const int numConvs = (int)param->convs()->size();
    shared->inputCount = param->convs()->GetAs<Convolution2D>(0)->common()->inputCount();

    std::unique_ptr<HexagonFusedLinear> exe(new HexagonFusedLinear(backend, shared));
    bool allGemv = true;
    for (int i = 0; i < numConvs; ++i) {
        shared->convOps.emplace_back(
            FusedProjCommon::makeConvOp(param->convs()->GetAs<Convolution2D>(i), fmt, op->externalPath()));
        auto convOp = FusedProjCommon::opOf(shared->convOps.back());
        auto common = convOp->main_as_Convolution2D()->common();
        // HexagonExecutionFactory tries T-MAC first; keep the same member execution choice.
        std::shared_ptr<Tensor> dummyIn(Tensor::createDevice<float>({1, common->inputCount(), 1, 1}));
        std::shared_ptr<Tensor> dummyOut(Tensor::createDevice<float>({1, common->outputCount(), 1, 1}));
        TensorUtils::getDescribe(dummyIn.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        TensorUtils::getDescribe(dummyOut.get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        Execution* child = HexagonTMac::create(backend, convOp, {dummyIn.get()}, {dummyOut.get()});
        if (child != nullptr) {
            allGemv = false;
        } else {
            auto res = HexagonConvolution::makeResource(backend, convOp);
            if (res == nullptr) {
                return nullptr;
            }
            shared->convResources.emplace_back(res);
            child = new HexagonConvolution(backend, res, convOp);
        }
        exe->mConvs.emplace_back(child);
    }
    if (!allGemv) {
        shared->convResources.clear();
    }
    if (shared->gateUp) {
        shared->mulSiluOp = FusedProjCommon::makeMulSiluOp(fmt);
        exe->mMulSilu.reset(HexagonBinary::create(backend, FusedProjCommon::opOf(shared->mulSiluOp)));
        if (exe->mMulSilu == nullptr) {
            return nullptr;
        }
    }
    if (shared->hasLn) {
        shared->layerNormOp = FusedProjCommon::makeLayerNormOp(param->ln(), fmt);
        exe->mLn.reset(HexagonLayerNorm::create(backend, FusedProjCommon::opOf(shared->layerNormOp)));
        if (exe->mLn == nullptr) {
            return nullptr;
        }
    }
    return exe.release();
}

bool HexagonFusedLinear::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (dst == nullptr) {
        // StaticModule strips the member weights once this probe says yes, and every later clone depends
        // on it having been truthful. So ask the children the same question the real clone below asks;
        // their probe branches ignore bn / op and only report their own validity.
        for (auto& conv : mConvs) {
            if (!conv->onClone(nullptr, nullptr, nullptr)) {
                return false;
            }
        }
        if (mMulSilu != nullptr && !mMulSilu->onClone(nullptr, nullptr, nullptr)) {
            return false;
        }
        if (mLn != nullptr && !mLn->onClone(nullptr, nullptr, nullptr)) {
            return false;
        }
        return true;
    }
    std::unique_ptr<HexagonFusedLinear> exe(new HexagonFusedLinear(bn, mShared));
    for (size_t i = 0; i < mConvs.size(); ++i) {
        Execution* child = nullptr;
        if (!mConvs[i]->onClone(bn, FusedProjCommon::opOf(mShared->convOps[i]), &child) || child == nullptr) {
            return false;
        }
        exe->mConvs.emplace_back(child);
    }
    if (mMulSilu) {
        Execution* child = nullptr;
        if (!mMulSilu->onClone(bn, FusedProjCommon::opOf(mShared->mulSiluOp), &child) || child == nullptr) {
            return false;
        }
        exe->mMulSilu.reset(child);
    }
    if (mLn) {
        Execution* child = nullptr;
        if (!mLn->onClone(bn, FusedProjCommon::opOf(mShared->layerNormOp), &child) || child == nullptr) {
            return false;
        }
        exe->mLn.reset(child);
    }
    *dst = exe.release();
    return true;
}

// The fused decode launch must compute exactly what each member's own M=1 GEMV_I8 command would:
// same weight blobs, same activation quantization, all members in VTCM at once.
bool HexagonFusedLinear::fusedDecodeOk(int rows) const {
    const auto& res = mShared->convResources;
    if (rows != 1 || res.empty() || res.size() != mConvs.size() || res.size() > DSP_GEMV_I8_MAX_SEGMENTS) {
        return false;
    }
    const auto& info = static_cast<const HexagonRuntime*>(backend()->getRuntime())->info();
    if ((info.capabilities & DSP_CAP_GEMV_I8_FUSED) == 0) {
        return false;
    }
    const int K = mShared->inputCount;
    const int nblk = res[0]->int4ScaleBlockNum;
    const bool asym = res[0]->int4ScaleAsymmetric;
    if (K <= 0 || (K % 64) != 0 || nblk <= 0 || (K / 32) % nblk != 0) {
        return false;
    }
    size_t memberBytes = 0;
    int firstOc = 0;
    for (size_t i = 0; i < res.size(); ++i) {
        auto common = FusedProjCommon::opOf(mShared->convOps[i])->main_as_Convolution2D()->common();
        const int oc = common->outputCount();
        // oc need not be a multiple of 32: the GEMV blob is padded to whole tiles (reorderInt4WeightForVrmpyGemv).
        if (!res[i]->useGemvI8 || res[i]->gemvI8Weight.first == nullptr || res[i]->int4ScaleBlockNum != nblk ||
            res[i]->int4ScaleAsymmetric != asym || common->inputCount() != K || common->relu() || common->relu6()) {
            return false;
        }
        if (i == 0) {
            firstOc = oc;
        } else if (mShared->gateUp && oc != firstOc) {
            return false;
        }
        const size_t np = UP_DIV(oc, 32);
        memberBytes += np * ((size_t)512 * (K / 32) + (size_t)nblk * 128);
    }
    // Mirrors the VTCM budget of matmul_q4block_gemv_i8.c's single-chain path.
    const size_t asymBytes =
        asym ? (size_t)nblk * 128 + (((size_t)(32 + nblk) * sizeof(int32_t) + 127) & ~(size_t)127) : 0;
    const size_t need = (size_t)K + (size_t)(K / 4) * 128 + asymBytes + memberBytes + 4096;
    return info.vtcmSize > 0 && need <= (size_t)info.vtcmSize;
}

ErrorCode HexagonFusedLinear::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto bn = backend();
    const bool gateUp = mShared->gateUp;
    const bool hasLn = mShared->hasLn;
    const int numConvs = (int)mConvs.size();
    const int numProjOut = gateUp ? 1 : numConvs;
    Tensor* hidden = hasLn ? inputs[1] : inputs[0];
    Tensor* projInput = hidden;
    // Intermediates are acquired for the children's resize and released at the end: the commands
    // capture their (fd, offset) and the dynamic arena keeps them valid until this op has run.
    mNormalized = nullptr;
    mGate = nullptr;
    mUp = nullptr;
    // Every DYNAMIC temp is handed back through releaseTemps, which the error paths below also run: the
    // flags (rather than the pointers) track what is still held, because onExecute keeps using the
    // pointers after the buffers have been returned to the allocator.
    bool gateHeld = false, upHeld = false, normHeld = false;
    auto releaseTemps = [&]() {
        if (gateHeld) {
            bn->onReleaseBuffer(mGate.get(), Backend::DYNAMIC);
            gateHeld = false;
        }
        if (upHeld) {
            bn->onReleaseBuffer(mUp.get(), Backend::DYNAMIC);
            upHeld = false;
        }
        if (normHeld) {
            bn->onReleaseBuffer(mNormalized.get(), Backend::DYNAMIC);
            normHeld = false;
        }
    };
    if (hasLn) {
        mNormalized = _makeLike(hidden, hidden->length(1));
        if (!bn->onAcquireBuffer(mNormalized.get(), Backend::DYNAMIC)) {
            mNormalized = nullptr;
            return OUT_OF_MEMORY;
        }
        normHeld = true;
        // Binary RMSNorm: in [residual, hidden], out [residual_out, normalized].
        auto code = mLn->onResize({inputs[0], inputs[1]}, {outputs[numProjOut], mNormalized.get()});
        if (code != NO_ERROR) {
            releaseTemps();
            return code;
        }
        projInput = mNormalized.get();
    }

    const int rows = mShared->inputCount > 0 ? (int)(hidden->elementSize() / mShared->inputCount) : 0;
    mUseFused = fusedDecodeOk(rows);
    ErrorCode code = NO_ERROR;
    if (mUseFused) {
        const auto& res = mShared->convResources;
        const int nmem = (int)res.size();
        int params[12 + DSP_GEMV_I8_MAX_SEGMENTS] = {0};
        int totalOc = 0;
        params[0] = 1;
        params[1] = mShared->inputCount;
        params[8] = res[0]->int4ScaleBlockNum;
        params[9] = res[0]->int4ScaleAsymmetric ? 1 : 0;
        params[10] = gateUp ? 1 : 0;
        params[11] = nmem;
        std::vector<std::pair<int, int>> inputFds = {HexagonBackend::getDevicePtr(projInput)};
        for (int p = 0; p < nmem; ++p) {
            const int oc =
                UP_DIV(FusedProjCommon::opOf(mShared->convOps[p])->main_as_Convolution2D()->common()->outputCount(),
                       32) *
                32;
            params[12 + p] = oc;
            totalOc += oc;
            inputFds.emplace_back(HexagonBackend::getDevicePtr(res[p]->gemvI8Weight));
            inputFds.emplace_back(res[p]->hasBias ? HexagonBackend::getDevicePtr(res[p]->bias) : std::make_pair(-1, 0));
        }
        params[2] = totalOc;
        std::vector<Tensor*> outs(outputs.begin(), outputs.begin() + numProjOut);
        std::vector<std::pair<int, int>> outputFds;
        for (auto t : outs) {
            outputFds.emplace_back(HexagonBackend::getDevicePtr(t));
        }
        mFusedCmd.build(static_cast<HexagonBackend*>(bn), DSP_OP_MATMUL_Q4A16_GEMV_I8_FUSED, params, sizeof(params),
                        inputFds, outputFds, {projInput}, outs);
    } else if (!gateUp) {
        for (int i = 0; i < numConvs && code == NO_ERROR; ++i) {
            code = mConvs[i]->onResize({projInput}, {outputs[i]});
        }
    } else {
        // act_silu_mul: gate = convs[0](x), up = convs[1](x), out = up * silu(gate).
        mGate = _makeLike(projInput, outputs[0]->length(1));
        mUp = _makeLike(projInput, outputs[0]->length(1));
        if (!bn->onAcquireBuffer(mGate.get(), Backend::DYNAMIC)) {
            mGate = nullptr;
            releaseTemps();
            return OUT_OF_MEMORY;
        }
        gateHeld = true;
        if (!bn->onAcquireBuffer(mUp.get(), Backend::DYNAMIC)) {
            mUp = nullptr;
            releaseTemps();
            return OUT_OF_MEMORY;
        }
        upHeld = true;
        code = mConvs[0]->onResize({projInput}, {mGate.get()});
        if (code == NO_ERROR) {
            code = mConvs[1]->onResize({projInput}, {mUp.get()});
        }
        if (code == NO_ERROR) {
            // MUL_SILU: out = in0 * silu(in1), so in0 = up and in1 = gate.
            code = mMulSilu->onResize({mUp.get(), mGate.get()}, {outputs[0]});
        }
    }
    releaseTemps();
    return code;
}

ErrorCode HexagonFusedLinear::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const bool gateUp = mShared->gateUp;
    const int numConvs = (int)mConvs.size();
    const int numProjOut = gateUp ? 1 : numConvs;
    Tensor* projInput = mShared->hasLn ? mNormalized.get() : inputs[0];
    ErrorCode code = NO_ERROR;
    if (mShared->hasLn) {
        code = mLn->onExecute({inputs[0], inputs[1]}, {outputs[numProjOut], mNormalized.get()});
        if (code != NO_ERROR) {
            return code;
        }
    }
    if (mUseFused) {
        return mFusedCmd.execute() == 0 ? NO_ERROR : OUT_OF_MEMORY;
    }
    if (!gateUp) {
        for (int i = 0; i < numConvs && code == NO_ERROR; ++i) {
            code = mConvs[i]->onExecute({projInput}, {outputs[i]});
        }
        return code;
    }
    code = mConvs[0]->onExecute({projInput}, {mGate.get()});
    if (code == NO_ERROR) {
        code = mConvs[1]->onExecute({projInput}, {mUp.get()});
    }
    if (code == NO_ERROR) {
        code = mMulSilu->onExecute({mUp.get(), mGate.get()}, {outputs[0]});
    }
    return code;
}

} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
