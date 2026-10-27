//
//  HexagonLinearAttention.cpp
//  MNN
//
//  See HexagonLinearAttention.hpp. State handling mirrors CPULinearAttention without the prefix-cache
//  files (HexagonAttention does not implement those either): the conv / recurrent state is zeroed when
//  the whole history is reset, kept for decode and reuse-kv continuations, and a partial erase is refused.
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "HexagonLinearAttention.hpp"

#include <cstring>

#include "backend/hexagon/backend/HexagonBackend.hpp"
#include "backend/hexagon/backend/HexagonRuntime.hpp"
#include "core/KVMeta.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

static bool _isC4(const Tensor* t) {
    return t != nullptr && TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
}

// Packed [tokens, channels, 1, 1] fp16 tensor.
static bool _isPackedTokens(const Tensor* t, int channels) {
    return _isC4(t) && t->dimensions() == 4 && t->length(2) == 1 && t->length(3) == 1 && t->length(0) > 0 &&
           t->length(1) == channels && HexagonBackend::getBytes(t) == 2;
}

HexagonLinearAttention::State::~State() {
    if (allocBackend == nullptr) {
        return;
    }
    for (auto t : {&convState, &rnnState, &gateConst}) {
        if (*t != nullptr) {
            allocBackend->onReleaseBuffer(t->get(), Backend::STATIC);
            t->reset();
        }
    }
}

HexagonLinearAttention::HexagonLinearAttention(Backend* backend, const DSPLinearAttnParam& param,
                                               std::vector<float> gateConst, std::shared_ptr<State> state)
    : HexagonExecution(backend), mParam(param), mGateConstHost(std::move(gateConst)), mState(std::move(state)) {
    mMeta = (KVMeta*)(backend->getMetaPtr());
}

// Shape checks shared by creation (when tensors are known) and resize. StaticModule pre-creates this op
// with no tensors so prefill and decode share one state, so creation alone cannot rely on them.
static bool _tensorsSupported(const DSPLinearAttnParam& p, const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs) {
    if (inputs.size() < 4 || outputs.size() != 1 || inputs[0] == nullptr || inputs[3] == nullptr) {
        return false;
    }
    auto qkv = inputs[0];
    auto convW = inputs[3];
    if (!_isC4(qkv) || !_isC4(outputs[0]) || HexagonBackend::getBytes(outputs[0]) != 2) {
        return false;
    }
    const int D = qkv->length(1);
    if (!_isPackedTokens(qkv, D) || _isC4(convW) || convW->dimensions() != 3 || convW->length(1) != 1 ||
        HexagonBackend::getBytes(convW) != 2 || convW->length(2) < 1 || convW->length(2) > 8) {
        return false;
    }
    if (p.type == DSP_LINEAR_ATTN_SHORT_CONV) {
        return D % 192 == 0 && convW->length(0) == D / 3;
    }
    return D == 2 * p.numKHeads * p.headKDim + p.numVHeads * p.headVDim && convW->length(0) == D &&
           _isPackedTokens(inputs[1], p.numVHeads) && _isPackedTokens(inputs[2], p.numVHeads) &&
           inputs[1]->length(0) == qkv->length(0) && inputs[2]->length(0) == qkv->length(0);
}

HexagonLinearAttention* HexagonLinearAttention::create(Backend* backend, const Op* op,
                                                       const std::vector<Tensor*>& inputs,
                                                       const std::vector<Tensor*>& outputs) {
    auto runtime = static_cast<const HexagonRuntime*>(backend->getRuntime());
    if (runtime == nullptr || (runtime->info().capabilities & DSP_CAP_LINEAR_ATTN) == 0) {
        return nullptr;
    }
    auto param = op->main_as_LinearAttentionParam();
    if (param == nullptr || param->attn_type() == nullptr) {
        return nullptr;
    }
    const auto type = param->attn_type()->str();
    DSPLinearAttnParam p;
    ::memset(&p, 0, sizeof(p));
    p.numKHeads = param->num_k_heads();
    p.numVHeads = param->num_v_heads();
    p.headKDim = param->head_k_dim();
    p.headVDim = param->head_v_dim();
    p.useL2Norm = param->use_qk_l2norm() ? 1 : 0;
    std::vector<float> gateConst;
    if (type == "short_conv") {
        p.type = DSP_LINEAR_ATTN_SHORT_CONV;
    } else if (type == "gated_delta_rule") {
        p.type = DSP_LINEAR_ATTN_GATED_DELTA;
        const int Hk = p.numKHeads, Hv = p.numVHeads, dk = p.headKDim, dv = p.headVDim;
        if (Hk <= 0 || Hv <= 0 || Hv % Hk != 0 || dk % 64 != 0 || dv % 64 != 0 || dk > 256 || dv > 256) {
            return nullptr;
        }
        if (param->gate_fold()) {
            // A folded graph has no gate chain left: malformed constants must be rejected, never ignored.
            if (param->gate_coef() == nullptr || param->gate_bias() == nullptr ||
                (int)param->gate_coef()->size() != Hv || (int)param->gate_bias()->size() != Hv) {
                return nullptr;
            }
            p.gateFold = 1;
            gateConst.assign(param->gate_coef()->begin(), param->gate_coef()->end());
            gateConst.insert(gateConst.end(), param->gate_bias()->begin(), param->gate_bias()->end());
        }
    } else {
        return nullptr;
    }
    if (!inputs.empty() && !_tensorsSupported(p, inputs, outputs)) {
        return nullptr;
    }
    return new HexagonLinearAttention(backend, p, std::move(gateConst), std::shared_ptr<State>(new State));
}

bool HexagonLinearAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (dst == nullptr) {
        return true;
    }
    // Prefill and decode executions share the recurrent state, as on the CPU.
    *dst = new HexagonLinearAttention(bn, mParam, mGateConstHost, mState);
    return true;
}

ErrorCode HexagonLinearAttention::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                             std::vector<HexagonCommand>& dst) {
    auto hexBackend = static_cast<HexagonBackend*>(backend());
    const bool gdr = mParam.type == DSP_LINEAR_ATTN_GATED_DELTA;
    if (!_tensorsSupported(mParam, inputs, outputs)) {
        MNN_ERROR("[Hexagon] LinearAttention: unsupported tensor layout (needs packed C4, batch 1)\n");
        return NOT_SUPPORT;
    }
    const int L = inputs[0]->length(0);
    if (mParam.convDim != 0 && (mParam.convDim != inputs[0]->length(1) || mParam.kernelSize != inputs[3]->length(2))) {
        return NOT_SUPPORT; // the persistent state was sized for another shape
    }
    mParam.convDim = inputs[0]->length(1);
    mParam.kernelSize = inputs[3]->length(2);
    mParam.seqLen = L;
    const int blocks = gdr ? mParam.convDim / 64 : mParam.convDim / 192;

    // Persistent buffers: allocated once and shared with the clones.
    auto acquireStatic = [&](std::shared_ptr<Tensor>& t, size_t bytes) {
        if (t != nullptr) {
            return true;
        }
        t.reset(Tensor::createDevice<int8_t>({(int)std::max<size_t>(bytes, 128)}));
        if (!backend()->onAcquireBuffer(t.get(), Backend::STATIC)) {
            t.reset();
            return false;
        }
        mState->allocBackend = backend();
        mState->needZero = true;
        return true;
    };
    if (!acquireStatic(mState->convState, (size_t)blocks * (mParam.kernelSize - 1) * 64 * sizeof(float))) {
        return OUT_OF_MEMORY;
    }
    if (gdr && !acquireStatic(mState->rnnState,
                              (size_t)mParam.numVHeads * mParam.headKDim * mParam.headVDim * sizeof(float))) {
        return OUT_OF_MEMORY;
    }
    if (mParam.gateFold && mState->gateConst == nullptr) {
        mState->gateConst.reset(Tensor::createDevice<float>({(int)mGateConstHost.size()}));
        if (!backend()->onAcquireBuffer(mState->gateConst.get(), Backend::STATIC)) {
            mState->gateConst.reset();
            return OUT_OF_MEMORY;
        }
        ::memcpy(HexagonBackend::getPtr(mState->gateConst.get()), mGateConstHost.data(),
                 mGateConstHost.size() * sizeof(float));
        hexBackend->markHostInput(mState->gateConst.get());
        mState->allocBackend = backend();
    }

    // Zeroing commands, issued from onExecute when the sequence restarts.
    mZeroCmd.clear();
    for (auto state : {mState->convState.get(), mState->rnnState.get()}) {
        if (state == nullptr) {
            continue;
        }
        int size = (int)state->elementSize();
        mZeroCmd.emplace_back();
        mZeroCmd.back().build(hexBackend, DSP_OP_ZERO, &size, sizeof(size), {}, {HexagonBackend::getDevicePtr(state)},
                              {}, {state});
    }

    mWorkspace.reset();
    if (gdr) {
        const size_t bytes = (size_t)L * mParam.convDim * sizeof(float) +
                             (size_t)DSP_LINEAR_ATTN_MAX_WORKERS * DSP_LINEAR_ATTN_SCRATCH_BYTES;
        mWorkspace.reset(Tensor::createDevice<int8_t>({(int)bytes}));
        if (!backend()->onAcquireBuffer(mWorkspace.get(), Backend::DYNAMIC)) {
            mWorkspace.reset();
            return OUT_OF_MEMORY;
        }
    }
    const std::pair<int, int> none = {-1, 0};
    std::vector<std::pair<int, int>> inputFds = {
        HexagonBackend::getDevicePtr(inputs[0]),
        gdr ? HexagonBackend::getDevicePtr(inputs[1]) : none,
        gdr ? HexagonBackend::getDevicePtr(inputs[2]) : none,
        HexagonBackend::getDevicePtr(inputs[3]),
        mState->gateConst ? HexagonBackend::getDevicePtr(mState->gateConst.get()) : none,
    };
    std::vector<std::pair<int, int>> outputFds = {
        HexagonBackend::getDevicePtr(outputs[0]),
        HexagonBackend::getDevicePtr(mState->convState.get()),
        gdr ? HexagonBackend::getDevicePtr(mState->rnnState.get()) : none,
        gdr ? HexagonBackend::getDevicePtr(mWorkspace.get()) : none,
    };
    std::vector<Tensor*> cmdInputs = {inputs[0], gdr ? inputs[1] : nullptr, gdr ? inputs[2] : nullptr, inputs[3],
                                      mState->gateConst.get()};
    std::vector<Tensor*> cmdOutputs = {outputs[0], mState->convState.get(), mState->rnnState.get(), mWorkspace.get()};
    dst.emplace_back();
    dst.back().build(hexBackend, DSP_OP_LINEAR_ATTN, &mParam, sizeof(mParam), inputFds, outputFds, cmdInputs,
                     cmdOutputs);
    if (mWorkspace) {
        backend()->onReleaseBuffer(mWorkspace.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

ErrorCode HexagonLinearAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (!mValid) {
        return NOT_SUPPORT;
    }
    // Same decisions as CPULinearAttention (checked here because onResize may be skipped). The recurrent
    // state folds every past token, so it can only continue or reset: erasing part of the history is refused.
    if (mMeta != nullptr && mMeta->remove > 0 && (mMeta->previous != mMeta->remove || mMeta->n_reserve > 0)) {
        MNN_ERROR("[Hexagon] LinearAttention: cannot erase part of the %zu-token history; reset all of it instead\n",
                  mMeta->previous);
        return NOT_SUPPORT;
    }
    const bool restart = mMeta == nullptr ? mParam.seqLen > 1 : mMeta->previous == mMeta->remove;
    if (mState->needZero || restart) {
        for (auto& cmd : mZeroCmd) {
            if (cmd.execute() != 0) {
                return OUT_OF_MEMORY;
            }
        }
        mState->needZero = false;
    }
    return HexagonExecution::onExecute(inputs, outputs);
}

} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
