//
//  dflash.cpp
//
//  Created by MNN on 2025/06/09.
//
//  DFlash: Block Diffusion based speculative decoding
//  Unlike MTP/Eagle, DFlash uses non-causal (bidirectional) attention
//  and generates an entire block of draft tokens in a single forward pass.
//

#include "generate.hpp"
#include "core/MNNFileUtils.h"
#include "llm/attention_mask.hpp"
#include <cstdlib>
#include <MNN/expr/Executor.hpp>
#include <MNN/Interpreter.hpp>


using namespace MNN::Express;
namespace MNN {
namespace Transformer {

// Debug helper: log each draft submodule forward's output count so an empty (NOT_SUPPORT)
// result can be attributed to a specific module. Returns true when the call produced outputs.
static bool dflashDbgOutputs(const char* tag, size_t n) {
    if (getenv("MNN_DFLASH_DBG_MOD")) {
        MNN_PRINT("[dflash-dbg] %s outputs=%zu\n", tag, n);
    }
    return n > 0;
}

// The fc projection (12800->2560 matmul) fails on the Hexagon backend at execute time for the
// per-step decode shape, so optionally run this small module on the CPU backend
// (MNN_DFLASH_FC_CPU=1) to keep the draft path correct; the target model stays on Hexagon.
static std::shared_ptr<MNN::Express::Executor::RuntimeManager> dflashCpuRuntime() {
    static std::shared_ptr<MNN::Express::Executor::RuntimeManager> rt = [] {
        MNN::ScheduleConfig cfg;
        cfg.type = MNN_FORWARD_CPU;
        cfg.numThread = 4;
        return std::shared_ptr<MNN::Express::Executor::RuntimeManager>(
            MNN::Express::Executor::RuntimeManager::createRuntimeManager(cfg));
    }();
    return rt;
}

// Optional (MNN_DFLASH_DRAFT_HEXAGON=1): run the draft modules on Hexagon while the target runs on
// CPU. Used to validate the draft's Hexagon path without competing with the 4B target for the
// ~4GB DSP rpcmem/mapping budget. Defined as a static member of DFlashGeneration, which is a friend
// of Llm, so it can read mConfig/mMeta/mRuntimeManager.
std::shared_ptr<MNN::Express::Executor::RuntimeManager> DFlashGeneration::draftRuntime(Llm* llm) {
    if (getenv("MNN_DFLASH_DRAFT_HEXAGON") == nullptr) {
        return llm->mRuntimeManager;
    }
    static std::shared_ptr<MNN::Express::Executor::RuntimeManager> rt = [llm] {
        MNN::ScheduleConfig cfg;
        cfg.type = MNN_FORWARD_HEXAGON;
        cfg.numThread = 4;
        auto mgr = std::shared_ptr<MNN::Express::Executor::RuntimeManager>(
            MNN::Express::Executor::RuntimeManager::createRuntimeManager(cfg));
        if (mgr != nullptr) {
            mgr->setHint(MNN::Interpreter::INIT_THREAD_NUMBER, 4);
            mgr->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);
            mgr->setHint(MNN::Interpreter::ATTENTION_OPTION, llm->mConfig->config_.value("attention_mode", 8));
            // The draft shares the session's KVMeta (see HexagonAttention's cross-attention note).
            mgr->setHintPtr(MNN::Interpreter::KVCACHE_INFO, llm->mMeta.get());
        }
        return mgr;
    }();
    return rt;
}


DFlashGeneration::DFlashGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config)
    : Generation(llm, context) {
    mBlockSize = config->dflash_block_size();
    mMaskTokenId = config->dflash_mask_token_id();
    mShiftLabel = config->dflash_shift_label() ? 1 : 0;
}

void DFlashGeneration::reset() {
    // Per turn: a stale draft KV cache no longer matches the reset target KV cache.
    mInitialized = false;
    mDraftKv.clear();
    // A new model/draft may expose a different vocab; drop the vocab-baked argmax graph.
    if (mArgmaxVocab > 0) {
        mArgmaxModule.reset();
        mArgmaxVocab = 0;
    }
}

bool DFlashGeneration::load(Module::Config module_config) {
    // lm_head resolution: the draft always reuses the target's lm_head.
    std::string sharedLmHeadInput = mLlm->mConfig->dflash_shared_lmhead_input();
    if (sharedLmHeadInput.empty()) {
        MNN_ERROR("DFlash: dflash_shared_lmhead_input is empty; the draft has no lm_head of its own. "
                  "Re-export the model with the current llmexport.\n");
        return false;
    }

    // Draft modules run on `draftRt`: the target's runtime by default, or a dedicated Hexagon
    // runtime when MNN_DFLASH_DRAFT_HEXAGON=1 (target then runs on CPU, e.g. backend_type=cpu).
    const bool draftOnHexagon = getenv("MNN_DFLASH_DRAFT_HEXAGON") != nullptr;
    auto draftRt = draftOnHexagon ? draftRuntime(mLlm) : mLlm->mRuntimeManager;

    // Loaded first: its output names give the draft layer count, which the draft graph's input list needs.
    {
        Module::Config kc;
        kc.shapeMutable = true;
        kc.rearrange = true;
        mKvMatModule.reset(Module::load({"new_context", "position_ids"}, {},
            mLlm->mConfig->dflash_kvmat().c_str(), draftRt, &kc));
        if (mKvMatModule == nullptr) {
            MNN_ERROR("DFlash: failed to load %s. The draft attention consumes per-layer "
                      "kv_k_i/kv_v_i, so this module is mandatory; re-export the draft with "
                      "the current llmexport.\n", mLlm->mConfig->dflash_kvmat().c_str());
            return false;
        }
    }
    const auto& kvOutNames = mKvMatModule->getInfo()->outputNames;
    int nKvLayers = 0;
    for (const auto& n : kvOutNames) {
        if (n.compare(0, 5, "kv_k_") == 0) {
            nKvLayers++;
        }
    }
    mKvMatSlot.assign(2 * nKvLayers, -1);
    for (int slot = 0; slot < (int)kvOutNames.size(); slot++) {
        const auto& n = kvOutNames[slot];
        bool isK = n.compare(0, 5, "kv_k_") == 0;
        if (!isK && n.compare(0, 5, "kv_v_") != 0) {
            continue;
        }
        int layer = atoi(n.c_str() + 5);
        if (layer < 0 || layer >= nKvLayers) {
            continue;
        }
        mKvMatSlot[2 * layer + (isK ? 0 : 1)] = slot;
    }
    for (int i = 0; i < (int)mKvMatSlot.size(); i++) {
        if (mKvMatSlot[i] < 0) {
            MNN_ERROR("DFlash: %s is missing output kv_%c_%d\n", mLlm->mConfig->dflash_kvmat().c_str(),
                      (i % 2) ? 'v' : 'k', i / 2);
            return false;
        }
    }

    // Load dflash main module
    std::vector<std::string> dflashInputNames{"noise_embedding"};
    for (int i = 0; i < nKvLayers; i++) {
        dflashInputNames.push_back("kv_k_" + std::to_string(i));
        dflashInputNames.push_back("kv_v_" + std::to_string(i));
    }
    dflashInputNames.insert(dflashInputNames.end(), {"attention_mask", "position_ids"});
    std::vector<std::string> dflashOutputNames{"hidden_states"};
    mDFlashModule.reset(Module::load(
        dflashInputNames, dflashOutputNames,
        mLlm->mConfig->dflash_model().c_str(),
        draftRt, &module_config));
    if (mDFlashModule == nullptr) {
        MNN_ERROR("DFlash: failed to load draft %s with %d per-layer KV inputs (output '%s'). "
                  "%s and the draft must come from the same export.\n",
                  mLlm->mConfig->dflash_model().c_str(), nKvLayers, dflashOutputNames[0].c_str(),
                  mLlm->mConfig->dflash_kvmat().c_str());
        mKvMatModule.reset();
        return false;
    }

    // lm_head module
    // Reuse the target's lm_head: load the subgraph {sharedLmHeadInput -> logits} from llm.mnn.
    // base + rearrange=true shares the target's lm_head weights via cloneBaseExecution
    // (rearrange must be true, otherwise base is silently ignored).
    {
        Module::Config lm_cfg;
        lm_cfg.shapeMutable = true;
        lm_cfg.rearrange = true;
        // Weight sharing via `base` requires the same backend; skip it when the draft has its own runtime.
        lm_cfg.base = draftOnHexagon ? nullptr : mLlm->mModule.get();
        std::vector<std::string> lmInputNames{sharedLmHeadInput};
        std::vector<std::string> lmOutputNames{"logits"};
        mLmHeadModule.reset(Module::load(
            lmInputNames, lmOutputNames,
            mLlm->mConfig->llm_model().c_str(),
            draftRt, &lm_cfg));
        if (mLmHeadModule == nullptr) {
            MNN_ERROR("DFlash: failed to load shared lm_head subgraph {%s -> logits} from %s. "
                      "Set dflash_shared_lmhead_input to the target's post-final-norm tensor name.\n",
                      sharedLmHeadInput.c_str(), mLlm->mConfig->llm_model().c_str());
            return false;
        }
        MNN_PRINT("DFlash: lm_head shared from target %s (subgraph %s -> logits)\n",
                  mLlm->mConfig->llm_model().c_str(), sharedLmHeadInput.c_str());
    }

    // Session runtime, so hidden_states -> fc -> context_hidden -> draft never leaves the device.
    {
        Module::Config fc_config;
        fc_config.shapeMutable = true;
        fc_config.rearrange = true;
        std::vector<std::string> fcInputNames{"target_hidden"};
        std::vector<std::string> fcOutputNames{"context_hidden"};
        auto fcRuntime = (getenv("MNN_DFLASH_FC_CPU") != nullptr) ? dflashCpuRuntime() : draftRt;
        mFcModule.reset(Module::load(
            fcInputNames, fcOutputNames,
            mLlm->mConfig->dflash_fc().c_str(),
            fcRuntime, &fc_config));
        if (mFcModule == nullptr) {
            MNN_ERROR("DFlash: failed to load %s (every draft step projects target hidden "
                      "states through it, so this module is mandatory)\n",
                      mLlm->mConfig->dflash_fc().c_str());
            return false;
        }
    }

    mHiddenStateIndex = mLlm->getOutputIndex("hidden_states");
    MNN_PRINT("DFlash: block_size=%d shift_label=%d (draft row %s predicts slot i), draft layers=%d\n",
              mBlockSize, mShiftLabel, mShiftLabel ? "i-1" : "i", nKvLayers);

    // Disable thinking mode for better draft acceptance rate.
    // Qwen3's chat template enables thinking by default, generating unpredictable
    // <think>...</think> tokens that the draft model cannot predict well.
    // Setting enable_thinking=false via jinja context skips the <think> prefix.
    mLlm->set_config("{\"jinja\": {\"context\": {\"enable_thinking\": false}}}");
    return true;
}


// Lazy: the draft's kv_heads/head_dim are only known from the first kvmat forward's output shape.
void DFlashGeneration::ensureKvConcatModule(int kvHeads, int headDim) {
    if (mKvConcatModule) {
        return;
    }
    std::vector<VARP> outputs;
    std::vector<std::string> inNames, outNames;
    for (int i = 0; i < (int)mKvMatSlot.size(); i++) {
        auto oldV = _Input({1, -1, kvHeads, headDim}, NCHW, halide_type_of<float>());
        auto newV = _Input({1, -1, kvHeads, headDim}, NCHW, halide_type_of<float>());
        inNames.push_back("old_" + std::to_string(i));
        inNames.push_back("new_" + std::to_string(i));
        oldV->setName(inNames[inNames.size() - 2]);
        newV->setName(inNames.back());
        auto o = _Concat({oldV, newV}, 1);
        outNames.push_back("out_" + std::to_string(i));
        o->setName(outNames.back());
        outputs.push_back(o);
    }
    auto buf = Variable::save(outputs);
    Module::Config cc;
    cc.shapeMutable = true;
    mKvConcatModule.reset(Module::load(inNames, outNames,
        (const uint8_t*)buf.data(), buf.size(), draftRuntime(mLlm), &cc));
}

// newContext rows take draft-local RoPE positions startPos..startPos+rows-1 (0-based,
// NOT target-absolute); RoPE only sees q-k position deltas, so the numbering is equivalent.
void DFlashGeneration::appendDraftKv(VARP newContext, int startPos) {
    int rows = newContext->getInfo()->dim[1];
    std::vector<int> posData(rows);
    for (int i = 0; i < rows; i++) {
        posData[i] = startPos + i;
    }
    auto posV = _Const(posData.data(), {1, rows}, NCHW, halide_type_of<int>());
    auto kvOut = mKvMatModule->onForward({newContext, posV});
    if (!dflashDbgOutputs("mKvMatModule", kvOut.size())) {
        return;
    }
    if (mDraftKv.empty()) {
        mDraftKv.resize(mKvMatSlot.size());
        for (int i = 0; i < (int)mKvMatSlot.size(); i++) {
            mDraftKv[i] = kvOut[mKvMatSlot[i]];
        }
        return;
    }
    auto kdim = kvOut[mKvMatSlot[0]]->getInfo()->dim;
    ensureKvConcatModule(kdim[2], kdim[3]);
    std::vector<VARP> concatIn;
    concatIn.reserve(mDraftKv.size() * 2);
    for (int i = 0; i < (int)mDraftKv.size(); i++) {
        concatIn.push_back(mDraftKv[i]);
        concatIn.push_back(kvOut[mKvMatSlot[i]]);
    }
    auto concatOut = mKvConcatModule->onForward(concatIn);
    if (!dflashDbgOutputs("mKvConcatModule", concatOut.size())) {
        return;
    }
    for (int i = 0; i < (int)mDraftKv.size(); i++) {
        mDraftKv[i] = concatOut[i];
    }
}

VARP DFlashGeneration::dflashForward(const std::vector<int>& block_ids) {
    // Embed block tokens
    auto noise_embedding = mLlm->embedding(block_ids);
    // noise_embedding shape: [block_size, 1, hidden_size] -> reshape to [1, block_size, hidden_size]
    int block_size = static_cast<int>(block_ids.size());
    int hidden_size = mLlm->mConfig->hidden_size();
    noise_embedding = _Reshape(noise_embedding, {1, block_size, hidden_size});

    int context_len = mDraftKv[0]->getInfo()->dim[1];
    int total_len = context_len + block_size;

    // Everything attends to everything. Kept dense until omitting the plane is verified on a dflash model.
    MaskRequest maskReq;
    maskReq.type       = AttentionMaskType::Bidirectional;
    maskReq.seqLen     = block_size;
    maskReq.kvSeqLen   = total_len;
    maskReq.rank       = 4;
    maskReq.forceDense = true;
    auto attention_mask = buildAttentionMask(maskReq, mLlm->mConfig->backend_type());

    // Draft-local RoPE positions: committed row r sits at r, so block slot i sits at context_len + i.
    auto position_ids = _Input({1, block_size}, NCHW, halide_type_of<int>());
    auto posPtr = position_ids->writeMap<int>();
    for (int i = 0; i < block_size; i++) {
        posPtr[i] = context_len + i;
    }

    // Forward through DFlash module
    std::vector<VARP> inputs{noise_embedding};
    inputs.insert(inputs.end(), mDraftKv.begin(), mDraftKv.end());
    inputs.insert(inputs.end(), {attention_mask, position_ids});
    auto outputs = mDFlashModule->onForward(inputs);
    if (!dflashDbgOutputs("mDFlashModule", outputs.size())) {
        return nullptr;
    }

    // The draft has no lm_head: run outputs[0] (hidden_states) through the target's lm_head subgraph.
    auto hidden_states_out = outputs[0];
    // The subgraph feeds the lm_head Convolution directly (4D NCHW); reshaping [1, block, hidden] to [block, hidden, 1, 1] is a numerical no-op.
    auto hsInfo = hidden_states_out->getInfo();
    VARP lm_input = _Reshape(hidden_states_out, {hsInfo->dim[0] * hsInfo->dim[1], hsInfo->dim[2], 1, 1});
    auto lmOut = mLmHeadModule->onForward({lm_input});
    if (!dflashDbgOutputs("mLmHeadModule", lmOut.size())) {
        return nullptr;
    }
    return lmOut[0];
}

// Build the speculation block [current_token, mask, ..., mask].
std::vector<int> DFlashGeneration::buildBlock(int currentToken) const {
    std::vector<int> block_ids(mBlockSize, mMaskTokenId);
    block_ids[0] = currentToken;
    return block_ids;
}

bool DFlashGeneration::rowArgmax(VARP logits, int rows, std::vector<int>& out) {
    int vocab = logits->getInfo()->size / rows;
    out.resize(rows);
#ifdef DUMP_PROFILE_INFO
    MNN::Timer _t;
#endif
    // The reshape bakes `vocab` into the graph, so a different vocab (model swap / draft swap after
    // reset) must rebuild the module rather than reuse the old one.
#ifdef DUMP_PROFILE_INFO
    MNN::Timer _tBuild;
#endif
    if (mArgmaxModule == nullptr || mArgmaxVocab != vocab) {
        mArgmaxModule.reset();
        mArgmaxVocab = 0;
        auto lv = _Input({1, -1, -1}, NCHW, halide_type_of<float>());
        lv->setName("logits");
        auto iv = _ArgMax(_Reshape(lv, {-1, vocab}), 1);
        iv->setName("pidx");
        auto buf = Variable::save({iv});
        Module::Config ac;
        ac.shapeMutable = true;
        mArgmaxModule.reset(Module::load({"logits"}, {"pidx"},
            (const uint8_t*)buf.data(), buf.size(), draftRuntime(mLlm), &ac));
        if (mArgmaxModule == nullptr) {
            MNN_ERROR("DFlash: failed to build the on-device argmax module (vocab=%d)\n", vocab);
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return false;
        }
        mArgmaxVocab = vocab;
        MNN_PRINT("DFlash: on-device argmax on (vocab=%d)\n", vocab);
    }
#ifdef DUMP_PROFILE_INFO
    phase_argmax_build_us += _tBuild.durationInUs();
    MNN::Timer _tFwd;
#endif
    auto argmaxOut = mArgmaxModule->onForward({logits});
#ifdef DUMP_PROFILE_INFO
    phase_argmax_fwd_us += _tFwd.durationInUs();
    MNN::Timer _tMap;
#endif
    if (!dflashDbgOutputs("mArgmaxModule", argmaxOut.size()) || argmaxOut[0].get() == nullptr) {
        MNN_ERROR("DFlash: on-device argmax returned no output\n");
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    const int* idxPtr = argmaxOut[0]->readMap<int>();
    if (idxPtr == nullptr) {
        MNN_ERROR("DFlash: on-device argmax output is not mappable\n");
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    ::memcpy(out.data(), idxPtr, rows * sizeof(int));
#ifdef DUMP_PROFILE_INFO
    phase_argmax_map_us += _tMap.durationInUs();
    phase_argmax_us += _t.durationInUs();
#endif
    return true;
}

int DFlashGeneration::acceptDraftPrefix(const std::vector<int>& targetArgmax,
                                             const std::vector<int>& block_ids, int blockSize) {
    int acceptance_length = 0;
    for (int i = 0; i < blockSize - 1; i++) {
        int target_prediction = targetArgmax[i];
        if (target_prediction != block_ids[i + 1]) {
            mContext->current_token = target_prediction;
            break;
        }
        acceptance_length++;
        if (mLlm->is_stop(target_prediction)) {
            mContext->current_token = target_prediction;
            break;
        }
    }
    if (acceptance_length == blockSize - 1) {
        mContext->current_token = targetArgmax[blockSize - 1];
    }
    return acceptance_length;
}

DFlashGeneration::VerifyResult DFlashGeneration::verifyAcceptCommit(
        const std::vector<int>& block_ids, int blockSize, int hiddenStateIndex) {
    VerifyResult vr;
    // Phase 4: Verify entire block with target model
#ifdef DUMP_PROFILE_INFO
    MNN::Timer _t_verify;
#endif
    // Defer the recurrent state update: part of this block may be rejected.
    mLlm->mMeta->spec_block = blockSize;
    auto verify_outputs = mLlm->forwardVec(block_ids);
    dflashDbgOutputs("mLlm->forwardVec(verify)", verify_outputs.size());
    mLlm->mMeta->spec_block = 0;
    if (verify_outputs.empty() || verify_outputs.size() < 2) {
        vr.stop = true;
        return vr;
    }
#ifdef DUMP_PROFILE_INFO
    int64_t verify_fwd_elapsed = _t_verify.durationInUs();
    phase_verify_fwd_us += verify_fwd_elapsed;
#endif

    auto verify_logits = verify_outputs[0];
    vr.newHiddenStates = verify_outputs[hiddenStateIndex];

    // Phase 5: Greedy prefix matching - compare target model's predictions with draft
    int acceptance_length = 0;
    {
        std::vector<int> targetArgmax;
        if (!rowArgmax(verify_logits, blockSize, targetArgmax)) {
            // The block's rows already entered the target KV cache; drop them all.
            mLlm->mMeta->remove = blockSize;
            vr.stop = true;
            return vr;
        }
        acceptance_length = acceptDraftPrefix(targetArgmax, block_ids, blockSize);
    }

#ifdef DUMP_PROFILE_INFO
    MNN::Timer _t_match;
#endif

    // Phase 6: Accept tokens and update state.
    // We accept block_ids[1..acceptance_length] + current_token
    vr.totalAccepted = acceptance_length + 1;
    for (int i = 1; i <= acceptance_length; i++) {
        int token = block_ids[i];
        {
            std::lock_guard<std::mutex> _l(mContext->mutex);
            mContext->history_tokens.push_back(token);
            mContext->output_tokens.push_back(token);
        }
        if (nullptr != mContext->os) {
            *mContext->os << mLlm->tokenizer_decode(token) << std::flush;
        }
        if (mLlm->is_stop(token)) {
            vr.stop = true;
            break;
        }
    }
    if (!vr.stop) {
        // Add the corrected/next token
        {
            std::lock_guard<std::mutex> _l(mContext->mutex);
            mContext->history_tokens.push_back(mContext->current_token);
            mContext->output_tokens.push_back(mContext->current_token);
        }
        if (nullptr != mContext->os) {
            if (mLlm->is_stop(mContext->current_token)) {
                *mContext->os << mContext->end_with << std::flush;
                vr.stop = true;
            } else {
                *mContext->os << mLlm->tokenizer_decode(mContext->current_token) << std::flush;
            }
        }
    }

#ifdef DUMP_PROFILE_INFO
    int64_t match_elapsed = _t_match.durationInUs();
    phase_verify_match_us += match_elapsed;
    phase_verify_us += verify_fwd_elapsed + match_elapsed;
#endif

    // Phase 7: Update KV cache - remove unaccepted tokens.
    // We fed blockSize tokens but only accepted vr.totalAccepted
    mLlm->mMeta->remove = blockSize - vr.totalAccepted;
    mLlm->updateContext(vr.totalAccepted, vr.totalAccepted);
#ifdef DUMP_PROFILE_INFO
    spl_decode += blockSize;
    spl_accept += vr.totalAccepted;
    spl_count++;
#endif
    return vr;
}

void DFlashGeneration::generate(GenerationParams& param) {
    int max_token = param.max_new_tokens;

    // First-time initialization: sample the first token and materialize the draft KV
    if (!mInitialized) {
        VARP prev_hidden_states = param.outputs[mHiddenStateIndex];

        // Sample first token from prefill logits
        mContext->current_token = mLlm->sample(param.outputs[0], param.validLogitStart, param.validLogitSize);
        {
            std::lock_guard<std::mutex> _l(mContext->mutex);
            mContext->history_tokens.push_back(mContext->current_token);
            mContext->output_tokens.push_back(mContext->current_token);
        }
        mLlm->updateContext(0, 1);

        if (mLlm->is_stop(mContext->current_token)) {
            if (nullptr != mContext->os) {
                *mContext->os << mContext->end_with << std::flush;
            }
            return;
        }

        // Output first token
        if (nullptr != mContext->os) {
            *mContext->os << mLlm->tokenizer_decode(mContext->current_token) << std::flush;
        }

        // The only time the prompt is projected: positions are frozen from here on.
        auto fcPrefill = mFcModule->onForward({prev_hidden_states});
        if (!dflashDbgOutputs("mFcModule(prefill)", fcPrefill.size())) {
            return;
        }
        appendDraftKv(fcPrefill[0], 0);
        mInitialized = true;

        // If max_token is 0, just do initialization and return
        if (max_token <= 0) {
            return;
        }
    }

#ifdef DUMP_PROFILE_INFO
    // The verify/sample/spl accumulators are members, shared with verifyAcceptCommit().
    int64_t phase_draft_us = 0, phase_fc_us = 0;
    int64_t phase_fc_fwd_us = 0, phase_kvmat_us = 0;
#endif

    int len = 0;
    while (len < max_token) {
        if (mContext->status == LlmStatus::USER_CANCEL) {
            break;
        }
        MNN::Timer _t;

        // Phase 1: Build block [last_accepted_token, mask, mask, ..., mask]
        auto block_ids = buildBlock(mContext->current_token);

        // Phase 2: DFlash forward - get draft logits
        // The draft head and lm_head are block-shaped too, so they get the same backend
        // treatment; reset after phase 3, where the lazy draft_logits is materialized.
        mLlm->mMeta->spec_block = mBlockSize;
#ifdef DUMP_PROFILE_INFO
        MNN::Timer _t_draft;
#endif
        VARP draft_logits = dflashForward(block_ids);
        if (draft_logits == nullptr) {
            mLlm->mMeta->spec_block = 0;
            break;
        }
#ifdef DUMP_PROFILE_INFO
        phase_draft_us += _t_draft.durationInUs();
#endif

        // Phase 3: Fast argmax for draft tokens (bypass expensive sampler)
        // draft_logits shape: [1, block_size, vocab_size]
#ifdef DUMP_PROFILE_INFO
        MNN::Timer _t_sample;
#endif
        {
            std::vector<int> draftArgmax;
            if (!rowArgmax(draft_logits, mBlockSize, draftArgmax)) {
                mLlm->mMeta->spec_block = 0;
                break;
            }
            // Slot 0 is the accepted token; with shift_label row i-1 predicts slot i, else row i.
            for (int i = 1; i < mBlockSize; i++) {
                block_ids[i] = draftArgmax[i - mShiftLabel];
            }
        }
        mLlm->mMeta->spec_block = 0;

        // Phase 4-7: target verify + greedy accept + emit + KV rollback
#ifdef DUMP_PROFILE_INFO
        phase_sample_us += _t_sample.durationInUs();
#endif
        auto vr = verifyAcceptCommit(block_ids, mBlockSize, mHiddenStateIndex);
        len += vr.totalAccepted;
        mContext->decode_us += _t.durationInUs();
        if (vr.stop) {
            break;
        }

        // Phase 8: Extend the draft KV cache with the rows just committed by the target.
#ifdef DUMP_PROFILE_INFO
        MNN::Timer _t_fc;
#endif
        int contextLen = mDraftKv[0]->getInfo()->dim[1];
        auto new_hidden_states = vr.newHiddenStates;
        if (vr.totalAccepted < mBlockSize) {
            auto hsInfo = new_hidden_states->getInfo();
            int hiddenDim = hsInfo->dim[2];
            std::vector<int> starts = {0, 0, 0};
            std::vector<int> sizes = {1, vr.totalAccepted, hiddenDim};
            new_hidden_states = _Slice(new_hidden_states,
                                       _Const(starts.data(), {3}, NHWC, halide_type_of<int>()),
                                       _Const(sizes.data(), {3}, NHWC, halide_type_of<int>()));
        }
        auto fcOut = mFcModule->onForward({new_hidden_states});
        if (!dflashDbgOutputs("mFcModule(decode)", fcOut.size())) {
            break;
        }
        auto new_context = fcOut[0];
#ifdef DUMP_PROFILE_INFO
        phase_fc_fwd_us += _t_fc.durationInUs();
        MNN::Timer _t_kvmat;
#endif
        appendDraftKv(new_context, contextLen);
#ifdef DUMP_PROFILE_INFO
        phase_kvmat_us += _t_kvmat.durationInUs();
        phase_fc_us += _t_fc.durationInUs();   // _t_fc already spans fc_forward + kvmat
#endif
    }

    if (len >= max_token) {
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
    }

#ifdef DUMP_PROFILE_INFO
    float spl_accept_rate = spl_decode > 0 ? 100.0f * spl_accept / spl_decode : 0.0f;
    MNN_PRINT("\n============== DFlash Decoding Statistics Start ===============\n");
    MNN_PRINT("Block size: %d\n", mBlockSize);
    MNN_PRINT("Average accept rate: %.2f%%\n", spl_accept_rate);
    MNN_PRINT("Average accepted per step: %.2f\n", spl_count > 0 ? (float)spl_accept / spl_count : 0.0f);
    MNN_PRINT("Total steps: %d, Total tokens: %d\n", spl_count, spl_accept);
    MNN_PRINT("Phase timing (ms): draft=%.1f, sample=%.1f, verify=%.1f, fc=%.1f\n",
              phase_draft_us / 1000.0f, phase_sample_us / 1000.0f,
              phase_verify_us / 1000.0f, phase_fc_us / 1000.0f);
    MNN_PRINT("Per-step avg (ms): draft=%.1f, sample=%.1f, verify=%.1f, fc=%.1f\n",
              spl_count > 0 ? phase_draft_us / 1000.0f / spl_count : 0,
              spl_count > 0 ? phase_sample_us / 1000.0f / spl_count : 0,
              spl_count > 0 ? phase_verify_us / 1000.0f / spl_count : 0,
              spl_count > 0 ? phase_fc_us / 1000.0f / spl_count : 0);
    float total_ms = (phase_draft_us + phase_sample_us + phase_verify_us + phase_fc_us) / 1000.0f;
    MNN_PRINT("Argmax detail (per step ms): build=%.2f, onForward=%.2f, readMap=%.2f, total=%.2f\n",
              spl_count > 0 ? phase_argmax_build_us / 1000.0f / spl_count : 0,
              spl_count > 0 ? phase_argmax_fwd_us / 1000.0f / spl_count : 0,
              spl_count > 0 ? phase_argmax_map_us / 1000.0f / spl_count : 0,
              spl_count > 0 ? phase_argmax_us / 1000.0f / spl_count : 0);
    MNN_PRINT("Phase breakdown %%: draft=%.1f%%, sample=%.1f%%, verify=%.1f%%, fc=%.1f%%\n",
              total_ms > 0 ? 100.0f * phase_draft_us / 1000.0f / total_ms : 0,
              total_ms > 0 ? 100.0f * phase_sample_us / 1000.0f / total_ms : 0,
              total_ms > 0 ? 100.0f * phase_verify_us / 1000.0f / total_ms : 0,
              total_ms > 0 ? 100.0f * phase_fc_us / 1000.0f / total_ms : 0);
    MNN_PRINT("Verify detail (ms): forward=%.1f, match=%.1f\n",
              phase_verify_fwd_us / 1000.0f, phase_verify_match_us / 1000.0f);
    MNN_PRINT("FC detail (ms): fc_forward=%.1f, kv_materialize=%.1f\n",
              phase_fc_fwd_us / 1000.0f, phase_kvmat_us / 1000.0f);
    MNN_PRINT("Argmax total (ms): %.1f\n", phase_argmax_us / 1000.0f);
    float per_accepted_ms = spl_accept > 0 ? total_ms / spl_accept : 0;
    MNN_PRINT("Effective per-token cost: %.2f ms/token (vs AR baseline)\n", per_accepted_ms);
    MNN_PRINT("============== DFlash Decoding Statistics End =================\n");
#endif
}

} // namespace Transformer
} // namespace MNN
