//
//  dflash.cpp
//
//  Created by MNN on 2025/06/09.
//
//  DFlash: Block Diffusion based speculative decoding
//  Unlike MTP/Eagle, DFlash uses non-causal (bidirectional) attention
//  and generates an entire block of draft tokens in a single forward pass.
//

#include <thread>
#include "generate.hpp"
#include "core/MNNFileUtils.h"
#include "llm/attention_mask.hpp"


using namespace MNN::Express;
namespace MNN {
namespace Transformer {


void MarkovSweepPool::ensureStarted(int nWorkers) {
    if (mWorkerCount == nWorkers) {
        return;
    }
    shutdown();
    mWorkerCount = nWorkers;
    mWorkers.reserve(nWorkers > 0 ? nWorkers - 1 : 0);
    for (int t = 1; t < nWorkers; t++) {
        mWorkers.emplace_back([this, t] {
            int seen = 0;
            for (;;) {
                std::function<void(int)> fn;
                {
                    std::unique_lock<std::mutex> lock(mMutex);
                    mCvWork.wait(lock, [this, &seen] { return mGeneration != seen; });
                    if (mGeneration < 0) {
                        return;   // shutdown
                    }
                    seen = mGeneration;
                    fn = mFn;
                }
                fn(t);
                {
                    std::lock_guard<std::mutex> lock(mMutex);
                    if (--mPending == 0) {
                        mCvDone.notify_all();
                    }
                }
            }
        });
    }
}

void MarkovSweepPool::shutdown() {
    if (mWorkers.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mGeneration = -1;
        mCvWork.notify_all();
    }
    for (auto& w : mWorkers) {
        w.join();
    }
    mWorkers.clear();
    mWorkerCount = 0;
}

DFlashGeneration::DFlashGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config)
    : Generation(llm, context) {
    mBlockSize = config->dflash_block_size();
    mMaskTokenId = config->dflash_mask_token_id();
    mShiftLabel = config->dflash_shift_label() ? 1 : 0;
    mMarkovFile = config->dflash_markov();
    mMarkovW1File = config->dflash_markov_w1();
    mMarkovW2File = config->dflash_markov_w2();
    // Bound the memory-bandwidth-limited host sweep instead of using every hardware thread.
    {
        const unsigned hw = std::thread::hardware_concurrency();
        mMarkovThreads = hw >= 12 ? 12 : (hw > 8 ? 8 : (hw > 2 ? 2 : 1));
    }
}

void DFlashGeneration::reset() {
    // Per turn: a stale draft KV cache no longer matches the reset target KV cache.
    mInitialized = false;
    mDraftKv.clear();
}

bool DFlashGeneration::load(Module::Config module_config) {
    // DSpark emits base logits; DFlash reuses the target's lm_head subgraph.
    std::string sharedLmHeadInput = mLlm->mConfig->dflash_shared_lmhead_input();

    // Loaded first: its output names give the draft layer count, which the draft graph's input list needs.
    {
        Module::Config kc;
        kc.shapeMutable = true;
        kc.rearrange = true;
        mKvMatModule.reset(Module::load({"new_context", "position_ids"}, {},
            mLlm->mConfig->dflash_kvmat().c_str(), mLlm->mRuntimeManager, &kc));
        if (mKvMatModule == nullptr) {
            MNN_ERROR("%s: failed to load %s. The draft attention consumes per-layer "
                      "kv_k_i/kv_v_i, so this module is mandatory; re-export the draft with "
                      "the current llmexport.\n", logTag(), mLlm->mConfig->dflash_kvmat().c_str());
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
            MNN_ERROR("%s: %s is missing output kv_%c_%d\n", logTag(), mLlm->mConfig->dflash_kvmat().c_str(),
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
    mSharedLmHead = !sharedLmHeadInput.empty();
    // Keep the DSpark input signature consistent with the exporter.
    if (!mSharedLmHead) {
        dflashInputNames.push_back("block_ids");
    }
    std::vector<std::string> dflashOutputNames{mSharedLmHead ? "hidden_states" : "block_logits"};
    mDFlashModule.reset(Module::load(
        dflashInputNames, dflashOutputNames,
        mLlm->mConfig->dflash_model().c_str(),
        mLlm->mRuntimeManager, &module_config));
    if (mDFlashModule == nullptr) {
        MNN_ERROR("%s: failed to load draft %s with %d per-layer KV inputs (output '%s'). "
                  "%s and the draft must come from the same export.\n",
                  logTag(), mLlm->mConfig->dflash_model().c_str(), nKvLayers, dflashOutputNames[0].c_str(),
                  mLlm->mConfig->dflash_kvmat().c_str());
        mKvMatModule.reset();
        return false;
    }

    // lm_head module
    // Reuse the target's lm_head: load the subgraph {sharedLmHeadInput -> logits} from llm.mnn.
    // base + rearrange=true shares the target's lm_head weights via cloneBaseExecution
    // (rearrange must be true, otherwise base is silently ignored).
    if (mSharedLmHead) {
        Module::Config lm_cfg;
        lm_cfg.shapeMutable = true;
        lm_cfg.rearrange = true;
        lm_cfg.base = mLlm->mModule.get();
        std::vector<std::string> lmInputNames{sharedLmHeadInput};
        std::vector<std::string> lmOutputNames{"logits"};
        mLmHeadModule.reset(Module::load(
            lmInputNames, lmOutputNames,
            mLlm->mConfig->llm_model().c_str(),
            mLlm->mRuntimeManager, &lm_cfg));
        if (mLmHeadModule == nullptr) {
            MNN_ERROR("%s: failed to load shared lm_head subgraph {%s -> logits} from %s. "
                      "Set dflash_shared_lmhead_input to the target's post-final-norm tensor name.\n",
                      logTag(), sharedLmHeadInput.c_str(), mLlm->mConfig->llm_model().c_str());
            return false;
        }
        MNN_PRINT("%s: lm_head shared from target %s (subgraph %s -> logits)\n",
                  logTag(), mLlm->mConfig->llm_model().c_str(), sharedLmHeadInput.c_str());
    }

    // Session runtime, so hidden_states -> fc -> context_hidden -> draft never leaves the device.
    {
        Module::Config fc_config;
        fc_config.shapeMutable = true;
        fc_config.rearrange = true;
        std::vector<std::string> fcInputNames{"target_hidden"};
        std::vector<std::string> fcOutputNames{"context_hidden"};
        mFcModule.reset(Module::load(
            fcInputNames, fcOutputNames,
            mLlm->mConfig->dflash_fc().c_str(),
            mLlm->mRuntimeManager, &fc_config));
        if (mFcModule == nullptr) {
            MNN_ERROR("%s: failed to load %s (every draft step projects target hidden "
                      "states through it, so this module is mandatory)\n",
                      logTag(), mLlm->mConfig->dflash_fc().c_str());
            return false;
        }
    }

    // Split Markov modules keep the coefficient gather and quantized bias projection on device.
    if (!mMarkovW1File.empty() && !mMarkovW2File.empty()) {
        Module::Config mkConfig;
        mkConfig.shapeMutable = true;
        mMarkovW1Module.reset(Module::load({"prev_ids"}, {"coef"},
            mMarkovW1File.c_str(), mLlm->mRuntimeManager, &mkConfig));
        if (mMarkovW1Module == nullptr) {
            MNN_ERROR("%s: failed to load %s (the draft's Markov head is mandatory for "
                      "the model this graph was exported from)\n", logTag(), mMarkovW1File.c_str());
            return false;
        }
        mMarkovW2Module.reset(Module::load({"coef"}, {"bias"},
            mMarkovW2File.c_str(), mLlm->mRuntimeManager, &mkConfig));
        if (mMarkovW2Module == nullptr) {
            MNN_ERROR("%s: failed to load %s (the draft's Markov head is mandatory for "
                      "the model this graph was exported from)\n", logTag(), mMarkovW2File.c_str());
            return false;
        }
        // Shapes come from one probe: an all-zero token still costs a gather plus a GEMV, and
        // doing it here keeps the chain itself free of first-call special cases.
        {
            int prevProbe = 0;
            auto prevVar = _Const(&prevProbe, {1}, NCHW, halide_type_of<int>());
            auto coefOut = mMarkovW1Module->onForward({prevVar});
            if (coefOut.empty() || coefOut[0] == nullptr) {
                MNN_ERROR("%s: %s did not produce a coefficient\n", logTag(), mMarkovW1File.c_str());
                return false;
            }
            mMarkovRank = coefOut[0]->getInfo()->size;
            auto biasOut = mMarkovW2Module->onForward({coefOut[0]});
            if (biasOut.empty() || biasOut[0] == nullptr) {
                MNN_ERROR("%s: %s did not produce a bias\n", logTag(), mMarkovW2File.c_str());
                return false;
            }
            mMarkovVocab = biasOut[0]->getInfo()->size;
        }
        MNN_PRINT("%s: Markov head on device (vocab=%d rank=%d)\n", logTag(), mMarkovVocab, mMarkovRank);
    } else if (!mMarkovFile.empty()) {
        Module::Config mkConfig;
        std::shared_ptr<Module> tables(Module::load({"dummy"}, {"w1", "w2"},
            mMarkovFile.c_str(), mLlm->mRuntimeManager, &mkConfig));
        if (tables == nullptr) {
            MNN_ERROR("%s: failed to load %s (the draft's Markov head is mandatory for "
                      "the model this graph was exported from)\n", logTag(), mMarkovFile.c_str());
            return false;
        }
        static float sDummyValue = 0.0f;
        auto dummy = _Const(&sDummyValue, {1}, NCHW, halide_type_of<float>());
        auto outs = tables->onForward({dummy});
        if (outs.size() < 2) {
            MNN_ERROR("%s: %s did not yield the Markov tables\n", logTag(), mMarkovFile.c_str());
            return false;
        }
        const auto w1Info = outs[0]->getInfo();
        const auto w2Info = outs[1]->getInfo();
        mMarkovRank = w1Info->dim[0];
        mMarkovVocab = w1Info->dim[1];
        if (w2Info->dim[0] != mMarkovRank || w2Info->dim[1] != mMarkovVocab) {
            MNN_ERROR("%s: Markov tables in %s have mismatched shapes\n", logTag(), mMarkovFile.c_str());
            return false;
        }
        const float* w1Ptr = outs[0]->readMap<float>();
        const float* w2Ptr = outs[1]->readMap<float>();
        if (w1Ptr == nullptr || w2Ptr == nullptr) {
            MNN_ERROR("%s: Markov tables in %s could not be read\n", logTag(), mMarkovFile.c_str());
            return false;
        }
        mMarkovW1.assign(w1Ptr, w1Ptr + (size_t)mMarkovVocab * mMarkovRank);
        // Host fallback path: one symmetric int8 scale per rank row keeps the rounding well
        // under the noise of a rank-256 sum feeding a 150k-way argmax.
        {
            const size_t n = (size_t)mMarkovVocab * mMarkovRank;
            mMarkovW2q.assign(n, 0);
            mMarkovW2Scale.assign(mMarkovRank, 1.0f);
            for (int k = 0; k < mMarkovRank; k++) {
                const float* row = w2Ptr + (size_t)k * mMarkovVocab;
                float amax = 0.0f;
                for (int v = 0; v < mMarkovVocab; v++) {
                    const float a = row[v] < 0 ? -row[v] : row[v];
                    amax = a > amax ? a : amax;
                }
                const float scale = amax > 0 ? amax / 127.0f : 1.0f;
                mMarkovW2Scale[k] = scale;
                const float inv = 1.0f / scale;
                int8_t* dst = mMarkovW2q.data() + (size_t)k * mMarkovVocab;
                for (int v = 0; v < mMarkovVocab; v++) {
                    float q = row[v] * inv;
                    q = q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q);
                    dst[v] = (int8_t)(q >= 0 ? q + 0.5f : q - 0.5f);
                }
            }
        }
        outs.clear();
        tables.reset();
        MNN_PRINT("%s: Markov tables loaded to host (vocab=%d rank=%d)\n", logTag(), mMarkovVocab, mMarkovRank);
    }

    mHiddenStateIndex = mLlm->getOutputIndex("hidden_states");
    MNN_PRINT("%s: block_size=%d shift_label=%d (draft row %s predicts slot i), draft layers=%d\n",
              logTag(), mBlockSize, mShiftLabel, mShiftLabel ? "i-1" : "i", nKvLayers);

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
        (const uint8_t*)buf.data(), buf.size(), mLlm->mRuntimeManager, &cc));
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

    if (mDraftKv.empty()) {
        MNN_ERROR("%s: draft KV is empty at dflashForward\n", logTag());
        return nullptr;
    }
    int context_len = mDraftKv[0]->getInfo()->dim[1];
    int total_len = context_len + block_size;

    // Keep an explicit bidirectional plane for the cacheless draft attention.
    MaskRequest maskReq;
    maskReq.type       = AttentionMaskType::Bidirectional;
    maskReq.seqLen     = block_size;
    maskReq.kvSeqLen   = total_len;
    maskReq.rank       = 4;
    maskReq.forceDense = true;
    auto attention_mask = buildAttentionMask(maskReq, mLlm->mConfig->backend_type());

    // DSpark slices block positions from the full range; context keys are already rotated by kvmat.
    const int posLen = mSharedLmHead ? block_size : total_len;
    // DSpark tables start at the context's own row 0; DFlash's block starts at context_len.
    const int posStart = mSharedLmHead ? context_len : 0;
    auto position_ids = _Input({1, posLen}, NCHW, halide_type_of<int>());
    auto posPtr = position_ids->writeMap<int>();
    for (int i = 0; i < posLen; i++) {
        posPtr[i] = posStart + i;
    }

    // Forward through DFlash module
    std::vector<VARP> inputs{noise_embedding};
    inputs.insert(inputs.end(), mDraftKv.begin(), mDraftKv.end());
    inputs.insert(inputs.end(), {attention_mask, position_ids});
    if (!mSharedLmHead) {
        // Retain the exporter's block_ids input; the Markov chain is evaluated separately.
        auto idsVar = _Input({1, block_size}, NCHW, halide_type_of<int>());
        auto idsPtr = idsVar->writeMap<int>();
        ::memcpy(idsPtr, block_ids.data(), block_size * sizeof(int));
        inputs.push_back(idsVar);
    }
    auto outputs = mDFlashModule->onForward(inputs);

    // A DFlash draft has no lm_head: run outputs[0] (hidden_states) through the target's
    // lm_head subgraph. A DSpark draft already emitted the block logits.
    auto draft_out = outputs[0];
    if (!mSharedLmHead) {
        return draft_out;
    }
    // The subgraph feeds the lm_head Convolution directly (4D NCHW); reshaping [1, block, hidden] to [block, hidden, 1, 1] is a numerical no-op.
    auto hsInfo = draft_out->getInfo();
    VARP lm_input = _Reshape(draft_out, {hsInfo->dim[0] * hsInfo->dim[1], hsInfo->dim[2], 1, 1});
    return mLmHeadModule->onForward({lm_input})[0];
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
    if (mArgmaxModule == nullptr) {
        auto lv = _Input({1, -1, -1}, NCHW, halide_type_of<float>());
        lv->setName("logits");
        auto iv = _ArgMax(_Reshape(lv, {-1, vocab}), 1);
        iv->setName("pidx");
        auto buf = Variable::save({iv});
        Module::Config ac;
        ac.shapeMutable = true;
        mArgmaxModule.reset(Module::load({"logits"}, {"pidx"},
            (const uint8_t*)buf.data(), buf.size(), mLlm->mRuntimeManager, &ac));
        if (mArgmaxModule == nullptr) {
            MNN_ERROR("%s: failed to build the on-device argmax module (vocab=%d)\n", logTag(), vocab);
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return false;
        }
        MNN_PRINT("%s: on-device argmax on (vocab=%d)\n", logTag(), vocab);
    }
    auto idx = mArgmaxModule->onForward({logits})[0];
    ::memcpy(out.data(), idx->readMap<int>(), rows * sizeof(int));
    return true;
}

// argmax(base + bias) as one on-device pass: the Markov chain never needs the 600 KB bias on
// the host, only the winning index.
bool DFlashGeneration::addArgmax(VARP base, VARP bias, int& outIdx) {
    if (base == nullptr || base->getInfo() == nullptr || bias == nullptr || bias->getInfo() == nullptr) {
        return false;
    }
    const int vocab = base->getInfo()->size;
    if (mAddArgmaxModule == nullptr) {
        auto bv = _Input({1, vocab}, NCHW, halide_type_of<float>());
        bv->setName("base");
        auto dv = _Input({1, vocab}, NCHW, halide_type_of<float>());
        dv->setName("bias");
        auto iv = _ArgMax(_Add(bv, dv), 1);
        iv->setName("pidx");
        auto buf = Variable::save({iv});
        Module::Config ac;
        ac.shapeMutable = true;
        mAddArgmaxModule.reset(Module::load({"base", "bias"}, {"pidx"},
            (const uint8_t*)buf.data(), buf.size(), mLlm->mRuntimeManager, &ac));
        if (mAddArgmaxModule == nullptr) {
            MNN_ERROR("%s: failed to build the on-device add+argmax module (vocab=%d)\n", logTag(), vocab);
            return false;
        }
    }
    auto out = mAddArgmaxModule->onForward({base, bias});
    if (out.empty() || out[0] == nullptr) {
        MNN_ERROR("%s: addArgmax produced nothing\n", logTag());
        return false;
    }
    const int* p = out[0]->readMap<int>();
    if (p == nullptr) {
        return false;
    }
    outIdx = *p;
    return true;
}

// DSpark walks the Markov chain: slot k's bias uses the token sampled at slot k-1, so each
// row's argmax must be taken after the previous row's token is known. Slot 0 is biased by the
// anchor (block_ids[0]); the rest by this forward's own predictions.
bool DFlashGeneration::markovSlot(int prev, VARP base, int& outIdx) {
    if (prev < 0 || prev >= mMarkovVocab) {
        prev = 0;
    }
    auto prevVar = _Const(&prev, {1}, NCHW, halide_type_of<int>());
    auto coefOut = mMarkovW1Module->onForward({prevVar});
    if (coefOut.empty() || coefOut[0] == nullptr) {
        return false;
    }
    auto biasOut = mMarkovW2Module->onForward({coefOut[0]});
    if (biasOut.empty() || biasOut[0] == nullptr) {
        return false;
    }
    return addArgmax(base, biasOut[0], outIdx);
}

bool DFlashGeneration::draftArgmaxChain(VARP logits, const std::vector<int>& block_ids,
                                        std::vector<int>& out) {
    const auto info = logits->getInfo();
    const int rows = info->dim[1];
    const int vocab = info->dim[2];
    if (mMarkovRank <= 0 || vocab != mMarkovVocab || (size_t)info->size < (size_t)rows * vocab) {
        MNN_ERROR("%s: draft logits %dx%d do not match the Markov tables (vocab=%d)\n",
                  logTag(), rows, vocab, mMarkovVocab);
        return false;
    }
    out.assign(rows, 0);
    int prev = block_ids.empty() ? 0 : block_ids[0];

    // Each Markov slot returns only its winning index rather than its vocab-wide bias.
    if (mMarkovW1Module != nullptr && mMarkovW2Module != nullptr) {
        // Materialise the block once so the slots below gather from computed data; without this
        // every slot re-runs the draft tail.
        auto block = _Reshape(logits, {rows, vocab});
        block->readMap<float>();
        for (int r = 0; r < rows; r++) {
            int rowIdx = r;
            auto rowVar = _Const(&rowIdx, {1}, NCHW, halide_type_of<int>());
            auto baseR = _Reshape(_GatherV2(block, rowVar, _Scalar<int>(0)), {1, vocab});
            int idx = 0;
            if (!markovSlot(prev, baseR, idx)) {
                MNN_ERROR("%s: Markov slot %d failed on device\n", logTag(), r);
                return false;
            }
            out[r] = idx;
            prev = idx;
        }
        return true;
    }

    // Host fallback for the older single-file tables.
    const float* src = logits->readMap<float>();
    if (src == nullptr) {
        return false;
    }
    mMarkovBias.assign(vocab, 0.0f);
    std::vector<float>& bias = mMarkovBias;
    const int nThreads = mMarkovThreads;
    const int perThread = (vocab + nThreads - 1) / nThreads;
    for (int r = 0; r < rows; r++) {
        if (prev < 0 || prev >= mMarkovVocab) {
            prev = 0;
        }
        auto sweep = [&](int beg, int end) {
            bool seeded = false;
            for (int k = 0; k < mMarkovRank; k++) {
                const float a = mMarkovW1[(size_t)k * mMarkovVocab + prev];
                if (a == 0.0f) {
                    continue;
                }
                const int8_t* w2Row = mMarkovW2q.data() + (size_t)k * mMarkovVocab;
                const float aw = a * mMarkovW2Scale[k];
                if (!seeded) {
                    for (int v = beg; v < end; v++) {
                        bias[v] = aw * (float)w2Row[v];
                    }
                    seeded = true;
                    continue;
                }
                for (int v = beg; v < end; v++) {
                    bias[v] += aw * (float)w2Row[v];
                }
            }
            if (!seeded) {
                std::fill(bias.begin() + beg, bias.begin() + end, 0.0f);
            }
        };
        if (nThreads == 1) {
            sweep(0, vocab);
        } else {
            mMarkovPool.run([&](int t) {
                const int beg = t * perThread;
                sweep(beg, std::min(vocab, beg + perThread));
            }, nThreads);
        }
        const float* row = src + (size_t)r * vocab;
        int best = 0;
        float bestVal = -INFINITY;
        for (int v = 0; v < vocab; v++) {
            const float x = row[v] + bias[v];
            if (x > bestVal) {
                bestVal = x;
                best = v;
            }
        }
        out[r] = best;
        prev = best;
    }
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
    // Defer the recurrent state update: part of this block may be rejected.
    mLlm->mMeta->spec_block = blockSize;
    auto verify_outputs = mLlm->forwardVec(block_ids);
    mLlm->mMeta->spec_block = 0;
    if (verify_outputs.empty() || verify_outputs.size() < 2) {
        vr.stop = true;
        return vr;
    }

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

    // Phase 7: Update KV cache - remove unaccepted tokens.
    // We fed blockSize tokens but only accepted vr.totalAccepted
    mLlm->mMeta->remove = blockSize - vr.totalAccepted;
    mLlm->updateContext(vr.totalAccepted, vr.totalAccepted);
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

        // Seed the context from prompt rows; the sampled anchor has not entered the target cache yet.
        auto fcOut = mFcModule->onForward({prev_hidden_states});
        if (fcOut.empty() || fcOut[0] == nullptr) {
            MNN_ERROR("%s: fc forward returned nothing\n", logTag());
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return;
        }
        appendDraftKv(fcOut[0], 0);
        mInitialized = true;

        // If max_token is 0, just do initialization and return
        if (max_token <= 0) {
            return;
        }
    }

    int len = 0;
    while (len < max_token) {
        if (mContext->status == LlmStatus::USER_CANCEL) {
            break;
        }
        // llm_bench requires decode timing in normal builds.
        MNN::Timer _t;

        // Phase 1: Build block [last_accepted_token, mask, mask, ..., mask]
        auto block_ids = buildBlock(mContext->current_token);

        // Phase 2: DFlash forward - get draft logits
        // The draft head and lm_head are block-shaped too, so they get the same backend
        // treatment; reset after phase 3, where the lazy draft_logits is materialized.
        mLlm->mMeta->spec_block = mBlockSize;
        VARP draft_logits = dflashForward(block_ids);
        if (draft_logits == nullptr) {
            mLlm->mMeta->spec_block = 0;
            break;
        }

        // Phase 3: Fast argmax for draft tokens (bypass expensive sampler)
        // draft_logits shape: [1, block_size, vocab_size]
        {
            std::vector<int> draftArgmax;
            // DSpark: slot k is biased by the token sampled at slot k-1, so the chain has to be
            // walked one slot at a time instead of one rowArgmax over the whole block.
            if (mMarkovRank > 0) {
                if (!draftArgmaxChain(draft_logits, block_ids, draftArgmax)) {
                    mLlm->mMeta->spec_block = 0;
                    break;
                }
            } else if (!rowArgmax(draft_logits, mBlockSize, draftArgmax)) {
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
        auto vr = verifyAcceptCommit(block_ids, mBlockSize, mHiddenStateIndex);
        len += vr.totalAccepted;
        mContext->decode_us += _t.durationInUs();
        if (vr.stop) {
            break;
        }

        // Phase 8: Extend the draft KV cache with the rows just committed by the target.
        int contextLen = mDraftKv[0]->getInfo()->dim[1];
        auto new_hidden_states = vr.newHiddenStates;
        if (vr.totalAccepted < mBlockSize) {
            // Keep the anchor and accepted draft rows; the corrected token has not been forwarded yet.
            auto hsInfo = new_hidden_states->getInfo();
            const int hiddenDim = hsInfo->dim[2];
            const int rows = std::min(vr.totalAccepted, (int)hsInfo->dim[1]);
            std::vector<int> starts = {0, 0, 0};
            std::vector<int> sizes = {1, rows, hiddenDim};
            new_hidden_states = _Slice(new_hidden_states,
                                       _Const(starts.data(), {3}, NHWC, halide_type_of<int>()),
                                       _Const(sizes.data(), {3}, NHWC, halide_type_of<int>()));
        }
        auto new_context = mFcModule->onForward({new_hidden_states})[0];
        appendDraftKv(new_context, contextLen);
    }

    if (len >= max_token) {
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
    }
}

} // namespace Transformer
} // namespace MNN
