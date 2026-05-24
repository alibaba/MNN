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

#define DFLASH_DEBUG 0

using namespace MNN::Express;
namespace MNN {
namespace Transformer {

DFlashGeneration::DFlashGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config)
    : Generation(llm, context) {
    mBlockSize = config->dflash_block_size();
    mMaskTokenId = config->dflash_mask_token_id();
}

void DFlashGeneration::load(Module::Config module_config) {
    // Check if separate lm_head model exists
    std::string lmheadPath = mLlm->mConfig->dflash_lmhead();
    // Guard against empty config value: base_dir + "" = directory path, which MNNFileExist would match
    bool hasSeparateLmHead =
        !lmheadPath.empty() && MNNFileExist(lmheadPath.c_str()) && !MNNDirExist(lmheadPath.c_str());

    // Load dflash main module
    std::vector<std::string> dflashInputNames{"noise_embedding", "context_hidden", "attention_mask", "q_position_ids", "k_position_ids"};
    std::vector<std::string> dflashOutputNames{hasSeparateLmHead ? "hidden_states" : "logits"};
    mDFlashModule.reset(Module::load(
        dflashInputNames, dflashOutputNames,
        mLlm->mConfig->dflash_model().c_str(),
        mLlm->mRuntimeManager, &module_config));

    // Load separate lm_head module if available (allows fp16 transformer + int4 lm_head)
    if (hasSeparateLmHead) {
        std::vector<std::string> lmInputNames{"hidden_states"};
        std::vector<std::string> lmOutputNames{"logits"};
        mLmHeadModule.reset(Module::load(
            lmInputNames, lmOutputNames,
            lmheadPath.c_str(),
            mLlm->mRuntimeManager, &module_config));
        MNN_PRINT("DFlash: loaded separate lm_head from %s\n", lmheadPath.c_str());
    }

    // Load fc module with dedicated CPU runtime to ensure fp32 precision
    // The fc layer has very high input dimension (num_layers * hidden_size),
    // which can cause NaN in fp16 dot products during prefill
    {
        // Create a dedicated CPU runtime for FC to guarantee fp32 execution
        ScheduleConfig fc_schedule;
        fc_schedule.type = MNN_FORWARD_CPU;
        fc_schedule.numThread = 4;
        BackendConfig fc_backend_config;
        fc_backend_config.precision = BackendConfig::Precision_High;
        fc_schedule.backendConfig = &fc_backend_config;
        mFcRuntimeManager.reset(Executor::RuntimeManager::createRuntimeManager(fc_schedule));
        mFcRuntimeManager->setHint(Interpreter::MEM_ALLOCATOR_TYPE, 0);

        Module::Config fc_config;
        fc_config.shapeMutable = true;
        fc_config.rearrange = true;
        std::vector<std::string> fcInputNames{"target_hidden"};
        std::vector<std::string> fcOutputNames{"context_hidden"};
        mFcModule.reset(Module::load(
            fcInputNames, fcOutputNames,
            mLlm->mConfig->dflash_fc().c_str(),
            mFcRuntimeManager, &fc_config));
        MNN_PRINT("DFlash: FC module loaded with dedicated CPU runtime (fp32)\n");
    }

    mHiddenStateIndex = mLlm->getOutputIndex("hidden_states");

    // Disable thinking mode for better draft acceptance rate.
    // Qwen3's chat template enables thinking by default, generating unpredictable
    // <think>...</think> tokens that the draft model cannot predict well.
    // Setting enable_thinking=false via jinja context skips the <think> prefix.
    mLlm->set_config("{\"jinja\": {\"context\": {\"enable_thinking\": false}}}");
}

VARP DFlashGeneration::fcForward(VARP hidden_states) {
#if DFLASH_DEBUG
    {
        auto hsInfo = hidden_states->getInfo();
        auto hsPtr = hidden_states->readMap<float>();
        int total = 1;
        for (auto d : hsInfo->dim) total *= d;
        float minV = hsPtr[0], maxV = hsPtr[0];
        int nanCount = 0, zeroCount = 0;
        for (int i = 0; i < total; i++) {
            if (std::isnan(hsPtr[i]) || std::isinf(hsPtr[i])) nanCount++;
            else {
                if (hsPtr[i] < minV) minV = hsPtr[i];
                if (hsPtr[i] > maxV) maxV = hsPtr[i];
                if (hsPtr[i] == 0.0f) zeroCount++;
            }
        }
        printf("fcForward input: dims=[");
        for (int d = 0; d < hsInfo->dim.size(); d++) {
            printf("%d%s", hsInfo->dim[d], d < hsInfo->dim.size()-1 ? ", " : "");
        }
        printf("], range=[%.4f, %.4f], nanCount=%d, zeroCount=%d/%d, type=%d\n",
               minV, maxV, nanCount, zeroCount, total, hsInfo->type.code);
        // Print first few values
        printf("  first 10 values: ");
        for (int i = 0; i < std::min(10, total); i++) printf("%.4f ", hsPtr[i]);
        printf("\n");
    }
#endif
    std::vector<VARP> inputs = {hidden_states};
    auto outputs = mFcModule->onForward(inputs);
    auto result = outputs[0];
#if DFLASH_DEBUG
    {
        auto rInfo = result->getInfo();
        auto rPtr = result->readMap<float>();
        int total = 1;
        for (auto d : rInfo->dim) total *= d;
        float minV = rPtr[0], maxV = rPtr[0];
        int nanCount = 0, zeroCount = 0;
        for (int i = 0; i < total; i++) {
            if (std::isnan(rPtr[i]) || std::isinf(rPtr[i])) nanCount++;
            else {
                if (rPtr[i] < minV) minV = rPtr[i];
                if (rPtr[i] > maxV) maxV = rPtr[i];
                if (rPtr[i] == 0.0f) zeroCount++;
            }
        }
        printf("fcForward output: dims=[");
        for (int d = 0; d < rInfo->dim.size(); d++) {
            printf("%d%s", rInfo->dim[d], d < rInfo->dim.size()-1 ? ", " : "");
        }
        printf("], range=[%.4f, %.4f], nanCount=%d, zeroCount=%d/%d, type=%d\n",
               minV, maxV, nanCount, zeroCount, total, rInfo->type.code);
        printf("  first 10 values: ");
        for (int i = 0; i < std::min(10, total); i++) printf("%.4f ", rPtr[i]);
        printf("\n");
    }
#endif
    // Sanitize: replace NaN values with 0 to prevent propagation
    auto info = result->getInfo();
    auto ptr = result->readMap<float>();
    int total = 1;
    for (auto d : info->dim) total *= d;
    int hiddenDim = info->dim[info->dim.size() - 1];
    int seqLen = total / hiddenDim;
    bool hasNaN = false;
    for (int i = 0; i < total; i++) {
        if (std::isnan(ptr[i]) || std::isinf(ptr[i])) {
            hasNaN = true;
#if DFLASH_DEBUG
            int pos = i / hiddenDim;
            int dim = i % hiddenDim;
            printf("  NaN at pos=%d, dim=%d (value=%f)\n", pos, dim, ptr[i]);
#endif
        }
    }
    if (hasNaN) {
        auto sanitized = _Input(info->dim, info->order, info->type);
        auto dst = sanitized->writeMap<float>();
        for (int i = 0; i < total; i++) {
            dst[i] = (std::isnan(ptr[i]) || std::isinf(ptr[i])) ? 0.0f : ptr[i];
        }
        return sanitized;
    }
    return result;
}

VARP DFlashGeneration::dflashForward(const std::vector<int>& block_ids, VARP context_hidden) {
    // Embed block tokens
    auto noise_embedding = mLlm->embedding(block_ids);
    // noise_embedding shape: [block_size, 1, hidden_size] -> reshape to [1, block_size, hidden_size]
    int block_size = static_cast<int>(block_ids.size());
    int hidden_size = mLlm->mConfig->hidden_size();
    noise_embedding = _Reshape(noise_embedding, {1, block_size, hidden_size});

    // context_hidden: [1, context_len, hidden_size] (already in correct shape)
    int context_len = context_hidden->getInfo()->dim[1];
    int total_len = context_len + block_size;

    // Non-causal attention mask: all zeros (everything attends to everything)
    auto attention_mask = _Input({1, 1, block_size, total_len}, NCHW, halide_type_of<float>());
    ::memset(attention_mask->writeMap<float>(), 0, block_size * total_len * sizeof(float));

    // Separate position IDs for Q (block only) and K (all positions)
    int pos_start = mContext->all_seq_len - context_len;

    // Q position IDs: positions for block tokens only [pos_start + context_len, ..., pos_start + total_len - 1]
    auto q_position_ids = _Input({1, block_size}, NCHW, halide_type_of<int>());
    auto qPosPtr = q_position_ids->writeMap<int>();
    for (int i = 0; i < block_size; i++) {
        qPosPtr[i] = pos_start + context_len + i;
    }

    // K position IDs: positions for all tokens [pos_start, ..., pos_start + total_len - 1]
    auto k_position_ids = _Input({1, total_len}, NCHW, halide_type_of<int>());
    auto kPosPtr = k_position_ids->writeMap<int>();
    for (int i = 0; i < total_len; i++) {
        kPosPtr[i] = pos_start + i;
    }

    // Forward through DFlash module
    std::vector<VARP> inputs = {noise_embedding, context_hidden, attention_mask, q_position_ids, k_position_ids};
    auto outputs = mDFlashModule->onForward(inputs);

    // If separate lm_head exists, outputs[0] is hidden_states -> pass through lm_head
    if (mLmHeadModule) {
        auto hidden_states_out = outputs[0]; // [1, block_size, hidden_size]
#if DFLASH_DEBUG
        {
            auto hsInfo = hidden_states_out->getInfo();
            auto hsPtr = hidden_states_out->readMap<float>();
            int total = 1;
            for (auto d : hsInfo->dim) total *= d;
            float minV = hsPtr[0], maxV = hsPtr[0];
            int nanCount = 0;
            for (int i = 0; i < total; i++) {
                if (std::isnan(hsPtr[i]) || std::isinf(hsPtr[i])) nanCount++;
                else { if (hsPtr[i] < minV) minV = hsPtr[i]; if (hsPtr[i] > maxV) maxV = hsPtr[i]; }
            }
            printf("dflashForward hidden_states: dims=[");
            for (int d = 0; d < hsInfo->dim.size(); d++) printf("%d%s", hsInfo->dim[d], d < hsInfo->dim.size()-1 ? ", " : "");
            printf("], range=[%.4f, %.4f], nanCount=%d/%d\n", minV, maxV, nanCount, total);
        }
#endif
        auto lm_outputs = mLmHeadModule->onForward({hidden_states_out});
        auto logits = lm_outputs[0]; // [1, block_size, vocab_size]
#if DFLASH_DEBUG
        {
            auto lInfo = logits->getInfo();
            auto lPtr = logits->readMap<float>();
            int total = 1;
            for (auto d : lInfo->dim) total *= d;
            int vocab = lInfo->dim[lInfo->dim.size()-1];
            // Check argmax for first few positions
            printf("lm_head logits: dims=[");
            for (int d = 0; d < lInfo->dim.size(); d++) printf("%d%s", lInfo->dim[d], d < lInfo->dim.size()-1 ? ", " : "");
            printf("], vocab=%d\n", vocab);
            for (int pos = 0; pos < std::min(3, lInfo->dim[1]); pos++) {
                const float* row = lPtr + pos * vocab;
                int bestIdx = 0; float bestVal = row[0];
                for (int j = 1; j < vocab; j++) { if (row[j] > bestVal) { bestVal = row[j]; bestIdx = j; } }
                printf("  pos %d: argmax=%d (val=%.4f), row[0]=%.4f, row[1]=%.4f\n", pos, bestIdx, bestVal, row[0], row[1]);
            }
        }
#endif
        return logits;
    }

    // outputs[0]: logits [1, block_size, vocab_size]
    return outputs[0];
}

void DFlashGeneration::generate(GenerationParams& param) {
    int max_token = param.max_new_tokens;

    // First-time initialization: sample first token and compute mContextHidden
    if (!mInitialized) {
        VARP prev_hidden_states = param.outputs[mHiddenStateIndex];

        // Sample first token from prefill logits
        mContext->current_token = mLlm->sample(param.outputs[0], param.validLogitStart, param.validLogitSize);
        mContext->history_tokens.push_back(mContext->current_token);
        mContext->output_tokens.push_back(mContext->current_token);
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

        // Compute mContextHidden from prefill hidden_states
        // Use full prefill hidden_states for better draft quality (matches reference implementation)
        {
            auto hsInfo = prev_hidden_states->getInfo();
            int seq_len = hsInfo->dim[1];
            int hiddenDim = hsInfo->dim[2];
#if DFLASH_DEBUG
            auto hsPtr = prev_hidden_states->readMap<float>();
            int hsTotal = seq_len * hiddenDim;
            int hsNanCount = 0;
            float hsMin = hsPtr[0], hsMax = hsPtr[0];
            for (int i = 0; i < hsTotal; i++) {
                if (std::isnan(hsPtr[i]) || std::isinf(hsPtr[i])) hsNanCount++;
                else { if (hsPtr[i] < hsMin) hsMin = hsPtr[i]; if (hsPtr[i] > hsMax) hsMax = hsPtr[i]; }
            }
            printf("DFlash prefill hidden_states: [1, %d, %d], range=[%.4f, %.4f], nanCount=%d/%d\n",
                   seq_len, hiddenDim, hsMin, hsMax, hsNanCount, hsTotal);
#endif
        }
        mContextHidden = fcForward(prev_hidden_states);
#if DFLASH_DEBUG
        {
            auto ctxInfo = mContextHidden->getInfo();
            auto ctxPtr = mContextHidden->readMap<float>();
            int total = 1;
            for (auto d : ctxInfo->dim) total *= d;
            float minV = ctxPtr[0], maxV = ctxPtr[0];
            int nanCount = 0;
            for (int i = 0; i < total; i++) {
                if (std::isnan(ctxPtr[i])) nanCount++;
                if (ctxPtr[i] < minV) minV = ctxPtr[i];
                if (ctxPtr[i] > maxV) maxV = ctxPtr[i];
            }
            printf("DFlash mContextHidden after fc: dims=[");
            for (int d = 0; d < ctxInfo->dim.size(); d++) {
                printf("%d%s", ctxInfo->dim[d], d < ctxInfo->dim.size()-1 ? ", " : "");
            }
            printf("], range=[%.4f, %.4f], nanCount=%d/%d\n", minV, maxV, nanCount, total);
        }
#endif
        mInitialized = true;

        // If max_token is 0, just do initialization and return
        if (max_token <= 0) {
            return;
        }
    }

#ifdef DUMP_PROFILE_INFO
    int spl_decode = 0, spl_accept = 0, spl_count = 0;
    int64_t phase_draft_us = 0, phase_verify_us = 0, phase_sample_us = 0, phase_fc_us = 0;
    int64_t phase_verify_fwd_us = 0, phase_verify_match_us = 0;
    int64_t phase_fc_fwd_us = 0, phase_fc_concat_us = 0;
#endif

    int len = 0;
    while (len < max_token) {
        if (mContext->status == LlmStatus::USER_CANCEL) {
            break;
        }
        MNN::Timer _t;

        // Phase 1: Build block [last_accepted_token, mask, mask, ..., mask]
        std::vector<int> block_ids(mBlockSize);
        block_ids[0] = mContext->current_token;
        for (int i = 1; i < mBlockSize; i++) {
            block_ids[i] = mMaskTokenId;
        }

        // Phase 2: DFlash forward - get draft logits
#if DFLASH_DEBUG
        if (len < 30) {
            printf("\n--- MNN Step %d ---\n", len / 1 + 1);
            printf("block_output_ids (before draft): [");
            for (int i = 0; i < mBlockSize; i++) printf("%d%s", block_ids[i], i < mBlockSize-1 ? ", " : "");
            printf("]\n");
            auto ctxInfo = mContextHidden->getInfo();
            auto ctxPtr = mContextHidden->readMap<float>();
            printf("context_hidden shape: [%d, %d, %d]\n", ctxInfo->dim[0], ctxInfo->dim[1], ctxInfo->dim[2]);
            printf("context_hidden[0,0,:5]: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   ctxPtr[0], ctxPtr[1], ctxPtr[2], ctxPtr[3], ctxPtr[4]);
            int pos_start = mContext->all_seq_len - ctxInfo->dim[1];
            printf("all_seq_len=%d, pos_start=%d\n", mContext->all_seq_len, pos_start);
            printf("q_position_ids: [");
            for (int i = 0; i < mBlockSize; i++) printf("%d%s", pos_start + ctxInfo->dim[1] + i, i < mBlockSize-1 ? ", " : "");
            printf("]\n");
            printf("k_position_ids: [%d, ..., %d] (len=%d)\n", pos_start, pos_start + ctxInfo->dim[1] + mBlockSize - 1, ctxInfo->dim[1] + mBlockSize);
        }
#endif
#ifdef DUMP_PROFILE_INFO
        MNN::Timer _t_draft;
#endif
        VARP draft_logits = dflashForward(block_ids, mContextHidden);
        if (draft_logits == nullptr) {
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
            auto logitsInfo = draft_logits->getInfo();
            int vocab_size = logitsInfo->dim[logitsInfo->dim.size() - 1];
            auto logitsPtr = draft_logits->readMap<float>();

            // Direct argmax for draft positions 1..block_size-1 (skip position 0)
            for (int i = 1; i < mBlockSize; i++) {
                const float* row = logitsPtr + i * vocab_size;
                int bestIdx = 0;
                float bestVal = row[0];
                for (int j = 1; j < vocab_size; j++) {
                    if (row[j] > bestVal) {
                        bestVal = row[j];
                        bestIdx = j;
                    }
                }
                block_ids[i] = bestIdx;
            }

#if DFLASH_DEBUG
            if (len < 30) {
                printf("draft_logits top5 per position:\n");
                for (int pos = 1; pos < std::min(mBlockSize, 4); pos++) {
                    const float* row = logitsPtr + pos * vocab_size;
                    // Find top 5
                    int top5_idx[5] = {0,0,0,0,0};
                    float top5_val[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
                    for (int j = 0; j < vocab_size; j++) {
                        for (int t = 0; t < 5; t++) {
                            if (row[j] > top5_val[t]) {
                                for (int s = 4; s > t; s--) { top5_idx[s] = top5_idx[s-1]; top5_val[s] = top5_val[s-1]; }
                                top5_idx[t] = j; top5_val[t] = row[j];
                                break;
                            }
                        }
                    }
                    printf("  pos %d top5: tokens=[%d,%d,%d,%d,%d], scores=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                           pos-1, top5_idx[0], top5_idx[1], top5_idx[2], top5_idx[3], top5_idx[4],
                           top5_val[0], top5_val[1], top5_val[2], top5_val[3], top5_val[4]);
                }
                printf("block_output_ids (after draft): [");
                for (int i = 0; i < mBlockSize; i++) printf("%d%s", block_ids[i], i < mBlockSize-1 ? ", " : "");
                printf("]\n");
            }
#endif
        }

        // Phase 4: Verify entire block with target model
#ifdef DUMP_PROFILE_INFO
        phase_sample_us += _t_sample.durationInUs();
        MNN::Timer _t_verify;
#endif
        auto verify_outputs = mLlm->forwardVec(block_ids);
        if (verify_outputs.empty() || verify_outputs.size() < 2) {
            break;
        }
#ifdef DUMP_PROFILE_INFO
        int64_t verify_fwd_elapsed = _t_verify.durationInUs();
        phase_verify_fwd_us += verify_fwd_elapsed;
#endif

        auto verify_logits = verify_outputs[0];
        auto new_hidden_states = verify_outputs[mHiddenStateIndex];

        // Phase 5: Greedy prefix matching - compare target model's predictions with draft
        int acceptance_length = 0;
        {
            auto verifyInfo = verify_logits->getInfo();
            int verify_vocab = verifyInfo->dim[verifyInfo->dim.size() - 1];
            auto verifyPtr = verify_logits->readMap<float>();

            for (int i = 0; i < mBlockSize - 1; i++) {
                const float* row = verifyPtr + i * verify_vocab;
                int target_prediction = 0;
                float bestVal = row[0];
                for (int j = 1; j < verify_vocab; j++) {
                    if (row[j] > bestVal) {
                        bestVal = row[j];
                        target_prediction = j;
                    }
                }

#if DFLASH_DEBUG
                if (len < 30 && i < 3) {
                    printf("  pos %d: draft=%d, target=%d %s\n", i, block_ids[i + 1], target_prediction,
                           target_prediction == block_ids[i + 1] ? "✓" : "✗");
                }
#endif

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

            if (acceptance_length == mBlockSize - 1) {
                const float* row = verifyPtr + (mBlockSize - 1) * verify_vocab;
                int target_prediction = 0;
                float bestVal = row[0];
                for (int j = 1; j < verify_vocab; j++) {
                    if (row[j] > bestVal) {
                        bestVal = row[j];
                        target_prediction = j;
                    }
                }
                mContext->current_token = target_prediction;
            }
        }

#ifdef DUMP_PROFILE_INFO
        MNN::Timer _t_match;
#endif

        // Phase 6: Accept tokens and update state
        // We accept block_ids[1..acceptance_length] + current_token
        int total_accepted = acceptance_length + 1;

        // Output accepted tokens
        bool stop = false;
        for (int i = 1; i <= acceptance_length; i++) {
            int token = block_ids[i];
            mContext->history_tokens.push_back(token);
            mContext->output_tokens.push_back(token);
            if (nullptr != mContext->os) {
                *mContext->os << mLlm->tokenizer_decode(token) << std::flush;
            }
            if (mLlm->is_stop(token)) {
                stop = true;
                break;
            }
        }

        if (!stop) {
            // Add the corrected/next token
            mContext->history_tokens.push_back(mContext->current_token);
            mContext->output_tokens.push_back(mContext->current_token);
            if (nullptr != mContext->os) {
                if (mLlm->is_stop(mContext->current_token)) {
                    *mContext->os << mContext->end_with << std::flush;
                    stop = true;
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

        // Phase 7: Update KV cache - remove unaccepted tokens
        // We fed mBlockSize tokens but only accepted total_accepted
        int remove_count = mBlockSize - total_accepted;
        mLlm->mMeta->remove = remove_count;
        mLlm->updateContext(total_accepted, total_accepted);

        len += total_accepted;
        mContext->decode_us += _t.durationInUs();

#ifdef DUMP_PROFILE_INFO
        spl_decode += mBlockSize;
        spl_accept += total_accepted;
        spl_count++;
#endif

        if (stop) {
            break;
        }

        // Phase 8: Update mContextHidden for next iteration
        // Accumulate context across iterations (simulates draft KV cache from reference)
        // The draft model needs full history context for good prediction quality
#ifdef DUMP_PROFILE_INFO
        MNN::Timer _t_fc;
#endif
        if (total_accepted < mBlockSize) {
            auto hsInfo = new_hidden_states->getInfo();
            int hiddenDim = hsInfo->dim[2];
            std::vector<int> starts = {0, 0, 0};
            std::vector<int> sizes = {1, total_accepted, hiddenDim};
            new_hidden_states = _Slice(new_hidden_states,
                                       _Const(starts.data(), {3}, NHWC, halide_type_of<int>()),
                                       _Const(sizes.data(), {3}, NHWC, halide_type_of<int>()));
        }
        auto new_context = fcForward(new_hidden_states);
#ifdef DUMP_PROFILE_INFO
        phase_fc_fwd_us += _t_fc.durationInUs();
        MNN::Timer _t_concat;
#endif
        auto concat_result = _Concat({mContextHidden, new_context}, 1);
        // Materialize to avoid lazy evaluation graph growth
        auto concatInfo = concat_result->getInfo();
        auto concatPtr = concat_result->readMap<float>();
        mContextHidden = _Const(concatPtr, concatInfo->dim, concatInfo->order, concatInfo->type);
#ifdef DUMP_PROFILE_INFO
        phase_fc_concat_us += _t_concat.durationInUs();
        phase_fc_us += _t_fc.durationInUs() + phase_fc_concat_us;
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
    MNN_PRINT("Phase breakdown %%: draft=%.1f%%, sample=%.1f%%, verify=%.1f%%, fc=%.1f%%\n",
              total_ms > 0 ? 100.0f * phase_draft_us / 1000.0f / total_ms : 0,
              total_ms > 0 ? 100.0f * phase_sample_us / 1000.0f / total_ms : 0,
              total_ms > 0 ? 100.0f * phase_verify_us / 1000.0f / total_ms : 0,
              total_ms > 0 ? 100.0f * phase_fc_us / 1000.0f / total_ms : 0);
    MNN_PRINT("Verify detail (ms): forward=%.1f, match=%.1f\n",
              phase_verify_fwd_us / 1000.0f, phase_verify_match_us / 1000.0f);
    MNN_PRINT("FC detail (ms): fc_forward=%.1f, concat=%.1f\n",
              phase_fc_fwd_us / 1000.0f, phase_fc_concat_us / 1000.0f);
    float per_accepted_ms = spl_accept > 0 ? total_ms / spl_accept : 0;
    MNN_PRINT("Effective per-token cost: %.2f ms/token (vs AR baseline)\n", per_accepted_ms);
    MNN_PRINT("============== DFlash Decoding Statistics End =================\n");
#endif
}

} // namespace Transformer
} // namespace MNN