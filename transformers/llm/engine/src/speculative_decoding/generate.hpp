//
//  generate.hpp
//
//  Created by MNN on 2025/06/09.
//

#ifndef SPEC_GENERATE_HPP
#define SPEC_GENERATE_HPP

#include <MNN/AutoTime.hpp>
#include "llm/llm.hpp"
#include "../llmconfig.hpp"
#include "../kvmeta.hpp"

//#define DUMP_PROFILE_INFO

namespace MNN {
namespace Transformer {
struct GenerationParams {
    int max_new_tokens;
    int timeout_ms = -1; // -1 means no timeout
    std::vector<int> input_ids;
    MNN::Express::VARP input_embeds;
    std::vector<MNN::Express::VARP> outputs;
    int validLogitStart = 0;
    int validLogitSize = 0;
};

class Generation {
public:
    Generation(Llm* llm, std::shared_ptr<LlmContext> context) {
        mLlm = llm;
        mContext = context;
    };
    virtual ~Generation() = default;
    // Returns false when a module the strategy's generate() depends on failed to
    // load; Llm::load() must then abort instead of entering RUNNING.
    virtual bool load(Module::Config module_config) {
        // do nothing
        return true;
    };
    virtual void generate(GenerationParams& param) = 0;
    virtual void reset() {
        // do nothing
    };
protected:
    int draftVerify(MNN::Express::VARP logits, const std::vector<int>& drafts, bool& stop);
    std::shared_ptr<LlmContext> mContext;
    Llm* mLlm;
};

class ArGeneration: public Generation {
public:
    ArGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    virtual ~ArGeneration() = default;
    virtual void generate(GenerationParams& param);
};

class LookaheadGeneration: public Generation {
public:
    LookaheadGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    virtual ~LookaheadGeneration() = default;
    virtual void generate(GenerationParams& param);
private:
    int mNgramKeyMaxLen = 4;
    MatchStrictLevel mStrictLevel;
    bool mUpdateNgram = false;
    NgramSelectRule mSelectRule;
};

class MtpGeneration: public Generation {
public:
    MtpGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    virtual ~MtpGeneration() = default;
    virtual bool load(Module::Config module_config) override;
    virtual void generate(GenerationParams& param) override;
private:
    std::vector<MNN::Express::VARP> mtpForward(const std::vector<int>& input_ids, MNN::Express::VARP hidden_states);
    std::vector<MNN::Express::VARP> mtpForward(MNN::Express::VARP input_embeds, MNN::Express::VARP hidden_states);

    std::vector<std::shared_ptr<MNN::Express::Module>> mMtpModules;
    std::map<std::pair<int, bool>, std::shared_ptr<MNN::Express::Module>> mMtpModulePool;
    std::shared_ptr<KVMeta> mMtpMeta;
    int mHiddenStateIndex = -1;
};

class EagleGeneration: public Generation {
public:
    EagleGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    virtual ~EagleGeneration() = default;
    virtual bool load(Module::Config module_config) override;
    virtual void generate(GenerationParams& param) override;
private:
    struct DraftInfo {
        std::vector<int> draftTokens;
        std::vector<std::vector<int>> retrieveIndices;
        VARP attentionMask;
        VARP positionIds;
    };
    struct AcceptInfo {
        std::vector<int> sampleTokens;
        std::vector<int> acceptIndices;
        std::vector<int> acceptTokens;
    };
    MNN::Express::VARPS eagleForwardRaw(const MNN::Express::VARPS& inputs);
    MNN::Express::VARPS eagleForward(const std::vector<int>& inputEmbeds, MNN::Express::VARP hiddenStates, bool allLogits = false);
    MNN::Express::VARPS eagleForward(MNN::Express::VARP inputEmbeds, MNN::Express::VARP hiddenStates, bool allLogits = false);
    DraftInfo topkGenerate(const std::vector<int>& inputIds, MNN::Express::VARP hiddenStates, MNN::Express::VARP inputEmbeds = nullptr);
    VARPS treeDecoding(const DraftInfo& draftInfo);
    AcceptInfo evaluatePosterior(const DraftInfo& drafInfo, VARP logits);
    DraftInfo updateDraft(const AcceptInfo& accpetInfo, VARP hiddenStates);
    MNN::Express::VARP getMask(std::vector<std::vector<bool>> mask, int seqLen);
    bool processTokens(const std::vector<int>& accpetTokens);
    void setPosition(int position);
    std::string tokenStr(int token);
    std::vector<std::shared_ptr<MNN::Express::Module>> mEagleModules;
    std::shared_ptr<KVMeta> mEagleMeta;
    MNN::Express::VARP mD2t, mTreePosition;
    int mTopK, mDepth;
    int mEaglePastLen = 0, mEagleRemove = 0;
};


class DFlashGeneration : public Generation {
public:
    DFlashGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    virtual ~DFlashGeneration() = default;
    virtual bool load(Module::Config module_config) override;
    virtual void generate(GenerationParams& param) override;
    virtual void reset() override;
private:
    // Row-wise argmax, shared by draft sampling and target verify; only the element count is read, so any rank binds.
    // False when the on-device argmax module could not be built: the caller must abort the step.
    bool rowArgmax(MNN::Express::VARP logits, int rows, std::vector<int>& out);

    // --- The verify / accept / emit / rollback half of the speculation step ---
    struct VerifyResult {
        int totalAccepted = 0;                  // tokens committed this step (acceptance_length + 1)
        bool stop = false;                      // a stop token was emitted
        MNN::Express::VARP newHiddenStates;     // target hidden_states, for the context update
    };
    // Target verify + greedy accept + emit + KV rollback for one speculation step.
    VerifyResult verifyAcceptCommit(const std::vector<int>& block_ids, int blockSize, int hiddenStateIndex);
    // Longest draft prefix the target agrees with; sets mContext->current_token to the corrected/next token.
    int acceptDraftPrefix(const std::vector<int>& targetArgmax, const std::vector<int>& block_ids, int blockSize);

    // --- Draft production and context update ---
    MNN::Express::VARP dflashForward(const std::vector<int>& block_ids);
    // startPos is the draft-local RoPE position of the first new row.
    void appendDraftKv(MNN::Express::VARP newContext, int startPos);
    void ensureKvConcatModule(int kvHeads, int headDim);
    std::vector<int> buildBlock(int currentToken) const;

    std::shared_ptr<MNN::Express::Module> mDFlashModule;   // dflash.mnn (transformer, no lm_head)
    std::shared_ptr<MNN::Express::Module> mFcModule;       // dflash_fc.mnn
    std::shared_ptr<MNN::Express::Module> mLmHeadModule;   // target lm_head subgraph (shared-from-target)
    std::shared_ptr<MNN::Express::Module> mKvMatModule;    // dflash_kvmat.mnn: context rows -> per-layer K/V
    std::shared_ptr<MNN::Express::Module> mKvConcatModule; // one dispatch appending all K/V pairs
    std::shared_ptr<MNN::Express::Module> mArgmaxModule;   // single _ArgMax over [rows, vocab]
    // Draft KV cache, interleaved [K_0, V_0, K_1, V_1, ...], each [1, ctx_len, kv_heads, head_dim].
    std::vector<MNN::Express::VARP> mDraftKv;
    // mDraftKv slot -> mKvMatModule output slot (the kvmat graph declares its own order).
    std::vector<int> mKvMatSlot;
    int mBlockSize;
    int mMaskTokenId;
    int mShiftLabel = 0;   // 1 when the draft head was trained with shift_label
    int mHiddenStateIndex = -1;
    bool mInitialized = false;
#ifdef DUMP_PROFILE_INFO
    int spl_decode = 0, spl_accept = 0, spl_count = 0;
    int64_t phase_sample_us = 0, phase_verify_us = 0, phase_verify_fwd_us = 0, phase_verify_match_us = 0;
    int64_t phase_argmax_us = 0;
#endif
};

class GenerationStrategyFactory {
public:
    static std::shared_ptr<Generation> create(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, bool canSpec);
};


} // namespace Transformer
} // namespace MNN
#endif
