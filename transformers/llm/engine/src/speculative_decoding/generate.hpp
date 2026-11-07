//
//  generate.hpp
//
//  Created by MNN on 2025/06/09.
//

#ifndef SPEC_GENERATE_HPP
#define SPEC_GENERATE_HPP

#include <MNN/AutoTime.hpp>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
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
    // True when the strategy tags its verify forward with KVMeta::spec_block.
    virtual bool marksSpecBlock() const {
        return false;
    };
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


// Reuse helper threads across the host Markov chain's serial slots.
class MarkovSweepPool {
public:
    ~MarkovSweepPool() { shutdown(); }
    template <typename Fn>
    void run(Fn&& fn, int nWorkers) {
        if (nWorkers <= 1) {
            fn(0);
            return;
        }
        ensureStarted(nWorkers);
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mFn = [&fn](int i) { fn(i); };
            // Worker 0 is the caller, so only nWorkers-1 helper threads report back.
            mPending = nWorkers - 1;
            mGeneration++;
            mCvWork.notify_all();
        }
        fn(0);
        std::unique_lock<std::mutex> lock(mMutex);
        mCvDone.wait(lock, [this] { return mPending == 0; });
    }
    // Workers are created lazily on first use and live until destruction.
    void ensureStarted(int nWorkers);
private:
    void shutdown();
    std::mutex mMutex;
    std::condition_variable mCvWork, mCvDone;
    std::function<void(int)> mFn;
    int mPending = 0;
    int mGeneration = 0;
    int mWorkerCount = 0;
    std::vector<std::thread> mWorkers;
};

class DFlashGeneration : public Generation {
public:
    DFlashGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    virtual ~DFlashGeneration() = default;
    virtual bool load(Module::Config module_config) override;
    virtual void generate(GenerationParams& param) override;
    virtual bool marksSpecBlock() const override {
        return true;
    }
    virtual void reset() override;
private:
    // Row-wise argmax, shared by draft sampling and target verify; only the element count is read, so any rank binds.
    // False when the on-device argmax module could not be built: the caller must abort the step.
    bool rowArgmax(MNN::Express::VARP logits, int rows, std::vector<int>& out);
    // Device-side Markov sampling: gather coefficients, project bias, then add and take argmax.
    bool addArgmax(MNN::Express::VARP base, MNN::Express::VARP bias, int& outIdx);
    bool markovSlot(int prev, MNN::Express::VARP base, int& outIdx);
    bool draftArgmaxChain(MNN::Express::VARP logits, const std::vector<int>& block_ids,
                          std::vector<int>& out);
    const char* logTag() const {
        return (mMarkovFile.empty() && mMarkovW1File.empty()) ? "DFlash" : "DSpark";
    }
    std::string mMarkovFile;
    std::string mMarkovW1File;
    std::string mMarkovW2File;
    std::shared_ptr<MNN::Express::Module> mMarkovW1Module;
    std::shared_ptr<MNN::Express::Module> mMarkovW2Module;

    // Target verification and cache rollback.
    struct VerifyResult {
        int totalAccepted = 0;
        bool stop = false;
        MNN::Express::VARP newHiddenStates;
    };
    VerifyResult verifyAcceptCommit(const std::vector<int>& block_ids, int blockSize, int hiddenStateIndex);
    // Longest draft prefix the target agrees with; sets mContext->current_token to the corrected/next token.
    int acceptDraftPrefix(const std::vector<int>& targetArgmax, const std::vector<int>& block_ids, int blockSize);

    // Draft production and context update.
    MNN::Express::VARP dflashForward(const std::vector<int>& block_ids);
    // startPos is the draft-local RoPE position of the first new row.
    void appendDraftKv(MNN::Express::VARP newContext, int startPos);
    void ensureKvConcatModule(int kvHeads, int headDim);
    std::vector<int> buildBlock(int currentToken) const;

    std::shared_ptr<MNN::Express::Module> mDFlashModule;
    std::shared_ptr<MNN::Express::Module> mFcModule;
    std::shared_ptr<MNN::Express::Module> mLmHeadModule;
    std::shared_ptr<MNN::Express::Module> mKvMatModule;
    std::shared_ptr<MNN::Express::Module> mKvConcatModule;
    std::shared_ptr<MNN::Express::Module> mArgmaxModule;
    std::shared_ptr<MNN::Express::Module> mAddArgmaxModule;
    // Legacy host fallback: flattened [rank, vocab] tables, with one int8 scale per w2 row.
    std::vector<float> mMarkovW1;
    std::vector<int8_t> mMarkovW2q;
    std::vector<float> mMarkovW2Scale;
    std::vector<float> mMarkovBias;
    int mMarkovThreads = 1;
    MarkovSweepPool mMarkovPool;
    int mMarkovRank = 0;
    int mMarkovVocab = 0;
    // Draft KV cache, interleaved [K_0, V_0, K_1, V_1, ...], each [1, ctx_len, kv_heads, head_dim].
    std::vector<MNN::Express::VARP> mDraftKv;
    // mDraftKv slot -> mKvMatModule output slot (the kvmat graph declares its own order).
    std::vector<int> mKvMatSlot;
    int mBlockSize;
    int mMaskTokenId;
    int mShiftLabel = 0;   // 1 when the draft head was trained with shift_label
    int mHiddenStateIndex = -1;
    bool mInitialized = false;
    // False when the draft emits logits itself (DSpark); true when its lm_head comes from the target.
    bool mSharedLmHead = true;
};

class GenerationStrategyFactory {
public:
    static std::shared_ptr<Generation> create(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, bool canSpec);
};


} // namespace Transformer
} // namespace MNN
#endif
