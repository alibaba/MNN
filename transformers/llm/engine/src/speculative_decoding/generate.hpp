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
    virtual void load(Module::Config module_config) {
        // do nothing
    };
    virtual void generate(GenerationParams& param) = 0;
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
    virtual void load(Module::Config module_config) override;
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
    virtual void load(Module::Config module_config) override;
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


class GenerationStrategyFactory {
public:
    static std::shared_ptr<Generation> create(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, bool canSpec);
};


} // namespace Transformer
} // namespace MNN
#endif
