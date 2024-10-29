

#ifndef PROMPT_Hpp
#define PROMPT_Hpp

#include "llm/llm.hpp"
#include "sampler/sampler.hpp"
#include "llmconfig.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <fstream>
#include <sstream>
#include <stdlib.h>

// <role, content>
#define PromptItem std::pair<std::string, std::string>


namespace MNN {
namespace Transformer {

/* PromptLib: history organization + input organization */
class MNN_PUBLIC PromptLib {
public:
    std::vector<PromptItem> mHistory;
    std::vector<PromptItem> mInputs;
public:
    static PromptLib* createPromptLib(const std::string prompt_type, const std::string& config_path);
    virtual void appendSystemPrompt(const std::string& sys_prompt) = 0;
    virtual void appendUserPrompt(const std::string& use_prompt) = 0;
    virtual void appendLLMOutput(std::string out_str) = 0;
    virtual std::string getLLMInput() = 0;
    virtual void reset();
};

class MNN_PUBLIC BaseChatPromptLib : public PromptLib {
protected:
    bool mReuseKV;
    std::string mSystemTemplate;
    std::string mUserTemplate;
    std::string mAssistantPrefix;
    std::string mAssistantSuffix;
    std::string applyTemplate(PromptItem item, std::string prompt_template, std::string placeholder = "%s");
    std::string applyTemplates(std::vector<PromptItem> inputs);
public:
    BaseChatPromptLib(std::shared_ptr<LlmConfig> config);
    virtual void appendSystemPrompt(const std::string& sys_prompt) override;
    virtual void appendUserPrompt(const std::string& user_prompt) override;
    virtual void appendLLMOutput(std::string out_str) override;
    virtual std::string getLLMInput() override;
};

}
}


#endif