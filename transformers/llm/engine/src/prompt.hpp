

#ifndef PROMPT_Hpp
#define PROMPT_Hpp

#include "llm/llm.hpp"
#include "llmconfig.hpp"
#include <fstream>
#include <sstream>
#include <stdlib.h>


namespace MNN {
namespace Transformer {

class Prompt {
private:
    class JinjaTemplate;
    std::shared_ptr<LlmContext> mContext;
    std::string mPromptTemplate; // for compatibility
    std::string mSystemPrompt;
    std::string mBos, mSystemTemplate, mUserTemplate, mAssistantTemplate;
    std::string mAssistantPrefix, mAssistantSuffix;
    std::string mSystemName = "system",
                mUserName = "user",
                mAssistantName = "assistant";
    std::shared_ptr<JinjaTemplate> mCommonTemplate;
public:
    static Prompt* createPrompt(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    Prompt(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    std::string getAssistantSuffix() const;
    void setParams(std::shared_ptr<LlmConfig> config);
    std::string applyTemplate(std::string user_content, bool add_system_prompt = false, bool add_generation_prompt = true);
    std::string applyTemplate(const std::vector<ChatMessage>& inputs, bool add_generation_prompt = true);
};

}
}


#endif
