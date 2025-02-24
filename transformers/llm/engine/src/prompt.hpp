

#ifndef PROMPT_Hpp
#define PROMPT_Hpp

#include "llm/llm.hpp"
#include "llmconfig.hpp"
#include <fstream>
#include <sstream>
#include <stdlib.h>


namespace MNN {
namespace Transformer {

class MNN_PUBLIC Prompt {
private:
    std::shared_ptr<LlmContext> mContext;
    std::string mPromptTemplate; // for compatibility
    std::string mSystemPrompt;
    std::string mBos, mSystemTemplate, mUserTemplate, mAssistantTemplate;
    std::string mAssistantPrefix, mAssistantSuffix;
    std::string mSystemName = "system",
                mUserName = "user",
                mAssistantName = "assistant";
    bool mReuseKV = false;
public:
    static Prompt* createPrompt(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    Prompt(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    std::string getAssistantSuffix() const;
    std::string applyTemplate(std::string user_content, bool add_system_prompt = false, bool add_generation_prompt = true);
    std::string applyTemplate(std::vector<ChatMessage> inputs, bool add_generation_prompt = true);
};

}
}


#endif