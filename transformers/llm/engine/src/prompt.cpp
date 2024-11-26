#include "prompt.hpp"

namespace MNN {
namespace Transformer {

/* ----------PromptLib---------- */
PromptLib* PromptLib::createPromptLib(Llm* llm, const std::string& config_path) {
    return createPromptLib(llm, std::shared_ptr<LlmConfig>(new LlmConfig(config_path)));
}
PromptLib* PromptLib::createPromptLib(Llm* llm, std::shared_ptr<LlmConfig> config) {
    if (config->app_type() == "chat" || config->app_type() == "perplexity") {
        return new BaseChatPromptLib(llm, config);
    } else {
        std::cout << "PromptLib not Implemented!\n" << std::endl; 
        return nullptr;
    }
}

/* ----------BaseChatPromptLib---------- */
BaseChatPromptLib::BaseChatPromptLib(Llm* llm, std::shared_ptr<LlmConfig> config) {
    mLlm = llm;
    mReuseKV = config->reuse_kv();
    mDefaultSystemPrompt = config->system_prompt();
    mSystemTemplate = config->system_prompt_template();
    mUserTemplate = config->user_prompt_template();
    mAssistantPrefix = config->assistant_prefix();
    mAssistantSuffix = config->assistant_suffix();
}

void BaseChatPromptLib::appendSystemPrompt() {
    appendSystemPrompt(mDefaultSystemPrompt);
}
void BaseChatPromptLib::appendSystemPrompt(const std::string sys_prompt) {
    mLlm->mLlmSessionInfos[0].mHistory.emplace_back(std::make_pair("system", sys_prompt));
    mLlm->mLlmSessionInfos[0].mInputs.emplace_back(std::make_pair("system", sys_prompt));
}
void BaseChatPromptLib::appendUserPrompt(const std::string user_prompt) {
    if (mLlm->mLlmSessionInfos[0].mHistory.empty()) { appendSystemPrompt(); } // prevent no system prompt appendix.
    mLlm->mLlmSessionInfos[0].mHistory.emplace_back(std::make_pair("user", user_prompt));
    mLlm->mLlmSessionInfos[0].mInputs.emplace_back(std::make_pair("user", user_prompt));
}
void BaseChatPromptLib::appendLLMOutput(std::string out_str) {
    mLlm->mLlmSessionInfos[0].mHistory.emplace_back(std::make_pair("assistant", out_str));
    if (mReuseKV) {
        // clear input
        mLlm->mLlmSessionInfos[0].mInputs.clear();
    } else {
        // keep input, append output
        mLlm->mLlmSessionInfos[0].mInputs.emplace_back(std::make_pair("assistant", out_str));
    }
}

std::string BaseChatPromptLib::getLLMInput() {
    std::string input_str;
    if (mReuseKV) {
        if (mLlm->mLlmSessionInfos[0].mHistory.size() != mLlm->mLlmSessionInfos[0].mInputs.size()) {
            // 1.1 not first prefill, add end of speech.
            input_str += mAssistantSuffix;
        }
    }
    // 1.2 generate from template
    input_str += applyTemplates(mLlm->mLlmSessionInfos[0].mInputs);
    input_str += mAssistantPrefix;
    return input_str;
}

std::string BaseChatPromptLib::applyTemplate(PromptItem item, std::string prompt_template, std::string placeholder) {
    size_t start_pos = prompt_template.find(placeholder);
    if (start_pos == std::string::npos) return item.first + "\n" + item.second + "\n";
    else {
        prompt_template.replace(start_pos, placeholder.length(), item.second);
        return prompt_template;
    }
}

std::string BaseChatPromptLib::applyTemplates(std::vector<PromptItem> inputs) {
    std::string input_str;
    for (auto input : inputs) {
        if (input.first == "") continue;
        if (input.first == "system") {
            if (input.second == "") continue;
            input_str += applyTemplate(input, mSystemTemplate, "%s");
            continue;
        } 
        if (input.first == "user") {
            input_str += applyTemplate(input, mUserTemplate, "%s");
            continue;
        }
        if (input.first == "assistant") {
            input_str += mAssistantPrefix + input.second + mAssistantSuffix;
            continue;
        }
        // Invalid role!!!
    }
    return input_str;
}

std::string BaseChatPromptLib::applyTemplate(std::string user_content) {
    std::vector<PromptItem> prompts;
    prompts.push_back(std::make_pair("system", mDefaultSystemPrompt));
    prompts.push_back(std::make_pair("user", user_content));
    return applyTemplates(prompts) + mAssistantPrefix;
}

std::string BaseChatPromptLib::getAssistantSuffix() const {
    return mAssistantSuffix;
}

}
}