#include "prompt/prompt.hpp"

namespace MNN {
namespace Transformer {

/* ----------PromptLib---------- */
PromptLib* PromptLib::createPromptLib(const std::string prompt_type, const std::string& config_path) {
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    if (prompt_type == "BaseChat") {
        return new BaseChatPromptLib(config);
    } else {
        std::cout << "PromptLib not Implemented!\n" << std::endl; 
        return nullptr;
    }
}

void PromptLib::reset() {
    mHistory.clear();
    mInputs.clear();
}

/* ----------BaseChatPromptLib---------- */
BaseChatPromptLib::BaseChatPromptLib(std::shared_ptr<LlmConfig> config) {
    mReuseKV = config->reuse_kv();
    mSystemTemplate = config->system_prompt_template();
    mUserTemplate = config->user_prompt_template();
    mAssistantPrefix = config->assistant_prefix();
    mAssistantSuffix = config->assistant_suffix();
    mHistory.clear();
    mInputs.clear();
}

void BaseChatPromptLib::appendSystemPrompt(const std::string& sys_prompt) {
    mHistory.emplace_back(std::make_pair("system", sys_prompt));
    mInputs.emplace_back(std::make_pair("system", sys_prompt));
}
void BaseChatPromptLib::appendUserPrompt(const std::string& user_prompt) {
    mHistory.emplace_back(std::make_pair("user", user_prompt));
    mInputs.emplace_back(std::make_pair("user", user_prompt));
}
void BaseChatPromptLib::appendLLMOutput(std::string out_str) {
    mHistory.emplace_back(std::make_pair("assistant", out_str));
    if (mReuseKV) {
        // clear input
        mInputs.clear();
    } else {
        // keep input, append output
        mInputs.emplace_back(std::make_pair("assistant", out_str));
    }
}

std::string BaseChatPromptLib::getLLMInput() {
    std::string input_str;
    if (mReuseKV) {
        if (mHistory.size() != mInputs.size()) {
            // 1.1 not first prefill, add end of speech.
            input_str += mAssistantSuffix;
        }
    }
    // 1.2 generate from template
    input_str += applyTemplates(mInputs);
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

}
}