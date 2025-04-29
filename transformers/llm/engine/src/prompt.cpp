#include "prompt.hpp"

namespace MNN {
namespace Transformer {

static std::string buildPrompt(ChatMessage item, std::string prompt_template, std::string placeholder) {
    size_t start_pos = prompt_template.find(placeholder);
    if (start_pos == std::string::npos) {
        return item.first + "\n" + item.second + "\n";
    } else {
        prompt_template.replace(start_pos, placeholder.length(), item.second);
        return prompt_template;
    }
}

Prompt* Prompt::createPrompt(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) {
    return new Prompt(context, config);
}

bool contains(const std::string& str, const std::string& substring) {
    return str.find(substring) != std::string::npos;
}
    
void Prompt::setParams(std::shared_ptr<LlmConfig> config) {
    mReuseKV = config->reuse_kv();
    mSystemPrompt = config->system_prompt();
    if (config->config_.document.HasMember("prompt_template")) {
        // std::cout << "legacy prompt_template" << std::endl;
        // legacy
        mPromptTemplate = config->prompt_template();
        if (contains(mPromptTemplate, "<reserved_106>")) {
            // Baichuan2
            mUserTemplate = "<reserved_106>%s";
            mAssistantTemplate = "<reserved_107>%s";
        } else if (contains(mPromptTemplate, "[Round 1]")) {
            // chatglm2
            mUserTemplate = "[Round 1]\n\n问：%s\n\n";
            mAssistantTemplate = "答：%s\n\n";
        } else if (contains(mPromptTemplate, "<|user|>\n%s\n<|assistant|>\n")) {
            // chatglm3/glm4
            mBos = "[gMASK]<sop>";
            mSystemTemplate = "<|system|>\n%s\n";
            mUserTemplate = "<|user|>\n%s\n";
            mAssistantTemplate = "<|assistant|>\n%s\n";
        } else if (contains(mPromptTemplate, "Assistant:")) {
            // deepseek-7b
            mBos = "<|begin_of_sentence|>";
            mSystemTemplate = "%s\n";
            mUserTemplate = "\nUser: %s\n";
            mAssistantTemplate = "\nAssistant: %s\n<|end_of_sentence|>";
        } else if (contains(mPromptTemplate, "<|begin_of_sentence|><|User|>")) {
            // DeepSeek
            mBos = "<|begin_of_sentence|>";
            mSystemTemplate = "%s";
            mUserTemplate = "<|User|>%s";
            mAssistantTemplate = "<|Assistant|>%s<|end_of_sentence|>";
        } else if (contains(mPromptTemplate, "[INST]")) {
            // llama2
            mBos = "[INST] ";
            mSystemTemplate = "<<SYS>>\n%s\n<</SYS>>\n\n";
            mUserTemplate = "%s [/INST]";
            mAssistantTemplate = "%s</s>";
        } else if (contains(mPromptTemplate, "<|start_header_id|>")) {
            // llama3.x.
            mSystemTemplate = "<|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|>";
            mUserTemplate = "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>";
            mAssistantTemplate = "<|start_header_id|>assistant<|end_header_id|>\n\n%s<|eot_id|>";
        } else if (contains(mPromptTemplate, "<s><|system|>")) {
            // TinyLlama
            mBos = "<s>";
            mSystemTemplate = "<|system|>\n%s</s>\n";
            mUserTemplate = "<|user|>\n%s</s>\n";
            mAssistantTemplate = "<|assistant|>\n%s</s>\n";
        } else if (contains(mPromptTemplate, "<start_of_turn>")) {
            // gemma2
            mBos = "<bos>";
            mSystemTemplate = "<start_of_turn>system\n%s<end_of_turn>\n";
            mUserTemplate = "<start_of_turn>user\n%s<end_of_turn>\n";
            mAssistantTemplate = "<start_of_turn>model\n%s<end_of_turn>\n";
        } else if (mPromptTemplate == "<bos>%s") {
            // gemma
            mBos = "<bos>";
            mSystemTemplate = "%s";
            mUserTemplate = "%s";
            mAssistantTemplate = "%s";
        } else if (contains(mPromptTemplate, "<|Bot|>")) {
            // internlm
            mUserTemplate = "<|User|>:%s<eoh>";
            mAssistantTemplate = "<|Bot|>%s<eoh>\n";
        } else if (contains(mPromptTemplate, "Instruct:")) {
            // phi
            mUserTemplate = "Instruct: %s\n";
            mAssistantTemplate = "Output:%s\n";
        } else {
            // default
            mSystemTemplate = "<|im_start|>system\n%s<|im_end|>\n";
            mUserTemplate = "<|im_start|>user\n%s<|im_end|>\n";
            mAssistantTemplate = "<|im_start|>assistant\n%s<|im_end|>\n";
        }
    } else {
        mBos = config->bos();
        mSystemTemplate = config->system_prompt_template();
        mUserTemplate = config->user_prompt_template();
        mAssistantTemplate = config->assistant_prompt_template();
    }
    mAssistantPrefix = mAssistantTemplate.substr(0, mAssistantTemplate.find("%s"));
    mAssistantSuffix = mAssistantTemplate.substr(mAssistantTemplate.find("%s") + 2);
}

std::string Prompt::getAssistantSuffix() const {
    return mAssistantSuffix;
}

Prompt::Prompt(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) {
    mContext = context;
    setParams(config);
}

std::string Prompt::applyTemplate(std::string user_content, bool add_system_prompt, bool add_generation_prompt) {
    std::vector<ChatMessage> prompts;
    if (add_system_prompt) {
        prompts.push_back(std::make_pair("system", mSystemPrompt));
    }
    prompts.push_back(std::make_pair("user", user_content));
    return applyTemplate(prompts, add_generation_prompt);
}

std::string Prompt::applyTemplate(std::vector<ChatMessage> inputs, bool add_generation_prompt) {
    std::string prompt_str = mBos;
    for (auto input : inputs) {
        if (input.first == "") continue;
        if (input.first == "system") {
            if (input.second == "") continue;
            prompt_str += buildPrompt(input, mSystemTemplate, "%s");
            continue;
        } else if (input.first == "user") {
            prompt_str += buildPrompt(input, mUserTemplate, "%s");
            continue;
        } else if (input.first == "assistant") {
            prompt_str += mAssistantPrefix + input.second + mAssistantSuffix;
            continue;
        } else {
            // don't support
        }
    }
    if (add_generation_prompt) {
        prompt_str += mAssistantPrefix;
    }
    return prompt_str;
}

}
}
