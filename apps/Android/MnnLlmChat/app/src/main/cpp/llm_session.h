//
// Created by ruoyi.sdj on 2025/4/18.
//
#pragma once
#include <vector>
#include <string>
#include "nlohmann/json.hpp"
#include "llm/llm.hpp"

using nlohmann::json;
using MNN::Transformer::Llm;

namespace mls {
using PromptItem = std::pair<std::string, std::string>;

class LlmSession {
public:
    LlmSession(std::string, json config, json extra_config, std::vector<std::string> string_history);
    void Reset();
    void Load();
    ~LlmSession();
    std::string getDebugInfo();
    void SetWavformCallback(std::function<bool(const float*, size_t, bool)> callback);
    const MNN::Transformer::LlmContext *
    Response(const std::string &prompt, const std::function<bool(const std::string &, bool is_eop)> &on_progress);
    void SetMaxNewTokens(int i);

    void setSystemPrompt(std::string system_prompt);

    void SetAssistantPrompt(const std::string& assistant_prompt);

    void enableAudioOutput(bool b);

private:
    std::string response_string_for_debug{};
    std::string model_path_;
    std::vector<PromptItem> history_{};
    json extra_config_{};
    json config_{};
    bool is_r1_{false};
    bool stop_requested_{false};
    bool generate_text_end_{false};
    bool keep_history_{true};
    std::vector<float> waveform{};
    Llm* llm_{nullptr};
    std::string prompt_string_for_debug{};
    int max_new_tokens_{2048};
    std::string system_prompt_;
    json current_config_{};
    bool enable_audio_output_{false};
};
}

