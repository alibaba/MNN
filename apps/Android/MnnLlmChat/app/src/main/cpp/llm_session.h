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
    LlmSession(std::string, json config, std::vector<std::string> string_history);
    void reset();
    void Load();
    ~LlmSession();
    std::string getDebugInfo();
    const MNN::Transformer::LlmContext* getContext();
    const MNN::Transformer::LlmContext *
    Response(const std::string &prompt, const std::function<bool(const std::string &, bool is_eop)> &on_progress);

private:
    std::string response_string_for_debug{};
    std::string model_path_;
    std::vector<PromptItem> history_{};
    json config_{};
    bool is_r1_{false};
    bool stop_requested_{false};
    bool keep_history_{true};
    Llm* llm_{nullptr};
    std::string prompt_string_for_debug{};
};
}

