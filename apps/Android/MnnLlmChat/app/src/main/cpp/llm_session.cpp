//
// Created by ruoyi.sjd on 2025/4/18.
//

#include "llm_session.h"
#include <utility>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <unistd.h>
#include "context_budget.hpp"
#include "MNN/MNNForwardType.h"
#include "MNN/expr/ExecutorScope.hpp"
#include "mls_log.h"
#include "mls_config.h"
#include "utf8_stream_processor.hpp"
#include "llm_stream_buffer.hpp"
#include "processor.h"

#ifdef __ANDROID__
#include "video/video_processor.h"
#endif

void ReportLlmSetConfigToFirebase(const std::string& stage, const std::string& config_json);

namespace mls {

namespace {

constexpr size_t kMaxTokenCacheEntries = 2048;
constexpr size_t kMaxSummaryBytes = 4096;
constexpr int kPeakSampleInterval = 16;

// Anonymous resident memory: statm's resident minus shared, i.e. excluding the
// file-backed pages of mmapped weights. GPU buffers are not part of it.
long readAnonRssKb() {
    FILE* fp = fopen("/proc/self/statm", "r");
    if (fp == nullptr) {
        return 0;
    }
    long size = 0, resident = 0, shared = 0;
    const int read = fscanf(fp, "%ld %ld %ld", &size, &resident, &shared);
    fclose(fp);
    if (read != 3 || resident <= shared) {
        return 0;
    }
    const long page_kb = sysconf(_SC_PAGESIZE) / 1024;
    return (resident - shared) * (page_kb > 0 ? page_kb : 4);
}

void restoreAndroidSteppingStatusIfNeeded(Llm* llm) {
    if (llm == nullptr) {
        return;
    }
    auto* context = llm->getContext();
    if (context == nullptr) {
        return;
    }
    if (context->status == MNN::Transformer::LlmStatus::MAX_TOKENS_FINISHED ||
        context->status == MNN::Transformer::LlmStatus::NORMAL_FINISHED) {
        // Android links the prebuilt libMNN.so runtime, so this wrapper locally resets
        // terminal step states instead of depending on a shared engine change.
        auto* mutable_context = const_cast<MNN::Transformer::LlmContext*>(context);
        mutable_context->status = MNN::Transformer::LlmStatus::RUNNING;
    }
}

struct AndroidSteppingStreamState {
    std::stringstream& response_buffer;
    const std::function<bool(const std::string&, bool is_eop)>& on_progress;
    bool& generate_text_end;
    bool& stop_requested;
    std::string& response_string_for_debug;
    std::function<void(const std::string&)> on_response_complete;
    const char* result_log_tag;
    bool pending_eop = false;

    void processChunk(const std::string& utf8Char) {
        const bool is_eop = utf8Char.find("<eop>") != std::string::npos;
        if (!is_eop) {
            response_buffer << utf8Char;
            if (on_progress) {
                stop_requested = stop_requested || on_progress(utf8Char, false);
            }
            return;
        }
        pending_eop = true;
    }

    void finalizePendingEop() {
        if (!pending_eop) {
            return;
        }
        std::string response_result = response_buffer.str();
        MNN_DEBUG("%s %s", result_log_tag, response_result.c_str());
        response_string_for_debug = response_result;
        on_response_complete(response_result);
        if (on_progress) {
            stop_requested = stop_requested || on_progress("<eop>", true);
        }
        generate_text_end = true;
        pending_eop = false;
    }
};

void resolveAndroidSteppingEop(
        Llm* llm,
        AndroidSteppingStreamState& stream_state,
        int current_size,
        int max_new_tokens) {
    auto* context = llm != nullptr ? llm->getContext() : nullptr;
    if (context != nullptr &&
        context->status == MNN::Transformer::LlmStatus::MAX_TOKENS_FINISHED &&
        !stream_state.stop_requested &&
        current_size < max_new_tokens) {
        // Android currently links a prebuilt runtime that emits <eop> at the end of
        // every generate(1) step. Treat that as an intermediate boundary, not final end.
        restoreAndroidSteppingStatusIfNeeded(llm);
        if (stream_state.pending_eop) {
            stream_state.generate_text_end = false;
            stream_state.pending_eop = false;
        }
        return;
    }
    if (context != nullptr &&
        context->status == MNN::Transformer::LlmStatus::NORMAL_FINISHED &&
        !stream_state.pending_eop &&
        !stream_state.stop_requested &&
        current_size < max_new_tokens) {
        // After one round finishes naturally, prefill-only response(..., 0) can leave the
        // next round's first generate(1) starting from NORMAL_FINISHED in the prebuilt runtime.
        restoreAndroidSteppingStatusIfNeeded(llm);
        return;
    }
    if (stream_state.pending_eop) {
        stream_state.finalizePendingEop();
    }
}

} // namespace

std::string trimLeadingWhitespace(const std::string& str) {
    auto it = std::find_if(str.begin(), str.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    });
    return {it, str.end()};
}

std::string getUserString(const char* user_content, bool for_history, bool is_r1) {
    if (is_r1) {
        return "<|User|>" + std::string(user_content) + "<|Assistant|>" + (for_history ? "" : "<think>\n");
    } else {
        return user_content;
    }
}

std::string GetSystemPromptString(std::string system_prompt,  bool is_r1) {
    if (is_r1) {
        return std::string("<|begin_of_sentence|>") + system_prompt;
    } else {
        return system_prompt;
    }
}

// Also handles templates that pre-open the think block (the stream then carries only
// "</think>"); otherwise the whole reasoning trace would stay in history_.
std::string deleteThinkPart(std::string assistant_content) {
    return context::removeThinking(std::move(assistant_content));
}

std::string getR1AssistantString(std::string assistant_content) {
    std::size_t pos = assistant_content.find("</think>");
    if (pos != std::string::npos) {
        assistant_content.erase(0, pos + std::string("</think>").length());
    }
    return trimLeadingWhitespace(assistant_content) + "<|end_of_sentence|>";
}

// Process multimodal prompt and convert to MultimodalPrompt structure
mls::PromptProcessingResult processMultimodalPrompt(const std::string& prompt_text) {
    mls::PromptProcessor processor;
    return processor.Process(prompt_text);
}

void LlmSession::Reset() {
    std::lock_guard<std::mutex> lock(session_mutex_);
    applyPendingConfigLocked();
    history_.resize(1);
    context_summary_.clear();
    rebuildSystemMessageLocked();
    prepared_valid_ = false;
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        last_stats_ = ResponseStats();
        last_stats_.context_limit = context_limit_.load();
    }
    if (llm_) {
        llm_->reset();
    }
}

void LlmSession::applyContextKeys(const json& cfg) {
    if (!cfg.is_object()) {
        return;
    }
    auto limit_it = cfg.find("context_token_limit");
    if (limit_it != cfg.end() && limit_it->is_number_integer()) {
        requested_limit_ = limit_it->get<int>();
    }
    auto compress_it = cfg.find("auto_compress_context");
    if (compress_it != cfg.end() && compress_it->is_boolean()) {
        auto_compress_.store(compress_it->get<bool>());
    }
    context_limit_.store(context::clampLimit(requested_limit_, model_max_context_.load()));
}

void LlmSession::rebuildSystemMessageLocked() {
    std::string base;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        base = GetSystemPromptString(system_prompt_, is_r1_);
    }
    const std::string content = context::systemWithSummary(base, context_summary_);
    if (history_.empty()) {
        history_.emplace_back("system", content);
    } else {
        history_[0] = PromptItem("system", content);
    }
}

void LlmSession::applyPendingConfigLocked() {
    json config = json::object();
    bool has_prompt = false;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        std::swap(config, pending_config_);
        if (has_pending_system_prompt_) {
            system_prompt_ = pending_system_prompt_;
            has_pending_system_prompt_ = false;
            has_prompt = true;
        }
    }
    if (has_prompt) {
        rebuildSystemMessageLocked();
        prepared_valid_ = false;
    }
    if (config.empty()) {
        return;
    }
    for (auto& [key, value] : config.items()) {
        current_config_[key] = value;
    }
    applyContextKeys(config);
    prepared_valid_ = false;
    if (llm_) {
        auto config_str = current_config_.dump();
        ReportLlmSetConfigToFirebase("update_config", config_str);
        llm_->set_config(config_str);
        refreshDumpedConfigLocked();
        MNN_DEBUG("Updated config applied: %s", config_str.c_str());
    }
}

LlmSession::LlmSession(std::string model_path, json config, json extra_config, std::vector<std::string> history):
        model_path_(std::move(model_path)), config_(std::move(config)), extra_config_(std::move(extra_config)) {
    // Typed lookups with defaults: a missing or mistyped field must not throw (that aborts).
    if (!config_.is_object()) {
        config_ = json::object();
    }
    if (!extra_config_.is_object()) {
        extra_config_ = json::object();
    }
    auto read_int = [](const json& j, const char* key, int fallback) {
        auto it = j.find(key);
        return it != j.end() && it->is_number_integer() ? it->get<int>() : fallback;
    };
    auto read_bool = [](const json& j, const char* key, bool fallback) {
        auto it = j.find(key);
        return it != j.end() && it->is_boolean() ? it->get<bool>() : fallback;
    };
    auto read_string = [](const json& j, const char* key, const char* fallback) {
        auto it = j.find(key);
        return it != j.end() && it->is_string() ? it->get<std::string>() : std::string(fallback);
    };
    max_new_tokens_ = std::max(1, read_int(config_, "max_new_tokens", 2048));
    keep_history_ = read_bool(extra_config_, "keep_history", true);
    is_r1_ = read_bool(extra_config_, "is_r1", false);
    system_prompt_ = read_string(config_, "system_prompt", "You are a helpful assistant.");
    applyContextKeys(config_);
    auto summary_it = extra_config_.find("context_summary");
    if (summary_it != extra_config_.end() && summary_it->is_string()) {
        context_summary_ = context::sanitizeUtf8(summary_it->get<std::string>());
    }
    history_.emplace_back("system", context::systemWithSummary(
            GetSystemPromptString(system_prompt_, is_r1_), context_summary_));
    if (!history.empty()) {
        for (int i = 0; i < history.size(); i++) {
            if (is_r1_) {
                if (i % 2 == 0) {
                    history_.emplace_back("user", getUserString(history[i].c_str(), true, is_r1_));
                } else {
                    history_.emplace_back("assistant", getR1AssistantString(history[i]));
                }
            } else {
                history_.emplace_back(i % 2 == 0 ? "user" : "assistant",
                                      i % 2 == 0 ? history[i] :
                                      deleteThinkPart(history[i]));
            }
        }
    }
}

bool LlmSession::Load() {
    std::lock_guard<std::mutex> lock(session_mutex_);
    last_load_error_.clear();
    auto mmap_it = extra_config_.find("mmap_dir");
    const std::string root_cache_dir_str =
            mmap_it != extra_config_.end() && mmap_it->is_string() ? mmap_it->get<std::string>() : std::string();
    const bool use_mmap = !root_cache_dir_str.empty();
    llm_ = Llm::createLLM(model_path_);
    if (llm_ == nullptr) {
        last_load_error_ = "createLLM failed for config path: " + model_path_ +
            " (config file missing or invalid)";
        MNN_DEBUG("Failed to create LLM instance: %s", last_load_error_.c_str());
        model_loaded_ = false;
        return false;
    }
    json config = config_;
    config["use_mmap"] = use_mmap;
    if (use_mmap) {
        std::string temp_dir = root_cache_dir_str;
        config["tmp_path"] = temp_dir;
    }
    if (is_r1_) {
        config["use_template"] = false;
        config["precision"] = "high";
    }
    current_config_ = config;
    auto config_str = config.dump();
    MNN_DEBUG("extra_config: %s", config_str.c_str());
    ReportLlmSetConfigToFirebase("load", config_str);
    llm_->set_config(config_str);
    MNN_DEBUG("dumped config: %s", llm_->dump_config().c_str());
    model_loaded_ = llm_->load();
    if (!model_loaded_) {
        last_load_error_ = "Module load failed for config: " + model_path_ +
            ". Common causes: model file (.mnn) missing or corrupted, wrong backend (e.g. NPU on CPU-only device), insufficient memory.";
        MNN_DEBUG("Model load() returned false: %s", last_load_error_.c_str());
        return model_loaded_;
    }
    // The model window is only known once the config is loaded; it caps the user's limit.
    json dumped = json::parse(llm_->dump_config(), nullptr, false);
    if (dumped.is_object()) {
        auto it = dumped.find("max_position_embeddings");
        if (it != dumped.end() && it->is_number_integer() && it->get<int>() > 0) {
            model_max_context_.store(it->get<int>());
        }
    }
    context_limit_.store(context::clampLimit(requested_limit_, model_max_context_.load()));
    refreshDumpedConfigLocked();
    MNN_DEBUG("context budget: limit=%d model_max=%d auto_compress=%d",
              context_limit_.load(), model_max_context_.load(), auto_compress_.load() ? 1 : 0);
    return model_loaded_;
}

void LlmSession::samplePeakMemory() {
    const long current = readAnonRssKb();
    long peak = peak_anon_kb_.load();
    while (current > peak && !peak_anon_kb_.compare_exchange_weak(peak, current)) {
    }
}

int LlmSession::messageTokensLocked(const PromptItem& message) {
    auto it = token_cache_.find(message.second);
    if (it != token_cache_.end()) {
        return it->second;
    }
    // Qualified call: Omni overrides tokenizer_encode(string) to run the vision/audio
    // encoders on media tags. Counting must never do that.
    const int tokens = static_cast<int>(llm_->Llm::tokenizer_encode(message.second).size()) +
                       context::kTemplateOverheadPerMessage;
    if (token_cache_.size() >= kMaxTokenCacheEntries) {
        token_cache_.clear();
    }
    token_cache_.emplace(message.second, tokens);
    return tokens;
}

int LlmSession::mediaCountLocked(const std::vector<PromptItem>& messages) const {
    int count = 0;
    for (const auto& message : messages) {
        count += context::countMediaMarkers(message.second);
    }
    return count;
}

int LlmSession::estimateTokensLocked(const std::vector<PromptItem>& messages) {
    int total = mediaCountLocked(messages) * media_tokens_per_item_;
    for (const auto& message : messages) {
        total += messageTokensLocked(message);
    }
    return total;
}

int LlmSession::countPromptTokensLocked(const std::vector<PromptItem>& messages, bool include_media) {
    if (llm_ == nullptr || messages.empty()) {
        return 0;
    }
    const std::string prompt = llm_->apply_chat_template(messages);
    const int text = static_cast<int>(llm_->Llm::tokenizer_encode(prompt).size());
    return include_media ? text + mediaCountLocked(messages) * media_tokens_per_item_ : text;
}

void LlmSession::refreshDumpedConfigLocked() {
    std::string dumped = llm_ != nullptr ? llm_->dump_config() : std::string("{}");
    std::lock_guard<std::mutex> lock(dumped_config_mutex_);
    dumped_config_ = std::move(dumped);
}

// Drops the oldest turns until the prompt fits `budget`. The per-message estimate picks
// how far to cut (cheap, cached); the exact count then confirms, and only if the estimate
// was too optimistic is one more turn dropped per exact recount.
bool LlmSession::fitHistoryToBudgetLocked(int budget, int* exact_tokens) {
    bool trimmed = false;
    auto cost = [this](const PromptItem& message) {
        return messageTokensLocked(message) +
               context::countMediaMarkers(message.second) * media_tokens_per_item_;
    };
    if (estimateTokensLocked(history_) > budget) {
        const int low_water = budget * context::kTrimLowWaterPercent / 100;
        trimmed = context::trimOldest(history_, low_water, cost) > 0;
        *exact_tokens = countPromptTokensLocked(history_);
    }
    while (*exact_tokens > budget && history_.size() > 2) {
        const int estimate = estimateTokensLocked(history_);
        if (context::trimOldest(history_, estimate - 1, cost) == 0) {
            break;
        }
        trimmed = true;
        *exact_tokens = countPromptTokensLocked(history_);
    }
    return trimmed;
}

ResponseStats LlmSession::lastResponseStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return last_stats_;
}

int LlmSession::countContextTokens() {
    std::lock_guard<std::mutex> lock(session_mutex_);
    applyPendingConfigLocked();
    if (llm_ == nullptr) {
        return 0;
    }
    return countPromptTokensLocked(history_);
}

ContextCheck LlmSession::prepareContext(const std::string& prompt) {
    std::lock_guard<std::mutex> lock(session_mutex_);
    cancel_context_work_.store(false);
    applyPendingConfigLocked();
    ContextCheck check;
    check.limit = context_limit_.load();
    check.budget = context::promptBudget(check.limit, max_new_tokens_.load());
    if (llm_ == nullptr) {
        return check;
    }
    const PromptItem pending("user", getUserString(prompt.c_str(), false, is_r1_));
    std::vector<PromptItem> candidate;
    if (keep_history_) {
        candidate = history_;
    } else {
        candidate.push_back(history_.front());
    }
    candidate.push_back(pending);
    pending_message_tokens_ = messageTokensLocked(pending) +
            context::countMediaMarkers(pending.second) * media_tokens_per_item_;
    check.tokens = countPromptTokensLocked(candidate);
    if (check.tokens > check.budget) {
        const std::vector<PromptItem> minimal{history_.front(), pending};
        if (countPromptTokensLocked(minimal, media_calibrated_) > check.budget) {
            check.action = ContextAction::TooLong;
            prepared_valid_ = false;
            return check;
        }
    }
    // Any completed turn is worth summarizing: compression keeps as many recent turns as
    // fit, so nothing that would otherwise be trimmed is lost unsummarized.
    if (keep_history_ && !is_r1_ && auto_compress_.load() &&
        check.tokens > check.budget * context::kCompressTriggerPercent / 100 &&
        context::countUserTurns(history_) >= 1) {
        check.action = ContextAction::Compress;
    }
    prepared_valid_ = true;
    prepared_prompt_ = prompt;
    prepared_tokens_ = check.tokens;
    return check;
}

LlmSession::~LlmSession() {
    MNN_DEBUG("LIFECYCLE: LlmSession DESTROYED at %p", this);
    delete llm_;
}

const MNN::Transformer::LlmContext * LlmSession::Response(const std::string &prompt,
                                                          const std::function<bool(const std::string&, bool is_eop)>& on_progress) {
    std::lock_guard<std::mutex> lock(session_mutex_);
    if (llm_ == nullptr) {
        return nullptr;
    }
    applyPendingConfigLocked();
    peak_anon_kb_.store(0);
    ResponseStats stats;
    stats.context_limit = context_limit_.load();
    auto publish_stats = [this, &stats]() {
        stats.peak_anon_kb = peak_anon_kb_.load();
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        last_stats_ = stats;
    };

    if (!keep_history_) {
        history_.resize(1);
    }
    int current_size = 0;
    stop_requested_ = false;
    generate_text_end_ = false;
    std::stringstream response_buffer;
    AndroidSteppingStreamState stream_state{
            response_buffer,
            on_progress,
            generate_text_end_,
            stop_requested_,
            response_string_for_debug,
            [this](const std::string& raw_response) {
                std::string response_result = raw_response;
                if (is_r1_) {
                    auto& last_message = history_.at(history_.size() - 1);
                    std::size_t user_think_pos = last_message.second.find("<think>\n");
                    if (user_think_pos != std::string::npos) {
                        last_message.second.erase(user_think_pos, std::string("<think>\n").length());
                    }
                    response_result = getR1AssistantString(response_result);
                }
                response_result = trimLeadingWhitespace(deleteThinkPart(response_result));
                history_.emplace_back("assistant", response_result);
            },
            "submitNative Result"
    };
    mls::Utf8StreamProcessor processor([&stream_state](const std::string& utf8Char) {
        stream_state.processChunk(utf8Char);
    });
    LlmStreamBuffer stream_buffer{[&processor](const char* str, size_t len){
        processor.processStream(str, len);
    }};
    std::ostream output_ostream(&stream_buffer);
//#define USE_DEBUG_PROMOT
#ifdef USE_DEBUG_PROMOT
    std::string debug_prompt = "<audio>/data/user/0/com.alibaba.mnnllm.android/files/history/1746690738111/record_1746690751335.wav</audio>";
    history_.emplace_back("user", getUserString(debug_prompt.c_str(), false, is_r1_));
#else
    history_.emplace_back("user", getUserString(prompt.c_str(), false, is_r1_));
#endif
    MNN_DEBUG("submitNative history count %zu", history_.size());

    // Context budget: drop the oldest turns before prefill. Trimming history_ is enough;
    // the engine sees the shorter prefix, drops the stale KV and re-prefills.
    const int context_limit = stats.context_limit;
    const int budget = context::promptBudget(context_limit, max_new_tokens_.load());
    int prompt_tokens = (prepared_valid_ && prepared_prompt_ == prompt)
            ? prepared_tokens_ : countPromptTokensLocked(history_);
    prepared_valid_ = false;
    if (prompt_tokens > budget && history_.size() > 2) {
        // Reject before trimming: if the request cannot fit even alone, trimming would
        // throw away the conversation for nothing.
        const std::vector<PromptItem> minimal{history_.front(), history_.back()};
        if (countPromptTokensLocked(minimal, media_calibrated_) > budget) {
            history_.pop_back();
            stats.context_too_long = true;
            publish_stats();
            MNN_DEBUG("context_too_long: request alone exceeds budget %d", budget);
            return llm_->getContext();
        }
    }
    if (prompt_tokens > budget) {
        stats.truncated = fitHistoryToBudgetLocked(budget, &prompt_tokens);
    }
    if (prompt_tokens > budget && !media_calibrated_ && mediaCountLocked(history_) > 0) {
        // The media reserve is still a guess; let the real prefill decide (guarded below).
        prompt_tokens = countPromptTokensLocked(history_, false);
    }
    if (prompt_tokens > budget) {
        // Only the system message and this request are left. Back the request out so
        // history_ matches what ran, and report instead of prefilling.
        history_.pop_back();
        stats.context_too_long = true;
        publish_stats();
        MNN_DEBUG("context_too_long: tokens=%d budget=%d", prompt_tokens, budget);
        return llm_->getContext();
    }

    // Generate full prompt string using Prompt::applyTemplate
    std::string full_prompt_text;
    prompt_string_for_debug.clear();  // per turn; appending forever grew without bound
    for (auto & it : history_) {
        full_prompt_text += it.second;
        prompt_string_for_debug += it.second;
    }

    MNN_DEBUG("submitNative prompt_string_for_debug count %s max_new_tokens_:%d", prompt_string_for_debug.c_str(), max_new_tokens_.load());
    
    // Check for multimodal content in the full prompt
    auto multimodal_result = processMultimodalPrompt(full_prompt_text);
    restoreAndroidSteppingStatusIfNeeded(llm_);
    if (multimodal_result.has_multimodal) {
        MNN_DEBUG("Detected multimodal content, using multimodal API prompt %s with %zu images",
             multimodal_result.multimodal_prompt.prompt_template.c_str(),
             multimodal_result.multimodal_prompt.images.size());
        if (!multimodal_result.error_message.empty()) {
            MNN_ERROR("Multimodal processing errors: %s", multimodal_result.error_message.c_str());
        }
        // Prefill only. Stepping decode one token at a time via llm_->generate(1) keeps
        // streaming active without letting the engine mark the session as finished up front.
        llm_->response(multimodal_result.multimodal_prompt, &output_ostream, "<eop>", 0);
    } else {
        MNN_DEBUG("No multimodal content detected, using regular text API");
        llm_->response(history_, &output_ostream, "<eop>", 0);
    }
    samplePeakMemory();
    const int prefilled = llm_->getContext() != nullptr ? llm_->getContext()->all_seq_len : 0;
    const int media_items = mediaCountLocked(history_);
    if (media_items > 0 && prefilled > 0) {
        // Calibrate the per-media reserve from what the prefill really took.
        int text_tokens = 0;
        for (const auto& message : history_) {
            text_tokens += messageTokensLocked(message);
        }
        if (prefilled > text_tokens) {
            media_tokens_per_item_ = context::clampMediaTokens((prefilled - text_tokens) / media_items);
            media_calibrated_ = true;
        }
    }
    if (prefilled >= context_limit - context::kDecodeHeadroom) {
        // Media turned out larger than estimated and the window is full. Undo the turn
        // (KV included) rather than decode with no room left.
        history_.pop_back();
        llm_->reset();
        stats.context_too_long = true;
        stats.context_tokens = 0;
        publish_stats();
        MNN_DEBUG("context_too_long after prefill: all_seq_len=%d limit=%d", prefilled, context_limit);
        return llm_->getContext();
    }
    const int decode_limit = std::max(1, std::min(max_new_tokens_.load(), context_limit - prefilled));
    resolveAndroidSteppingEop(llm_, stream_state, current_size, decode_limit);
    while (!stop_requested_ && !generate_text_end_ && current_size < decode_limit) {
        llm_->generate(1);
        current_size++;
        if (current_size % kPeakSampleInterval == 0) {
            samplePeakMemory();
        }
        resolveAndroidSteppingEop(llm_, stream_state, current_size, decode_limit);
    }
    samplePeakMemory();
    if (!stop_requested_ && enable_audio_output_) {
        llm_->generateWavform();
    }
    if (llm_->getContext() != nullptr) {
        stats.context_tokens = llm_->getContext()->all_seq_len;
    }
    // Two post-decode outcomes:
    // 1. Natural <eop>: finalizePendingEop fires the on_response_complete
    //    callback which adds the assistant reply to history_ — sync cache.
    // 2. No <eop> (max-tokens truncation OR user cancel): the partial
    //    response was streamed to the UI and the caller always persists
    //    the assistant item, so add it to history_ and sync to keep
    //    engine state aligned with the visible conversation.
    size_t history_after = history_.size();
    stream_state.finalizePendingEop();
    if (history_.size() > history_after) {
        // (1) Natural <eop> — callback already added the assistant reply
        llm_->syncPromptCache(history_);
    } else if (current_size > 0) {
        // (2) No <eop> — add the partial response to history_ using the
        //     same processing as the on_response_complete callback.
        std::string response_result = response_buffer.str();
        if (is_r1_) {
            auto& last_message = history_.at(history_.size() - 1);
            std::size_t user_think_pos = last_message.second.find("<think>\n");
            if (user_think_pos != std::string::npos) {
                last_message.second.erase(user_think_pos, std::string("<think>\n").length());
            }
            response_result = getR1AssistantString(response_result);
        }
        response_result = trimLeadingWhitespace(deleteThinkPart(response_result));
        if (!response_result.empty()) {
            history_.emplace_back("assistant", response_result);
        }
        llm_->syncPromptCache(history_);
    }

    auto context = llm_->getContext();
    float prefill_s = context->prefill_us / 1e6f;
    float decode_s = context->decode_us / 1e6f;
    float prefill_tps = (prefill_s > 0) ? context->prompt_len / prefill_s : 0;
    float decode_tps = (decode_s > 0) ? context->gen_seq_len / decode_s : 0;
    MNN_DEBUG("PERF | prefill: %d tok in %.2fs (%.1f t/s) | decode: %d tok in %.2fs (%.1f t/s) | history: %zu msgs",
              context->prompt_len, prefill_s, prefill_tps,
              context->gen_seq_len, decode_s, decode_tps,
              history_.size());
    publish_stats();
    return context;
}

bool LlmSession::generateSummaryLocked(const std::string& request, int max_tokens, std::string* summary) {
    const std::vector<PromptItem> messages{{"user", request}};
    std::stringstream output;
    restoreAndroidSteppingStatusIfNeeded(llm_);
    // end_with must be explicit: nullptr keeps the previous turn's "<eop>".
    llm_->response(messages, &output, "", max_tokens);
    restoreAndroidSteppingStatusIfNeeded(llm_);
    std::string text = context::removeAll(output.str(), "<eop>");
    text = context::trimWhitespace(context::removeThinking(std::move(text)));
    text = context::utf8Prefix(context::sanitizeUtf8(text), kMaxSummaryBytes);
    if (text.empty()) {
        return false;
    }
    *summary = std::move(text);
    return true;
}

CompressResult LlmSession::compressContext() {
    std::lock_guard<std::mutex> lock(session_mutex_);
    CompressResult result;
    if (llm_ == nullptr || !keep_history_ || is_r1_) {
        return result;
    }
    applyPendingConfigLocked();
    const int limit = context_limit_.load();
    const int summary_tokens = context::summaryTokens(limit);
    // What remains after compression (summary + kept turns + the pending message) must sit
    // under the trim low-water mark, or Response() would drop kept turns unsummarized.
    const int budget = context::promptBudget(limit, max_new_tokens_.load());
    const int keep_target = budget * context::kTrimLowWaterPercent / 100 - summary_tokens -
                            pending_message_tokens_ - messageTokensLocked(history_.front());
    auto cost = [this](const PromptItem& message) {
        return messageTokensLocked(message) +
               context::countMediaMarkers(message.second) * media_tokens_per_item_;
    };
    const size_t keep_from = context::keepStartWithinBudget(
            history_, context::kKeepTurnsAfterCompress, keep_target, cost);
    if (keep_from <= 1) {
        return result;
    }
    const int input_budget = limit - summary_tokens - context::kDecodeHeadroom;

    // Summaries are written with thinking off (a reasoning trace would use up the whole
    // summary budget). Config merging can add or overwrite but never delete a key, so the
    // switch is only touched when it already exists; an absent key could not be restored
    // and templates differ on what "absent" means. Any reasoning is stripped regardless.
    bool has_thinking_switch = false;
    bool thinking_was_on = false;
    json dumped = json::parse(llm_->dump_config(), nullptr, false);
    if (dumped.is_object()) {
        auto jinja = dumped.find("jinja");
        if (jinja != dumped.end() && jinja->is_object()) {
            auto ctx = jinja->find("context");
            if (ctx != jinja->end() && ctx->is_object()) {
                auto flag = ctx->find("enable_thinking");
                if (flag != ctx->end() && flag->is_boolean()) {
                    has_thinking_switch = true;
                    thinking_was_on = flag->get<bool>();
                }
            }
        }
    }
    auto set_thinking = [this, has_thinking_switch](bool enabled) {
        if (!has_thinking_switch) {
            return;
        }
        json patch;
        patch["jinja"]["context"]["enable_thinking"] = enabled;
        llm_->set_config(patch.dump());
    };
    set_thinking(false);

    std::string merged = context_summary_;
    size_t cursor = 1;
    while (cursor < keep_from && !cancel_context_work_.load()) {
        const std::vector<size_t> ends = context::turnBoundaries(history_, cursor, keep_from);
        auto request_for = [&](size_t end) {
            return context::buildSummaryRequest(merged, context::buildTranscript(history_, cursor, end));
        };
        // Largest whole-turn batch whose request fits next to the summary output.
        size_t low = 0, high = ends.size(), fit = 0;
        while (low < high) {
            const size_t mid = low + (high - low) / 2;
            const std::vector<PromptItem> probe{{"user", request_for(ends[mid])}};
            if (countPromptTokensLocked(probe) <= input_budget) {
                fit = mid + 1;
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        std::string request;
        size_t batch_end;
        if (fit > 0) {
            batch_end = ends[fit - 1];
            request = request_for(batch_end);
        } else {
            // A single turn is larger than the window: summarize a prefix of it.
            batch_end = ends.front();
            const size_t max_bytes = static_cast<size_t>(std::max(256, input_budget)) * 2;
            request = context::buildSummaryRequest(
                    merged, context::utf8Prefix(context::buildTranscript(history_, cursor, batch_end), max_bytes));
        }
        std::string piece;
        if (!generateSummaryLocked(request, summary_tokens, &piece)) {
            MNN_DEBUG("compressContext: empty summary for messages [%zu, %zu)", cursor, batch_end);
            break;
        }
        merged = std::move(piece);
        cursor = batch_end;
    }

    set_thinking(thinking_was_on);
    refreshDumpedConfigLocked();
    // Summary generation replaced the KV and the prompt cache text; start clean.
    llm_->reset();
    prepared_valid_ = false;

    if (cursor > 1) {
        // Only completed batches are committed, and every batch ends on a user message,
        // so the kept history still starts with a user turn.
        context_summary_ = merged;
        history_.erase(history_.begin() + 1, history_.begin() + static_cast<long>(cursor));
        rebuildSystemMessageLocked();
        result.success = true;
        result.summary = merged;
        result.kept_user_turns = context::countUserTurns(history_);
    }
    MNN_DEBUG("compressContext: success=%d kept_user_turns=%d summary_bytes=%zu",
              result.success ? 1 : 0, result.kept_user_turns, result.summary.size());
    return result;
}

std::string LlmSession::getDebugInfo() {
    // The debug strings are written during generation; never block the UI to read them.
    std::unique_lock<std::mutex> lock(session_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        return "generating...";
    }
    return ("last_prompt:\n" + prompt_string_for_debug + "\nlast_response:\n" + response_string_for_debug);
}

void LlmSession::SetWavformCallback(std::function<bool(const float *, size_t, bool)> callback) {
    if (llm_ != nullptr && callback != nullptr) {
        waveform.clear();
        llm_->setWavformCallback([this, callback = std::move(callback)](const float *ptr, size_t size, bool last_chunk) {
#if DEBUG_SAVE_WAV
            waveform.reserve(waveform.size() + size);
            waveform.insert(waveform.end(), ptr, ptr + size);
            MNN_DEBUG("waveform size %zu", waveform.size());
            if (last_chunk) {
                auto waveform_var = MNN::Express::_Const(waveform.data(), {(int)waveform.size()}, MNN::Express::NCHW, halide_type_of<float>());
                MNN::AUDIO::save("/data/data/com.alibaba.mnnllm.android/files/output.wav", waveform_var, 24000);
                waveform.clear();
            }
#endif
            if (!enable_audio_output_ || stop_requested_) {
                return false;
            }
            if (callback) {
                return !callback(ptr, size, last_chunk);
            }
            return false;
        });
    } else {
        MNN_ERROR("no llm instance");
    }
}

void LlmSession::SetMaxNewTokens(int i) {
    max_new_tokens_ = std::max(1, i);
}

// The setters below are called from the UI thread. They queue the change and apply it
// right away only if no generation holds session_mutex_; otherwise the next locked
// operation applies it before it starts.
void LlmSession::setSystemPrompt(std::string system_prompt) {
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_system_prompt_ = std::move(system_prompt);
        has_pending_system_prompt_ = true;
    }
    std::unique_lock<std::mutex> session(session_mutex_, std::try_to_lock);
    if (session.owns_lock()) {
        applyPendingConfigLocked();
    }
}

void LlmSession::SetAssistantPrompt(const std::string& assistant_prompt) {
    json patch = json::object();
    patch["assistant_prompt_template"] = assistant_prompt;
    updateConfig(patch.dump());
}

void LlmSession::updateConfig(const std::string& config_json) {
    json new_config = json::parse(config_json, nullptr, false);
    if (!new_config.is_object()) {
        MNN_ERROR("Failed to parse config JSON: invalid JSON format");
        return;
    }
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        for (auto& [key, value] : new_config.items()) {
            pending_config_[key] = value;
        }
    }
    std::unique_lock<std::mutex> session(session_mutex_, std::try_to_lock);
    if (session.owns_lock()) {
        applyPendingConfigLocked();
    } else {
        MNN_DEBUG("Session busy, config queued for the next turn");
    }
}

void LlmSession::enableAudioOutput(bool enable) {
    enable_audio_output_ = enable;
}

const MNN::Transformer::LlmContext * LlmSession::ResponseWithHistory(
        const std::vector<PromptItem>& full_history,
        const std::function<bool(const std::string&, bool is_eop)>& on_progress) {
    std::lock_guard<std::mutex> lock(session_mutex_);
    if (llm_ == nullptr) {
        return nullptr;
    }
    applyPendingConfigLocked();

    // Create temporary history without modifying member variables
    std::vector<PromptItem> temp_history;

    // Directly use the provided full history without saving to member variables
    temp_history.insert(temp_history.end(), full_history.begin(), full_history.end());

    int current_size = 0;
    stop_requested_ = false;
    generate_text_end_ = false;
    std::stringstream response_buffer;

    // Stream processing logic without modifying history_ member
    AndroidSteppingStreamState stream_state{
            response_buffer,
            on_progress,
            generate_text_end_,
            stop_requested_,
            response_string_for_debug,
            [this](const std::string& raw_response) {
                std::string response_result = raw_response;
                if (is_r1_) {
                    response_result = getR1AssistantString(response_result);
                }
                response_result = trimLeadingWhitespace(deleteThinkPart(response_result));
            },
            "ResponseWithHistory Result"
    };
    mls::Utf8StreamProcessor processor([&stream_state](const std::string& utf8Char) {
        stream_state.processChunk(utf8Char);
    });

    LlmStreamBuffer stream_buffer{[&processor](const char* str, size_t len){
        processor.processStream(str, len);
    }};
    std::ostream output_ostream(&stream_buffer);



    MNN_DEBUG("submitNative history count %zu", temp_history.size());
    prompt_string_for_debug.clear(); // Clear old data to avoid duplicate appending
    
    // Generate full prompt text from history
    std::string full_prompt_text;
    for (const auto& it : temp_history) {
        prompt_string_for_debug += "[" + it.first + "]: " + it.second + "\n";
        full_prompt_text += it.second;
    }
    
    MNN_DEBUG("submitNative prompt_string_for_debug:\n%s\nmax_new_tokens_:%d", prompt_string_for_debug.c_str(), max_new_tokens_.load());
    
    // Check for multimodal content in the full prompt
    auto multimodal_result = processMultimodalPrompt(full_prompt_text);
    restoreAndroidSteppingStatusIfNeeded(llm_);
    if (multimodal_result.has_multimodal) {
        MNN_DEBUG("ResponseWithHistory: Detected multimodal content, using multimodal API");
        if (!multimodal_result.error_message.empty()) {
            MNN_ERROR("ResponseWithHistory: Multimodal processing errors: %s", multimodal_result.error_message.c_str());
        }
        llm_->response(multimodal_result.multimodal_prompt, &output_ostream, "<eop>", 0);
    } else {
        MNN_DEBUG("ResponseWithHistory: No multimodal content detected, using regular text API");
        llm_->response(temp_history, &output_ostream, "<eop>", 0);
    }
    size_t kv_before_decode = llm_->getCurrentHistory();
    resolveAndroidSteppingEop(llm_, stream_state, current_size, max_new_tokens_);

    while (!stop_requested_ && !generate_text_end_ && current_size < max_new_tokens_) {
        llm_->generate(1);
        current_size++;
        resolveAndroidSteppingEop(llm_, stream_state, current_size, max_new_tokens_);
    }

    if (!stop_requested_ && enable_audio_output_) {
        llm_->generateWavform();
    }

    // Sync prompt cache if generation completed normally: either by <eop>
    // (generate_text_end_) or by reaching max_new_tokens. Only skip sync
    // when explicitly stopped (disconnect/cancel) since the caller typically
    // won't include that partial reply in the next request.
    stream_state.finalizePendingEop();  // flush any pending eop
    if (!stop_requested_) {
        std::string response_result = response_buffer.str();
        if (is_r1_) {
            response_result = getR1AssistantString(response_result);
        }
        response_result = trimLeadingWhitespace(deleteThinkPart(response_result));
        if (!response_result.empty()) {
            temp_history.emplace_back("assistant", response_result);
        }
        llm_->syncPromptCache(temp_history);
    } else if (current_size > 0) {
        // Cancelled — engine's KV/history is ahead of what the caller will
        // include in the next request. Roll back to pre-decode state.
        llm_->eraseHistory(kv_before_decode, 0);
    }

    return llm_->getContext();
}

void LlmSession::clearHistory(int numToKeep) {
    std::lock_guard<std::mutex> lock(session_mutex_);
    if (numToKeep < 0) {
        numToKeep = 0;
    }
    if (history_.size() > static_cast<size_t>(numToKeep)) {
        history_.erase(history_.begin() + numToKeep, history_.end());
    }
    // 清空相关缓存
    prompt_string_for_debug.clear();
    //response_string_for_debug.clear();
    context_summary_.clear();
    if (!history_.empty()) {
        rebuildSystemMessageLocked();
    }
    prepared_valid_ = false;
    if (llm_) {
        llm_->reset();
    }
}

std::string LlmSession::getSystemPrompt() const {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    return has_pending_system_prompt_ ? pending_system_prompt_ : system_prompt_;
}

std::string LlmSession::dumpConfig() const {
    std::lock_guard<std::mutex> lock(dumped_config_mutex_);
    return dumped_config_;
}

// Pure C++ benchmark implementation following llm_bench.cpp exactly
LlmSession::BenchmarkResult LlmSession::runBenchmark(int backend, int threads, bool useMmap, int power, 
                                                    int precision, int memory, int dynamicOption, int nPrompt, 
                                                    int nGenerate, int nRepeat, bool kvCache, 
                                                    const BenchmarkCallback& callback) {
    MNN_DEBUG("BENCHMARK: runBenchmark() STARTED! this=%p", this);
    MNN_DEBUG("BENCHMARK: Parameters - nPrompt=%d, nGenerate=%d, nRepeat=%d, kvCache=%s", 
                nPrompt, nGenerate, nRepeat, kvCache ? "true" : "false");
    
    // Initialize result structure
    MNN_DEBUG("BENCHMARK: Initializing benchmark result structure");
    BenchmarkResult result = initializeBenchmarkResult(nPrompt, nGenerate, nRepeat, kvCache);
    
    // Initialize LLM for benchmark
    MNN_DEBUG("BENCHMARK: About to initialize LLM for benchmark");
    if (!initializeLlmForBenchmark(result, callback)) {
        MNN_DEBUG("BENCHMARK: initializeLlmForBenchmark FAILED!");
        return result;
    }
    MNN_DEBUG("BENCHMARK: initializeLlmForBenchmark SUCCESS - entering benchmark loop");

    // Run benchmark iterations
    MNN_DEBUG("BENCHMARK: Starting benchmark loop for %d iterations", nRepeat + 1);
    for (int i = 0; i < nRepeat + 1; ++i) {
        MNN_DEBUG("BENCHMARK: Starting iteration %d/%d", i, nRepeat);
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Report progress
        MNN_DEBUG("BENCHMARK: Reporting progress for iteration %d", i);
        reportBenchmarkProgress(i, nRepeat, nPrompt, nGenerate, callback);
        
        // Run the actual test
        if (kvCache) {
            if (!runKvCacheTest(i, nPrompt, nGenerate, start_time, result, callback)) {
                return result;
            }
        } else {
            if (!runLlamaBenchTest(i, nPrompt, nGenerate, start_time, result, callback)) {
                return result;
            }
        }
    }
    
    // Report completion
    if (callback.onProgress) {
        BenchmarkProgressInfo completionInfo;
        completionInfo.progress = 100;
        completionInfo.statusMessage = "Benchmark completed!";
        completionInfo.progressType = ProgressType::COMPLETED;
        callback.onProgress(completionInfo);
    }

    result.success = true;
    return result;
}

// Initialize benchmark result structure
LlmSession::BenchmarkResult LlmSession::initializeBenchmarkResult(int nPrompt, int nGenerate, int nRepeat, bool kvCache) {
    BenchmarkResult result;
    result.prompt_tokens = nPrompt;
    result.generate_tokens = nGenerate;
    result.repeat_count = nRepeat;
    result.kv_cache_enabled = kvCache;
    result.success = false;
    return result;
}

// Initialize LLM for benchmark and verify it's ready
bool LlmSession::initializeLlmForBenchmark(BenchmarkResult& result, const BenchmarkCallback& callback) {
    // Validate this pointer first
    // Note: this check is removed as it's always false in well-defined C++ code

    // Get underlying Llm object for direct access
    auto *llm = this->getLlm();
    if (!llm) {
        result.error_message = "Underlying LLM object is not initialized";
        if (callback.onError) callback.onError(result.error_message);
        return false;
    }

    // Basic validation that the llm object pointer looks reasonable
    // Check if the pointer is in a reasonable memory range (basic sanity check)
    if (reinterpret_cast<uintptr_t>(llm) < 0x1000) {
        result.error_message = "LLM object pointer appears invalid (too low address)";
        if (callback.onError) callback.onError(result.error_message);
        return false;
    }

    // Verify LLM context is valid before proceeding
    auto context = llm->getContext();
    if (!context) {
        result.error_message = "LLM context is not valid - model may not be properly loaded";
        if (callback.onError) callback.onError(result.error_message);
        return false;
    }

    // Reset session state for clean benchmark
    this->Reset();
    
    // Re-verify context after reset
    context = llm->getContext();
    if (!context) {
        result.error_message = "LLM context became invalid after reset";
        if (callback.onError) callback.onError(result.error_message);
        return false;
    }

    return true;
}

// Report benchmark progress
void LlmSession::reportBenchmarkProgress(int iteration, int nRepeat, int nPrompt, int nGenerate, const BenchmarkCallback& callback) {
    if (callback.onProgress) {
        BenchmarkProgressInfo progressInfo;
        
        if (iteration == 0) {
            progressInfo.progress = 0;
            progressInfo.statusMessage = "Warming up...";
            progressInfo.progressType = ProgressType::WARMING_UP;
        } else {
            progressInfo.progress = (iteration * 100) / nRepeat;
            progressInfo.statusMessage = "Running test " + std::to_string(iteration) + "/" + std::to_string(nRepeat) + 
                                        " (prompt=" + std::to_string(nPrompt) + ", generate=" + std::to_string(nGenerate) + ")";
            progressInfo.progressType = ProgressType::RUNNING_TEST;
        }
        
        // Set structured data
        progressInfo.currentIteration = iteration;
        progressInfo.totalIterations = nRepeat;
        progressInfo.nPrompt = nPrompt;
        progressInfo.nGenerate = nGenerate;
        
        callback.onProgress(progressInfo);
    }
}

// Run KV cache test iteration
bool LlmSession::runKvCacheTest(int iteration, int nPrompt, int nGenerate, 
                                std::chrono::high_resolution_clock::time_point start_time,
                                BenchmarkResult& result, const BenchmarkCallback& callback) {
    auto *llm = this->getLlm();
    const int tok = 16; // Same token ID as used in llm_bench.cpp
    
    std::vector<int> tokens(nPrompt, tok);
    
    // Validate token vector
    if (tokens.empty() || nPrompt <= 0) {
        result.error_message = "Invalid token configuration for kv-cache test";
        if (callback.onError) callback.onError(result.error_message);
        return false;
    }
    
    llm->response(tokens, nullptr, nullptr, nGenerate);
    
    // Re-get context after response to ensure it's still valid
    auto context = llm->getContext();
    if (!context) {
        result.error_message = "Context became invalid after response in kv-cache test " + std::to_string(iteration);
        if (callback.onError) callback.onError(result.error_message);
        return false;
    }
    
    if (iteration > 0) { // Exclude the first performance value
        auto end_time = std::chrono::high_resolution_clock::now();
        processBenchmarkResults(context->prefill_us, context->decode_us, start_time, end_time, iteration, nPrompt, nGenerate, 
                                result, callback, true);
    }
    return true;
}

// Run llama-bench test iteration (without kv cache)
bool LlmSession::runLlamaBenchTest(int iteration, int nPrompt, int nGenerate,
                                    std::chrono::high_resolution_clock::time_point start_time,
                                    BenchmarkResult& result, const BenchmarkCallback& callback) {
    auto *llm = this->getLlm();
    const int tok = 500;
    int64_t prefill_us = 0;
    int64_t decode_us = 0;
    std::vector<int> tokens(nPrompt, tok);
    std::vector<int> tokens1(1, tok);
    
    // Validate token vectors
    if ((nPrompt > 0 && tokens.empty()) || tokens1.empty()) {
        result.error_message = "Invalid token configuration for llama-bench test " + std::to_string(iteration);
        if (callback.onError) callback.onError(result.error_message);
        return false;
    }
    MNN_DEBUG("runLlamaBenchTest nPrompt:%d, nGenerate:%d tokens:\n", nPrompt, nGenerate);
    if (nPrompt > 0) {
        MNN_DEBUG("runLlamaBenchTest prefill begin");
        llm->response(tokens, nullptr, nullptr, 1);
        MNN_DEBUG("runLlamaBenchTest prefill beginx");
        auto context = llm->getContext();
        if (!context) {
            result.error_message = "Context became invalid after prefill response in llama-bench test " + std::to_string(iteration);
            if (callback.onError) callback.onError(result.error_message);
            return false;
        }
        MNN_DEBUG("runLlamaBenchTest prefill end");
        prefill_us = context->prefill_us;
    }
    
    if (nGenerate > 0) {
        MNN_DEBUG("runLlamaBenchTest generate begin");
        llm->response(tokens1, nullptr, nullptr, nGenerate);

        auto context = llm->getContext();
        if (!context) {
            result.error_message = "Context became invalid after decode response in llama-bench test " + std::to_string(iteration);
            if (callback.onError) callback.onError(result.error_message);
            return false;
        }
        decode_us = context->decode_us;
        MNN_DEBUG("runLlamaBenchTest generate end");
    }
    
    if (iteration > 0) { // Exclude the first performance value
        auto end_time = std::chrono::high_resolution_clock::now();
        
        processBenchmarkResults(prefill_us, decode_us,
                                start_time, end_time, iteration, nPrompt, nGenerate, 
                                result, callback, false);
        
        result.sample_times_us.push_back(prefill_us + decode_us);
        result.decode_times_us.push_back(decode_us);
        result.prefill_times_us.push_back(prefill_us);
    }
    return true;
}

// Process and report benchmark results
void LlmSession::processBenchmarkResults(int64_t prefillTime, int64_t decodeTime,
                                        std::chrono::high_resolution_clock::time_point start_time,
                                        std::chrono::high_resolution_clock::time_point end_time,
                                        int iteration, int nPrompt, int nGenerate,
                                        BenchmarkResult& result, const BenchmarkCallback& callback,
                                        bool isKvCache) {
    auto runTime = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    if (isKvCache) {
        result.prefill_times_us.push_back(prefillTime);
        result.decode_times_us.push_back(decodeTime);
    }
    
    // Convert times to seconds
    float runTimeSeconds = runTime / 1000000.0f;
    float prefillTimeSeconds = prefillTime / 1000000.0f;
    float decodeTimeSeconds = decodeTime / 1000000.0f;
    
    // Calculate speeds (tokens per second)
    float prefillSpeed = (prefillTime > 0 && nPrompt > 0) ? ((float)nPrompt / prefillTimeSeconds) : 0.0f;
    float decodeSpeed = (decodeTime > 0 && nGenerate > 0) ? ((float)nGenerate / decodeTimeSeconds) : 0.0f;
    
    // Report detailed results with structured data
    BenchmarkProgressInfo detailedInfo;
    detailedInfo.progress = (iteration * 100) / result.repeat_count;
    detailedInfo.progressType = ProgressType::RUNNING_TEST;
    detailedInfo.currentIteration = iteration;
    detailedInfo.totalIterations = result.repeat_count;
    detailedInfo.nPrompt = nPrompt;
    detailedInfo.nGenerate = nGenerate;
    detailedInfo.runTimeSeconds = runTimeSeconds;
    detailedInfo.prefillTimeSeconds = prefillTimeSeconds;
    detailedInfo.decodeTimeSeconds = decodeTimeSeconds;
    detailedInfo.prefillSpeed = prefillSpeed;
    detailedInfo.decodeSpeed = decodeSpeed;
    
    // Keep detailed message for backward compatibility
    char detailedMsg[1024];
    snprintf(detailedMsg, sizeof(detailedMsg), 
        "BenchmarkService: Native Progress [%dp+%dg] (%d%%): Running test %d/%d (prompt=%d, generate=%d) runTime:%.3fs, prefillTime:%.3fs, decodeTime:%.3fs, prefillSpeed:%.2f tok/s, decodeSpeed:%.2f tok/s",
        nPrompt, nGenerate, detailedInfo.progress, iteration, result.repeat_count, nPrompt, nGenerate, 
        runTimeSeconds, prefillTimeSeconds, decodeTimeSeconds, prefillSpeed, decodeSpeed);
    
    detailedInfo.statusMessage = std::string(detailedMsg);
    
    MNN_DEBUG("%s", detailedMsg);
    
    if (callback.onProgress) {
        callback.onProgress(detailedInfo);
    }
    
    if (callback.onIterationComplete) {
        callback.onIterationComplete(std::string(detailedMsg));
    }
}

}
