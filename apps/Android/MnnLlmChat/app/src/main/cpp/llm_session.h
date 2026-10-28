//
// Created by ruoyi.sdj on 2025/4/18.
//
#pragma once
#include <atomic>
#include <vector>
#include <string>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include "nlohmann/json.hpp"
#include "llm/llm.hpp"

// Forward declarations for JNI types
#ifdef __cplusplus
extern "C" {
#endif
typedef struct _JNIEnv JNIEnv;
typedef struct _jobject* jobject;
#ifdef __cplusplus
}
#endif

using nlohmann::json;
using MNN::Transformer::Llm;

namespace mls {
using PromptItem = std::pair<std::string, std::string>;

enum class ContextAction : int {
    Send = 0,     // the prompt fits
    Compress = 1, // close to the budget: summarize older turns first
    TooLong = 2,  // the pending message alone exceeds the budget
};

struct ContextCheck {
    int tokens = 0;
    int budget = 0;
    int limit = 0;
    ContextAction action = ContextAction::Send;
};

struct CompressResult {
    bool success = false;
    std::string summary;
    // User turns still kept verbatim after the summary. Kotlin counts this many user rows
    // back from the end of the stored conversation to find the summary's anchor row.
    int kept_user_turns = 0;
};

// Telemetry of the last Response(), read by JNI after it returns.
struct ResponseStats {
    int context_tokens = 0;
    int context_limit = 0;
    bool truncated = false;
    bool context_too_long = false;
    long peak_anon_kb = 0;
};

class LlmSession {
public:
    LlmSession(std::string, json config, json extra_config, std::vector<std::string> string_history);
    void Reset();
    bool Load();
    bool isModelReady() const { return llm_ != nullptr && model_loaded_; }
    /** Last error message when Load() fails. Cleared on success. */
    const std::string& getLastLoadError() const { return last_load_error_; }
    ~LlmSession();
    std::string getDebugInfo();
    void SetWavformCallback(std::function<bool(const float*, size_t, bool)> callback);
    const MNN::Transformer::LlmContext *
    Response(const std::string &prompt, const std::function<bool(const std::string &, bool is_eop)> &on_progress);
    void SetMaxNewTokens(int i);

    // Context budget. These take session_mutex_ and may run for a while (prepare
    // tokenizes the history, compress generates a summary), so they are called from a
    // worker thread only. The setters below never block on that lock.
    ContextCheck prepareContext(const std::string& prompt);
    CompressResult compressContext();
    int countContextTokens();
    // Asks a running compressContext to stop after the current summary batch.
    void cancelContextWork() { cancel_context_work_.store(true); }
    ResponseStats lastResponseStats() const;
    int contextLimit() const { return context_limit_.load(); }
    int modelMaxContext() const { return model_max_context_.load(); }

    void setSystemPrompt(std::string system_prompt);

    void SetAssistantPrompt(const std::string& assistant_prompt);

    void updateConfig(const std::string& config_json);

    void enableAudioOutput(bool b);

    // 新增：API服务历史消息推理方法
    const MNN::Transformer::LlmContext *
    ResponseWithHistory(const std::vector<PromptItem>& full_history,
                        const std::function<bool(const std::string &, bool is_eop)> &on_progress);

    std::string getSystemPrompt() const;

    void clearHistory(int numToKeep = 1);

    std::string dumpConfig() const;

    // Add getter method for underlying Llm object for benchmarking purposes
    Llm* getLlm() const { return llm_; }
    
    // Platform-independent benchmark result structure
    struct BenchmarkResult {
        bool success;
        std::string error_message;
        std::vector<int64_t> prefill_times_us;
        std::vector<int64_t> decode_times_us;
        std::vector<int64_t> sample_times_us;
        int prompt_tokens;
        int generate_tokens;
        int repeat_count;
        bool kv_cache_enabled;
    };
    
    // Progress type enumeration for structured reporting
    enum class ProgressType {
        UNKNOWN = 0,
        INITIALIZING = 1,
        WARMING_UP = 2,
        RUNNING_TEST = 3,
        PROCESSING_RESULTS = 4,
        COMPLETED = 5,
        STOPPING = 6
    };
    
    // Structured progress information
    struct BenchmarkProgressInfo {
        int progress;              // 0-100
        std::string statusMessage; // Keep for backward compatibility
        ProgressType progressType;
        int currentIteration;
        int totalIterations;
        int nPrompt;
        int nGenerate;
        float runTimeSeconds;
        float prefillTimeSeconds;
        float decodeTimeSeconds;
        float prefillSpeed;
        float decodeSpeed;
        
        BenchmarkProgressInfo() : progress(0), statusMessage(""), progressType(ProgressType::UNKNOWN),
                                currentIteration(0), totalIterations(0), nPrompt(0), nGenerate(0),
                                runTimeSeconds(0.0f), prefillTimeSeconds(0.0f), decodeTimeSeconds(0.0f),
                                prefillSpeed(0.0f), decodeSpeed(0.0f) {}
    };
    
    // Platform-independent benchmark callback interface
    struct BenchmarkCallback {
        std::function<void(const BenchmarkProgressInfo& progressInfo)> onProgress;
        std::function<void(const std::string& error)> onError;
        std::function<void(const std::string& detailed_stats)> onIterationComplete;
    };
    
    // Pure C++ benchmark method (platform-independent)
    BenchmarkResult runBenchmark(int backend, int threads, bool useMmap, int power, 
                                int precision, int memory, int dynamicOption, int nPrompt, 
                                int nGenerate, int nRepeat, bool kvCache, 
                                const BenchmarkCallback& callback);

private:
    // All *Locked helpers require session_mutex_.
    void applyPendingConfigLocked();
    void applyContextKeys(const json& cfg);
    void rebuildSystemMessageLocked();
    int messageTokensLocked(const PromptItem& message);
    int estimateTokensLocked(const std::vector<PromptItem>& messages);
    // include_media=false counts text only; used before the per-media cost is calibrated
    // so a message is never rejected on the strength of a guess.
    int countPromptTokensLocked(const std::vector<PromptItem>& messages, bool include_media = true);
    void refreshDumpedConfigLocked();
    int mediaCountLocked(const std::vector<PromptItem>& messages) const;
    bool fitHistoryToBudgetLocked(int budget, int* exact_tokens);
    bool generateSummaryLocked(const std::string& request, int max_tokens, std::string* summary);
    void samplePeakMemory();

    // Benchmark helper functions
    BenchmarkResult initializeBenchmarkResult(int nPrompt, int nGenerate, int nRepeat, bool kvCache);
    bool initializeLlmForBenchmark(BenchmarkResult& result, const BenchmarkCallback& callback);
    void reportBenchmarkProgress(int iteration, int nRepeat, int nPrompt, int nGenerate, const BenchmarkCallback& callback);
    bool runKvCacheTest(int iteration, int nPrompt, int nGenerate, 
                       std::chrono::high_resolution_clock::time_point start_time,
                       BenchmarkResult& result, const BenchmarkCallback& callback);
    bool runLlamaBenchTest(int iteration, int nPrompt, int nGenerate,
                          std::chrono::high_resolution_clock::time_point start_time,
                          BenchmarkResult& result, const BenchmarkCallback& callback);
    void processBenchmarkResults(int64_t prefillTime, int64_t decodeTime,
                               std::chrono::high_resolution_clock::time_point start_time,
                               std::chrono::high_resolution_clock::time_point end_time,
                               int iteration, int nPrompt, int nGenerate,
                               BenchmarkResult& result, const BenchmarkCallback& callback,
                               bool isKvCache);

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
    bool model_loaded_{false};
    std::string prompt_string_for_debug{};
    std::atomic<int> max_new_tokens_{2048};
    std::string system_prompt_;
    json current_config_{};
    std::atomic<bool> enable_audio_output_{false};
    std::string last_load_error_{};

    // Serializes everything that touches llm_ or history_: tokenizer, KV cache and
    // history are not thread safe, and ChatPresenter dispatches from Dispatchers.IO.
    // Stopping a generation goes through on_progress and never needs this lock.
    mutable std::mutex session_mutex_;
    // Config changes from the UI thread are queued here and applied by whoever holds
    // session_mutex_ next, so a settings tap never waits for a running generation.
    mutable std::mutex pending_mutex_;
    json pending_config_ = json::object();
    bool has_pending_system_prompt_{false};
    std::string pending_system_prompt_;

    int requested_limit_{8192}; // user's choice, before the model-window cap
    std::atomic<int> context_limit_{8192};
    std::atomic<int> model_max_context_{0};
    std::atomic<bool> auto_compress_{true};
    std::atomic<bool> cancel_context_work_{false};
    std::string context_summary_;
    std::unordered_map<std::string, int> token_cache_;
    int media_tokens_per_item_{1024};
    bool media_calibrated_{false};
    // Copy of llm_->dump_config() for UI-thread readers; refreshed under session_mutex_.
    mutable std::mutex dumped_config_mutex_;
    std::string dumped_config_{"{}"};
    // Estimated cost of the message prepareContext checked; compression leaves room for it.
    int pending_message_tokens_{0};
    // prepareContext's exact count, reused by the Response() that follows it.
    bool prepared_valid_{false};
    std::string prepared_prompt_;
    int prepared_tokens_{0};

    mutable std::mutex stats_mutex_;
    ResponseStats last_stats_;
    std::atomic<long> peak_anon_kb_{0};
};
}

