//
// Created by ruoyi.sdj on 2025/4/18.
//
#pragma once
#include <vector>
#include <string>
#include <chrono>
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

    // 新增：API服务历史消息推理方法
    const MNN::Transformer::LlmContext *
    ResponseWithHistory(const std::vector<PromptItem>& full_history,
                        const std::function<bool(const std::string &, bool is_eop)> &on_progress);

    std::string getSystemPrompt() const;

    void clearHistory(int numToKeep = 1);

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
    std::string prompt_string_for_debug{};
    int max_new_tokens_{2048};
    std::string system_prompt_;
    json current_config_{};
    bool enable_audio_output_{false};
};
}

