#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <llm/llm.hpp>

namespace mnn_harmony {

struct GenerateRequest {
    std::string requestId;
    MNN::Transformer::ChatMessages messages;
    int maxTokens = 256;
    // When set, omni models with a talker write the spoken waveform to this WAV path.
    std::string audioOutputPath;
};

struct GenerateResult {
    std::string requestId;
    std::string text;
    int promptTokens = 0;
    int generatedTokens = 0;
    double prefillMs = 0.0;
    double decodeMs = 0.0;
    double totalMs = 0.0;
    double firstTokenMs = 0.0;
    bool canceled = false;
};

struct BenchmarkSample {
    int iteration = 0;
    int promptTokens = 0;
    int generatedTokens = 0;
    double prefillMs = 0.0;
    double decodeMs = 0.0;
    double totalMs = 0.0;
};

struct BenchmarkRequest {
    std::string requestId;
    int promptTokens = 128;
    int generateTokens = 128;
    int repeats = 3;
};

struct BenchmarkResult {
    std::string requestId;
    std::vector<BenchmarkSample> samples;
    bool canceled = false;
};

class LlmSession {
public:
    using CancellationFlag = std::shared_ptr<std::atomic<bool>>;
    using ChunkCallback = std::function<void(const std::string&)>;
    using BenchmarkCallback = std::function<void(const BenchmarkSample&)>;

    static LlmSession& instance();

    void load(const std::string& configPath, const std::string& runtimeConfigJson, double& loadMs, bool& reused,
              std::string& error);
    void registerRequest(const std::string& requestId);
    void generate(const GenerateRequest& request, const ChunkCallback& onChunk, GenerateResult& result,
                  std::string& error);
    void benchmark(const BenchmarkRequest& request, const BenchmarkCallback& onProgress, BenchmarkResult& result,
                   std::string& error);
    void cancel(const std::string& requestId);
    void reset();
    void release();
    std::string dumpConfig();

private:
    CancellationFlag getCancellation(const std::string& requestId);
    void completeRequest(const std::string& requestId, const CancellationFlag& cancellation);
    void prepareForResponse();

    std::mutex mOperationMutex;
    std::mutex mCancellationMutex;
    // Cached at load time so reading the config from JS never blocks behind a running generation.
    std::mutex mConfigMutex;
    std::string mConfigDump;
    std::unique_ptr<MNN::Transformer::Llm> mLlm;
    std::unordered_map<std::string, CancellationFlag> mCancellations;
    std::string mLoadIdentity;
    double mLoadMs = 0.0;
};

} // namespace mnn_harmony
