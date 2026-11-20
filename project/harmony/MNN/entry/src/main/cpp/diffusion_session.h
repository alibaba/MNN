#pragma once

#include <functional>
#include <mutex>
#include <string>

namespace mnn_harmony {

struct DiffusionRequest {
    std::string requestId;
    std::string resourcePath;
    std::string prompt;
    std::string negativePrompt;
    std::string imageInput;
    std::string outputPath;
    int modelType = 2;  // 0 = SD1.5, 2 = Sana
    int backendType = 3;  // MNN_FORWARD_OPENCL
    int memoryMode = 0;
    int width = 512;
    int height = 512;
    int steps = 10;
    int seed = -1;
    bool useCfg = true;
    float cfgScale = 4.5f;
};

struct DiffusionResult {
    std::string requestId;
    bool success = false;
    double totalMs = 0.0;
};

class DiffusionSession {
public:
    using ProgressCallback = std::function<void(int)>;

    static DiffusionSession& instance();

    void run(const DiffusionRequest& request, const ProgressCallback& onProgress, DiffusionResult& result,
             std::string& error);

private:
    // Both the Sana LLM encoder and the diffusion modules are peak-memory heavy; only one
    // generation may run at a time.
    std::mutex mOperationMutex;
};

} // namespace mnn_harmony