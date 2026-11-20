#include "diffusion_session.h"

#include <chrono>
#include <memory>
#include <string>

#include "diffusion/diffusion.hpp"
#include "diffusion/sana_llm.hpp"

namespace mnn_harmony {

using MNN::DIFFUSION::Diffusion;
using MNN::DIFFUSION::DiffusionModelType;
using MNN::DIFFUSION::SANA_DIFFUSION;
using MNN::DIFFUSION::SanaLlm;
using MNN::Express::VARP;

namespace {

constexpr int kSanaModelType = 2;

std::string describedFailure(const char* stage) {
    return std::string("Diffusion failed while ") + stage;
}

} // namespace

DiffusionSession& DiffusionSession::instance() {
    static DiffusionSession session;
    return session;
}

void DiffusionSession::run(const DiffusionRequest& request, const ProgressCallback& onProgress,
                           DiffusionResult& result, std::string& error) {
    std::lock_guard<std::mutex> lock(mOperationMutex);
    const auto start = std::chrono::steady_clock::now();
    result.requestId = request.requestId;
    result.success = false;

    if (request.resourcePath.empty() || request.prompt.empty() || request.outputPath.empty()) {
        error = "Diffusion request is missing model path, prompt or output path";
        return;
    }

    try {
        if (request.modelType == kSanaModelType) {
            // Sana needs the Qwen text encoder first. Ensure the LLM and diffusion
            // modules never stay alive together to keep peak memory bounded.
            auto textEncoder = std::make_unique<SanaLlm>(request.resourcePath + "/llm");
            VARP embeddings = request.useCfg ? textEncoder->process(request.prompt, true, request.negativePrompt)
                                             : textEncoder->process(request.prompt, false);
            if (embeddings.get() == nullptr) {
                error = describedFailure("encoding the prompt");
                return;
            }
            std::unique_ptr<Diffusion> diffusion(Diffusion::createDiffusion(
                request.resourcePath, SANA_DIFFUSION, static_cast<MNNForwardType>(request.backendType),
                request.memoryMode));
            if (diffusion.get() == nullptr) {
                embeddings = nullptr;
                error = describedFailure("creating the model");
                return;
            }
            if (!diffusion->load()) {
                embeddings = nullptr;
                diffusion.reset();
                error = describedFailure("loading the model");
                return;
            }
            const std::string mode = request.imageInput.empty() ? "text2img" : "img2img";
            result.success = diffusion->run(embeddings, mode, request.imageInput, request.outputPath, request.width,
                                            request.height, request.steps, request.seed, request.useCfg,
                                            request.cfgScale, [&onProgress](int progress) {
                                                if (onProgress) {
                                                    onProgress(progress);
                                                }
                                            });
            // Release the feature tensor before the diffusion graph to avoid dangling state.
            embeddings = nullptr;
            textEncoder.reset();
            diffusion.reset();
        } else {
            std::unique_ptr<Diffusion> diffusion(Diffusion::createDiffusion(
                request.resourcePath, static_cast<DiffusionModelType>(request.modelType),
                static_cast<MNNForwardType>(request.backendType), request.memoryMode));
            if (diffusion.get() == nullptr) {
                error = describedFailure("creating the model");
                return;
            }
            if (!diffusion->load()) {
                diffusion.reset();
                error = describedFailure("loading the model");
                return;
            }
            result.success = diffusion->run(request.prompt, request.outputPath, request.steps, request.seed,
                                            [&onProgress](int progress) {
                                                if (onProgress) {
                                                    onProgress(progress);
                                                }
                                            });
            diffusion.reset();
        }
    } catch (const std::exception& exception) {
        error = exception.what();
    } catch (...) {
        error = "Diffusion failed with an unknown error";
    }

    if (!result.success) {
        if (error.empty()) {
            error = "Diffusion generation failed";
        }
        return;
    }
    if (onProgress) {
        onProgress(100);
    }
    result.totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

} // namespace mnn_harmony