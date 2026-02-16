#include "sana_session.h"
#include "mls_log.h"

namespace mls {

SanaSession::SanaSession(std::string resource_path, int memory_mode, int backend_type, int width, int height, int grid_size) 
    : resource_path_(std::move(resource_path)), memory_mode_(memory_mode), backend_type_(backend_type), width_(width), height_(height), grid_size_(grid_size) {
}

SanaSession::~SanaSession() {
    diffusion_.reset();
    sana_llm_.reset();
}

void SanaSession::Load() {
    // Defer loading to Run() to optimize peak memory usage.
    // We strictly serialize LLM and Diffusion lifecycles to avoid OOM.
    MNN_DEBUG("SanaSession::Load deferred to Run");
    loaded_ = true;
}

bool SanaSession::Run(const std::string& prompt,
                      const std::string& image_path,
                      const std::string& output_path,
                      int steps,
                      int seed,
                      bool use_cfg,
                      float cfg_scale,
                      const std::function<void(int)>& progressCallback) {

    MNN_DEBUG("SanaSession::Run ENTERED");
    MNN_DEBUG("SanaSession::Run prompt=%s, output=%s, steps=%d, seed=%d",
              prompt.c_str(), output_path.c_str(), steps, seed);

    // 0. Ensure Clean Slate (Free everything possible)
    diffusion_.reset();
    sana_llm_.reset();

    // 1. Initialize and Run SanaLlm
    MNN_DEBUG("SanaSession::Run [Step 1] Loading LLM...");
    std::string llm_path = resource_path_ + "/llm";
    MNN_DEBUG("SanaSession::Run LLM path: %s", llm_path.c_str());
    sana_llm_ = std::make_unique<SanaLlm>(llm_path);
    MNN_DEBUG("SanaSession::Run SanaLlm created successfully");

    MNN_DEBUG("SanaSession::Run [Step 1] Generating features...");
    // process() now internally performs deep copy and releases mLlm memory
    // 根据use_cfg决定是否生成正负样本对
    VARP llm_out;
    if (use_cfg) {
        MNN_DEBUG("SanaSession::Run CFG mode enabled, generating positive and negative samples");
        llm_out = sana_llm_->process(prompt, true, "");
    } else {
        MNN_DEBUG("SanaSession::Run CFG mode disabled, generating single sample");
        llm_out = sana_llm_->process(prompt, false);
    }
    MNN_DEBUG("SanaSession::Run LLM process() returned, llm_out=%p", llm_out.get());

    // Free the wrapper immediately
    sana_llm_.reset();
    MNN_DEBUG("SanaSession::Run [Step 1] LLM Finished and Released.");

    if (llm_out.get() == nullptr) {
        MNN_ERROR("SanaSession::Run LLM process failed (returned null).");
        return false;
    }

    // 2. Initialize and Run Diffusion
    MNN_DEBUG("SanaSession::Run [Step 2] Loading Diffusion...");
    MNN_DEBUG("SanaSession::Run resource_path_: %s", resource_path_.c_str());
    MNN_DEBUG("SanaSession::Run backend_type_: %d", backend_type_);
    MNN_DEBUG("SanaSession::Run memory_mode_: %d", memory_mode_);
    // Model Type 2 is Sana
    diffusion_.reset(Diffusion::createDiffusion(resource_path_, (DiffusionModelType)2, (MNNForwardType)backend_type_, memory_mode_));
    MNN_DEBUG("SanaSession::Run createDiffusion returned, diffusion_: %p", diffusion_.get());

    if (!diffusion_) {
        MNN_ERROR("SanaSession::Run Failed to create Diffusion.");
        return false;
    }
    MNN_DEBUG("SanaSession::Run About to call diffusion_->load()");

    diffusion_->load();
    MNN_DEBUG("SanaSession::Run diffusion_->load() completed");

    MNN_DEBUG("SanaSession::Run [Step 2] Running Diffusion...");
    // Determine mode based on whether image_path is provided
    std::string mode = image_path.empty() ? "text2img" : "img2img";
    MNN_DEBUG("SanaSession::Run mode: %s, image_path: %s, use_cfg: %d, cfg_scale: %.2f", mode.c_str(), image_path.c_str(), use_cfg, cfg_scale);
    bool success = diffusion_->run(llm_out, mode, image_path, output_path, width_, height_, steps, seed, use_cfg, cfg_scale, progressCallback);
    MNN_DEBUG("SanaSession::Run diffusion_->run() completed, success=%d", success);

    if (!success) {
        MNN_ERROR("SanaSession::Run Diffusion::run() failed.");
        return false;
    }
    MNN_DEBUG("SanaSession::Run About to call progressCallback(100)");
    progressCallback(100); // Ensure final progress is sent
    MNN_DEBUG("SanaSession::Run progressCallback(100) completed");

    // 3. Cleanup Diffusion (Optional: Keep it if we want faster consecutive runs? 
    // But since we need to run LLM next time, we'd have to free it anyway. 
    // Might as well free it now to be safe.)
    MNN_DEBUG("SanaSession::Run [Step 2] Diffusion Finished. Cleaning up...");
    diffusion_.reset();

    MNN_DEBUG("SanaSession::Run completed successfully");
    return true;
}

} // namespace mls