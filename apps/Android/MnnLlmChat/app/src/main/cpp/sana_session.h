#pragma once
#include <string>
#include <memory>
#include <functional>
#include "diffusion/diffusion.hpp"
#include "diffusion/sana_llm.hpp"

using namespace MNN::DIFFUSION;

namespace mls {

class SanaSession {
public:
    explicit SanaSession(std::string resource_path, int memory_mode, int backend_type, int width, int height, int grid_size);
    ~SanaSession();
    
    // Initialize the models (can be heavy)
    void Load();

    bool Run(const std::string& prompt,
             const std::string& image_path,
             const std::string& output_path,
             int steps,
             int seed,
             bool use_cfg,
             float cfg_scale,
             const std::function<void(int)>& progressCallback);

private:
    std::string resource_path_;
    int memory_mode_;
    int backend_type_;
    int width_;
    int height_;
    int grid_size_;
    bool loaded_{false};
    
    std::unique_ptr<SanaLlm> sana_llm_{nullptr};
    std::unique_ptr<Diffusion> diffusion_{nullptr};
};

} // namespace mls
