#include <random>
#include <fstream>
#include <chrono>
#include "diffusion/diffusion.hpp"
#include "diffusion/stable_diffusion.hpp"
#include "diffusion/sana_diffusion.hpp"
#include "diffusion/wan_diffusion.hpp"

#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#endif

namespace MNN {
namespace DIFFUSION {

// Helper function for progress display
void display_progress(int cur, int total){
    putchar('\r');
    MNN_PRINT("[");
    for (int i = 0; i < cur; i++) putchar('#');
    for (int i = 0; i < total - cur; i++) putchar('-');
    MNN_PRINT("]");
    fprintf(stdout, "  [%3d%%]", cur * 100 / total);
    if (cur == total) putchar('\n');
    fflush(stdout);
}

// Base Diffusion Implementation
Diffusion::Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode) :
mModelPath(modelPath), mModelType(modelType), mBackendType(backendType), mMemoryMode(memoryMode) {
}

Diffusion::~Diffusion() {
    mModules.clear();
    runtime_manager_.reset();
}

bool Diffusion::runVideo(const std::string& prompt, const std::string& outputDir, int width, int height, int frames,
                         int steps, int seed, float cfgScale, std::function<void(int)> progressCallback) {
    (void)prompt;
    (void)outputDir;
    (void)width;
    (void)height;
    (void)frames;
    (void)steps;
    (void)seed;
    (void)cfgScale;
    (void)progressCallback;
    MNN_ERROR("This diffusion model does not support video generation.\n");
    return false;
}

// Factory Method
Diffusion* Diffusion::createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode) {
    if (modelType == SANA_DIFFUSION) {
        return new SanaDiffusion(modelPath, modelType, backendType, memoryMode);
    } else if (modelType == WAN2_1_T2V) {
        return new WanDiffusion(modelPath, modelType, backendType, memoryMode);
    } else {
        return new StableDiffusion(modelPath, modelType, backendType, memoryMode);
    }
}

} // namespace DIFFUSION
} // namespace MNN
