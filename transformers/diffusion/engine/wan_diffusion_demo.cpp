#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "diffusion/wan_diffusion.hpp"

#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

using namespace MNN::DIFFUSION;

int main(int argc, const char* argv[]) {
    if (argc < 12) {
        MNN_PRINT(
            "Usage: ./wan_diffusion_demo <resource_path> <memory_mode> <backend_type> <steps> <seed> "
            "<width> <height> <frames> <cfg_scale> <output_dir> <prompt_text>\n");
        return 0;
    }

    std::string resourcePath = argv[1];
    int memoryMode = atoi(argv[2]);
    auto backendType = (MNNForwardType)atoi(argv[3]);
    int steps = atoi(argv[4]);
    int seed = atoi(argv[5]);
    int width = atoi(argv[6]);
    int height = atoi(argv[7]);
    int frames = atoi(argv[8]);
    float cfgScale = (float)atof(argv[9]);
    std::string outputDir = argv[10];

    std::string prompt;
    for (int i = 11; i < argc; ++i) {
        if (!prompt.empty()) {
            prompt += " ";
        }
        prompt += argv[i];
    }

    MNN_PRINT("Wan2.1-T2V resource path: %s\n", resourcePath.c_str());
    MNN_PRINT("Backend type: %d, memory mode: %d\n", (int)backendType, memoryMode);
    MNN_PRINT("Video: %dx%d, frames=%d, steps=%d, seed=%d, cfg_scale=%.3f\n", width, height, frames, steps, seed,
              cfgScale);
    MNN_PRINT("Output dir: %s\n", outputDir.c_str());
    MNN_PRINT("Prompt: %s\n", prompt.c_str());

    std::unique_ptr<Diffusion> diffusion(Diffusion::createDiffusion(resourcePath, WAN2_1_T2V, backendType, memoryMode));
    if (!diffusion) {
        MNN_ERROR("Failed to create Wan diffusion instance\n");
        return -1;
    }
    if (!diffusion->load()) {
        MNN_ERROR("Failed to load Wan diffusion models\n");
        return -1;
    }

    bool success =
        diffusion->runVideo(prompt, outputDir, width, height, frames, steps, seed, cfgScale, [](int progress) {
            printf("Progress: %d%%\r", progress);
            fflush(stdout);
        });
    printf("\n");
    if (!success) {
        MNN_ERROR("Wan video generation failed\n");
        return -1;
    }
    MNN_PRINT("Wan frame sequence saved to %s\n", outputDir.c_str());
    return 0;
}
