//
//  minimax_h3_demo.cpp
//
//  Runs the MiniMax-H3 denoising loop from precomputed conditioning and writes the latents.
//
#include <cstdlib>
#include <iostream>
#include <string>

#include "diffusion/minimax_h3_diffusion.hpp"

using namespace MNN::DIFFUSION;

static void usage(const char* program) {
    std::cout
        << "Usage: " << program << " <resource_dir> <prompt_embeds.bin> <output_dir> [backend] [seed] [memory] [resident_groups] [precision]\n"
        << "\n"
        << "  resource_dir      Holds h3_manifest.json, h3_adaln.{json,bin}, h3_rope.bin and the .mnn modules,\n"
        << "                    as produced by transformers/diffusion/export/minimax_h3/h3_build_mnn.py.\n"
        << "  prompt_embeds.bin num_text_tokens * text_dim little-endian float32 encoder hidden states.\n"
        << "  output_dir        Where the video and audio latent rows are written.\n"
        << "  backend           cpu (default), cuda, opencl, vulkan or metal.\n"
        << "  seed              Noise seed; -1 draws one. Reproduces this runtime, not the torch reference.\n"
        << "  memory            0 saves memory (default), 1 favours speed.\n"
        << "  resident_groups   Block partitions kept resident; 0 (default) keeps all. MNN's CUDA backend\n"
        << "                    materializes int4 weights as fp16, so a 50-layer stack needs a window there.\n"
        << "  precision         high (default), bf16, normal or fp16.\n";
}

static MNNForwardType parseBackend(const std::string& name) {
    if (name == "cuda") {
        return MNN_FORWARD_CUDA;
    }
    if (name == "opencl") {
        return MNN_FORWARD_OPENCL;
    }
    if (name == "vulkan") {
        return MNN_FORWARD_VULKAN;
    }
    if (name == "metal") {
        return MNN_FORWARD_METAL;
    }
    return MNN_FORWARD_CPU;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }
    const std::string resourceDir = argv[1];
    const std::string promptEmbeds = argv[2];
    const std::string outputDir = argv[3];
    const auto backend = parseBackend(argc > 4 ? argv[4] : "cpu");
    const int seed = argc > 5 ? std::atoi(argv[5]) : 0;
    const int memoryMode = argc > 6 ? std::atoi(argv[6]) : 0;
    const int residentGroups = argc > 7 ? std::atoi(argv[7]) : 0;
    const std::string precisionName = argc > 8 ? argv[8] : "high";
    auto precision = MNN::BackendConfig::Precision_High;
    if (precisionName == "bf16") {
        precision = MNN::BackendConfig::Precision_Low_BF16;
    } else if (precisionName == "normal") {
        precision = MNN::BackendConfig::Precision_Normal;
    } else if (precisionName == "fp16") {
        precision = MNN::BackendConfig::Precision_Low;
    }

    MinimaxH3Diffusion diffusion(resourceDir, MINIMAX_H3, backend, memoryMode);
    diffusion.setResidentGroups(residentGroups);
    diffusion.setPrecision(precision);
    if (!diffusion.load()) {
        std::cerr << "Failed to load the MiniMax-H3 resources from " << resourceDir << "\n";
        return 1;
    }
    const bool ok = diffusion.runFromPromptEmbeds(promptEmbeds, outputDir, seed, nullptr);
    if (!ok) {
        std::cerr << "Generation failed\n";
        return 1;
    }
    std::cout << "wrote the latents of a " << diffusion.numSteps() << "-step run to " << outputDir << "\n";
    return 0;
}
