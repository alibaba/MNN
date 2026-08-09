//
//  minimax_h3_video_demo.cpp
//
//  End-to-end MiniMax-H3 video generation: denoise the packed sequence, decode the latents, write a video.
//
//  The two stages are loaded and released one at a time. That is not just tidiness -- the transformer and the
//  VAE decoder do not fit a 24 GB device together, and the same serial lifecycle is what the on-device
//  pipeline needs.
//
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "diffusion/minimax_h3_diffusion.hpp"
#include "diffusion/minimax_h3_text_encoder.hpp"
#include "diffusion/minimax_h3_vae.hpp"

using namespace MNN::DIFFUSION;

namespace {

void usage(const char* program) {
    std::cout
        << "Usage: " << program << " <resource_dir> <prompt_embeds.bin> <output.y4m> [backend] [seed]\n"
        << "                       [resident_groups] [fps]\n"
        << "\n"
        << "  resource_dir      Holds the transformer resources (h3_manifest.json, h3_adaln.*, h3_rope.bin,\n"
        << "                    h3_embed/head/blocks_*.mnn) and the VAE ones (h3_vae_plan.json,\n"
        << "                    h3_vae_rope.bin, h3_vae_decoder.mnn).\n"
        << "  prompt            The text prompt. A path to a .bin file is taken as precomputed\n"
        << "                    num_text_tokens * text_dim float32 conditioning instead, skipping the\n"
        << "                    conditioner -- useful when its resources are not built.\n"
        << "  output.y4m        Uncompressed YUV4MPEG2, playable as is. For MP4:\n"
        << "                      ffmpeg -i output.y4m -c:v libx264 -pix_fmt yuv420p output.mp4\n"
        << "  backend           cpu (default) or cuda.\n"
        << "  seed              Noise seed; -1 draws one.\n"
        << "  resident_groups   Block partitions kept resident; 0 keeps all. CUDA at fp32 needs 1.\n"
        << "  fps               Frame rate written into the header, 24 by default.\n"
        << "  reuse_latents     1 skips denoising and decodes the latents already in <resource_dir>/latents,\n"
        << "                    which is how to iterate on the decoder without re-running the transformer.\n"
        << "  precision         high (default), normal, low or low_bf16. Anything below high runs the residual\n"
        << "                    stream in 16 bits, which is much faster but can overflow to NaN.\n";
}

MNN::BackendConfig::PrecisionMode parsePrecision(const std::string& name) {
    if (name == "normal") {
        return MNN::BackendConfig::Precision_Normal;
    }
    if (name == "low") {
        return MNN::BackendConfig::Precision_Low;
    }
    if (name == "low_bf16") {
        return MNN::BackendConfig::Precision_Low_BF16;
    }
    return MNN::BackendConfig::Precision_High;
}

MNNForwardType parseBackend(const std::string& name) {
    if (name == "cuda") {
        return MNN_FORWARD_CUDA;
    }
    if (name == "opencl") {
        return MNN_FORWARD_OPENCL;
    }
    if (name == "vulkan") {
        return MNN_FORWARD_VULKAN;
    }
    return MNN_FORWARD_CPU;
}

uint8_t clampToByte(float value) {
    const int rounded = static_cast<int>(value + 0.5f);
    return static_cast<uint8_t>(rounded < 0 ? 0 : (rounded > 255 ? 255 : rounded));
}

// YUV4MPEG2 with 4:2:0 chroma: a text header per stream and per frame, then planar Y, U, V. Uncompressed, so
// it needs no encoder to be a playable file.
bool writeY4m(const std::string& path, const std::vector<float>& rgb, int numFrames, int height, int width,
              int fps) {
    if (height % 2 != 0 || width % 2 != 0) {
        std::cerr << "Y4M 4:2:0 needs even dimensions, got " << width << "x" << height << "\n";
        return false;
    }
    std::ofstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        std::cerr << "Cannot write " << path << "\n";
        return false;
    }
    stream << "YUV4MPEG2 W" << width << " H" << height << " F" << fps << ":1 Ip A1:1 C420jpeg\n";

    const size_t plane = static_cast<size_t>(height) * width;
    std::vector<uint8_t> luma(plane);
    std::vector<uint8_t> chromaU(plane / 4);
    std::vector<uint8_t> chromaV(plane / 4);
    for (int frame = 0; frame < numFrames; ++frame) {
        const float* source = rgb.data() + static_cast<size_t>(frame) * plane * 3;
        for (size_t pixel = 0; pixel < plane; ++pixel) {
            const float r = source[pixel * 3 + 0] * 255.0f;
            const float g = source[pixel * 3 + 1] * 255.0f;
            const float b = source[pixel * 3 + 2] * 255.0f;
            luma[pixel] = clampToByte(0.299f * r + 0.587f * g + 0.114f * b);
        }
        // Chroma is averaged over each 2x2 block rather than point-sampled, which is what C420jpeg means.
        for (int y = 0; y < height; y += 2) {
            for (int x = 0; x < width; x += 2) {
                float red = 0.0f;
                float green = 0.0f;
                float blue = 0.0f;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const size_t pixel = static_cast<size_t>(y + dy) * width + (x + dx);
                        red += source[pixel * 3 + 0];
                        green += source[pixel * 3 + 1];
                        blue += source[pixel * 3 + 2];
                    }
                }
                red = red * 255.0f / 4.0f;
                green = green * 255.0f / 4.0f;
                blue = blue * 255.0f / 4.0f;
                const size_t index = static_cast<size_t>(y / 2) * (width / 2) + (x / 2);
                chromaU[index] = clampToByte(-0.168736f * red - 0.331264f * green + 0.5f * blue + 128.0f);
                chromaV[index] = clampToByte(0.5f * red - 0.418688f * green - 0.081312f * blue + 128.0f);
            }
        }
        stream << "FRAME\n";
        stream.write(reinterpret_cast<const char*>(luma.data()), static_cast<std::streamsize>(luma.size()));
        stream.write(reinterpret_cast<const char*>(chromaU.data()), static_cast<std::streamsize>(chromaU.size()));
        stream.write(reinterpret_cast<const char*>(chromaV.data()), static_cast<std::streamsize>(chromaV.size()));
    }
    return stream.good();
}

bool readLatents(const std::string& path, std::vector<float>* out) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream.is_open()) {
        return false;
    }
    const auto bytes = static_cast<size_t>(stream.tellg());
    stream.seekg(0);
    out->resize(bytes / sizeof(float));
    stream.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(bytes));
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }
    const std::string resourceDir = argv[1];
    const std::string prompt = argv[2];
    const std::string outputPath = argv[3];
    const auto backend = parseBackend(argc > 4 ? argv[4] : "cpu");
    const int seed = argc > 5 ? std::atoi(argv[5]) : 0;
    const int residentGroups = argc > 6 ? std::atoi(argv[6]) : 0;
    const int fps = argc > 7 ? std::atoi(argv[7]) : 24;
    const bool reuseLatents = argc > 8 && std::atoi(argv[8]) != 0;
    const auto precision = parsePrecision(argc > 9 ? argv[9] : "high");

    const std::string latentDir = resourceDir + "/latents";
    ::system(("mkdir -p " + latentDir).c_str());

    // 0. Condition. The conditioner is 25B parameters and is released before the transformer loads: the two
    // never coexist, which is the same lifecycle the on-device pipeline needs.
    const bool precomputed = prompt.size() > 4 && prompt.compare(prompt.size() - 4, 4, ".bin") == 0;
    std::string promptEmbeds = prompt;
    if (!precomputed && !reuseLatents) {
        MinimaxH3TextEncoder encoder(resourceDir, backend);
        encoder.setResidentGroups(residentGroups);
        if (!encoder.load()) {
            std::cerr << "Failed to load the MiniMax-H3 conditioner from " << resourceDir << "\n";
            return 1;
        }
        std::vector<float> conditioning;
        int numTokens = 0;
        if (!encoder.encode(prompt, &conditioning, &numTokens)) {
            std::cerr << "Conditioning failed\n";
            return 1;
        }
        promptEmbeds = latentDir + "/prompt_embeds.bin";
        std::ofstream stream(promptEmbeds, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(conditioning.data()),
                     static_cast<std::streamsize>(conditioning.size() * sizeof(float)));
        if (!stream.good()) {
            std::cerr << "Cannot write " << promptEmbeds << "\n";
            return 1;
        }
        std::cout << "conditioned " << numTokens << " token(s) into " << conditioning.size() / numTokens
                  << " channels\n";
    }

    // 1. Denoise. The transformer is released before the VAE is loaded.
    int conditionRows = 0;
    if (!reuseLatents) {
        MinimaxH3Diffusion diffusion(resourceDir, MINIMAX_H3, backend, 1);
        diffusion.setResidentGroups(residentGroups);
        diffusion.setPrecision(precision);
        if (!diffusion.load()) {
            std::cerr << "Failed to load the MiniMax-H3 transformer from " << resourceDir << "\n";
            return 1;
        }
        if (!diffusion.runFromPromptEmbeds(promptEmbeds, latentDir, seed, nullptr)) {
            std::cerr << "Denoising failed\n";
            return 1;
        }
        conditionRows = diffusion.numConditionVideoRows();
        std::cout << "denoised " << diffusion.numSteps() << " steps over " << diffusion.sequenceLength()
                  << " rows\n";
    }

    // 2. Decode. The transformer's weights are gone by now, so the VAE has the device to itself.
    std::vector<float> rows;
    if (!readLatents(latentDir + "/video_latent_rows.bin", &rows)) {
        std::cerr << "Cannot read the video latent rows\n";
        return 1;
    }

    MinimaxH3Vae vae(resourceDir, backend);
    if (!vae.load()) {
        std::cerr << "Failed to load the MiniMax-H3 VAE from " << resourceDir << "\n";
        return 1;
    }
    std::vector<float> latents;
    if (!vae.unpackLatentRows(rows, conditionRows, &latents)) {
        std::cerr << "Cannot unpack the latent rows for the VAE\n";
        return 1;
    }

    std::vector<float> rgb;
    int numFrames = 0;
    int height = 0;
    int width = 0;
    if (!vae.decode(latents, &rgb, &numFrames, &height, &width)) {
        std::cerr << "VAE decode failed\n";
        return 1;
    }
    if (!writeY4m(outputPath, rgb, numFrames, height, width, fps)) {
        return 1;
    }
    std::cout << "wrote " << outputPath << ": " << numFrames << " frames at " << width << "x" << height << ", "
              << fps << " fps\n";
    return 0;
}
