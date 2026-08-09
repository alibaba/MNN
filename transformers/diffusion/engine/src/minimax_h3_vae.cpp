//
//  minimax_h3_vae.cpp
//
//  MiniMax-H3 video VAE decoder: replays the reference decode plan over a fixed-shape tile graph.
//
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

#include "diffusion/minimax_h3_vae.hpp"

#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#include "rapidjson/document.h"

namespace MNN {
namespace DIFFUSION {

using namespace MNN::Express;

namespace {

// The VAE decodes into ImageNet-normalized RGB over a [0, 1] base range.
constexpr float kPixelMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kPixelStd[3] = {0.229f, 0.224f, 0.225f};

bool readFile(const std::string& path, std::string* out) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    *out = buffer.str();
    return true;
}

std::vector<int> readIntArray(const rapidjson::Value& value) {
    std::vector<int> result;
    for (const auto& item : value.GetArray()) {
        result.push_back(item.GetInt());
    }
    return result;
}

} // namespace

struct MinimaxH3Vae::Plan {
    int latentChannels = 0;
    int spatialRatio = 0;
    int temporalRatio = 0;
    int numLatentFrames = 0;
    int latentHeight = 0;
    int latentWidth = 0;
    int tokensChunkSize = 0;
    int tokenOverlap = 0;
    int tokenDrop = 0;
    int framePrePadding = 0;
    int frameOverlap = 0;
    int chunkNumFrames = 0;
    int numChunks = 0;
    int padTokens = 0;
    int padFrames = 0;
    int tileLatentFrames = 0;
    int tileLatentHeight = 0;
    int tileLatentWidth = 0;
    std::vector<int> yStarts;
    std::vector<int> yOverlaps;
    std::vector<int> xStarts;
    std::vector<int> xOverlaps;
    std::vector<float> latentsMean;
    std::vector<float> latentsStd;
    std::vector<int> videoPatchSize;

    int tileFrames() const {
        return tileLatentFrames * temporalRatio;
    }
    int tileHeight() const {
        return tileLatentHeight * spatialRatio;
    }
    int tileWidth() const {
        return tileLatentWidth * spatialRatio;
    }
    int height() const {
        return latentHeight * spatialRatio;
    }
    int width() const {
        return latentWidth * spatialRatio;
    }
};

MinimaxH3Vae::MinimaxH3Vae(std::string resourcePath, MNNForwardType backendType)
    : mResourcePath(std::move(resourcePath)), mBackendType(backendType) {
    mPlan.reset(new Plan);
}

MinimaxH3Vae::~MinimaxH3Vae() = default;

int MinimaxH3Vae::numFrames() const {
    // Every chunk contributes `chunkNumFrames - framePrePadding`, plus the trailing overlap chunk.
    const int perChunk = mPlan->chunkNumFrames - mPlan->framePrePadding;
    return mPlan->numChunks * perChunk + mPlan->frameOverlap - mPlan->padFrames;
}
int MinimaxH3Vae::height() const {
    return mPlan->height();
}
int MinimaxH3Vae::width() const {
    return mPlan->width();
}

bool MinimaxH3Vae::load() {
    AUTOTIME;
    std::string text;
    if (!readFile(mResourcePath + "/h3_vae_plan.json", &text)) {
        MNN_ERROR("Cannot read %s/h3_vae_plan.json\n", mResourcePath.c_str());
        return false;
    }
    rapidjson::Document document;
    document.Parse(text.c_str());
    if (document.HasParseError() || !document.IsObject()) {
        MNN_ERROR("h3_vae_plan.json is not valid JSON\n");
        return false;
    }
    auto& plan = *mPlan;
    plan.latentChannels = document["latent_channels"].GetInt();
    plan.spatialRatio = document["spatial_ratio"].GetInt();
    plan.temporalRatio = document["temporal_ratio"].GetInt();
    plan.numLatentFrames = document["num_latent_frames"].GetInt();
    plan.latentHeight = document["latent_height"].GetInt();
    plan.latentWidth = document["latent_width"].GetInt();
    plan.tokensChunkSize = document["tokens_chunk_size"].GetInt();
    plan.tokenOverlap = document["token_overlap"].GetInt();
    plan.tokenDrop = document["token_drop"].GetInt();
    plan.framePrePadding = document["frame_pre_padding"].GetInt();
    plan.frameOverlap = document["frame_overlap"].GetInt();
    plan.chunkNumFrames = document["chunk_num_frames"].GetInt();
    plan.numChunks = document["num_chunks"].GetInt();
    plan.padTokens = document["pad_tokens"].GetInt();
    plan.padFrames = document["pad_frames"].GetInt();
    plan.tileLatentFrames = document["tile_latent_frames"].GetInt();
    plan.tileLatentHeight = document["tile_latent_height"].GetInt();
    plan.tileLatentWidth = document["tile_latent_width"].GetInt();
    plan.yStarts = readIntArray(document["y_starts"]);
    plan.yOverlaps = readIntArray(document["y_overlaps"]);
    plan.xStarts = readIntArray(document["x_starts"]);
    plan.xOverlaps = readIntArray(document["x_overlaps"]);
    plan.videoPatchSize = readIntArray(document["video_patch_size"]);
    for (const auto& item : document["latents_mean"].GetArray()) {
        plan.latentsMean.push_back(item.GetFloat());
    }
    for (const auto& item : document["latents_std"].GetArray()) {
        plan.latentsStd.push_back(item.GetFloat());
    }
    if (static_cast<int>(plan.latentsMean.size()) != plan.latentChannels ||
        plan.latentsStd.size() != plan.latentsMean.size()) {
        MNN_ERROR("h3_vae_plan.json carries %zu means and %zu deviations for %d channels\n",
                  plan.latentsMean.size(), plan.latentsStd.size(), plan.latentChannels);
        return false;
    }

    ScheduleConfig config;
    config.type = mBackendType;
    // Same reason as the transformer: the cutlass convolutions stage their dequantized weights in the CUDA
    // static pool while loading, and Memory_Low makes that fail silently.
    mBackendConfig.memory = BackendConfig::Memory_Normal;
    mBackendConfig.precision = BackendConfig::Precision_High;
    config.numThread = mBackendType == MNN_FORWARD_CPU ? 4 : 1;
    config.backendConfig = &mBackendConfig;
    mRuntime.reset(Executor::RuntimeManager::createRuntimeManager(config));
    if (!mRuntime) {
        MNN_ERROR("Failed to create the MiniMax-H3 VAE runtime manager\n");
        return false;
    }
    Module::Config moduleConfig;
    moduleConfig.shapeMutable = false;
    const std::string path = mResourcePath + "/h3_vae_decoder.mnn";
    mModule.reset(Module::load({}, {}, path.c_str(), mRuntime, &moduleConfig));
    if (mModule == nullptr) {
        MNN_ERROR("Failed to load %s\n", path.c_str());
        return false;
    }

    // The rotary tables are baked because their grids are float64 normalized coordinates.
    const int numVoxels = plan.tileLatentFrames * plan.tileLatentHeight * plan.tileLatentWidth;
    std::ifstream rope(mResourcePath + "/h3_vae_rope.bin", std::ios::binary | std::ios::ate);
    if (!rope.is_open()) {
        MNN_ERROR("Cannot read %s/h3_vae_rope.bin\n", mResourcePath.c_str());
        return false;
    }
    const size_t bytes = static_cast<size_t>(rope.tellg());
    rope.seekg(0);
    std::vector<float> table(bytes / sizeof(float));
    rope.read(reinterpret_cast<char*>(table.data()), static_cast<std::streamsize>(bytes));
    // cos then sin, each (numTokens, rotaryDim).
    const size_t half = table.size() / 2;
    // The register tokens and the zero token follow the voxels, so the token count comes from the table.
    const int rotaryDim = static_cast<int>(half) % (numVoxels + 5) == 0 ? static_cast<int>(half) / (numVoxels + 5) : 0;
    if (rotaryDim == 0) {
        MNN_ERROR("h3_vae_rope.bin does not match a %d-voxel tile\n", numVoxels);
        return false;
    }
    const int numTokens = static_cast<int>(half) / rotaryDim;

    auto makeVar = [](std::vector<int> shape, const float* source) {
        auto variable = _Input(shape, NCHW, halide_type_of<float>());
        int count = 1;
        for (auto dimension : shape) {
            count *= dimension;
        }
        if (source != nullptr) {
            ::memcpy(variable->writeMap<float>(), source, static_cast<size_t>(count) * sizeof(float));
        } else {
            ::memset(variable->writeMap<float>(), 0, static_cast<size_t>(count) * sizeof(float));
        }
        variable.fix(VARP::CONSTANT);
        return variable;
    };
    mRopeCos = makeVar({1, numTokens, 1, rotaryDim}, table.data());
    mRopeSin = makeVar({1, numTokens, 1, rotaryDim}, table.data() + half);
    // Non-causal attention needs a materialized all-zero additive mask, as in the transformer.
    mMask = makeVar({1, 1, numTokens, numTokens}, nullptr);

    MNN_PRINT("MiniMax-H3 VAE: %d chunk(s) x %zu tile(s), tile %dx%dx%d -> %d frames at %dx%d\n", plan.numChunks,
              plan.yStarts.size() * plan.xStarts.size(), plan.tileFrames(), plan.tileHeight(), plan.tileWidth(),
              numFrames(), plan.width(), plan.height());
    return true;
}

bool MinimaxH3Vae::unpackLatentRows(const std::vector<float>& rows, int numConditionRows,
                                    std::vector<float>* latents) const {
    const auto& plan = *mPlan;
    const int patchT = plan.videoPatchSize[0];
    const int patchH = plan.videoPatchSize[1];
    const int patchW = plan.videoPatchSize[2];
    const int patchDim = plan.latentChannels * patchT * patchH * patchW;
    const int rowsPerFrame = (plan.latentHeight / patchH) * (plan.latentWidth / patchW);
    const int generatedRows = plan.numLatentFrames / patchT * rowsPerFrame;
    const size_t offset = static_cast<size_t>(numConditionRows) * patchDim;
    if (rows.size() < offset + static_cast<size_t>(generatedRows) * patchDim) {
        MNN_ERROR("The latent rows hold %zu values, too few for %d generated rows of %d\n", rows.size(),
                  generatedRows, patchDim);
        return false;
    }

    const size_t framePlane = static_cast<size_t>(plan.latentHeight) * plan.latentWidth;
    latents->assign(static_cast<size_t>(plan.latentChannels) * plan.numLatentFrames * framePlane, 0.0f);
    // Rows are frame-major then row-major within a frame, and each row holds one (C, patchT, patchH, patchW)
    // block -- the inverse of the exporter's patchify.
    for (int frame = 0; frame < plan.numLatentFrames / patchT; ++frame) {
        for (int blockY = 0; blockY < plan.latentHeight / patchH; ++blockY) {
            for (int blockX = 0; blockX < plan.latentWidth / patchW; ++blockX) {
                const size_t row = offset +
                                   ((static_cast<size_t>(frame) * (plan.latentHeight / patchH) + blockY) *
                                        (plan.latentWidth / patchW) +
                                    blockX) *
                                       patchDim;
                for (int channel = 0; channel < plan.latentChannels; ++channel) {
                    for (int dt = 0; dt < patchT; ++dt) {
                        for (int dy = 0; dy < patchH; ++dy) {
                            for (int dx = 0; dx < patchW; ++dx) {
                                const size_t from = row +
                                                    (((static_cast<size_t>(channel) * patchT + dt) * patchH + dy) *
                                                         patchW +
                                                     dx);
                                const size_t to = static_cast<size_t>(channel) * plan.numLatentFrames * framePlane +
                                                  static_cast<size_t>(frame * patchT + dt) * framePlane +
                                                  static_cast<size_t>(blockY * patchH + dy) * plan.latentWidth +
                                                  (blockX * patchW + dx);
                                (*latents)[to] = rows[from] * plan.latentsStd[channel] + plan.latentsMean[channel];
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool MinimaxH3Vae::decodeTile(const std::vector<float>& latents, int latentFrameStart, int latentYStart,
                              int latentXStart, std::vector<float>* tile) {
    const auto& plan = *mPlan;
    auto input = _Input(
        {1, plan.latentChannels, plan.tileLatentFrames, plan.tileLatentHeight, plan.tileLatentWidth}, NCHW,
        halide_type_of<float>());
    float* destination = input->writeMap<float>();

    const size_t framePlane = static_cast<size_t>(plan.latentHeight) * plan.latentWidth;
    const size_t channelStride = framePlane * plan.numLatentFrames;
    for (int channel = 0; channel < plan.latentChannels; ++channel) {
        for (int frame = 0; frame < plan.tileLatentFrames; ++frame) {
            // Latent frames past the end repeat the last one, which is how the reference pads a chunk.
            const int source = std::min(latentFrameStart + frame, plan.numLatentFrames - 1);
            for (int row = 0; row < plan.tileLatentHeight; ++row) {
                const size_t from = channel * channelStride + static_cast<size_t>(source) * framePlane +
                                    static_cast<size_t>(latentYStart + row) * plan.latentWidth + latentXStart;
                const size_t to = ((static_cast<size_t>(channel) * plan.tileLatentFrames + frame) *
                                       plan.tileLatentHeight +
                                   row) *
                                  plan.tileLatentWidth;
                ::memcpy(destination + to, latents.data() + from,
                         static_cast<size_t>(plan.tileLatentWidth) * sizeof(float));
            }
        }
    }
    input.fix(VARP::CONSTANT);

    auto outputs = mModule->onForward({input, mRopeCos, mRopeSin, mMask});
    if (outputs.empty() || outputs[0] == nullptr) {
        MNN_ERROR("The MiniMax-H3 VAE tile forward failed\n");
        return false;
    }
    auto info = outputs[0]->getInfo();
    const float* source = outputs[0]->readMap<float>();
    if (source == nullptr) {
        MNN_ERROR("The MiniMax-H3 VAE tile produced no data\n");
        return false;
    }
    tile->assign(source, source + info->size);
    return true;
}

bool MinimaxH3Vae::decode(const std::vector<float>& latents, std::vector<float>* rgb, int* numFramesOut,
                          int* heightOut, int* widthOut) {
    AUTOTIME;
    const auto& plan = *mPlan;
    const size_t expected = static_cast<size_t>(plan.latentChannels) * plan.numLatentFrames * plan.latentHeight *
                            plan.latentWidth;
    if (latents.size() != expected) {
        MNN_ERROR("The VAE expected %zu latent values, got %zu\n", expected, latents.size());
        return false;
    }

    const int tileFrames = plan.tileFrames();
    const int tileHeight = plan.tileHeight();
    const int tileWidth = plan.tileWidth();
    const int height = plan.height();
    const int width = plan.width();
    const int perChunk = plan.chunkNumFrames - plan.framePrePadding;

    // One canvas per chunk-slot, in (frame, y, x, channel) order.
    const int total = numFrames() + plan.padFrames;
    rgb->assign(static_cast<size_t>(total) * height * width * 3, 0.0f);

    // `overlap` carries the tail the previous chunk decoded past its own span, to cross-fade into the next.
    std::vector<float> overlap;
    bool haveOverlap = false;
    int written = 0;

    std::vector<float> tile;
    // Stitched canvas of one clip: (tileFrames, height, width, 3).
    std::vector<float> clip;

    for (int chunk = 0; chunk < plan.numChunks; ++chunk) {
        clip.assign(static_cast<size_t>(tileFrames) * height * width * 3, 0.0f);
        const int latentFrameStart = chunk * plan.tokensChunkSize;

        for (size_t yIndex = 0; yIndex < plan.yStarts.size(); ++yIndex) {
            for (size_t xIndex = 0; xIndex < plan.xStarts.size(); ++xIndex) {
                if (!decodeTile(latents, latentFrameStart, plan.yStarts[yIndex] / plan.spatialRatio,
                                plan.xStarts[xIndex] / plan.spatialRatio, &tile)) {
                    return false;
                }
                // The tile arrives as (1, 3, F, H, W); place it channel-last with a linear cross-fade over the
                // overlap with the tiles already written to its left and above.
                const int yStart = plan.yStarts[yIndex];
                const int xStart = plan.xStarts[xIndex];
                const int yBlend = yIndex > 0 ? plan.yOverlaps[yIndex - 1] : 0;
                const int xBlend = xIndex > 0 ? plan.xOverlaps[xIndex - 1] : 0;
                const size_t tilePlane = static_cast<size_t>(tileHeight) * tileWidth;
                for (int frame = 0; frame < tileFrames; ++frame) {
                    for (int row = 0; row < tileHeight; ++row) {
                        const int y = yStart + row;
                        if (y >= height) {
                            break;
                        }
                        const float weightY = yBlend > 0 && row < yBlend
                                                  ? static_cast<float>(row) / static_cast<float>(yBlend)
                                                  : 1.0f;
                        for (int column = 0; column < tileWidth; ++column) {
                            const int x = xStart + column;
                            if (x >= width) {
                                break;
                            }
                            const float weightX = xBlend > 0 && column < xBlend
                                                      ? static_cast<float>(column) / static_cast<float>(xBlend)
                                                      : 1.0f;
                            const float weight = weightY * weightX;
                            const size_t destination =
                                ((static_cast<size_t>(frame) * height + y) * width + x) * 3;
                            for (int channel = 0; channel < 3; ++channel) {
                                const size_t from = (static_cast<size_t>(channel) * tileFrames + frame) * tilePlane +
                                                    static_cast<size_t>(row) * tileWidth + column;
                                clip[destination + channel] =
                                    clip[destination + channel] * (1.0f - weight) + tile[from] * weight;
                            }
                        }
                    }
                }
            }
        }

        // `token_drop` left each clip covering two chunk slots: the chunk itself and the overlap of the next.
        const size_t rowBytes = static_cast<size_t>(height) * width * 3;
        for (int slot = 0; slot < (plan.tokenDrop > 0 ? 2 : 1); ++slot) {
            const int frameStart = slot * plan.chunkNumFrames + plan.framePrePadding;
            const int available = std::max(0, std::min(tileFrames - frameStart,
                                                       plan.chunkNumFrames - plan.framePrePadding));
            if (available <= 0) {
                continue;
            }
            if (slot == 0) {
                for (int frame = 0; frame < available; ++frame) {
                    if (written >= total) {
                        break;
                    }
                    float* destination = rgb->data() + static_cast<size_t>(written) * rowBytes;
                    const float* source = clip.data() + static_cast<size_t>(frameStart + frame) * rowBytes;
                    if (haveOverlap && frame < plan.frameOverlap) {
                        const float weight = static_cast<float>(frame) / static_cast<float>(plan.frameOverlap);
                        const float* previous = overlap.data() + static_cast<size_t>(frame) * rowBytes;
                        for (size_t index = 0; index < rowBytes; ++index) {
                            destination[index] = previous[index] * (1.0f - weight) + source[index] * weight;
                        }
                    } else {
                        ::memcpy(destination, source, rowBytes * sizeof(float));
                    }
                    ++written;
                }
            } else {
                overlap.assign(clip.begin() + static_cast<size_t>(frameStart) * rowBytes,
                               clip.begin() + static_cast<size_t>(frameStart + available) * rowBytes);
                haveOverlap = true;
            }
        }
    }
    if (haveOverlap) {
        const size_t rowBytes = static_cast<size_t>(height) * width * 3;
        const int available = static_cast<int>(overlap.size() / rowBytes);
        for (int frame = 0; frame < available && written < total; ++frame) {
            ::memcpy(rgb->data() + static_cast<size_t>(written) * rowBytes,
                     overlap.data() + static_cast<size_t>(frame) * rowBytes, rowBytes * sizeof(float));
            ++written;
        }
    }

    rgb->resize(static_cast<size_t>(numFrames()) * height * width * 3);
    // Revert the VAE's ImageNet normalization.
    for (size_t index = 0; index + 2 < rgb->size(); index += 3) {
        for (int channel = 0; channel < 3; ++channel) {
            float value = (*rgb)[index + channel] * kPixelStd[channel] + kPixelMean[channel];
            (*rgb)[index + channel] = std::min(1.0f, std::max(0.0f, value));
        }
    }
    *numFramesOut = numFrames();
    *heightOut = height;
    *widthOut = width;
    MNN_PRINT("MiniMax-H3 VAE: decoded %d frames at %dx%d\n", numFrames(), width, height);
    return true;
}

} // namespace DIFFUSION
} // namespace MNN
