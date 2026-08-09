//
//  minimax_h3_diffusion.cpp
//
//  MiniMax-H3 joint video + audio generation runtime.
//
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>

#include "diffusion/minimax_h3_diffusion.hpp"

#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include "rapidjson/document.h"

namespace MNN {
namespace DIFFUSION {

namespace {

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

bool readFloats(const std::string& path, std::vector<float>* out) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream.is_open()) {
        return false;
    }
    const auto bytes = static_cast<size_t>(stream.tellg());
    if (bytes % sizeof(float) != 0) {
        MNN_ERROR("%s is not a whole number of floats\n", path.c_str());
        return false;
    }
    stream.seekg(0);
    out->resize(bytes / sizeof(float));
    stream.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(bytes));
    return stream.good() || stream.eof();
}

std::vector<float> readFloatArray(const rapidjson::Value& value) {
    std::vector<float> result;
    for (const auto& item : value.GetArray()) {
        result.push_back(item.GetFloat());
    }
    return result;
}

} // namespace

struct MinimaxH3Diffusion::Resources {
    int sequenceLength = 0;
    int numTextTokens = 0;
    int numVideoRows = 0;
    int numAudioRows = 0;
    int numConditionVideoRows = 0;
    int hiddenSize = 0;
    int textDim = 0;
    int videoPatchDim = 0;
    int audioInChannels = 0;
    int rotaryDim = 0;
    int numLayers = 0;
    int layersPerGroup = 0;
    int numLatentFrames = 0;
    int latentHeight = 0;
    int latentWidth = 0;
    int numAudioLatents = 0;
    int numSteps = 0;
    int height = 0;
    int width = 0;
    int numFrames = 0;
    BackendConfig backendConfig;
    std::vector<std::pair<int, int>> groups; // (first layer, layer count)
    std::vector<float> timesteps;
    std::vector<float> audioTimesteps;
    std::vector<float> videoSigmas;
    std::vector<float> audioSigmas;
    // [step][layer][6] modulation tensors, and [step] final-norm shift/scale.
    std::vector<std::vector<std::vector<VARP>>> modulation;
    std::vector<std::vector<VARP>> headModulation;
};

MinimaxH3Diffusion::MinimaxH3Diffusion(std::string modelPath, DiffusionModelType modelType,
                                       MNNForwardType backendType, int memoryMode)
    : Diffusion(modelPath, modelType, backendType, memoryMode) {
    mResources.reset(new Resources);
}

MinimaxH3Diffusion::~MinimaxH3Diffusion() = default;

int MinimaxH3Diffusion::sequenceLength() const {
    return mResources->sequenceLength;
}
int MinimaxH3Diffusion::numVideoRows() const {
    return mResources->numVideoRows;
}
int MinimaxH3Diffusion::numAudioRows() const {
    return mResources->numAudioRows;
}
int MinimaxH3Diffusion::numSteps() const {
    return mResources->numSteps;
}
int MinimaxH3Diffusion::numConditionVideoRows() const {
    return mResources->numConditionVideoRows;
}

bool MinimaxH3Diffusion::parseManifest() {
    auto& resources = *mResources;
    std::string text;
    if (!readFile(mModelPath + "/h3_manifest.json", &text)) {
        MNN_ERROR("Cannot read %s/h3_manifest.json\n", mModelPath.c_str());
        return false;
    }
    rapidjson::Document manifest;
    manifest.Parse(text.c_str());
    if (manifest.HasParseError() || !manifest.IsObject()) {
        MNN_ERROR("h3_manifest.json is not valid JSON\n");
        return false;
    }
    const auto& config = manifest["config"];
    resources.hiddenSize = config["hidden_size"].GetInt();
    resources.textDim = config["text_dim"].GetInt();
    resources.audioInChannels = config["audio_in_channels"].GetInt();
    int patchProduct = 1;
    for (const auto& item : config["patch_size"].GetArray()) {
        patchProduct *= item.GetInt();
    }
    resources.videoPatchDim = config["in_channels"].GetInt() * patchProduct;

    resources.sequenceLength = manifest["sequence_length"].GetInt();
    resources.numTextTokens = manifest["num_text_tokens"].GetInt();
    resources.numVideoRows = manifest["num_video_rows"].GetInt();
    resources.numAudioRows = manifest["num_audio_rows"].GetInt();
    resources.numConditionVideoRows = manifest["num_condition_video_rows"].GetInt();
    resources.rotaryDim = manifest["rotary_dim"].GetInt();
    resources.numLayers = manifest["num_layers"].GetInt();
    resources.layersPerGroup = manifest["layers_per_group"].GetInt();
    resources.numLatentFrames = manifest["num_latent_frames"].GetInt();
    resources.latentHeight = manifest["latent_height"].GetInt();
    resources.latentWidth = manifest["latent_width"].GetInt();
    resources.numAudioLatents = manifest["num_audio_latents"].GetInt();
    resources.height = manifest["height"].GetInt();
    resources.width = manifest["width"].GetInt();
    resources.numFrames = manifest["num_frames"].GetInt();
    resources.timesteps = readFloatArray(manifest["timesteps"]);
    resources.audioTimesteps = readFloatArray(manifest["audio_timesteps"]);
    resources.videoSigmas = readFloatArray(manifest["video_sigmas"]);
    resources.audioSigmas = readFloatArray(manifest["audio_sigmas"]);
    resources.numSteps = static_cast<int>(resources.timesteps.size());
    for (const auto& group : manifest["groups"].GetArray()) {
        resources.groups.emplace_back(group["start"].GetInt(), group["num_layers"].GetInt());
    }

    return true;
}

// Materializes the AdaLN, rotary and mask constants.
//
// This runs *after* the modules are loaded, and the order matters on CUDA: the cutlass convolutions stage
// their whole dequantized weight in the backend's static pool while loading -- 616 MB for the widest
// feed-forward projection -- and a thousand small constants allocated first fragment that pool enough for the
// staging to fail. The conv resource only logs and returns when it does, leaving its weights unfilled, so the
// symptom is a NaN forward rather than an error.
bool MinimaxH3Diffusion::buildResourceTensors() {
    auto& resources = *mResources;
    std::string text;
    if (!readFile(mModelPath + "/h3_adaln.json", &text)) {
        MNN_ERROR("Cannot read %s/h3_adaln.json\n", mModelPath.c_str());
        return false;
    }
    rapidjson::Document index;
    index.Parse(text.c_str());
    if (index.HasParseError() || std::string(index["dtype"].GetString()) != "float32") {
        MNN_ERROR("h3_adaln.json is invalid, or its table is not float32\n");
        return false;
    }
    std::vector<float> adaln;
    std::vector<float> rope;
    if (!readFloats(mModelPath + "/h3_adaln.bin", &adaln) ||
        !readFloats(mModelPath + "/h3_rope.bin", &rope)) {
        MNN_ERROR("Cannot read the AdaLN or rotary tables from %s\n", mModelPath.c_str());
        return false;
    }

    // Turn every table entry into a variable once. The whole table is tens of MB, and rebuilding these per
    // step would copy the same constants 50 times a step.
    auto slice = [](const std::vector<float>& blob, const rapidjson::Value& entry, std::vector<int> shape) {
        const size_t offset = entry["offset"].GetUint64() / sizeof(float);
        int count = 1;
        for (auto dimension : shape) {
            count *= dimension;
        }
        auto variable = _Input(shape, NCHW, halide_type_of<float>());
        ::memcpy(variable->writeMap<float>(), blob.data() + offset, static_cast<size_t>(count) * sizeof(float));
        variable.fix(VARP::CONSTANT);
        return variable;
    };

    resources.modulation.assign(resources.numSteps, {});
    resources.headModulation.assign(resources.numSteps, {});
    for (const auto& entry : index["entries"].GetArray()) {
        const std::string name = entry["name"].GetString();
        const auto& shapeValue = entry["shape"];
        const int rows = shapeValue[1].GetInt();
        int step = 0;
        int layer = 0;
        const bool isHead = name.find(".head") != std::string::npos;
        if (isHead) {
            if (::sscanf(name.c_str(), "step%d.head", &step) != 1) {
                MNN_ERROR("Unexpected AdaLN entry %s\n", name.c_str());
                return false;
            }
        } else if (::sscanf(name.c_str(), "step%d.layer%d", &step, &layer) != 2) {
            MNN_ERROR("Unexpected AdaLN entry %s\n", name.c_str());
            return false;
        }
        if (step < 0 || step >= resources.numSteps) {
            MNN_ERROR("AdaLN entry %s is outside the %d-step schedule\n", name.c_str(), resources.numSteps);
            return false;
        }
        // Stored as (parts, rows, hidden); the graph takes one variable per part.
        const int parts = shapeValue[0].GetInt();
        std::vector<VARP> variables;
        const size_t base = entry["offset"].GetUint64() / sizeof(float);
        const size_t stride = static_cast<size_t>(rows) * resources.hiddenSize;
        for (int part = 0; part < parts; ++part) {
            auto variable = _Input({rows, resources.hiddenSize}, NCHW, halide_type_of<float>());
            ::memcpy(variable->writeMap<float>(), adaln.data() + base + static_cast<size_t>(part) * stride,
                     stride * sizeof(float));
            variable.fix(VARP::CONSTANT);
            variables.push_back(variable);
        }
        if (isHead) {
            resources.headModulation[step] = std::move(variables);
        } else {
            auto& perStep = resources.modulation[step];
            if (static_cast<int>(perStep.size()) <= layer) {
                perStep.resize(layer + 1);
            }
            perStep[layer] = std::move(variables);
        }
    }

    for (const auto& entry : index["rope_entries"].GetArray()) {
        const std::string name = entry["name"].GetString();
        auto variable = slice(rope, entry, {1, resources.sequenceLength, 1, resources.rotaryDim});
        if (name == "rope_cos") {
            mRopeCos = variable;
        } else if (name == "rope_sin") {
            mRopeSin = variable;
        }
    }
    if (mRopeCos == nullptr || mRopeSin == nullptr) {
        MNN_ERROR("h3_rope.bin is missing rope_cos or rope_sin\n");
        return false;
    }

    // A partition takes six modulation tensors per layer, and the Euler update reads sigmas[step + 1], so a
    // truncated table would otherwise surface as a shape error deep inside a forward.
    for (int step = 0; step < resources.numSteps; ++step) {
        if (static_cast<int>(resources.modulation[step].size()) != resources.numLayers) {
            MNN_ERROR("AdaLN table step %d covers %zu of %d layers\n", step, resources.modulation[step].size(),
                      resources.numLayers);
            return false;
        }
        for (int layer = 0; layer < resources.numLayers; ++layer) {
            if (resources.modulation[step][layer].size() != 6) {
                MNN_ERROR("AdaLN table step %d layer %d has %zu of 6 parts\n", step, layer,
                          resources.modulation[step][layer].size());
                return false;
            }
        }
        if (resources.headModulation[step].size() != 2) {
            MNN_ERROR("AdaLN table step %d is missing the final norm's shift/scale pair\n", step);
            return false;
        }
    }
    if (static_cast<int>(resources.videoSigmas.size()) != resources.numSteps + 1 ||
        static_cast<int>(resources.audioSigmas.size()) != resources.numSteps + 1) {
        MNN_ERROR("a %d-step schedule needs %d sigmas, got %zu video and %zu audio\n", resources.numSteps,
                  resources.numSteps + 1, resources.videoSigmas.size(), resources.audioSigmas.size());
        return false;
    }

    return true;
}

// MNN's fused attention treats a missing mask as lower-triangular, so an attention op that still takes a mask
// needs a materialized all-zero additive one. `h3_rebuild.py --drop_attention_mask` instead removes the mask
// from the op and collapses this input to a single element, so the size comes from the module itself.
static VARP zeroMask(const Module* module, const char* name, int rows) {
    auto info = module->getInfo();
    if (info != nullptr) {
        for (size_t index = 0; index < info->inputNames.size() && index < info->inputs.size(); ++index) {
            if (info->inputNames[index] == name && !info->inputs[index].dim.empty()) {
                rows = info->inputs[index].dim.back();
                break;
            }
        }
    }
    auto variable = _Input({1, 1, rows, rows}, NCHW, halide_type_of<float>());
    ::memset(variable->writeMap<float>(), 0, static_cast<size_t>(rows) * rows * sizeof(float));
    variable.fix(VARP::CONSTANT);
    return variable;
}

bool MinimaxH3Diffusion::load() {
    AUTOTIME;
    ScheduleConfig config;
    // The runtime manager keeps this pointer, so it has to outlive `load`.
    auto& backendConfig = mResources->backendConfig;
    config.type = mBackendType;
    // Memory_Low selects CUDA's weight-only quantized convolution, which keeps the int4 weights packed on the
    // device -- 385 MB per H3 partition instead of 3.1 GB of dequantized fp32. It is not worth taking yet: its
    // GEMM is a one-output-per-thread dequantize-and-multiply kernel that measures 9x slower than the cutlass
    // one, so a partition costs 19.5 s instead of 6.5 s and that dwarfs the reload it saves.
    backendConfig.memory = (mMemoryMode == 0) ? BackendConfig::Memory_Low : BackendConfig::Memory_Normal;
    // Set by `setPrecision`; float16 is not usable here -- see the header.
    backendConfig.precision = mPrecision;
    if (config.type == MNN_FORWARD_CPU) {
        config.numThread = 4;
    } else if (config.type == MNN_FORWARD_OPENCL) {
        config.mode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST;
    } else {
        config.numThread = 1;
    }
    config.backendConfig = &backendConfig;

    // The global Express executor stays on CPU and only the modules' runtime targets the accelerator, which
    // is what tools/cpp/ModuleBasic does. Pointing the global executor at CUDA as well stands up a second
    // backend whose pool competes with the modules' weight staging, and the cutlass convolutions only log and
    // return when that staging fails -- leaving their weights unfilled and the forward NaN.
    auto executor = ExecutorScope::Current();
    executor->lazyEval = false;

    runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
    if (runtime_manager_ != nullptr) {
        // H3's residual stream needs fp32's exponent range, but not its mantissa: the released checkpoint is
        // bfloat16, which carries fewer mantissa bits than TF32 does. So let the GEMMs use the tensor cores.
        runtime_manager_->setHint(Interpreter::ALLOW_TF32, 1);
    }
    if (!runtime_manager_) {
        MNN_ERROR("Failed to create the MiniMax-H3 runtime manager\n");
        return false;
    }

    if (!parseManifest()) {
        return false;
    }
    auto& resources = *mResources;

    mModuleConfig.shapeMutable = false;

    mEmbed = loadModule("h3_embed");
    mHead = loadModule("h3_head");
    if (mEmbed == nullptr || mHead == nullptr) {
        return false;
    }
    mBlockGroups.resize(resources.groups.size());
    const bool windowed = mResidentGroups > 0 && mResidentGroups < static_cast<int>(resources.groups.size());
    if (!windowed) {
        for (size_t index = 0; index < resources.groups.size(); ++index) {
            mBlockGroups[index] = loadModule("h3_blocks_" + std::to_string(index));
            if (mBlockGroups[index] == nullptr) {
                return false;
            }
        }
    }
    // Only now, so the modules' weight staging gets an unfragmented static pool.
    if (!buildResourceTensors()) {
        return false;
    }
    MNN_PRINT("MiniMax-H3: %d rows, %d layers in %zu partitions%s, %d steps, %dx%d / %d frames\n",
              resources.sequenceLength, resources.numLayers, mBlockGroups.size(),
              windowed ? " (windowed)" : "", resources.numSteps, resources.width, resources.height,
              resources.numFrames);
    return true;
}

void MinimaxH3Diffusion::setResidentGroups(int groups) {
    mResidentGroups = groups;
}

void MinimaxH3Diffusion::setPrecision(BackendConfig::PrecisionMode precision) {
    mPrecision = precision;
}

std::shared_ptr<Module> MinimaxH3Diffusion::loadModule(const std::string& name) {
    const std::string path = mModelPath + "/" + name + ".mnn";
    Timer timer;
    std::shared_ptr<Module> module(Module::load({}, {}, path.c_str(), runtime_manager_, &mModuleConfig));
    mAccounting.moduleLoads += 1;
    mAccounting.loadMs += timer.durationInUs() / 1000.0;
    if (module == nullptr) {
        MNN_ERROR("Failed to load %s\n", path.c_str());
    }
    return module;
}

VARP MinimaxH3Diffusion::packedForward(VARP videoRows, VARP audioRows, VARP* audioVelocity, int step) {
    auto& resources = *mResources;
    const bool windowed = mResidentGroups > 0 && mResidentGroups < static_cast<int>(mBlockGroups.size());

    if (mEmbed == nullptr) {
        if (windowed) {
            // The partition the previous step ended on is dead, and the embed's float weights are larger than
            // any of them. Letting the two coexist is what puts the second step over the device.
            for (auto& group : mBlockGroups) {
                group.reset();
            }
        }
        mEmbed = loadModule("h3_embed");
        if (mEmbed == nullptr) {
            return nullptr;
        }
    }
    if (mTextMask == nullptr) {
        mTextMask = zeroMask(mEmbed.get(), "text_mask", resources.numTextTokens);
    }
    auto hidden = mEmbed->onForward({videoRows, audioRows, mPromptEmbeds, mTextMask})[0];
    if (windowed) {
        // The embed stage runs once per step and its weights are float, so holding them through the block
        // stack costs more than reloading them. At a 21763-row layout that is the difference between fitting a
        // 24 GB device and failing in the attention op's score buffer.
        hidden->readMap<float>();
        mEmbed.reset();
    }

    for (size_t index = 0; index < mBlockGroups.size(); ++index) {
        const int start = resources.groups[index].first;
        const int count = resources.groups[index].second;
        if (windowed && mBlockGroups[index] == nullptr) {
            // Free the partitions outside the trailing window before pulling the next one in, so the two
            // never coexist on the device. Execution is strictly forward-sequential, so the window that
            // matters ends at the current partition.
            for (size_t other = 0; other < mBlockGroups.size(); ++other) {
                if (other > index || other + mResidentGroups <= index) {
                    mBlockGroups[other].reset();
                }
            }
            mBlockGroups[index] = loadModule("h3_blocks_" + std::to_string(index));
            if (mBlockGroups[index] == nullptr) {
                return nullptr;
            }
        }
        if (mMask == nullptr) {
            mMask = zeroMask(mBlockGroups[index].get(), "mask", resources.sequenceLength);
        }
        std::vector<VARP> inputs{hidden, mRopeCos, mRopeSin, mMask};
        for (int layer = 0; layer < count; ++layer) {
            const auto& parts = resources.modulation[step][start + layer];
            inputs.insert(inputs.end(), parts.begin(), parts.end());
        }
        hidden = mBlockGroups[index]->onForward(inputs)[0];
        // The output has to be materialized before the partition that produced it goes away.
        if (windowed) {
            hidden->readMap<float>();
        }
    }

    const auto& head = resources.headModulation[step];
    // One complete pass over embed, every block and the head: the unit another implementation calls a step.
    mAccounting.forwardCount += 1;
    auto outputs = mHead->onForward({hidden, head[0], head[1]});
    *audioVelocity = outputs[1];
    return outputs[0];
}

VARP MinimaxH3Diffusion::eulerStep(VARP sample, VARP velocity, int step, bool audio, int skipRows) {
    const auto& resources = *mResources;
    const auto& sigmas = audio ? resources.audioSigmas : resources.videoSigmas;
    const auto& timesteps = audio ? resources.audioTimesteps : resources.timesteps;
    // The sigma the transformer was conditioned on and the sigma of the Euler ratio are kept apart: for
    // sigma < 0.5 the float32 round trip 1 - (1 - sigma) is not exact, and the reference keeps both sources.
    const float sigmaFromTimestep = 1.0f - timesteps[step];
    const float ratio = sigmas[step + 1] / sigmas[step];

    auto info = sample->getInfo();
    const int count = info->size;
    const int rowWidth = info->dim.back();
    // Only the generated rows are ever written, so keyframe anchors survive the loop by construction.
    const int start = skipRows * rowWidth;
    const float* samplePtr = sample->readMap<float>();
    const float* velocityPtr = velocity->readMap<float>();
    auto result = _Input(info->dim, NCHW, halide_type_of<float>());
    float* out = result->writeMap<float>();
    ::memcpy(out, samplePtr, static_cast<size_t>(start) * sizeof(float));
    for (int index = start; index < count; ++index) {
        // Data-ward velocity: x0 = x_t + (1 - t) * v, then the x_t / x0 blend.
        const float denoised = samplePtr[index] + sigmaFromTimestep * velocityPtr[index];
        out[index] = ratio * samplePtr[index] + (1.0f - ratio) * denoised;
    }
    return result;
}

bool MinimaxH3Diffusion::denoiseStep(int step, VARP* videoRows, VARP* audioRows) {
    auto& resources = *mResources;
    VARP audioVelocity;
    auto videoVelocity = packedForward(*videoRows, *audioRows, &audioVelocity, step);
    if (videoVelocity == nullptr || audioVelocity == nullptr) {
        MNN_ERROR("MiniMax-H3 forward failed at step %d\n", step);
        return false;
    }
    *videoRows = eulerStep(*videoRows, videoVelocity, step, false, resources.numConditionVideoRows);
    *audioRows = eulerStep(*audioRows, audioVelocity, step, true, 0);
    return true;
}

bool MinimaxH3Diffusion::writeLatents(VARP videoRows, VARP audioRows, const std::string& outputDir) {
    const auto& resources = *mResources;
    auto dump = [](const std::string& path, VARP variable) {
        std::ofstream stream(path, std::ios::binary);
        if (!stream.is_open()) {
            MNN_ERROR("Cannot write %s\n", path.c_str());
            return false;
        }
        auto info = variable->getInfo();
        stream.write(reinterpret_cast<const char*>(variable->readMap<float>()),
                     static_cast<std::streamsize>(info->size) * sizeof(float));
        return stream.good();
    };
    if (!dump(outputDir + "/video_latent_rows.bin", videoRows) ||
        !dump(outputDir + "/audio_latent_rows.bin", audioRows)) {
        return false;
    }
    std::ofstream meta(outputDir + "/h3_latents.json");
    if (!meta.is_open()) {
        MNN_ERROR("Cannot write %s/h3_latents.json\n", outputDir.c_str());
        return false;
    }
    // The rows stay patchified; unpatchifying belongs to whoever feeds the VAE.
    meta << "{\n"
         << "  \"video_rows\": " << resources.numVideoRows << ",\n"
         << "  \"video_patch_dim\": " << resources.videoPatchDim << ",\n"
         << "  \"num_condition_video_rows\": " << resources.numConditionVideoRows << ",\n"
         << "  \"audio_rows\": " << resources.numAudioRows << ",\n"
         << "  \"audio_in_channels\": " << resources.audioInChannels << ",\n"
         << "  \"num_latent_frames\": " << resources.numLatentFrames << ",\n"
         << "  \"latent_height\": " << resources.latentHeight << ",\n"
         << "  \"latent_width\": " << resources.latentWidth << ",\n"
         << "  \"num_audio_latents\": " << resources.numAudioLatents << ",\n"
         << "  \"height\": " << resources.height << ",\n"
         << "  \"width\": " << resources.width << ",\n"
         << "  \"num_frames\": " << resources.numFrames << "\n"
         << "}\n";
    return meta.good();
}

bool MinimaxH3Diffusion::runFromPromptEmbeds(const std::string& promptEmbedsPath, const std::string& outputDir,
                                             int seed, std::function<void(int)> progressCallback) {
    AUTOTIME;
    auto& resources = *mResources;
    std::vector<float> embeds;
    if (!readFloats(promptEmbedsPath, &embeds)) {
        MNN_ERROR("Cannot read the prompt embeddings from %s\n", promptEmbedsPath.c_str());
        return false;
    }
    const size_t expected = static_cast<size_t>(resources.numTextTokens) * resources.textDim;
    if (embeds.size() != expected) {
        MNN_ERROR("%s holds %zu floats but the resources were exported for %d x %d = %zu\n",
                  promptEmbedsPath.c_str(), embeds.size(), resources.numTextTokens, resources.textDim, expected);
        return false;
    }
    mPromptEmbeds = _Input({1, resources.numTextTokens, resources.textDim}, NCHW, halide_type_of<float>());
    ::memcpy(mPromptEmbeds->writeMap<float>(), embeds.data(), expected * sizeof(float));
    mPromptEmbeds.fix(VARP::CONSTANT);

    // sigma starts at 1, so the first sample is pure noise.
    std::mt19937 generator(seed < 0 ? std::random_device{}() : static_cast<unsigned>(seed));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    auto noise = [&](std::vector<int> shape) {
        auto variable = _Input(shape, NCHW, halide_type_of<float>());
        auto pointer = variable->writeMap<float>();
        int count = 1;
        for (auto dimension : shape) {
            count *= dimension;
        }
        for (int index = 0; index < count; ++index) {
            pointer[index] = normal(generator);
        }
        return variable;
    };
    auto videoRows = noise({1, resources.numVideoRows, resources.videoPatchDim});
    auto audioRows = noise({1, resources.numAudioRows, resources.audioInChannels});

    mAccounting.reset();
    for (int step = 0; step < resources.numSteps; ++step) {
        Timer timer;
        const double loadBefore = mAccounting.loadMs;
        const int rebuildsBefore = mAccounting.moduleLoads;
        if (!denoiseStep(step, &videoRows, &audioRows)) {
            return false;
        }
        const double stepMs = timer.durationInUs() / 1000.0;
        mAccounting.forwardMs += stepMs;
        const double stepLoadMs = mAccounting.loadMs - loadBefore;
        MNN_PRINT("step %d/%d: t=%.4f audio_t=%.4f, %.1f ms (compute %.1f ms + %d module rebuild(s) %.1f ms = "
                  "%.0f%%)\n",
                  step + 1, resources.numSteps, resources.timesteps[step], resources.audioTimesteps[step], stepMs,
                  stepMs - stepLoadMs, mAccounting.moduleLoads - rebuildsBefore, stepLoadMs,
                  stepMs > 0 ? stepLoadMs / stepMs * 100.0 : 0.0);
        if (progressCallback) {
            progressCallback((step + 1) * 100 / resources.numSteps);
        }
    }
    reportAccounting();
    return writeLatents(videoRows, audioRows, outputDir);
}

void MinimaxH3Diffusion::reportAccounting() const {
    const double total = mAccounting.forwardMs;
    MNN_PRINT("MiniMax-H3 accounting: %d complete DiT forward(s), %d module rebuild(s)\n"
              "  denoise %.1f s | module construction %.1f s (%.0f%%) | compute %.1f s (%.0f%%)\n",
              mAccounting.forwardCount, mAccounting.moduleLoads, total / 1000.0, mAccounting.loadMs / 1000.0,
              total > 0 ? mAccounting.loadMs / total * 100.0 : 0.0, (total - mAccounting.loadMs) / 1000.0,
              total > 0 ? (total - mAccounting.loadMs) / total * 100.0 : 0.0);
}

bool MinimaxH3Diffusion::runVideo(const std::string& prompt, const std::string& outputDir, int width, int height,
                                  int frames, int steps, int seed, float cfgScale,
                                  std::function<void(int)> progressCallback) {
    (void)prompt;
    (void)cfgScale;
    const auto& resources = *mResources;
    if ((width && width != resources.width) || (height && height != resources.height) ||
        (frames && frames != resources.numFrames) || (steps && steps != resources.numSteps)) {
        MNN_ERROR("MiniMax-H3 resources are exported for a fixed layout: %dx%d, %d frames, %d steps\n",
                  resources.width, resources.height, resources.numFrames, resources.numSteps);
        return false;
    }
    // The H3 encoder is not part of the device pipeline yet, so conditioning arrives as a tensor.
    return runFromPromptEmbeds(mModelPath + "/prompt_embeds.bin", outputDir, seed, progressCallback);
}

bool MinimaxH3Diffusion::run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed,
                             std::function<void(int)> progressCallback) {
    (void)prompt;
    (void)imagePath;
    (void)iterNum;
    (void)randomSeed;
    (void)progressCallback;
    MNN_ERROR("MiniMax-H3 generates video; use runVideo or runFromPromptEmbeds\n");
    return false;
}

bool MinimaxH3Diffusion::run(const VARP input_embeds, const std::string& mode, const std::string& inputImagePath,
                             const std::string& outputImagePath, int width, int height, int iterNum, int randomSeed,
                             bool use_cfg, float cfg_scale, std::function<void(int)> progressCallback) {
    (void)input_embeds;
    (void)mode;
    (void)inputImagePath;
    (void)outputImagePath;
    (void)width;
    (void)height;
    (void)iterNum;
    (void)randomSeed;
    (void)use_cfg;
    (void)cfg_scale;
    (void)progressCallback;
    MNN_ERROR("MiniMax-H3 generates video; use runVideo or runFromPromptEmbeds\n");
    return false;
}

} // namespace DIFFUSION
} // namespace MNN
