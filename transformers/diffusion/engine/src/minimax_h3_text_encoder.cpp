//
//  minimax_h3_text_encoder.cpp
//
//  MiniMax-H3 text conditioner: the first 50 decoder layers of Qwen3-VL-32B.
//
#include <cstring>
#include <fstream>
#include <sstream>

#include "diffusion/minimax_h3_text_encoder.hpp"
#include "tokenizer.hpp"

#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#include "rapidjson/document.h"

namespace MNN {
namespace DIFFUSION {

using namespace MNN::Express;

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

float bfloat16ToFloat(uint16_t bits) {
    // bfloat16 is the top half of a float32, so widening is a shift.
    const uint32_t widened = static_cast<uint32_t>(bits) << 16;
    float result;
    ::memcpy(&result, &widened, sizeof(result));
    return result;
}

} // namespace

struct MinimaxH3TextEncoder::Resources {
    int hiddenSize = 0;
    int vocabSize = 0;
    int maxTokens = 0;
    int numLayers = 0;
    int layersPerGroup = 0;
    int rotaryDim = 0;
    std::vector<std::pair<int, int>> groups;
    std::string embedPath;
};

MinimaxH3TextEncoder::MinimaxH3TextEncoder(std::string resourcePath, MNNForwardType backendType)
    : mResourcePath(std::move(resourcePath)), mBackendType(backendType) {
    mResources.reset(new Resources);
}

MinimaxH3TextEncoder::~MinimaxH3TextEncoder() = default;

int MinimaxH3TextEncoder::hiddenSize() const {
    return mResources->hiddenSize;
}
int MinimaxH3TextEncoder::maxTokens() const {
    return mResources->maxTokens;
}
void MinimaxH3TextEncoder::setResidentGroups(int groups) {
    mResidentGroups = groups;
}

std::shared_ptr<Module> MinimaxH3TextEncoder::loadModule(const std::string& name) {
    const std::string path = mResourcePath + "/" + name + ".mnn";
    std::shared_ptr<Module> module(Module::load({}, {}, path.c_str(), mRuntime, &mModuleConfig));
    if (module == nullptr) {
        MNN_ERROR("Failed to load %s\n", path.c_str());
    }
    return module;
}

bool MinimaxH3TextEncoder::load() {
    AUTOTIME;
    auto& resources = *mResources;
    std::string text;
    if (!readFile(mResourcePath + "/h3_text_manifest.json", &text)) {
        MNN_ERROR("Cannot read %s/h3_text_manifest.json\n", mResourcePath.c_str());
        return false;
    }
    rapidjson::Document manifest;
    manifest.Parse(text.c_str());
    if (manifest.HasParseError() || !manifest.IsObject()) {
        MNN_ERROR("h3_text_manifest.json is not valid JSON\n");
        return false;
    }
    resources.hiddenSize = manifest["hidden_size"].GetInt();
    resources.vocabSize = manifest["vocab_size"].GetInt();
    resources.maxTokens = manifest["max_tokens"].GetInt();
    resources.numLayers = manifest["num_layers"].GetInt();
    resources.layersPerGroup = manifest["layers_per_group"].GetInt();
    resources.rotaryDim = manifest["rotary_dim"].GetInt();
    for (const auto& group : manifest["groups"].GetArray()) {
        resources.groups.emplace_back(group["start"].GetInt(), group["num_layers"].GetInt());
    }
    resources.embedPath = mResourcePath + "/h3_text_embed.bin";

    ScheduleConfig config;
    BackendConfig backendConfig;
    config.type = mBackendType;
    // Memory_Low starves the CUDA static pool the cutlass convolutions stage their dequantized weights in, and
    // a failed staging only logs before returning -- leaving weights unfilled and the forward NaN.
    backendConfig.memory = BackendConfig::Memory_Normal;
    // The conditioner is a bfloat16 checkpoint; float32 activations are the safe choice, as in the transformer.
    backendConfig.precision = BackendConfig::Precision_High;
    config.numThread = mBackendType == MNN_FORWARD_CPU ? 4 : 1;
    config.backendConfig = &backendConfig;
    mRuntime.reset(Executor::RuntimeManager::createRuntimeManager(config));
    if (!mRuntime) {
        MNN_ERROR("Failed to create the MiniMax-H3 conditioner runtime manager\n");
        return false;
    }
    mModuleConfig.shapeMutable = false;

    mGroups.resize(resources.groups.size());
    const bool windowed = mResidentGroups > 0 && mResidentGroups < static_cast<int>(resources.groups.size());
    if (!windowed) {
        for (size_t index = 0; index < resources.groups.size(); ++index) {
            mGroups[index] = loadModule("h3_text_layers_" + std::to_string(index));
            if (mGroups[index] == nullptr) {
                return false;
            }
        }
    }

    // The rotary tables are baked: Qwen3-VL's interleaved mrope is captured from the reference rather than
    // reimplemented here.
    std::ifstream rope(mResourcePath + "/h3_text_rope.bin", std::ios::binary | std::ios::ate);
    if (!rope.is_open()) {
        MNN_ERROR("Cannot read %s/h3_text_rope.bin\n", mResourcePath.c_str());
        return false;
    }
    const size_t bytes = static_cast<size_t>(rope.tellg());
    rope.seekg(0);
    std::vector<float> table(bytes / sizeof(float));
    rope.read(reinterpret_cast<char*>(table.data()), static_cast<std::streamsize>(bytes));
    const size_t half = table.size() / 2;
    if (half != static_cast<size_t>(resources.maxTokens) * resources.rotaryDim) {
        MNN_ERROR("h3_text_rope.bin holds %zu values per table, expected %d x %d\n", half, resources.maxTokens,
                  resources.rotaryDim);
        return false;
    }

    auto makeVar = [](std::vector<int> shape, const float* source, float fill) {
        auto variable = _Input(shape, NCHW, halide_type_of<float>());
        int count = 1;
        for (auto dimension : shape) {
            count *= dimension;
        }
        float* pointer = variable->writeMap<float>();
        if (source != nullptr) {
            ::memcpy(pointer, source, static_cast<size_t>(count) * sizeof(float));
        } else {
            for (int index = 0; index < count; ++index) {
                pointer[index] = fill;
            }
        }
        variable.fix(VARP::CONSTANT);
        return variable;
    };
    mRopeCos = makeVar({1, resources.maxTokens, 1, resources.rotaryDim}, table.data(), 0.0f);
    mRopeSin = makeVar({1, resources.maxTokens, 1, resources.rotaryDim}, table.data() + half, 0.0f);

    // Causal: a token attends to itself and everything before it. Built here rather than shipped because it is
    // a constant of the sequence length.
    mMask = _Input({1, 1, resources.maxTokens, resources.maxTokens}, NCHW, halide_type_of<float>());
    float* mask = mMask->writeMap<float>();
    for (int row = 0; row < resources.maxTokens; ++row) {
        for (int column = 0; column < resources.maxTokens; ++column) {
            mask[static_cast<size_t>(row) * resources.maxTokens + column] =
                column > row ? -std::numeric_limits<float>::infinity() : 0.0f;
        }
    }
    mMask.fix(VARP::CONSTANT);

    // MtokTokenizer takes the directory holding tokenizer.mtok, not the file.
    mTokenizer.reset(new MtokTokenizer(MtokTokenizer::Style::kSingle, -1, -1));
    if (!mTokenizer->load(mResourcePath)) {
        MNN_ERROR("Cannot load %s/tokenizer.mtok\n", mResourcePath.c_str());
        return false;
    }

    MNN_PRINT("MiniMax-H3 conditioner: %d layers in %zu partitions%s, %d tokens, hidden %d\n", resources.numLayers,
              resources.groups.size(), windowed ? " (windowed)" : "", resources.maxTokens, resources.hiddenSize);
    return true;
}

bool MinimaxH3TextEncoder::gatherEmbeddings(const std::vector<int>& tokenIds, float* destination) const {
    const auto& resources = *mResources;
    std::ifstream stream(resources.embedPath, std::ios::binary);
    if (!stream.is_open()) {
        MNN_ERROR("Cannot read %s\n", resources.embedPath.c_str());
        return false;
    }
    // Only the rows the prompt uses are read; the table is 1.5 GB and a prompt touches a few hundred KB of it.
    std::vector<uint16_t> row(resources.hiddenSize);
    for (size_t index = 0; index < tokenIds.size(); ++index) {
        const int token = tokenIds[index];
        if (token < 0 || token >= resources.vocabSize) {
            MNN_ERROR("Token id %d is outside the %d-entry vocabulary\n", token, resources.vocabSize);
            return false;
        }
        stream.seekg(static_cast<std::streamoff>(token) * resources.hiddenSize * sizeof(uint16_t));
        stream.read(reinterpret_cast<char*>(row.data()),
                    static_cast<std::streamsize>(row.size() * sizeof(uint16_t)));
        if (!stream) {
            MNN_ERROR("Short read of embedding row %d\n", token);
            return false;
        }
        float* out = destination + index * resources.hiddenSize;
        for (int channel = 0; channel < resources.hiddenSize; ++channel) {
            out[channel] = bfloat16ToFloat(row[channel]);
        }
    }
    return true;
}

bool MinimaxH3TextEncoder::encodeTokens(const std::vector<int>& tokenIds, std::vector<float>* hiddenStates) {
    AUTOTIME;
    const auto& resources = *mResources;
    const int numTokens = static_cast<int>(tokenIds.size());
    if (numTokens == 0 || numTokens > resources.maxTokens) {
        MNN_ERROR("The conditioner takes 1 to %d tokens, got %d\n", resources.maxTokens, numTokens);
        return false;
    }

    auto hidden = _Input({1, resources.maxTokens, resources.hiddenSize}, NCHW, halide_type_of<float>());
    float* pointer = hidden->writeMap<float>();
    ::memset(pointer, 0, static_cast<size_t>(resources.maxTokens) * resources.hiddenSize * sizeof(float));
    if (!gatherEmbeddings(tokenIds, pointer)) {
        return false;
    }
    hidden.fix(VARP::CONSTANT);

    const bool windowed = mResidentGroups > 0 && mResidentGroups < static_cast<int>(mGroups.size());
    for (size_t index = 0; index < mGroups.size(); ++index) {
        if (windowed && mGroups[index] == nullptr) {
            // Free the partitions outside the trailing window before pulling the next one in.
            for (size_t other = 0; other < mGroups.size(); ++other) {
                if (other > index || other + mResidentGroups <= index) {
                    mGroups[other].reset();
                }
            }
            mGroups[index] = loadModule("h3_text_layers_" + std::to_string(index));
            if (mGroups[index] == nullptr) {
                return false;
            }
        }
        auto outputs = mGroups[index]->onForward({hidden, mRopeCos, mRopeSin, mMask});
        if (outputs.empty() || outputs[0] == nullptr) {
            MNN_ERROR("The conditioner partition %zu forward failed\n", index);
            return false;
        }
        hidden = outputs[0];
        if (windowed) {
            // Materialize before the partition that produced it goes away.
            hidden->readMap<float>();
        }
    }

    const float* source = hidden->readMap<float>();
    if (source == nullptr) {
        MNN_ERROR("The conditioner produced no data\n");
        return false;
    }
    // The padding rows never influenced a real token, so they are simply dropped.
    hiddenStates->assign(source, source + static_cast<size_t>(numTokens) * resources.hiddenSize);
    return true;
}

bool MinimaxH3TextEncoder::encode(const std::string& prompt, std::vector<float>* hiddenStates, int* numTokens) {
    // MiniMax-H3 tokenizes the prompt with no special tokens: the presentation it builds carries no template.
    auto tokenIds = mTokenizer->encode(prompt);
    if (tokenIds.empty()) {
        MNN_ERROR("The prompt tokenized to nothing\n");
        return false;
    }
    if (static_cast<int>(tokenIds.size()) > mResources->maxTokens) {
        MNN_ERROR("The prompt is %zu tokens but the conditioner was exported for %d\n", tokenIds.size(),
                  mResources->maxTokens);
        return false;
    }
    *numTokens = static_cast<int>(tokenIds.size());
    MNN_PRINT("MiniMax-H3 conditioner: %d token(s)\n", *numTokens);
    return encodeTokens(tokenIds, hiddenStates);
}

} // namespace DIFFUSION
} // namespace MNN
