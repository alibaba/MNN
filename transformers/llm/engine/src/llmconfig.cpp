#include "llmconfig.hpp"
#include "core/MNNFileUtils.h"
#include <algorithm>

namespace MNN {
namespace Transformer {
namespace {

uint32_t read32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

uint32_t adler32(const std::vector<uint8_t>& data) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < data.size();) {
        const size_t end = std::min(i + 5552, data.size());
        for (; i < end; ++i) {
            a += data[i];
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    return (b << 16) | a;
}

class LlmFile : public LlmResources {
public:
    Resource resource(const std::string& path) const override {
        const auto it = mEntries.find(path);
        return it == mEntries.end() ? Resource() : it->second;
    }

    bool read(const std::string& path, std::istream& input) {
        uint8_t header[12];
        if (!input.read(reinterpret_cast<char*>(header), sizeof(header))) {
            return false;
        }
        const auto version = read32(header);
        uint32_t headerSize = 20;
        if (version == 3) {
            uint8_t extension[8];
            if (!input.read(reinterpret_cast<char*>(extension), sizeof(extension))) {
                return false;
            }
            headerSize = read32(extension);
            const auto requiredFeatures = read32(extension + 4);
            if (headerSize < 28) {
                return false;
            }
            if (requiredFeatures != 0) {
                MNN_ERROR("Unsupported .mpack required_features: 0x%08x\n", requiredFeatures);
                return false;
            }
        } else if (version != 2) {
            return false;
        }
        const auto size = read32(header + 4), checksum = read32(header + 8);
        constexpr uint32_t limit = 64 * 1024 * 1024;
        if (size < 4 || size > limit) {
            return false;
        }
        FileLoader directory(FileLoader::fileRegion(path, headerSize, size).c_str(), true);
        if (!directory.valid()) {
            return false;
        }
        mDirectory.resize(size);
        if (!directory.read(reinterpret_cast<char*>(mDirectory.data()), size)) {
            return false;
        }
        return checksum == adler32(mDirectory) && readEntries(path, uint64_t(headerSize) + size);
    }

private:
    bool readEntries(const std::string& path, uint64_t fileEnd) {
        const uint32_t count = read32(mDirectory.data());
        if (count == 0 || count > 1024) {
            return false;
        }
        size_t pos = 4;
        for (uint32_t i = 0; i < count; ++i) {
            if (mDirectory.size() - pos < 16) {
                return false;
            }
            const auto p = mDirectory.data() + pos;
            const uint32_t nameSize = read32(p), storage = read32(p + 4);
            Resource r;
            r.size = read32(p + 8) | (uint64_t(read32(p + 12)) << 32);
            pos += 16;
            if (!nameSize || nameSize > mDirectory.size() - pos || !r.size) {
                return false;
            }
            std::string name(reinterpret_cast<const char*>(mDirectory.data() + pos), nameSize);
            pos += nameSize;
            if (LlmConfig::relativeResourcePath("", name) != name) {
                return false;
            }
            if (storage == 0) {
                pos = (pos + 7) & ~size_t(7);
                if (pos > mDirectory.size() || r.size > mDirectory.size() - pos) {
                    return false;
                }
                r.data = mDirectory.data() + pos;
                pos += size_t(r.size);
            } else if (storage == 1) {
                r.path = path;
                r.offset = (fileEnd + 16383) & ~uint64_t(16383);
                r.bounded = true;
                if (!r.valid()) {
                    return false;
                }
                fileEnd = r.offset + r.size;
            } else {
                return false;
            }
            if (!mEntries.emplace(std::move(name), std::move(r)).second) {
                return false;
            }
        }
        // FileLoader checks the last byte with 64-bit offsets on 32-bit Android as well.
        return pos == mDirectory.size() &&
               FileLoader(FileLoader::fileRegion(path, int64_t(fileEnd - 1), 1).c_str(), true).valid();
    }

    std::vector<uint8_t> mDirectory;
    std::unordered_map<std::string, Resource> mEntries;
};

} // namespace

bool tryReadLlmPackage(const std::string& path, std::shared_ptr<LlmResources>& result) {
    std::ifstream input(path, std::ios::binary);
    uint8_t signature[8];
    if (!input.read(reinterpret_cast<char*>(signature), sizeof(signature)) ||
        read32(signature) != 430 || read32(signature + 4) != 1) {
        return false;
    }
    auto file = std::make_shared<LlmFile>();
    if (file->read(path, input)) {
        result = std::move(file);
    }
    return true;
}

Express::Module* LlmConfig::loadModule(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs,
                                       const std::string& path,
                                       const std::shared_ptr<Express::Executor::RuntimeManager>& runtime,
                                       const Express::Module::Config* config, const std::string& weight) const {
    const auto graph = resource(path);
    if (!graph.valid()) {
        MNN_ERROR("Missing model resource: %s\n", path.c_str());
        return nullptr;
    }
    runtime->setExternalFile("");
    if (!weight.empty() || resources_) {
        const auto weights = resource(weight.empty() ? path + ".weight" : weight);
        if (weights.data || (!weight.empty() && !weights.valid())) {
            MNN_ERROR("External weights require a file region: %s\n", path.c_str());
            return nullptr;
        }
        if (weights.bounded) {
            if (!runtime->setExternalFile(weights.path, int64_t(weights.offset), int64_t(weights.size))) {
                return nullptr;
            }
        } else {
            runtime->setExternalFile(weights.path);
        }
    }
    Express::Module* module = nullptr;
    if (graph.data) {
        module = Express::Module::load(inputs, outputs, graph.data, size_t(graph.size), runtime, config);
    } else if (graph.bounded) {
        module = Express::Module::load(inputs, outputs, graph.path.c_str(), int64_t(graph.offset), int64_t(graph.size),
                                       runtime, config);
    } else {
        module = Express::Module::load(inputs, outputs, graph.path.c_str(), runtime, config);
    }
    runtime->setExternalFile("");
    return module;
}

std::map<std::string, Express::VARP> LlmConfig::loadVariables(const std::string& path) const {
    const auto source = resource(path);
    if (!source.valid()) {
        return {};
    }
    return source.data ? Express::Variable::loadMap(source.data, size_t(source.size))
                       : Express::Variable::loadMap(source.filePath().c_str());
}

} // namespace Transformer
} // namespace MNN
