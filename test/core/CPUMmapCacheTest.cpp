// Copyright (c) Alibaba Group Holding Limited. All rights reserved.

#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "MNNTestSuite.h"
#include "core/Backend.hpp"
#include "core/MNNFileUtils.h"

using namespace MNN;

// These lifecycle cases require POSIX file unlink semantics.
#if !defined(_WIN32)
namespace {
// Each case owns a fresh directory. No model or external test data is needed.
class MmapTestDirectory {
public:
    MmapTestDirectory() {
        char name[] = "./mnn_mmap_cache_XXXXXX";
        if (mkdtemp(name)) {
            path = name;
        }

    }
    ~MmapTestDirectory() {
        for (const auto& name : files()) {
            MNNRemoveFile(MNNFilePathConcat(path, name).c_str());
        }
        if (!path.empty()) {
            rmdir(path.c_str());

        }
    }
    std::vector<std::string> files() const {
        std::vector<std::string> result;
        if (path.empty()) {
            return result;
        }
        auto directory = opendir(path.c_str());
        if (directory) {
            while (auto entry = readdir(directory)) {
                if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
                    result.emplace_back(entry->d_name);
                }
            }
            closedir(directory);
        }

        return result;
    }
    // Discover the file the real runtime created; do not assert a prefix string.
    std::string findSuffix(const char* suffix) const {
        const size_t length = strlen(suffix);
        for (const auto& name : files()) {
            if (name.size() >= length && name.compare(name.size() - length, length, suffix) == 0) {
                return MNNFilePathConcat(path, name);
            }
        }
        return "";
    }
    std::string path;
};

// Allocate through the production CPU runtime/backend. The STATIC tensor stands
// in for an execution's packed weights, and onClearBuffer seals/syncs the pool.
bool runStaticTensor(const std::string& directory, bool cached, bool expectTrust, bool write, int value) {
    BackendConfig config;
    config.precision = BackendConfig::Precision_High;
    config.flags = 4; // Default CPU backend; no ISA-specific packed format needed.
    Backend::Info info;
    info.type = MNN_FORWARD_CPU;
    info.numThread = 1;
    info.user = &config;
    auto creator = MNNGetExtraRuntimeCreator(MNN_FORWARD_CPU);
    MNNTEST_ASSERT(creator != nullptr);
    std::unique_ptr<Runtime> runtime(creator->onCreate(info));
    MNNTEST_ASSERT(runtime != nullptr);
    RuntimeHint hint;
    hint.weightMemoryPath = directory;
    hint.useCachedMmap = cached ? 1 : 0;
    hint.mmapFileSize = 1;
    runtime->setRuntimeHint(hint);
    std::unique_ptr<Backend> backend(runtime->onCreate(&config));
    MNNTEST_ASSERT(backend != nullptr);
    if ((runtime->hint().useCachedMmap > 1) != expectTrust) {
        MNN_ERROR("mmap cache trust mismatch: cached=%d hint=%d expected=%d\n", cached,
                  runtime->hint().useCachedMmap, expectTrust);
        return false;
    }
    std::unique_ptr<Tensor> weights(Tensor::createDevice<int>({64}));
    MNNTEST_ASSERT(backend->onAcquireBuffer(weights.get(), Backend::STATIC));
    MNNTEST_ASSERT(weights->host<int>() != nullptr);
    if (write) {
        for (int i = 0; i < 64; ++i) {
            weights->host<int>()[i] = value + i;
        }
    } else {
        for (int i = 0; i < 64; ++i) {
            if (weights->host<int>()[i] != value + i) {
                MNN_ERROR("mmap cached weight mismatch at %d: %d != %d\n", i, weights->host<int>()[i], value + i);
                return false;
            }
        }
    }
    MNNTEST_ASSERT(backend->onClearBuffer());
    // Destruction order is tensor -> backend -> runtime, including allocator.
    return true;
}

bool checkSavedWeights(const std::string& fileName, int value) {
    auto file = MNNOpenFile(fileName.c_str(), MNN_FILE_READ);
    MNNTEST_ASSERT(file != INVALID_FILE);
    int actual[64] = {0};
    auto count = MNNReadFile(file, actual, sizeof(actual));
    MNNCloseFile(file);
    MNNTEST_ASSERT(count == sizeof(actual));
    for (int i = 0; i < 64; ++i) {
        MNNTEST_ASSERT(actual[i] == value + i);
    }
    return true;
}
} // namespace

class CPUMmapCacheModeSwitchTest : public MNNTestCase {
public:
    bool run(int precision) override {
        MmapTestDirectory directory;
        MNNTEST_ASSERT(!directory.path.empty());
        const int cachedValue = 0x13572468;
        MNNTEST_ASSERT(runStaticTensor(directory.path, true, false, true, cachedValue));
        auto dataFile = directory.findSuffix("0.static");
        auto marker = directory.findSuffix("sync.static");
        MNNTEST_ASSERT(!dataFile.empty() && !marker.empty());
        MNNTEST_ASSERT(checkSavedWeights(dataFile, cachedValue));
        MNNTEST_ASSERT(runStaticTensor(directory.path, true, true, false, cachedValue));
        // The temporary allocator may write different weights and delete its own
        // files. It must preserve the persistent cache's bytes and marker.
        MNNTEST_ASSERT(runStaticTensor(directory.path, false, false, true, 0x24681357));
        MNNTEST_ASSERT(MNNFileExist(dataFile.c_str()));
        MNNTEST_ASSERT(MNNFileExist(marker.c_str()));
        MNNTEST_ASSERT(checkSavedWeights(dataFile, cachedValue));
        MNNTEST_ASSERT(runStaticTensor(directory.path, true, true, false, cachedValue));
        return true;
    }
};

class CPUMmapCacheMissingDataTest : public MNNTestCase {
public:
    bool run(int precision) override {
        MmapTestDirectory directory;
        MNNTEST_ASSERT(!directory.path.empty());
        const int value = 0x13572468;
        MNNTEST_ASSERT(runStaticTensor(directory.path, true, false, true, value));
        auto dataFile = directory.findSuffix("0.static");
        auto marker = directory.findSuffix("sync.static");
        MNNTEST_ASSERT(!dataFile.empty() && !marker.empty());
        MNNRemoveFile(dataFile.c_str());
        MNNTEST_ASSERT(!MNNFileExist(dataFile.c_str()) && MNNFileExist(marker.c_str()));
        // A marker alone cannot authorize executions to skip weight loading.
        MNNTEST_ASSERT(runStaticTensor(directory.path, true, false, true, value));
        MNNTEST_ASSERT(checkSavedWeights(dataFile, value));
        MNNTEST_ASSERT(runStaticTensor(directory.path, true, true, false, value));
        return true;
    }
};

MNNTestSuiteRegister(CPUMmapCacheModeSwitchTest, "core/cpu/mmap_cache/mode_switch");
MNNTestSuiteRegister(CPUMmapCacheMissingDataTest, "core/cpu/mmap_cache/missing_first_data");
#endif // !defined(_WIN32)
