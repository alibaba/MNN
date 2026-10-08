// Copyright (c) Alibaba Group Holding Limited. All rights reserved.

#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "MNNTestSuite.h"
#include "core/Backend.hpp"
#include "core/BufferAllocator.hpp"
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
bool runStaticTensor(const std::string& directory, bool cached, bool expectTrust, bool write, int value,
                     int count = 1, int length = 64) {
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
    std::vector<std::unique_ptr<Tensor>> tensors;
    for (int t = 0; t < count; ++t) {
        tensors.emplace_back(Tensor::createDevice<int>({length}));
        auto weights = tensors.back().get();
        MNNTEST_ASSERT(backend->onAcquireBuffer(weights, Backend::STATIC));
        MNNTEST_ASSERT(weights->host<int>() != nullptr);
        for (int i = 0; i < length; ++i) {
            if (write) {
                weights->host<int>()[i] = value + i + t;
            } else if (weights->host<int>()[i] != value + i + t) {
                MNN_ERROR("mmap cached weight mismatch at tensor=%d index=%d\n", t, i);
                return false;
            }
        }
    }
    MNNTEST_ASSERT(backend->onClearBuffer());
    // Destruction order is tensor -> backend -> runtime, including allocator.
    return true;
}

std::vector<uint8_t> readSavedBytes(const std::string& name) {
    auto file = MNNOpenFile(name.c_str(), MNN_FILE_READ);
    if (file == INVALID_FILE) return {};
    const auto length = MNNGetFileSize(file);
    std::vector<uint8_t> bytes;
    if (length != INVALID_SIZE) {
        bytes.resize(length);
        if (MNNReadFile(file, bytes.data(), bytes.size()) != bytes.size()) bytes.clear();
    }
    if (MNNCloseFile(file) != NO_ERROR) bytes.clear();
    return bytes;
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

class CPUMmapCacheManifestTest : public MNNTestCase {
public:
    bool run(int precision) override {
        for (int mutation = 0; mutation < 3; ++mutation) {
            MmapTestDirectory directory;
            MNNTEST_ASSERT(!directory.path.empty());
            // Each tensor exceeds the pool minimum, forcing three mmap files.
            MNNTEST_ASSERT(runStaticTensor(directory.path, true, false, true, 1234, 3, 300000));
            MNNTEST_ASSERT(runStaticTensor(directory.path, true, true, false, 1234, 3, 300000));
            auto later = directory.findSuffix("2.static");
            MNNTEST_ASSERT(!later.empty());
            if (mutation == 0) {
                MNNTEST_ASSERT(MNNRemoveFile(later.c_str()) == NO_ERROR);
            } else {
                auto file = MNNOpenFile(later.c_str(), MNN_FILE_READ | MNN_FILE_WRITE);
                MNNTEST_ASSERT(file != INVALID_FILE);
                auto size = MNNGetFileSize(file);
                MNNTEST_ASSERT(size != INVALID_SIZE);
                MNNTEST_ASSERT(MNNSetFileSize(file, mutation == 1 ? size - 32 : size + 32) == NO_ERROR);
                MNNTEST_ASSERT(MNNCloseFile(file) == NO_ERROR);
            }
            // A valid first file must not hide corruption in a later file.
            MNNTEST_ASSERT(runStaticTensor(directory.path, true, false, true, 5678, 3, 300000));
            MNNTEST_ASSERT(runStaticTensor(directory.path, true, true, false, 5678, 3, 300000));
        }
        return true;
    }
};

class CPUMmapCacheInvalidMarkerTest : public MNNTestCase {
public:
    bool run(int precision) override {
        for (int mutation = 0; mutation < 9; ++mutation) {
            MmapTestDirectory directory;
            MNNTEST_ASSERT(!directory.path.empty());
            MNNTEST_ASSERT(runStaticTensor(directory.path, true, false, true, 1234));
            auto marker = directory.findSuffix("sync.static");
            auto file = MNNOpenFile(marker.c_str(), MNN_FILE_READ | MNN_FILE_WRITE);
            MNNTEST_ASSERT(file != INVALID_FILE);
            uint64_t saved[4] = {};
            MNNTEST_ASSERT(MNNReadFile(file, saved, sizeof(saved)) == sizeof(saved));
            MNNTEST_ASSERT(MNNCloseFile(file) == NO_ERROR);
            // Empty legacy marker, truncated payload, trailing payload,
            // unsupported version, overflowing count, zero count, zero size.
            if (mutation == 3) saved[1] += 1;
            if (mutation == 4) saved[2] = UINT64_MAX;
            if (mutation == 5) saved[2] = 0;
            if (mutation == 6) saved[3] = 0;
            if (mutation == 7) saved[0] ^= 1;
            if (mutation == 8) saved[3] += 1;
            file = MNNCreateFile(marker.c_str());
            MNNTEST_ASSERT(file != INVALID_FILE);
            const auto bytes = mutation == 0 ? 0 : mutation == 1 ? sizeof(saved) - 1 : sizeof(saved);
            if (bytes) {
                MNNTEST_ASSERT(MNNWriteFile(file, saved, bytes) == bytes);
            }
            if (mutation == 2) {
                uint64_t trailing = 42;
                MNNTEST_ASSERT(MNNWriteFile(file, &trailing, sizeof(trailing)) == sizeof(trailing));
            }
            MNNTEST_ASSERT(MNNCloseFile(file) == NO_ERROR);
            MNNTEST_ASSERT(runStaticTensor(directory.path, true, false, true, 5678));
            MNNTEST_ASSERT(runStaticTensor(directory.path, true, true, false, 5678));
        }
        return true;
    }
};

class CPUMmapCacheAllocatorTest : public MNNTestCase {
public:
    bool run(int precision) override {
        MmapTestDirectory directory;
        MNNTEST_ASSERT(!directory.path.empty());
        auto allocator = BufferAllocator::Allocator::createMmap(directory.path.c_str(), "", "static", false);
        allocator->sync();
        MNNTEST_ASSERT(!MNNFileExist(MNNFilePathConcat(directory.path, "sync.static").c_str()));
        auto first = allocator->onAlloc(32, 32);
        auto second = allocator->onAlloc(64, 32);
        MNNTEST_ASSERT(!first.invalid() && !second.invalid());
        memset(first.first, 1, 32);
        memset(second.first, 2, 64);
        allocator->onRelease(first);
        // Releasing one mapping must not reuse its index while others live.
        auto third = allocator->onAlloc(96, 32);
        MNNTEST_ASSERT(!third.invalid());
        MNNTEST_ASSERT(MNNFileExist(MNNFilePathConcat(directory.path, "2.static").c_str()));
        allocator->sync();
        MNNTEST_ASSERT(BufferAllocator::Allocator::validateMmapCache(directory.path.c_str(), "", "static"));
        allocator->onRelease(second);
        allocator->onRelease(third);
        // Once all mappings are gone, the next generation starts at index zero.
        auto again = allocator->onAlloc(16, 16);
        MNNTEST_ASSERT(!again.invalid());
        MNNTEST_ASSERT(!MNNFileExist(MNNFilePathConcat(directory.path, "sync.static").c_str()));
        allocator->sync();
        MNNTEST_ASSERT(BufferAllocator::Allocator::validateMmapCache(directory.path.c_str(), "", "static"));
        auto manifest = MNNOpenFile(MNNFilePathConcat(directory.path, "sync.static").c_str(), MNN_FILE_READ);
        MNNTEST_ASSERT(manifest != INVALID_FILE);
        uint64_t record[4] = {};
        MNNTEST_ASSERT(MNNReadFile(manifest, record, sizeof(record)) == sizeof(record));
        MNNTEST_ASSERT(MNNCloseFile(manifest) == NO_ERROR);
        // The old 32-byte file is retained while only 16 bytes are mapped.
        // The manifest records its actual EOF, not the new mapping length.
        MNNTEST_ASSERT(record[2] == 1 && record[3] == 32);
        allocator->onRelease(again);
        auto marker = MNNFilePathConcat(directory.path, "sync.static");
        MNNTEST_ASSERT(MNNRemoveFile(marker.c_str()) == NO_ERROR);
        // A marker path that is a directory makes publication fail safely.
        auto failed = allocator->onAlloc(32, 32);
        MNNTEST_ASSERT(!failed.invalid());
        MNNTEST_ASSERT(mkdir(marker.c_str(), 0700) == 0);
        allocator->sync();
        MNNTEST_ASSERT(!BufferAllocator::Allocator::validateMmapCache(directory.path.c_str(), "", "static"));
        allocator->onRelease(failed);
        // An unremovable marker also blocks rebuilding underlying weights.
        MNNTEST_ASSERT(allocator->onAlloc(32, 32).invalid());
        MNNTEST_ASSERT(rmdir(marker.c_str()) == 0);
        return true;
    }
};

MNNTestSuiteRegister(CPUMmapCacheManifestTest, "core/cpu/mmap_cache/later_data_manifest");
MNNTestSuiteRegister(CPUMmapCacheInvalidMarkerTest, "core/cpu/mmap_cache/invalid_marker");
MNNTestSuiteRegister(CPUMmapCacheAllocatorTest, "core/cpu/mmap_cache/allocator_lifecycle");

// A validated manifest is only permission to reuse the recorded files, never
// permission to synthesize missing bytes after executions skip weight loading.
template<int Mutation>
class CPUMmapCacheWarmAllocatorGuardTest : public MNNTestCase {
public:
    bool run(int precision) override {
        for (int mutation = Mutation; mutation <= Mutation; ++mutation) {
            MmapTestDirectory directory;
            auto cold = BufferAllocator::Allocator::createMmap(directory.path.c_str(), "", "static", false);
            auto seed = cold->onAlloc(32, 32);
            MNNTEST_ASSERT(!seed.invalid());
            memset(seed.first, 7, 32);
            cold->sync();
            auto warm = BufferAllocator::Allocator::createMmap(directory.path.c_str(), "", "static", false, true);
            const auto data = MNNFilePathConcat(directory.path, "0.static");
            const auto marker = MNNFilePathConcat(directory.path, "sync.static");
            if (mutation == 2) {
                MNNTEST_ASSERT(MNNRemoveFile(data.c_str()) == NO_ERROR);
            } else if (mutation == 3) {
                auto file = MNNOpenFile(data.c_str(), MNN_FILE_READ | MNN_FILE_WRITE);
                MNNTEST_ASSERT(file != INVALID_FILE);
                MNNTEST_ASSERT(MNNSetFileSize(file, 16) == NO_ERROR);
                MNNTEST_ASSERT(MNNCloseFile(file) == NO_ERROR);
            }
            const auto dataBefore = readSavedBytes(data);
            const auto markerBefore = readSavedBytes(marker);
            MNNTEST_ASSERT(!markerBefore.empty());
            MemChunk first;
            if (mutation == 1) {
                first = warm->onAlloc(32, 32);
                MNNTEST_ASSERT(!first.invalid());
            }
            MNNTEST_ASSERT(warm->onAlloc(mutation == 0 ? 64 : 32, 32).invalid());
            MNNTEST_ASSERT(!MNNFileExist(MNNFilePathConcat(directory.path, "1.static").c_str()));
            MNNTEST_ASSERT(MNNFileExist(marker.c_str()));
            if (mutation != 2) {
                auto file = MNNOpenFile(data.c_str(), MNN_FILE_READ);
                MNNTEST_ASSERT(file != INVALID_FILE);
                MNNTEST_ASSERT(MNNGetFileSize(file) == (mutation == 3 ? 16 : 32));
                MNNTEST_ASSERT(MNNCloseFile(file) == NO_ERROR);
            } else {
                MNNTEST_ASSERT(!MNNFileExist(data.c_str()));
            }
            warm->sync();
            MNNTEST_ASSERT(readSavedBytes(data) == dataBefore);
            MNNTEST_ASSERT(readSavedBytes(marker) == markerBefore);
            if (!first.invalid()) {
                warm->onRelease(first);
                MNNTEST_ASSERT(warm->onAlloc(32, 32).invalid());
                MNNTEST_ASSERT(readSavedBytes(data) == dataBefore);
                MNNTEST_ASSERT(readSavedBytes(marker) == markerBefore);
            }
        }
        return true;
    }
};

template<int Mutation>
class CPUMmapCacheRuntimeGuardTest : public MNNTestCase {
public:
    bool run(int precision) override {
        for (int mutation = Mutation; mutation <= Mutation; ++mutation) {
            MmapTestDirectory directory;
            MNNTEST_ASSERT(runStaticTensor(directory.path, true, false, true, 1234));
            BackendConfig config;
            config.precision = BackendConfig::Precision_High;
            config.flags = 4;
            Backend::Info info;
            info.type = MNN_FORWARD_CPU;
            info.numThread = 1;
            info.user = &config;
            auto creator = MNNGetExtraRuntimeCreator(MNN_FORWARD_CPU);
            std::unique_ptr<Runtime> runtime(creator->onCreate(info));
            RuntimeHint hint;
            hint.weightMemoryPath = directory.path;
            hint.useCachedMmap = 1;
            hint.mmapFileSize = 1;
            runtime->setRuntimeHint(hint);
            std::unique_ptr<Backend> backend(runtime->onCreate(&config));
            MNNTEST_ASSERT(runtime->hint().useCachedMmap > 1);
            auto data = directory.findSuffix("0.static");
            auto marker = directory.findSuffix("sync.static");
            const auto markerBefore = readSavedBytes(marker);
            std::unique_ptr<Tensor> first;
            if (mutation == 1) {
                first.reset(Tensor::createDevice<int>({300000}));
                // The recorded minimum pool is only one MiB.
                MNNTEST_ASSERT(!backend->onAcquireBuffer(first.get(), Backend::STATIC));
            } else {
                if (mutation >= 2) {
                    if (mutation == 2) {
                        MNNTEST_ASSERT(MNNRemoveFile(data.c_str()) == NO_ERROR);
                    } else {
                        auto file = MNNOpenFile(data.c_str(), MNN_FILE_READ | MNN_FILE_WRITE);
                        MNNTEST_ASSERT(file != INVALID_FILE);
                        MNNTEST_ASSERT(MNNSetFileSize(file, 16) == NO_ERROR);
                        MNNTEST_ASSERT(MNNCloseFile(file) == NO_ERROR);
                    }
                } else {
                    first.reset(Tensor::createDevice<int>({262144}));
                    MNNTEST_ASSERT(backend->onAcquireBuffer(first.get(), Backend::STATIC));
                }
                std::unique_ptr<Tensor> rejected(Tensor::createDevice<int>({64}));
                MNNTEST_ASSERT(!backend->onAcquireBuffer(rejected.get(), Backend::STATIC));
            }
            MNNTEST_ASSERT(MNNFileExist(marker.c_str()));
            MNNTEST_ASSERT(directory.findSuffix("1.static").empty());
            MNNTEST_ASSERT(readSavedBytes(marker) == markerBefore);
            if (mutation < 2) {
                MNNTEST_ASSERT(checkSavedWeights(data, 1234));
            }
        }
        // A cold runtime may still fall back to ordinary memory when mmap fails.
        MmapTestDirectory directory;
        const auto blocked = MNNFilePathConcat(directory.path, "0_0_0_0_sync.static");
        MNNTEST_ASSERT(mkdir(blocked.c_str(), 0700) == 0);
        MNNTEST_ASSERT(runStaticTensor(directory.path, true, false, true, 9876));
        MNNTEST_ASSERT(rmdir(blocked.c_str()) == 0);
        return true;
    }
};
typedef CPUMmapCacheWarmAllocatorGuardTest<0> WarmAllocatorGrow;
typedef CPUMmapCacheWarmAllocatorGuardTest<1> WarmAllocatorCount;
typedef CPUMmapCacheWarmAllocatorGuardTest<2> WarmAllocatorMissing;
typedef CPUMmapCacheWarmAllocatorGuardTest<3> WarmAllocatorTruncated;
typedef CPUMmapCacheRuntimeGuardTest<0> WarmRuntimeCount;
typedef CPUMmapCacheRuntimeGuardTest<1> WarmRuntimeGrow;
typedef CPUMmapCacheRuntimeGuardTest<2> WarmRuntimeMissing;
typedef CPUMmapCacheRuntimeGuardTest<3> WarmRuntimeTruncated;
MNNTestSuiteRegister(WarmAllocatorGrow, "core/cpu/mmap_cache/warm_allocator_grow");
MNNTestSuiteRegister(WarmAllocatorCount, "core/cpu/mmap_cache/warm_allocator_count_and_release");
MNNTestSuiteRegister(WarmAllocatorMissing, "core/cpu/mmap_cache/warm_allocator_missing");
MNNTestSuiteRegister(WarmAllocatorTruncated, "core/cpu/mmap_cache/warm_allocator_truncated");
MNNTestSuiteRegister(WarmRuntimeCount, "core/cpu/mmap_cache/warm_runtime_count");
MNNTestSuiteRegister(WarmRuntimeGrow, "core/cpu/mmap_cache/warm_runtime_grow");
MNNTestSuiteRegister(WarmRuntimeMissing, "core/cpu/mmap_cache/warm_runtime_missing");
MNNTestSuiteRegister(WarmRuntimeTruncated, "core/cpu/mmap_cache/warm_runtime_truncated");

class CPUMmapCacheMixedFallbackTest : public MNNTestCase {
public:
    bool run(int precision) override {
        MmapTestDirectory directory;
        MNNTEST_ASSERT(!directory.path.empty());
        // First load: A maps file 0; B falls back to RAW; C maps file 1.
        // Publishing those two files would make the next load read C as B.
        // Second load must rebuild all weights; only the third may trust them.
        const int lengths[] = {262144, 64, 64};
        const int values[] = {0x12341000, 0x23452000, 0x34563000};
        for (int phase = 0; phase < 3; ++phase) {
            BackendConfig config;
            config.precision = BackendConfig::Precision_High;
            config.flags = 4;
            Backend::Info info;
            info.type = MNN_FORWARD_CPU;
            info.numThread = 1;
            info.user = &config;
            auto creator = MNNGetExtraRuntimeCreator(MNN_FORWARD_CPU);
            MNNTEST_ASSERT(creator != nullptr);
            std::unique_ptr<Runtime> runtime(creator->onCreate(info));
            MNNTEST_ASSERT(runtime != nullptr);
            RuntimeHint hint;
            hint.weightMemoryPath = directory.path;
            hint.useCachedMmap = 1;
            hint.mmapFileSize = 1;
            runtime->setRuntimeHint(hint);
            std::unique_ptr<Backend> backend(runtime->onCreate(&config));
            MNNTEST_ASSERT(backend != nullptr);
            MNNTEST_ASSERT(runtime->hint().useCachedMmap == (phase == 2 ? 2 : 1));
            std::vector<std::unique_ptr<Tensor>> tensors;
            std::string blocked;
            for (int t = 0; t < 3; ++t) {
                if (phase == 0 && t == 1) {
                    const auto zero = directory.findSuffix("0.static");
                    MNNTEST_ASSERT(!zero.empty());
                    blocked = zero.substr(0, zero.size() - strlen("0.static")) + "1.static";
                    MNNTEST_ASSERT(mkdir(blocked.c_str(), 0700) == 0);
                }
                tensors.emplace_back(Tensor::createDevice<int>({lengths[t]}));
                auto weights = tensors.back().get();
                MNNTEST_ASSERT(backend->onAcquireBuffer(weights, Backend::STATIC));
                MNNTEST_ASSERT(weights->host<int>() != nullptr);
                for (int i = 0; i < lengths[t]; ++i) {
                    if (phase < 2) weights->host<int>()[i] = values[t] + i;
                    MNNTEST_ASSERT(weights->host<int>()[i] == values[t] + i);
                }
                if (phase == 0 && t == 1) {
                    MNNTEST_ASSERT(rmdir(blocked.c_str()) == 0);
                }
                if (phase == 0 && t == 2) {
                    // C really resumed mmap, rather than falling back as well.
                    MNNTEST_ASSERT(checkSavedWeights(blocked, values[2]));
                }
            }
            MNNTEST_ASSERT(backend->onClearBuffer());
            MNNTEST_ASSERT(directory.findSuffix("sync.static").empty() == (phase == 0));
        }
        return true;
    }
};

class CPUMmapCacheColdFailureStickyTest : public MNNTestCase {
public:
    bool run(int precision) override {
        for (int blockedMarker = 0; blockedMarker < 2; ++blockedMarker) {
            MmapTestDirectory directory;
            MNNTEST_ASSERT(!directory.path.empty());
            auto allocator = BufferAllocator::Allocator::createMmap(directory.path.c_str(), "", "static", false);
            const auto blocked = MNNFilePathConcat(directory.path, blockedMarker ? "sync.static" : "0.static");
            MNNTEST_ASSERT(mkdir(blocked.c_str(), 0700) == 0);
            MNNTEST_ASSERT(allocator->onAlloc(32, 32).invalid());
            MNNTEST_ASSERT(rmdir(blocked.c_str()) == 0);
            auto recovered = allocator->onAlloc(32, 32);
            MNNTEST_ASSERT(!recovered.invalid());
            memset(recovered.first, 3, 32);
            allocator->sync();
            MNNTEST_ASSERT(!MNNFileExist(MNNFilePathConcat(directory.path, "sync.static").c_str()));
            // Releasing all mmap buffers cannot prove RAW fallback buffers are
            // gone: failure stays sticky for the allocator's entire lifetime.
            allocator->onRelease(recovered);
            auto next = allocator->onAlloc(32, 32);
            MNNTEST_ASSERT(!next.invalid());
            memset(next.first, 4, 32);
            allocator->sync();
            MNNTEST_ASSERT(!MNNFileExist(MNNFilePathConcat(directory.path, "sync.static").c_str()));
            allocator->onRelease(next);
            allocator.reset();
            allocator = BufferAllocator::Allocator::createMmap(directory.path.c_str(), "", "static", false);
            auto rebuilt = allocator->onAlloc(32, 32);
            MNNTEST_ASSERT(!rebuilt.invalid());
            memset(rebuilt.first, 5, 32);
            allocator->sync();
            MNNTEST_ASSERT(BufferAllocator::Allocator::validateMmapCache(directory.path.c_str(), "", "static"));
            allocator->onRelease(rebuilt);
        }
        return true;
    }
};
MNNTestSuiteRegister(CPUMmapCacheMixedFallbackTest, "core/cpu/mmap_cache/mixed_raw_fallback");
MNNTestSuiteRegister(CPUMmapCacheColdFailureStickyTest, "core/cpu/mmap_cache/cold_failure_sticky");

#endif // !defined(_WIN32)
