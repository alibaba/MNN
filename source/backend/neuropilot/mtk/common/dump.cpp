#include "dump.h"

#include "cpp11_compat.h"
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>

namespace fs = mtk::cpp11_compat::fs;

// Default disable dump
static constexpr int kDefaultDumpLevel = 0;

// Provides runtime dump level
class DumpOptions {
private:
    static constexpr char kPropertyKey[] = "debug.llm.dumplevel";
    explicit DumpOptions() {
        char propValue[PROP_VALUE_MAX] = "0";
        if (__system_property_get(kPropertyKey, propValue)) {
            propValue[1] = '\0';
            mDumpLevel = atoi(propValue);
        }
    }

public:
    static DumpOptions& instance() {
        static DumpOptions dumpOptions;
        return dumpOptions;
    }

    int getDumpLevel() const { return mDumpLevel; }

private:
    int mDumpLevel = kDefaultDumpLevel;
};

DumpHandler& DumpHandler::instance() {
    static DumpHandler dumpHandler;
    return dumpHandler;
}

bool DumpHandler::isEnabled() const {
    const int dumpLevel = DumpOptions::instance().getDumpLevel();

    // Cache dump requires dump level 2
    if (mDumpType == DumpType::CACHE) {
        return dumpLevel >= 2;
    }
    return dumpLevel >= 1;
}

DumpHandler& DumpHandler::setDumpType(const DumpType dumpType) {
    mDumpType = dumpType;
    return *this;
}

DumpHandler& DumpHandler::setIndex(const size_t index) {
    mIndex = index;
    return *this;
}

DumpHandler& DumpHandler::setChunkIndex(const size_t chunkIndex) {
    mChunkIndex = chunkIndex;
    return *this;
}

void DumpHandler::fromString(const std::string& name, const std::string& str) {
    if (!isEnabled()) {
        return;
    }
    const auto outpath = getSavePath(name + ".txt");
    std::fstream fout(outpath, std::ios::out | std::ios::trunc);
    fout << str;
}

template <typename T>
void DumpHandler::fromValue(const std::string& name, const T& val) {
    if (!isEnabled()) {
        return;
    }
    const auto outpath = getSavePath(name + ".txt");
    std::fstream fout(outpath, std::ios::out | std::ios::trunc);
    fout << val;
}

template <typename T>
void DumpHandler::fromVector(const std::string& name, const std::vector<T>& vec) {
    if (!isEnabled()) {
        return;
    }
    const auto outpath = getSavePath(name + ".txt");
    std::fstream fout(outpath, std::ios::out | std::ios::trunc);
    auto iter = vec.cbegin();
    fout << "{" << *iter++;
    while (iter != vec.cend()) {
        fout << ", " << *iter++;
    }
    fout << "}";
}

void DumpHandler::fromBinary(const std::string& name, const void* buffer, const size_t size) {
    if (!isEnabled()) {
        return;
    }
    const auto outpath = getSavePath(name + ".bin");
    std::fstream fout(outpath, std::ios::out | std::ios::trunc | std::ios::binary);
    fout.write(reinterpret_cast<const char*>(buffer), size);
}

std::string DumpHandler::getIndexStr() const {
    return std::to_string(mIndex);
}

std::string DumpHandler::getChunkName() const {
    return "chunk_" + std::to_string(mChunkIndex);
}

std::string DumpHandler::getSavePath(const std::string& fileName) const {
    fs::path saveDir(kDumpDir);
    switch (mDumpType) {
        case DumpType::PROMPT:
            saveDir = saveDir / "prompt";
            break;
        case DumpType::RESPONSE:
            saveDir = saveDir / "inference" / getIndexStr() / "response";
            break;
        case DumpType::INPUTS:
            saveDir = saveDir / "inference" / getIndexStr();
            break;
        case DumpType::CHUNK_OUT:
            saveDir = saveDir / "inference" / getIndexStr() / getChunkName();
            break;
        case DumpType::CACHE:
            saveDir = saveDir / "inference" / getIndexStr() / getChunkName() / "cache_outputs";
            break;
    }
    if (!fs::exists(saveDir.string())) {
        fs::create_directories(saveDir.string()); // mkdir -p <saveDir>
    }
    return (saveDir / fileName).string();
}

// Explicit instantiation of templated dump functions for integral types
#define __DECL__(Type)                                                              \
    template void DumpHandler::fromValue(const std::string& name, const Type& vec); \
    template void DumpHandler::fromVector(const std::string& name, const std::vector<Type>& vec);

__DECL__(int32_t)
__DECL__(int64_t)
__DECL__(uint32_t)
__DECL__(uint64_t)

#undef __DECL__
