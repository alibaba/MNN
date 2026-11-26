//
//  FileLoader.hpp
//  MNN
//
//  Created by MNN on 2019/07/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef MNN_FileLoader_hpp
#define MNN_FileLoader_hpp
#include <vector>
#include <mutex>
#include <string>
#include <functional>
#include <memory>

#include "core/AutoStorage.h"
namespace MNN {

class BaseLoader {
public:
    BaseLoader() = default;
    virtual ~BaseLoader() = default;
    virtual bool read(char* buffer, int64_t size) = 0;
};

class MNN_PUBLIC FileLoader : public BaseLoader {
public:
    FileLoader(const char* file, bool init = false);

    ~FileLoader();

    bool read();

    static bool write(const char* filePath, std::pair<const void*, size_t> cacheInfo);

    // The returned locator preserves bounds when an external op reopens path().
    static std::string fileRegion(const std::string& file, int64_t offset, int64_t length);
    static bool isFileRegion(const std::string& file);

    // Retain the returned locator while readers may reopen it.
    // Filters receive absolute file offsets and may run concurrently.
    using ReadFilter = std::function<bool(uint8_t*, size_t, int64_t)>;
    static std::shared_ptr<const std::string> registerFile(const std::string& file, ReadFilter filter);

    bool valid() const {
        return mFile != nullptr;
    }
    inline size_t size() const {
        return mTotalSize;
    }
    inline std::string path() const {
        return mFilePath;
    }

    bool merge(AutoStorage<uint8_t>& buffer);

    int offset(int64_t offset);

    bool read(char* buffer, int64_t size);
private:
    struct ReadSource;
    std::shared_ptr<ReadSource> mSource;
    void _init();
    std::vector<std::pair<size_t, void*>> mBlocks;
    FILE* mFile                 = nullptr;
    static const int gCacheSize = 4096;
    size_t mTotalSize           = 0;
    std::string mFilePath;
    bool mInited = false;
    int64_t mRegionOffset = 0;
    int64_t mRegionLength = -1;
    int64_t mPosition = 0;
};

class MemoryLoader : public BaseLoader {
public:
    MemoryLoader(unsigned char* ptr) : buffer_(ptr) {}
    virtual bool read(char *dst, int64_t size) override {
        ::memcpy(dst, buffer_, size);
        buffer_ += size;
        return true;
    }
private:
    unsigned char* buffer_ = nullptr;
};
} // namespace MNN
#endif
