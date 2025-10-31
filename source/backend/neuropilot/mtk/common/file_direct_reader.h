#pragma once

#include "thread_pool.h"

#include <string>

// File reader using Direct IO
class FileDirectReader {
private:
    static constexpr size_t kMemoryAlignment = 4096UL; // 4096 Bytes
    static constexpr size_t kBlockSize = 1048576UL;    // 1 MB

public:
    explicit FileDirectReader(const std::string& filepath, const size_t numWorkers = 8);

    ~FileDirectReader();

    explicit operator bool() const;

    bool valid() const;

    size_t size() const;

    bool load(void* dstBuffer, const size_t dstSize);

private:
    static size_t legalizeNumWorkers(const size_t numWorkers);

private:
    const std::string kFilePath;
    const size_t kNumWorkers;
    BasicThreadPool mThreadPool;

    int mFileFd = -1;
    size_t mFileSize = 0;
    size_t mWholeBlocksSize = 0;
    size_t mLastBlockSize = 0;
};
