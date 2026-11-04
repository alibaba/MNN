//
//  FileLoader.cpp
//  MNN
//
//  Created by MNN on 2019/07/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/FileLoader.hpp"
#include <climits>
#include <algorithm>
#include <limits>
#ifdef __ANDROID__
#include <cerrno>
#include <unistd.h>
#endif
#if defined(_MSC_VER)
#include "Windows.h"
#endif
namespace MNN {

static constexpr char kRegionPrefix[] = "mnn-region:1:";

bool FileLoader::isFileRegion(const std::string& file) {
    return file.compare(0, sizeof(kRegionPrefix) - 1, kRegionPrefix) == 0;
}

static bool readRegionNumber(const char*& cursor, int64_t& value) {
    value = 0;
    if (*cursor < '0' || *cursor > '9') {
        return false;
    }
    while (*cursor >= '0' && *cursor <= '9') {
        const int digit = *cursor++ - '0';
        if (value > (INT64_MAX - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
    }
    if (*cursor != ':')
        return false;
    ++cursor;
    return true;
}

std::string FileLoader::fileRegion(const std::string& file, int64_t offset, int64_t length) {
    if (file.empty() || offset < 0 || length <= 0 || offset > INT64_MAX - length ||
        file.compare(0, sizeof(kRegionPrefix) - 1, kRegionPrefix) == 0) {
        return {};
    }
    return std::string(kRegionPrefix) + std::to_string(offset) + ":" + std::to_string(length) + ":" + file;
}

static int seekFile(FILE* file, int64_t offset, int origin) {
#if defined(_MSC_VER)
    return _fseeki64(file, offset, origin);
#elif defined(__ANDROID__) && __ANDROID_API__ >= 24
    return fseeko64(file, offset, origin);
#elif defined(__ANDROID__) && !defined(__LP64__)
    return lseek64(fileno(file), offset, origin) < 0 ? -1 : 0;
#else
    if (offset > std::numeric_limits<off_t>::max())
        return -1;
    return fseeko(file, static_cast<off_t>(offset), origin);
#endif
}

static int64_t tellFile(FILE* file) {
#if defined(_MSC_VER)
    return _ftelli64(file);
#elif defined(__ANDROID__) && __ANDROID_API__ >= 24
    return ftello64(file);
#elif defined(__ANDROID__) && !defined(__LP64__)
    return lseek64(fileno(file), 0, SEEK_CUR);
#else
    return ftello(file);
#endif
}

static size_t readFile(FILE* file, void* buffer, size_t size, bool& error) {
#if defined(__ANDROID__) && !defined(__LP64__) && __ANDROID_API__ < 24
    // Old 32-bit bionic lacks 64-bit stdio seek; keep all reads on the descriptor too.
    size_t count = 0;
    while (count < size) {
        const auto result =
            ::read(fileno(file), static_cast<char*>(buffer) + count, std::min(size - count, size_t(SSIZE_MAX)));
        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0) {
            error = result < 0;
            break;
        }
        count += result;
    }
    return count;
#else
    const size_t count = fread(buffer, 1, size, file);
    error = ferror(file) != 0;
    return count;
#endif
}

static FILE* _OpenFile(const char* file, bool read) {
#if defined(_MSC_VER)
    wchar_t wFilename[1024];
    if (0 == MultiByteToWideChar(CP_ACP, 0, file, -1, wFilename, sizeof(wFilename))) {
        return nullptr;
    }
#if _MSC_VER >= 1400
    FILE* mFile = nullptr;
    if (read) {
        if (0 != _wfopen_s(&mFile, wFilename, L"rb")) {
            return nullptr;
        }
    } else {
        if (0 != _wfopen_s(&mFile, wFilename, L"wb")) {
            return nullptr;
        }
    }
    return mFile;
#else
    if (read) {
        return _wfopen(wFilename, L"rb");
    } else {
        return _wfopen(wFilename, L"wb");
    }
#endif
#else
    if (read) {
        return fopen(file, "rb");
    } else {
        return fopen(file, "wb");
    }
#endif
    return nullptr;
}
FileLoader::FileLoader(const char* file, bool init) {
    if (nullptr == file) {
        return;
    }
    mFilePath = file;
    if (init) {
        _init();
    }
}

FileLoader::~FileLoader() {
    if (nullptr != mFile) {
        fclose(mFile);
    }
    for (auto iter : mBlocks) {
        MNNMemoryFreeAlign(iter.second);
    }
}

bool FileLoader::read() {
    for (const auto& block : mBlocks) {
        MNNMemoryFreeAlign(block.second);
    }
    mBlocks.clear();
    mTotalSize = 0;
    _init();
    if (nullptr == mFile) {
        return false;
    }
    auto block = MNNMemoryAllocAlign(gCacheSize, MNN_MEMORY_ALIGN_DEFAULT);
    if (nullptr == block) {
        MNN_PRINT("Memory Alloc Failed\n");
        return false;
    }
    bool error = false;
    const auto readBlock = [&](void* dst) -> size_t {
        const size_t count = mRegionLength < 0
                                 ? gCacheSize
                                 : static_cast<size_t>(std::min<int64_t>(gCacheSize, mRegionLength - mPosition));
        const size_t result = readFile(mFile, dst, count, error);
        mPosition += result;
        return result;
    };
    auto size = readBlock(block);
    mTotalSize = size;
    mBlocks.push_back(std::make_pair(size, block));

    while (size == gCacheSize && (mRegionLength < 0 || mPosition < mRegionLength)) {
        block = MNNMemoryAllocAlign(gCacheSize, MNN_MEMORY_ALIGN_DEFAULT);
        if (nullptr == block) {
            MNN_PRINT("Memory Alloc Failed\n");
            return false;
        }
        size = readBlock(block);
        if (size > gCacheSize) {
            MNN_PRINT("Read file Error\n");
            MNNMemoryFreeAlign(block);
            return false;
        }
        mTotalSize += size;
        mBlocks.push_back(std::make_pair(size, block));
    }

    if (error) {
        return false;
    }
    return mRegionLength < 0 || mPosition == mRegionLength;
}

bool FileLoader::write(const char* filePath, std::pair<const void*, size_t> cacheInfo) {
    FILE* f = _OpenFile(filePath, false);
    if (nullptr == f) {
        MNN_ERROR("Open %s error\n", filePath);
        return false;
    }
    // Write Cache
    static const size_t block = 4096;
    size_t totalSize          = cacheInfo.second;
    size_t blockSize          = UP_DIV(totalSize, block);
    for (size_t i = 0; i < blockSize; ++i) {
        size_t sta = block * i;
        size_t fin = (sta + block >= totalSize) ? totalSize : (sta + block);
        if (fin > sta) {
            auto realSize = fwrite((const char*)(cacheInfo.first) + sta, 1, fin - sta, f);
            if (realSize != fin - sta) {
                MNN_ERROR("Write %s error\n", filePath);
                fclose(f);
                return false;
            }
        }
    }
    fclose(f);
    return true;
}

bool FileLoader::merge(AutoStorage<uint8_t>& buffer) {
    if (mTotalSize == 0 || mTotalSize > INT_MAX) {
        return false;
    }
    buffer.reset((int)mTotalSize);
    if (buffer.get() == nullptr) {
        MNN_PRINT("Memory Alloc Failed\n");
        return false;
    }
    auto dst   = buffer.get();
    int offset = 0;
    for (auto iter : mBlocks) {
        ::memcpy(dst + offset, iter.second, iter.first);
        offset += iter.first;
    }
    return true;
}

void FileLoader::_init() {
    if (mInited) {
        return;
    }
    mInited = true;
    const char* file = mFilePath.c_str();
    if (mFilePath.compare(0, sizeof(kRegionPrefix) - 1, kRegionPrefix) == 0) {
        file += sizeof(kRegionPrefix) - 1;
        if (!readRegionNumber(file, mRegionOffset) || !readRegionNumber(file, mRegionLength) || *file == '\0' ||
            mRegionOffset > INT64_MAX - mRegionLength || strncmp(file, kRegionPrefix, sizeof(kRegionPrefix) - 1) == 0) {
            return;
        }
    }
    if (*file != '\0') {
        mFile = _OpenFile(file, true);
    }
    if (mFile && mRegionLength >= 0) {
        bool valid = seekFile(mFile, 0, SEEK_END) == 0;
        const int64_t fileSize = tellFile(mFile);
        valid = valid && fileSize >= 0 && mRegionOffset <= fileSize && mRegionLength <= fileSize - mRegionOffset &&
                seekFile(mFile, mRegionOffset, SEEK_SET) == 0;
        if (!valid) {
            fclose(mFile);
            mFile = nullptr;
        }
    }
    if (nullptr == mFile) {
        MNN_ERROR("Can't open file:%s\n", mFilePath.c_str());
    }
}
int FileLoader::offset(int64_t offset) {
    _init();
    if (nullptr == mFile || offset < 0 || offset > INT64_MAX - mRegionOffset ||
        (mRegionLength >= 0 && offset > mRegionLength)) {
        return -1;
    }
    const int result = seekFile(mFile, mRegionOffset + offset, SEEK_SET);
    if (result == 0) {
        mPosition = offset;
    }
    return result;
}

bool FileLoader::read(char* buffer, int64_t size) {
    _init();
    if (nullptr == mFile || size < 0 || static_cast<uint64_t>(size) > SIZE_MAX ||
        (mRegionLength >= 0 && size > mRegionLength - mPosition)) {
        return false;
    }
    bool error = false;
    const size_t count = readFile(mFile, buffer, static_cast<size_t>(size), error);
    mPosition += count;
    return !error && count == static_cast<uint64_t>(size);
}

} // namespace MNN
