//
//  FileLoader.cpp
//  MNN
//
//  Created by MNN on 2019/07/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/FileLoader.hpp"
#if defined(_MSC_VER)
#include "Windows.h"
#endif
namespace MNN {
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
    _init();
    if (nullptr == mFile) {
        return false;
    }
    auto block = MNNMemoryAllocAlign(gCacheSize, MNN_MEMORY_ALIGN_DEFAULT);
    if (nullptr == block) {
        MNN_PRINT("Memory Alloc Failed\n");
        return false;
    }
    auto size  = fread(block, 1, gCacheSize, mFile);
    mTotalSize = size;
    mBlocks.push_back(std::make_pair(size, block));

    while (size == gCacheSize) {
        block = MNNMemoryAllocAlign(gCacheSize, MNN_MEMORY_ALIGN_DEFAULT);
        if (nullptr == block) {
            MNN_PRINT("Memory Alloc Failed\n");
            return false;
        }
        size = fread(block, 1, gCacheSize, mFile);
        if (size > gCacheSize) {
            MNN_PRINT("Read file Error\n");
            MNNMemoryFreeAlign(block);
            return false;
        }
        mTotalSize += size;
        mBlocks.push_back(std::make_pair(size, block));
    }

    if (ferror(mFile)) {
        return false;
    }
    return true;
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
    if (!mFilePath.empty()) {
        mFile = _OpenFile(mFilePath.c_str(), true);
    }
    if (nullptr == mFile) {
        MNN_ERROR("Can't open file:%s\n", mFilePath.c_str());
    }
}
int FileLoader::offset(int64_t offset) {
    _init();
    if (nullptr == mFile) {
        return 0;
    }
    return fseek(mFile, offset, SEEK_SET);
}

bool FileLoader::read(char* buffer, int64_t size) {
    _init();
    if (nullptr == mFile) {
        return false;
    }
    return fread(buffer, 1, size, mFile) == size;
}

} // namespace MNN
