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
FileLoader::FileLoader(const char* file) {
#if defined(_MSC_VER)
    wchar_t wFilename[1024];
    if (0 == MultiByteToWideChar(CP_ACP, 0, file, -1, wFilename, sizeof(wFilename))) {
        mFile = nullptr;
        return;
    }
#if _MSC_VER >= 1400
    if (0 != _wfopen_s(&mFile, wFilename, L"rb")) {
        mFile = nullptr;
        return;
    }
#else
    mFile = _wfopen(wFilename, L"rb");
#endif
#else
    mFile = fopen(file, "rb");
#endif
    mFilePath = file;
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
    FILE* f = fopen(filePath, "wb");
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

int FileLoader::offset(int64_t offset) {
    return fseek(mFile, offset, SEEK_SET);
}

bool FileLoader::read(char* buffer, int64_t size) {
    return fread(buffer, 1, size, mFile) == size;
}

} // namespace MNN
