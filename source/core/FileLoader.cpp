//
//  FileLoader.cpp
//  MNN
//
//  Created by MNN on 2019/07/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/FileLoader.hpp"
namespace MNN {
FileLoader::FileLoader(const char* file) {
    mFile = fopen(file, "rb");
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

} // namespace MNN
