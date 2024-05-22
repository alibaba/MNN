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

#include "core/AutoStorage.h"
namespace MNN {
class MNN_PUBLIC FileLoader {
public:
    FileLoader(const char* file, bool init = false);

    ~FileLoader();

    bool read();
    
    static bool write(const char* filePath, std::pair<const void*, size_t> cacheInfo);

    bool valid() const {
        return mFile != nullptr;
    }
    inline size_t size() const {
        return mTotalSize;
    }

    bool merge(AutoStorage<uint8_t>& buffer);

    int offset(int64_t offset);

    bool read(char* buffer, int64_t size);
private:
    void _init();
    std::vector<std::pair<size_t, void*>> mBlocks;
    FILE* mFile                 = nullptr;
    static const int gCacheSize = 4096;
    size_t mTotalSize           = 0;
    std::string mFilePath;
    bool mInited = false;
};
} // namespace MNN
#endif
