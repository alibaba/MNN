//
//  FileLoader.hpp
//  MNN
//
//  Created by MNN on 2019/07/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <vector>
#include "core/AutoStorage.h"
namespace MNN {
class MNN_PUBLIC FileLoader {
public:
    FileLoader(const char* file);

    ~FileLoader();

    bool read();

    bool valid() const {
        return mFile != nullptr;
    }
    inline size_t size() const {
        return mTotalSize;
    }

    bool merge(AutoStorage<uint8_t>& buffer);

private:
    std::vector<std::pair<size_t, void*>> mBlocks;
    FILE* mFile                 = nullptr;
    static const int gCacheSize = 4096;
    size_t mTotalSize           = 0;
};
} // namespace MNN
