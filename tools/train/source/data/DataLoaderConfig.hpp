//
//  DataLoaderConfig.hpp
//  MNN
//
//  Created by MNN on 2019/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DataLoaderConfig_hpp
#define DataLoaderConfig_hpp
#include <MNN/MNNDefine.h>
namespace MNN {
namespace Train {

class MNN_PUBLIC DataLoaderConfig {
public:
    DataLoaderConfig() = default;
    DataLoaderConfig(size_t batchSize, size_t nWorkers = 0) : batchSize(batchSize), numWorkers(nWorkers) {
    }

    size_t batchSize  = 1;
    size_t numWorkers = 0;
    size_t numJobs    = numWorkers * 2;
    bool dropLast     = false;
};

} // namespace Train
} // namespace MNN

#endif // DataLoaderConfig
